# PyTorch SFT for dense text

The SFT configs initialize from an existing i1 checkpoint, train on original
images grouped into rectangular buckets, and accept captions up to 1,024 tokens.
The VAE and T5Gemma encoder remain frozen; the DiT and its text adapter are trained.
The JSONL SFT path does not require TensorFlow; the existing TFRecord training path
is unchanged. For the current unified v4 dataset, follow the
[end-to-end production pipeline](#end-to-end-production-pipeline-gpu-captioning-to-sft).
The later [CPU/GPU preflight workflow](#preflight-cpu-preparation-gpu-checks-and-training-smoke-test)
provides reusable diagnostics and smoke tests.

## Prepare the training environment

The following setup targets Linux and a single server with eight A800 80 GB GPUs.
Use the same environment for caption auditing, training, and inference. These are
setup instructions, not a claim that full pretrained eight-GPU training has been
validated in this environment; the existing regression tests use tiny CPU models.

### 1. Check the host and create an isolated environment

```bash
cc --version

conda create -n i1_sft python=3.11 -y
conda activate i1_sft
python --version
which python
python -m pip install --upgrade pip
```

Use Conda to install Python 3.11 into this environment; a system-wide `python3.11`
installation is not required. The commands work from the existing `(base)` shell;
there is no need to deactivate it first. If you already deactivated `base` and
`python` is no longer found, run the same Conda commands above. Python can exist
only inside Conda environments on this node.

After activation, `python --version` should report 3.11.x and `which python` should
point into the `i1_sft` environment. If the environment already exists, skip
`conda create` and activate it. Do not also create a `venv` or use
`source ~/envs/i1_sft/bin/activate`; those belong to a different setup method.
Keep `i1_sft` activated for every installation and training command below.

The host also needs an NVIDIA driver compatible with the selected CUDA wheel and
a working C/C++ compiler for `torch.compile`.
GPU visibility, topology, shared memory, compilation and communication are
checked together in the preflight section below. On a CPU-only preparation host,
skip the GPU-specific host requirements until moving to the GPU server.

The wheel below uses CUDA 12.8. Installing it does not update the host driver.
Do not rely only on the CUDA version printed by `nvidia-smi`: check the
[NVIDIA driver compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
including its restrictions for PTX/JIT compilation on older drivers. Choose a
driver that supports CUDA 12.8 compilation workloads, or select a compatible wheel
from the [official PyTorch version table](https://pytorch.org/get-started/previous-versions/).

### 2. Install the Python dependencies

```bash
python -m pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  "numpy==1.26.4" pillow tqdm \
  "transformers==4.57.1" "diffusers==0.35.1" \
  accelerate safetensors sentencepiece "huggingface_hub>=0.34,<1.0" \
  "pyarrow==22.0.0"
python -m pip check
```

PyTorch 2.9.1 is pinned here to avoid accidentally installing a CPU-only or changing
major-version environment. The other version pins follow the repository's training
instructions. The model uses PyTorch scaled-dot-product attention; a separate
`flash-attn` installation is not required. TensorFlow, torchvision, and torchaudio
are not needed for the JSONL or Parquet SFT paths. PyArrow is required for the
prepared GPT-Image Parquet shards.

Install `wandb` only if enabling `config.wandb.log_wandb`:

```bash
python -m pip install wandb
wandb login
```

For the original TFRecord input path only, also install:

```bash
python -m pip install "tensorflow-cpu==2.19.0" "tensorflow-datasets==4.9.9"
python -m pip install --no-deps "tensorflow-metadata==1.16.1"
python -m pip check
```

### 3. Configure model access and storage

Choose writable cache directories with sufficient free space, preferably on fast
local storage. Set these variables before downloading or launching any process,
and keep them consistent in later shells and batch jobs:

```bash
# Replace /path/to/fast-disk with an actual writable storage location.
export HF_HOME=/path/to/fast-disk/i1/huggingface
export TORCHINDUCTOR_CACHE_DIR=/path/to/fast-disk/i1/torchinductor
export TRITON_CACHE_DIR=/path/to/fast-disk/i1/triton
mkdir -p "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

hf auth login
hf auth whoami
```

Use an account with access to
[google/t5gemma-2b-2b-ul2-it](https://huggingface.co/google/t5gemma-2b-2b-ul2-it)
and [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev).
Complete any required access/license acceptance on their model pages before
downloading; logging in alone does not grant gated-model access.

Prefetch the text encoder/tokenizer and VAE once, rather than having eight workers
start the downloads concurrently. Download the i1 checkpoint for the intended run:

```bash
hf download google/t5gemma-2b-2b-ul2-it
hf download black-forest-labs/FLUX.2-dev --include "vae/*"

# Download either checkpoint, or both if running both resolutions.
hf download zlab-princeton/i1-3B 512_resolution_checkpoint_torch.pt \
  --local-dir /path/to/checkpoints/i1-3B
hf download zlab-princeton/i1-3B 1024_resolution_checkpoint_torch.pt \
  --local-dir /path/to/checkpoints/i1-3B
```

Point `--init_from` at the downloaded `.pt` file. Allocate separate space for the
source images, output checkpoints, and retained checkpoint copies; training
checkpoints include optimizer and EMA state and are larger than inference weights.
Keep the full repository checkout: inference imports shared helpers from
`torch_train` and should not be run from a copied `generate.py` alone.

## Data formats and image policy

GPT-Image-200K is stored in `/nfs_yaoyuan/liuxinyu/GPT-Image-200K/out_images`, with
100 `shard_*/shard_*.parquet` files and 200,000 rows containing `id`, `prompt`,
`size` and `image_bytes`. This corpus has incorrect `size` metadata: use the
corrected-index workflow below before training. Do not use stale absolute
`image_path` values from generation manifests or extract the entire 446 GB corpus.

The generic alternative is a JSONL manifest:

```json
{"image_path":"images/poster.png","caption":"A poster with the heading ...","width":1600,"height":1200}
```

`prompt` is accepted instead of `caption`; captions must be nonempty. Relative
image paths resolve against the manifest directory unless `input.image_root` is
set. Optional dimensions must describe the image after EXIF orientation. Corrected
Parquet indexes contain `id`, `caption`, `width`, `height`, `parquet_path`,
`row_group` and `row_in_group`; relative Parquet paths resolve against the index.

The default transform fits the entire image into its nearest eligible bucket and
adds white padding, which receives normal image loss. It does not stretch, flip
or randomly crop images. `resize_mode="crop"` explicitly enables center cropping;
captions must then match the visible content. Source area and optional minimum
short-side filters run before assignment; upscaling is disabled by default. Pixel
area alone does not establish text readability. Square TFRecords cannot restore
text removed by an earlier crop. See the resolution/batching reference below for
bucket geometry and sampling details.

## End-to-end production pipeline: GPU captioning to SFT

This section is the production runbook for the unified dense-text v4 dataset. It
covers the complete handoff between three machines/stages:

1. an eight-GPU machine selects and captions unified Parquet rows;
2. a CPU machine exports captioned Parquet, audits every caption and image, and
   builds an accepted-only pixel cache;
3. an eight-A800 80 GB machine fine-tunes i1 at 1024 resolution and reports to
   Weights & Biases online.

The stages communicate through NFS and Ceph. Do not copy partial JSONL files
between machines, and do not start a downstream stage before its upstream
completion artifact exists.

### Fixed production paths and expected result

```bash
export DENSE_PROJECT=/cephfs/liuxinyu/DenseText-Project
export UNIFIED_SOURCE=/nfs_yaoyuan/liuxinyu/textdense_primary_english_unified_v2
export CAPTION_OUTPUT=/nfs_yaoyuan/liuxinyu/textdense_primary_english_captioned_v4
export CAPTION_CACHE="$DENSE_PROJECT/artifacts/textdense_primary_english_captioned_v4_precompute/cache_1024"
export SFT_MANIFEST="$CAPTION_CACHE/cache.jsonl"
export SFT_CONFIG="$DENSE_PROJECT/i1/torch_train/configs/sft_1024_captioned.py"
export SFT_INIT=/cephfs/liuxinyu/.cache/data_juicer/models/i1-3B/1024_resolution_checkpoint_torch.pt
export SFT_WORKDIR="$DENSE_PROJECT/artifacts/sft_densetext_captioned_v4_1024"
# Private task-manager credentials used to release the GPU VM after completion.
export GPU_COMPLETION_CONFIG="$DENSE_PROJECT/local_captioning/completion_config.json"
```

The completed v4 preparation selected 202,812 rows from four sources. The CPU
audit accepted 199,095 rows and rejected 3,717: 1,325 captions over the 1,024-token
T5Gemma limit, 1,314 images with a side over 4,096 pixels, and 1,078 images over
4,096 squared pixels. The accepted cache has 45 populated buckets and resolves to
6,245 optimizer steps for one epoch at global batch 32.

If the source selection, caption model/prompt, token limit, image limits, bucket
frontier, or transform policy changes, use new caption/cache/work directories.
The manifest and transform fingerprints intentionally reject incompatible reuse.

### Stage 1: prepare and caption on the GPU machine

Use the existing `qwen` environment. The caption launcher starts one independent
Gemma replica per visible GPU (`TP_SIZE=1`) and writes one resumable part file per
worker. Keep the same GPU count and GPU ordering when resuming because part
ownership depends on the worker count.

First create or validate the deterministic input selection. This is a metadata
and UID scan and does not consume GPU inference:

```bash
cd "$DENSE_PROJECT/local_captioning"

/root/miniconda3/envs/qwen/bin/python run_unified_production.py \
  --stage prepare \
  --dataset "$UNIFIED_SOURCE" \
  --output-dir "$CAPTION_OUTPUT" \
  --sources danqing monet paper2fig100k chartgalaxy_real
```

Then launch captioning on eight GPUs:

```bash
cd "$DENSE_PROJECT/local_captioning"

export HF_HUB_CACHE=/cephfs/liuxinyu/.cache/data_juicer/models
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

GPUS=0,1,2,3,4,5,6,7 \
/root/miniconda3/envs/qwen/bin/python run_unified_production.py \
  --stage caption \
  --dataset "$UNIFIED_SOURCE" \
  --output-dir "$CAPTION_OUTPUT" \
  --sources danqing monet paper2fig100k chartgalaxy_real \
  --completion-config "$GPU_COMPLETION_CONFIG"
```

The optional completion configuration contains GPU-task-manager credentials. Keep
it outside the repository with mode 0600, and omit `--completion-config` when
automatic VM release is not wanted. A successful notified run releases the
configured GPU VMs only after all workers finish and `captions.jsonl` has been
atomically merged.

The command is resumable. Relaunch the same command after interruption; completed
IDs in the eight `captions.part-*-of-00008.jsonl` files are skipped. Do not alter
the prompt, OCR injection, model settings, or worker count inside one output
directory.

Completion evidence:

```bash
test -f "$CAPTION_OUTPUT/captions.jsonl"
wc -l "$CAPTION_OUTPUT/caption_input.jsonl" "$CAPTION_OUTPUT/captions.jsonl"

/root/miniconda3/envs/qwen/bin/python - <<'PY'
import collections
import json
import os
from pathlib import Path

path = Path(os.environ["CAPTION_OUTPUT"]) / "captions.jsonl"
statuses = collections.Counter()
with path.open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            statuses[json.loads(line).get("caption_status")] += 1
print(dict(statuses))
PY
```

For the frozen v4 run, both files contain 202,812 rows and the status count is
`{"ok": 202812}`. `ok` means that generation returned a nonempty caption; the CPU
stage applies the authoritative training-token and image checks.

### Stage 2: export, audit, and build the cache on the CPU machine

Run the fused CPU stage only after `captions.jsonl` exists. It first exports
captioned Parquet with original compressed image bytes to NFS, then invokes the
accepted-only cache builder with the T5Gemma tokenizer. Both phases are resumable.

```bash
cd "$DENSE_PROJECT/local_captioning"

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_CACHE=/cephfs/liuxinyu/.cache/data_juicer/models \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
/root/miniconda3/envs/i1_sft/bin/python run_unified_cpu.py \
  --dataset "$UNIFIED_SOURCE" \
  --output-dir "$CAPTION_OUTPUT" \
  --cache-output-dir "$CAPTION_CACHE" \
  --resolution 1024 \
  --cache-workers 8 \
  --token-limit 1024
```

The NFS export retains every captioned row. The Ceph training cache contains only
accepted rows. Captions over 1,024 T5Gemma tokens are not truncated: they are
written to `rejected.jsonl` with reason `caption_too_long` and omitted from
`cache.jsonl`. Broken images, inconsistent dimensions, oversized sources, and
ineligible bucket geometry are handled the same way with explicit reason codes.

Verify the completed summaries and manifest counts:

```bash
/root/miniconda3/envs/i1_sft/bin/python - <<'PY'
import json
import os
from pathlib import Path

caption_output = Path(os.environ["CAPTION_OUTPUT"])
cache = Path(os.environ["CAPTION_CACHE"])
export = json.loads((caption_output / "export_summary.json").read_text())
summary = json.loads((cache / "summary.json").read_text())
assert export["count"] == 202812
assert summary["status"] == "complete"
assert summary["inspected"] == 202812
assert summary["count"] == 199095
assert summary["rejected"] == 3717
assert summary["verified_all_written_bytes"] is True
print("export", export["count"], export["caption_status_counts"])
print("cache", summary["count"], "accepted;", summary["rejected"], "rejected")
print("reasons", summary["rejected_counts"])
print("tokens", summary["caption_tokens"])
PY

wc -l "$CAPTION_CACHE/cache.jsonl" "$CAPTION_CACHE/rejected.jsonl"
```

Do not train from `captions.jsonl`, the NFS Parquet directory, or
`rejected.jsonl`. The production training input is exactly
`$CAPTION_CACHE/cache.jsonl`; its records point into verified binary RGB cache
shards under `$CAPTION_CACHE/parts`.

### Stage 3: preflight the eight-A800 training machine

Activate `i1_sft`, restore the path variables from the first subsection, and
verify the checkpoint, cache, CUDA visibility, BF16 support, shared memory, and
free checkpoint space. The default eight-rank DataLoader can queue roughly 3 GiB
of float32 images; allocate about 16 GiB of `/dev/shm` rather than the common 64
MiB container default.

```bash
conda activate i1_sft
cd "$DENSE_PROJECT/i1/torch_train"

test -f "$SFT_MANIFEST"
test -f "$SFT_INIT"
test -f "$CAPTION_CACHE/summary.json"
test -f "$GPU_COMPLETION_CONFIG"
test "$(stat -c '%a' "$GPU_COMPLETION_CONFIG")" = 600
mkdir -p "$SFT_WORKDIR"
df -h "$SFT_WORKDIR" /dev/shm

python - <<'PY'
import torch

assert torch.cuda.device_count() == 8, torch.cuda.device_count()
for index in range(8):
    props = torch.cuda.get_device_properties(index)
    with torch.cuda.device(index):
        assert torch.cuda.is_bf16_supported(), index
    print(index, props.name, f"{props.total_memory / 2**30:.1f} GiB")
PY
```

Run the NCCL check and all-bucket compiled smoke test in the GPU preflight below
before a new production run. The conservative validated starting topology is
FSDP-8, tensor parallelism 1, global batch 32, and four accumulation passes, which
gives one image per GPU per microbatch. `grad_accum=2` or `1` retains global batch
32 but increases the microbatch to two or four images per GPU; use it only after
an all-bucket memory/throughput smoke test in a separate workdir.

### Stage 4: start training with online Weights & Biases

Use a new workdir. An existing `checkpoint.pt` takes precedence and causes a
resume, so explicitly stop if the intended fresh directory already contains one.
Run inside a durable scheduler allocation or `tmux` session.

The config enables W&B project `DenseText-SFT` with run name
`i1-1024-captioned-v4`. Once per machine, install and authenticate it with
`python -m pip install wandb` and `wandb login`. For unattended jobs, provide
`WANDB_API_KEY` through the cluster secret manager, never through repository
files or logs.

```bash
conda activate i1_sft
cd "$DENSE_PROJECT/i1/torch_train"

if test -e "$SFT_WORKDIR/checkpoint.pt"; then
  echo "Refusing a fresh launch: $SFT_WORKDIR/checkpoint.pt already exists" >&2
  exit 1
fi

mkdir -p "$SFT_WORKDIR" "$SFT_WORKDIR/wandb"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_CACHE=/cephfs/liuxinyu/.cache/data_juicer/models
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export WANDB_MODE=online
export WANDB_PROJECT=DenseText-SFT
export WANDB_DIR="$SFT_WORKDIR/wandb"

set -o pipefail
/root/miniconda3/envs/i1_sft/bin/torchrun \
  --standalone \
  --nproc_per_node=8 \
  -m training.main \
  --config "$SFT_CONFIG" \
  --manifest "$SFT_MANIFEST" \
  --init_from "$SFT_INIT" \
  --workdir "$SFT_WORKDIR" \
  --fsdp 8 \
  --batch_size 32 \
  --grad_accum 4 \
  --no_compile \
  --completion-config "$GPU_COMPLETION_CONFIG" \
  2>&1 | tee "$SFT_WORKDIR/train.log"
```

Rank zero prints the online W&B URL; the other ranks do not create duplicate
runs. Expected startup values are 199,095 images, 45 buckets, global batch 32,
microbatch 1 per rank, four accumulation passes, and 6,245 optimizer steps.
`--no_compile` avoids a PyTorch 2.9.1 TorchInductor stride assertion when compiled
backward graphs move between rectangular buckets. It retains eager-mode training
semantics at lower throughput; remove it only after the all-bucket compiled smoke
test passes with the installed PyTorch build. After the final optimizer step and
checkpoint finish on every rank, rank zero flushes pending W&B logs and posts the
task-finished notification using
`/cephfs/liuxinyu/DenseText-Project/local_captioning/completion_config.json`.
An exception, signal, failed rank, or incomplete checkpoint never reaches this
notification path. A notification failure is retried three times and then makes
the otherwise completed command exit nonzero instead of silently leaving the VM
running.

Monitor both the durable log and W&B:

```bash
tail -f "$SFT_WORKDIR/train.log"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv -l 5
```

Evaluate held-out rendering prompts at retained checkpoints; healthy loss and
gradient metrics alone do not establish generation quality.

### Stage 5: resume an interrupted training run

Restore the fixed path variables and Stage 4 environment exports, keeping the
same config, manifest, topology, batch settings, seed, and workdir. Then replace
`--init_from` with `--resume`:

```bash
cd "$DENSE_PROJECT/i1/torch_train"

test -f "$SFT_WORKDIR/checkpoint.pt"
set -o pipefail
/root/miniconda3/envs/i1_sft/bin/torchrun \
  --standalone \
  --nproc_per_node=8 \
  -m training.main \
  --config "$SFT_CONFIG" \
  --manifest "$SFT_MANIFEST" \
  --resume "$SFT_WORKDIR/checkpoint.pt" \
  --workdir "$SFT_WORKDIR" \
  --fsdp 8 \
  --batch_size 32 \
  --grad_accum 4 \
  --no_compile \
  --completion-config "$GPU_COMPLETION_CONFIG" \
  2>&1 | tee -a "$SFT_WORKDIR/train.log"
```

This restores model, optimizer, EMA, step, and sampler position. The completion
notification is sent only when the resumed run reaches its configured final step.
Noise and dropout RNG are not restored bit-for-bit. The current trainer starts a
new W&B run after process restart because it does not persist the W&B run ID;
checkpoint resumption is unaffected.

### Destructive smoke test for automatic VM shutdown

`smoke_test_completion_shutdown.sh` runs three eager-mode steps with the real
production model, manifest, eight-rank FSDP topology, and completion credentials.
It writes to a separate timestamped workdir, saves a final checkpoint, and then
calls the real task-finished endpoint. The configured VM is expected to power off.
The script refuses to start when a GPU compute process is present and requires an
explicit destructive-operation acknowledgement:

```bash
cd "$DENSE_PROJECT/i1/torch_train"
CONFIRM_GPU_SHUTDOWN=YES ./smoke_test_completion_shutdown.sh
```

Do not run this alongside production training. After the VM is started again,
inspect the durable Ceph artifacts from the most recent test:

```bash
SFT_SMOKE_WORKDIR=$(cat \
  "$DENSE_PROJECT/artifacts/sft_completion_shutdown_smoke/latest_run.txt")
test -f "$SFT_SMOKE_WORKDIR/checkpoint.pt"
grep -E 'saved checkpoint|task-finished notification acknowledged' \
  "$SFT_SMOKE_WORKDIR/train.log"
```

Success requires a final-step checkpoint, an HTTP acknowledgement in the log,
and the VM becoming unavailable. The script does not reuse or modify the
production checkpoint. Override `SFT_SMOKE_STEPS` only when more than three steps
are needed.

An already-running trainer has imported its code and cannot acquire this update
in place. To switch it over, wait for a `saved checkpoint` line in `train.log`,
send one `Ctrl-C` to the foreground `torchrun` process (or `kill -INT` to its
launcher PID), and wait until all `training.main` workers exit. Then use the
Stage 5 resume command above, including
`--completion-config "$GPU_COMPLETION_CONFIG"`. Interrupting between checkpoints
discards the steps since the most recent saved checkpoint but does not corrupt
that atomic checkpoint.

## Preflight: CPU preparation, GPU checks, and training smoke test

Follow this section in order before a full SFT run. CPU preparation produces a
validated training input; GPU checks exercise the actual pretrained models,
distributed execution, checkpoint/resume, and inference. Neither successful CPU
checks nor a falling smoke-test loss establishes dense-text generation quality.

| Phase | Where | Completion evidence |
| --- | --- | --- |
| CPU 1–2: environment, audit, cache | CPU host | Passing tests and accepted index/caption/cache reports. |
| CPU 3–4: review, split, optional VAE check | CPU host | Reviewed text preservation and a fixed held-out set. |
| GPU 1–2: host and communication | Eight-GPU server | CUDA/BF16/compile and NCCL checks pass. |
| GPU 3–5: bucket coverage and training | Eight-GPU server | Encoder checks, all-bucket training, checkpoint and resume pass. |
| GPU 6: inference and profiling | GPU server | Valid held-out samples and representative throughput/memory measurements. |

### 0. Set paths and choose the resolution

Activate the environment from the setup section. All commands in this section run
from `i1/torch_train` unless a command explicitly changes directories. Replace
placeholder paths once, and restore these variables when moving to the GPU host:

```bash
conda activate i1_sft
export I1_ROOT=/cephfs/liuxinyu/DenseText-Project/i1
export GPT_IMAGE_200K=/nfs_yaoyuan/liuxinyu/GPT-Image-200K/out_images
export SFT_AUDIT=/path/to/sft_precompute
export SFT_RESOLUTION=1024
export SFT_IMAGE_INDEX="$SFT_AUDIT/corrected_images.jsonl"
export SFT_CACHE="$SFT_AUDIT/cache_${SFT_RESOLUTION}"
export SFT_CONFIG="$I1_ROOT/torch_train/configs/sft_${SFT_RESOLUTION}.py"
export SFT_INIT=/path/to/checkpoints/i1-3B/1024_resolution_checkpoint_torch.pt
export SFT_WORKERS=8
export SFT_WORKDIR=/path/to/new_sft_run
mkdir -p "$SFT_AUDIT"
cd "$I1_ROOT/torch_train"
```

For a 512 run, set `SFT_RESOLUTION=512`, update `SFT_CACHE` and `SFT_CONFIG`, and use
the 512 initialization checkpoint. Build separate pixel caches for each resolution.
Preserve the environment, bucket configuration and Pillow version used to build a
cache. The existing run's location is listed in the artifact appendix; inspect its
`status.json` before starting another pipeline against the same output directory.

### CPU 1. Check the environment and available resources

```bash
python -m pip check
python - <<'PY'
import torch, transformers, diffusers, numpy, PIL, pyarrow
from transformers import T5GemmaModel
from diffusers import AutoencoderKL
from torch.distributed.fsdp import fully_shard
print("torch", torch.__version__, "transformers", transformers.__version__)
print("diffusers", diffusers.__version__, "Pillow", PIL.__version__)
print("NumPy", numpy.__version__, "PyArrow", pyarrow.__version__)
PY
python -m unittest discover -s "$I1_ROOT/torch_train/tests" -v
python -m pip freeze > "$SFT_AUDIT/environment.txt"
df -h "$SFT_AUDIT" /dev/shm
```

The regression tests use tiny CPU models and synthetic inputs; they do not load
the production checkpoint or prove CUDA/FSDP correctness. They cover geometry,
caption overflow, cache parity/integrity, sampling, checkpoint round trips, and
CPU distributed communication.

Check the job/container CPU quota rather than relying only on the host's reported
CPU count. The current CPU container has an eight-core quota and 64 MiB `/dev/shm`.
Use eight cache workers as a starting point. Budget approximately **622 GB at
1024**, or **155 GB at 512**, for uint8 pixels, plus metadata and inspection images.
The 446 GB source corpus remains in place. Do not store normalized float32 pixels:
they require four times the pixel-cache space.

### CPU 2. Build and audit the input, then precompute pixels

Use **either** the automated pipeline **or** the individual commands below. Do not
run both simultaneously against the same outputs. Prefetch the tokenizer using the
setup section first; the pipeline runs Hugging Face operations offline.

**Automated path:**

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m datasets.run_cpu_precompute \
  --source "$GPT_IMAGE_200K" --output_dir "$SFT_AUDIT" \
  --resolution "$SFT_RESOLUTION" --workers "$SFT_WORKERS"
```

It runs index construction, caption auditing, pixel precomputation, cache/duplicate
checks, concurrent-reader benchmarking, and a 150-example gallery. Each stage has
a log under `SFT_AUDIT`; `status.json` must report `complete` before treating the
pipeline as finished. It does not perform manual visual review, create a held-out
split, run the VAE diagnostic, or run any GPU training.

**Individual stages / recovery:** run the required stage rather than restarting an
already successful full decode. The pipeline itself starts with index construction
when relaunched normally; the pixel exporter can reuse verified completed parts.

For the production unified-caption export, use the fused caption/image audit instead of
the GPT-Image-specific corrected-index pipeline:

```bash
cd /cephfs/liuxinyu/DenseText-Project/local_captioning
python run_unified_cpu.py --cache-workers 8
```

That command first performs the resumable captioned-Parquet export to NFS, then invokes
the fused audit/cache pass below in the `i1_sft` environment. To recover or rerun only the
second phase after a completed export, invoke it directly:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_HUB_CACHE=/cephfs/liuxinyu/.cache/data_juicer/models \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m datasets.precompute_captioned \
  --source /nfs_yaoyuan/liuxinyu/textdense_primary_english_captioned_v4 \
  --output-dir /cephfs/liuxinyu/DenseText-Project/artifacts/\
textdense_primary_english_captioned_v4_precompute/cache_1024 \
  --resolution 1024 --workers 8 --token-limit 1024
```

Train this cache with `configs/sft_1024_captioned.py`. Its 45-shape, native-32px
frontier is deliberately separate from `sft_1024.py`, preserving compatibility with the
existing GPT-Image-200K cache's 25-shape transform fingerprint.

This pass writes only accepted records to `cache.jsonl`. Acceptance requires an `ok`
caption status, a nonempty caption no longer than 1,024 T5Gemma tokens, a decodable image,
matching EXIF-oriented declared dimensions, at most 4,096 pixels on either side and at
most 4,096² source pixels, and an eligible training bucket. The size limits are checked
before full image decoding and can be overridden with `--max-source-side` and
`--max-source-pixels`. All failures are retained in `rejected.jsonl` with reason codes.
`bucket_plan.json` is a metadata-only
coverage/storage report generated before transformation; `summary.json` records final
accepted counts, token distribution, buckets, padding, and checksums. Completed parts are
checksum-validated and reused on restart.

```bash
# a. Decode all original images and repair their dimension metadata.
python -m datasets.build_image_index \
  --manifest "$GPT_IMAGE_200K" --output "$SFT_IMAGE_INDEX" \
  --workers "$SFT_WORKERS"

# b. Audit all retained captions with the actual tokenizer; no encoder weights.
python -m datasets.inspect_captions \
  --manifest "$SFT_IMAGE_INDEX" --token_len 1024 \
  --report "$SFT_AUDIT/caption_audit.json"

# c. Apply the exact training transform to every eligible image and cache it.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m datasets.precompute_images \
  --manifest "$SFT_IMAGE_INDEX" --output_dir "$SFT_CACHE" \
  --resolution "$SFT_RESOLUTION" --workers "$SFT_WORKERS"

# d. Check cache parity and identify duplicate candidates.
python -m datasets.check_precompute \
  --manifest "$SFT_CACHE/cache.jsonl" --resolution "$SFT_RESOLUTION" \
  --report "$SFT_AUDIT/cache_checks.json"

# e. Compare source versus cache loading with 1, 4, and 8 concurrent readers.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m datasets.benchmark_images \
  --source_manifest "$SFT_IMAGE_INDEX" --cache_manifest "$SFT_CACHE/cache.jsonl" \
  --resolution "$SFT_RESOLUTION" --report "$SFT_AUDIT/io_benchmark.json"

# f. Produce the visual-review material used in CPU 3.
python -m datasets.build_inspection_gallery \
  --manifest "$SFT_IMAGE_INDEX" --output_dir "$SFT_AUDIT/gallery" --count 150
```

Review the outputs before proceeding:

| Output | Required check |
| --- | --- |
| `corrected_images.report.json` | Every shard was processed; inspect corrected dimensions and all excluded records. |
| `caption_audit.json` | Zero unintended overflows; inspect the longest captions and decoded tokenizer round trips. |
| `cache_<resolution>/summary.json` | Expected retained/filter counts and bucket counts; `verified_all_written_bytes` is true. |
| `cache_checks.json` | Pixel parity passed; review duplicate IDs, exact image/caption duplicates, and near-duplicate candidates. |
| `io_benchmark.json` | Record throughput and tail latency; understand the measurement limits below. |
| `gallery/index.html` | Material for human review, not an automatic quality approval. |

The index builder fully decodes images, applies EXIF orientation, repairs sizes,
and excludes undecodable images or empty captions. It aborts publication on a
worker failure, unreadable row group, or missing required columns. The resulting
JSONL references the original Parquet rows; it does not copy image bytes. Keep the
source shards unchanged for later parity checks and galleries.

`datasets.validate_images --manifest "$GPT_IMAGE_200K" --workers "$SFT_WORKERS"
--report "$SFT_AUDIT/image_validation.json"` remains available as a strict audit of
the original metadata: it exits nonzero on any issue. It repeats the full decode,
so it is unnecessary when building a fresh corrected index. Merely constructing
`BucketedImages` does not decode every image; the index and exporter perform that
work explicitly.

The pixel exporter stores the final RGB transform in per-bucket binary shards of
at most 256 MiB, unless one image alone is larger. Every shard is read back and
SHA256-checked against the bytes produced by the online transform. The final
`cache.jsonl` is published only after every part succeeds. The independent checker
compares online/cached pixels and normalized training tensors on a deterministic
sample covering every populated bucket; it does not independently recompute every
cached image. Resume reuses verified parts only with matching inputs/settings.
Changed sources or transforms require a new output directory.
Parts whose images are all filtered retain their exclusion counts and can be
resumed. If the entire input is filtered, export fails before publishing a cache
manifest. JSONL image paths honor `input.image_root` just as the training loader does.

Cached training avoids PNG decoding, resizing and padding. Float normalization,
frozen-VAE encoding, caption tokenization and frozen-text encoding remain online.
The loader checks the transform fingerprint and preserves original-source filtering
and bucket assignment. Keep `cache.jsonl` and `parts/` together when moving the cache.

Caption limits are token counts, not character counts. The configs use 1,024 tokens
and reject overflow. Resolve overflows before caching/training; do not silently
summarize away intended text. `caption_overflow="truncate"` is an explicit opt-in,
not the default. See the caption-conditioning notes below before changing the limit.

The CPU benchmark includes filesystem-cache effects and concurrent read/transform
work. It excludes DataLoader shared-memory queues, tokenization, GPU transfer and
model computation. Repeat it on the GPU host's actual storage path. With global
batch 32 and optimizer-step time `T` seconds, loading must sustain `32/T` images/s
with headroom; the GPU smoke run determines `T`.

### CPU 3. Review text preservation, duplicates, and evaluation separation

Open `gallery/index.html`. It shows originals, 512/1024 processed images, full
captions, token counts and assigned dimensions; its JSON sidecars preserve the
selection. Sparse buckets receive fewer samples rather than duplicated examples.
Selection covers the populated 512 buckets. Each source is checked separately
against the 1024 training policy; ineligible sources are marked as excluded instead
of showing an upscaled 1024 preview.
Inspect 100–200 examples across all populated shapes, including small text, long
documents, numbers/formulas, extreme aspect ratios and text at image edges. Use
native pixels and enlarged text crops; thumbnails can hide missing characters.
Check caption–image alignment and record rejected IDs and reasons.

White padding preserves image extent but cannot prevent downsampling from erasing
small characters. Avoid stretching or cropping captioned text. If readability is
already poor after resizing, revise resolution/data selection before training.

Review duplicate reports before fixing a held-out evaluation set. Exact image
hashes use decoded RGB pixels; identical captions are not automatically duplicate
images. The dHash screen is **non-exhaustive**, with dense-bin and candidate limits;
its matches require review. The tools do not automatically remove records or make
a split. Keep duplicates/related variants within one split, and include different
text densities and aspects in evaluation. Do not tune on held-out examples.

Write the approved training/evaluation manifests **beside `cache.jsonl`** to retain
relative `parts/` references, or rewrite those references as absolute paths. Set:

```bash
export SFT_TRAIN_MANIFEST="$SFT_CACHE/train.jsonl"
export SFT_EVAL_PROMPTS="$SFT_AUDIT/held_out_prompts.txt"
```

These are files you create after review, not automatic pipeline outputs. Preserve
full cache-record fields when filtering a manifest. The prompt file contains one
held-out prompt per line; include long prompts with exact intended strings.

### CPU 4. Optional small VAE diagnostic and deferred compute

If the frozen VAE weights are cached, run a small deterministic reconstruction
check on CPU. This is useful before GPU allocation but can be slow:

```bash
HF_HUB_OFFLINE=1 python -m datasets.inspect_vae_reconstructions \
  --manifest "$SFT_IMAGE_INDEX" --output_dir "$SFT_AUDIT/vae_cpu" --count 2
```

The current helper is CPU-only and checks up to two populated buckets at **each**
resolution, using one example per selected bucket. It saves processed/reconstructed
PNGs, finiteness/normalization-round-trip results and pixel metrics. Inspect the
text: PSNR is not a transcription metric. Use the GPU probe below for broader
coverage. Diagnostics use the VAE posterior's mode; training samples its posterior.

Do not precompute T5Gemma embeddings or VAE latents as a prerequisite. Their cache
readers are not implemented. Start with online encoding and profile on GPUs first.
The existing pixel cache already removes image decoding and resize costs. Details
and storage estimates for possible future embedding caches are below.

**CPU exit criteria:** accepted index/exclusions, no unintended caption overflow,
verified cache, reviewed visual examples/duplicates, and a fixed held-out split.
An automated `complete` status alone does not satisfy the manual-review criteria.

### GPU 1. Stage inputs and check CUDA, compilation, and shared memory

On the allocated eight-A800 server, activate the same environment and restore the
paths from step 0 plus `SFT_TRAIN_MANIFEST` and `SFT_EVAL_PROMPTS`. If moving the
cache, copy its complete directory and use paths valid on this host. Pre-stage the
VAE, tokenizer/text-encoder weights and matching i1 checkpoint; eight workers
should not begin by downloading the same model. Keep TorchInductor/Triton cache
directories writable. Repeat CPU imports/tests if this is a different environment.

```bash
nvidia-smi
nvidia-smi topo -m
cc --version
df -h /dev/shm
test -s "$SFT_TRAIN_MANIFEST"
test -s "$SFT_INIT"
python - <<'PY'
import torch
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8, "Expected eight visible GPUs."
@torch.compile
def compiled_op(x):
    return torch.nn.functional.silu(x @ x)
for index in range(8):
    with torch.cuda.device(index):
        assert torch.cuda.is_bf16_supported(), f"BF16 unavailable on GPU {index}"
        props = torch.cuda.get_device_properties(index)
        x = torch.randn(128, 128, device=f"cuda:{index}", dtype=torch.bfloat16)
        assert torch.isfinite(compiled_op(x)).all().item()
        torch.cuda.synchronize()
        print(index, props.name, f"{props.total_memory / 2**30:.1f} GiB", "passed")
PY
```

Confirm the expected NVLink topology and compatible driver. Resolve compile failures
before testing the compiled trainer; `config.compile=False` in a copied config is
an eager-mode diagnostic, not proof that the default compiled setup works.

The default eight-rank 1024 DataLoader can queue approximately **3 GiB of float32
images**, before active batches, pinned copies and other overhead. A 64 MiB
`/dev/shm` mount is insufficient. Allocate several GiB with headroom (for example,
16 GiB for this starting configuration) through the container/job launcher.
`config.input.num_workers=0` is a debugging fallback, not the production throughput
test. Confirm the actual target-host CPU quota, RAM, shared memory and storage rate.

### GPU 2. Verify communication between all eight ranks

```bash
i1_check_dir=$(mktemp -d)
cat > "$i1_check_dir/check_nccl.py" <<'PY'
import os
from datetime import timedelta
import torch
import torch.distributed as dist
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", timeout=timedelta(minutes=2))
rank, world = dist.get_rank(), dist.get_world_size()
assert world == 8
value = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
dist.all_reduce(value)
assert value.item() == 36.0
torch.cuda.synchronize()
if rank == 0:
    print("Eight-GPU NCCL all-reduce passed")
dist.destroy_process_group()
PY
torchrun --standalone --nproc_per_node=8 "$i1_check_dir/check_nccl.py"
rm "$i1_check_dir/check_nccl.py"
rmdir "$i1_check_dir"
```

This verifies launch/collectives, not communication bandwidth or full FSDP training.

### GPU 3. Prepare a smoke input that actually covers every bucket

A short count-weighted run on the original corpus can miss rare buckets. Create a
**diagnostic-only** manifest with 32 entries per populated bucket, using shorter
and longer captions by character length as candidate examples. CPU caption auditing
has already checked actual token lengths; character length here only selects
examples. Repetition is intentional for this hardware test, not data augmentation.
All cache paths are made absolute so this manifest can live outside the cache.

```bash
export SFT_SMOKE_DIR="$SFT_AUDIT/gpu_smoke_${SFT_RESOLUTION}"
mkdir -p "$SFT_SMOKE_DIR"
python - <<'PY'
import json, os
from dataclasses import asdict
from pathlib import Path
from training.main import load_config
from datasets.bucketed import BucketedImages, BucketBatchSampler
config = load_config(os.environ["SFT_CONFIG"])
config.input.manifest = os.environ["SFT_TRAIN_MANIFEST"]
data = BucketedImages(config.input)
root = Path(os.environ["SFT_SMOKE_DIR"])
manifest = root / "manifest.jsonl"
with manifest.open("w") as handle:
    for group in data.groups:
        if not group:
            continue
        candidates = [min(group, key=lambda i: len(data.records[i][0].caption)),
                      max(group, key=lambda i: len(data.records[i][0].caption))]
        for j in range(32):
            record = data.records[candidates[j % 2]][0]
            row = {key: value for key, value in asdict(record).items() if value is not None}
            row["id"] = row.pop("identifier")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
config.input.manifest = str(manifest)
smoke = BucketedImages(config.input)
expected = {i for i, group in enumerate(smoke.groups) if group}
seen = set()
# Same global batch, seed and step numbering as the eight-rank trainer.
for step, indices in enumerate(BucketBatchSampler(smoke.groups, 32, 0, 1, 4096,
                                                seed=config.seed), 1):
    seen.add(smoke.records[indices[0]][3])
    if seen == expected:
        steps = max(100, step + 10)
        break
else:
    raise RuntimeError("Could not establish deterministic bucket coverage.")
(root / "steps.txt").write_text(str(steps) + "\n")
(root / "coverage.json").write_text(json.dumps(dict(
    seed=config.seed, global_batch=32, first_full_coverage_step=step,
    smoke_steps=steps, buckets=[smoke.buckets[i] for i in sorted(seen)]), indent=2))
print(f"Prepared {len(smoke)} entries; {len(seen)} buckets covered by step {step}.")
PY
export SFT_SMOKE_STEPS=$(cat "$SFT_SMOKE_DIR/steps.txt")
```

Keep this manifest/config/seed unchanged through the smoke and resume checks.
The coverage calculation is invalid if these settings or global batch size change.
A bucket with only one source cannot provide two different caption lengths; use
the longest available captions in other buckets and long held-out inference prompts.

### GPU 4. Check the frozen encoders and VAE reconstructions

Run this **single-process** probe before `torchrun`. It checks one shorter/longer
candidate pair per populated bucket, verifies tensor contracts/finiteness, and
saves VAE reconstructions for native-pixel inspection. It uses posterior mode and
reverses the training normalization before decoding. Review text-heavy cases;
add examples if the selected subset does not represent difficult small text.

```bash
python - <<'PY'
import os
from pathlib import Path
from PIL import Image
import torch
from training.main import load_config
from datasets.bucketed import BucketedImages
from datasets.captions import tokenize_captions
from text_encoder.text_encoder import TextEncoder, encode_text_encoder
from vae.vae import load_vae, encode_images_to_latents, scale_latents, reverse_scale_latents
config = load_config(os.environ["SFT_CONFIG"])
root = Path(os.environ["SFT_SMOKE_DIR"])
config.input.manifest = str(root / "manifest.jsonl")
data = BucketedImages(config.input)
device = torch.device("cuda:0")
bundle = TextEncoder(config, config.text_encoder_type, config.token_len,
                     weight_dtype=torch.bfloat16, device=device)
vae = load_vae(config, device, dtype=torch.float32)
output = root / "vae_gpu"
output.mkdir(exist_ok=True)
with torch.inference_mode():
    for bucket, group in enumerate(data.groups):
        for ordinal, index in enumerate(group[:2]):
            pixels, caption = data[index]
            h, w = pixels.shape[:2]
            tokens = tokenize_captions(bundle.tokenizer, [caption], config.token_len)
            ids, mask = (tokens[key].to(device) for key in ("input_ids", "attention_mask"))
            hidden = encode_text_encoder(bundle.text_encoder, ids, mask)
            assert ids.shape == mask.shape == (1, 1024)
            assert hidden.shape == (1, 1024, 2304) and torch.isfinite(hidden).all()
            latents = encode_images_to_latents(vae, pixels[None].to(device), sample=False)
            normalized = scale_latents(latents, config)
            assert normalized.shape == (1, 32, h // 8, w // 8)
            assert torch.isfinite(normalized).all()
            restored = reverse_scale_latents(normalized, config.vae_type)
            torch.testing.assert_close(restored, latents, rtol=1e-5, atol=1e-5)
            decoded = vae.decode(restored).sample[0].permute(1, 2, 0)
            assert decoded.shape == pixels.shape and torch.isfinite(decoded).all()
            for label, image in (("processed", pixels), ("reconstructed", decoded.cpu())):
                rgb = ((image.float().clamp(-1, 1) + 1) * 127.5).round().byte().numpy()
                Image.fromarray(rgb).save(output / f"b{bucket:02d}_{ordinal}_{label}.png")
            print(bucket, (h, w), "caption tokens", mask.sum().item(), "passed")
PY
```

The assertions target the supplied 1,024-token configs. Adjust them deliberately
when testing a different token limit. These checks do not replace DiT forward and
backward validation in the distributed training run below.

### GPU 5. Run compiled eight-GPU training, save, and resume

Use a **new** smoke workdir: an existing `checkpoint.pt` takes precedence over fresh
initialization. For the default global batch 32, eight data-parallel ranks and four
accumulation steps, each rank processes four images with microbatch size one.

```bash
set -o pipefail
torchrun --standalone --nproc_per_node=8 -m training.main \
  --config "$SFT_CONFIG" --manifest "$SFT_SMOKE_DIR/manifest.jsonl" \
  --init_from "$SFT_INIT" --workdir "$SFT_SMOKE_DIR/train" \
  --fsdp 8 --batch_size 32 --grad_accum 4 \
  --total_steps "$SFT_SMOKE_STEPS" --ckpt_steps "$SFT_SMOKE_STEPS" --log_every 1 \
  2>&1 | tee "$SFT_SMOKE_DIR/train.log"

# total_steps is the cumulative stopping step, not the number of extra steps.
torchrun --standalone --nproc_per_node=8 -m training.main \
  --config "$SFT_CONFIG" --manifest "$SFT_SMOKE_DIR/manifest.jsonl" \
  --resume "$SFT_SMOKE_DIR/train/checkpoint.pt" --workdir "$SFT_SMOKE_DIR/train" \
  --fsdp 8 --batch_size 32 --grad_accum 4 \
  --total_steps "$((SFT_SMOKE_STEPS + 20))" --ckpt_steps 20 --log_every 1 \
  2>&1 | tee "$SFT_SMOKE_DIR/resume.log"
```

**Pass criteria:** every scheduled bucket executes; loss, `l2_grads` and
`l2_updates` remain finite; updates are nonzero; no OOM, loader stalls, collective
hangs or repeated compile failures occur; the checkpoint is saved; and resume logs
`resumed at step <SFT_SMOKE_STEPS>` and continues to the larger target step. Inspect
the logs—these commands do not install an automatic finite-value assertion for
every intermediate model tensor. The frozen-encoder probe above checks embeddings
and latents explicitly. Resume restores optimizer and EMA state, but the trainer
does not checkpoint all noise/dropout RNG state, so bitwise continuation is not a
pass criterion.

### GPU 6. Generate, then measure representative training throughput

Generate from the saved smoke checkpoint using held-out long prompts, with prompt
rewriting disabled. This tests checkpoint/inference compatibility, not final SFT
quality. Repeat at portrait/landscape shapes used by the intended run:

```bash
python "$I1_ROOT/torch_inference/generate.py" \
  --checkpoint "$SFT_SMOKE_DIR/train/checkpoint.pt" \
  --height 832 --width 1248 --prompts-file "$SFT_EVAL_PROMPTS" \
  --rewrite-prompt false --num-steps 50 --outdir "$SFT_SMOKE_DIR/samples"
```

The example shape is for 1024; use `416 x 624` for its 512 counterpart. Inspect
exact spelling, numbers, line breaks, layout and edge text. Compare production
experiments against the original checkpoint at matched output shapes and inference
settings; a decreasing training loss alone is insufficient.

Finally, use the **approved full training split**, its actual storage location and
production DataLoader settings for a throughput trial in another fresh workdir:

```bash
torchrun --standalone --nproc_per_node=8 -m training.main \
  --config "$SFT_CONFIG" --manifest "$SFT_TRAIN_MANIFEST" \
  --init_from "$SFT_INIT" --workdir "$SFT_SMOKE_DIR/throughput" \
  --fsdp 8 --batch_size 32 --grad_accum 4 \
  --total_steps 300 --log_every 10 --no_save \
  2>&1 | tee "$SFT_SMOKE_DIR/throughput.log"
```

Exclude model startup, compilation and storage warmup; retain approximately 200
steady-state steps, extending the trial if 300 total steps do not provide that
window. The balanced smoke input deliberately oversamples rare buckets, so its
throughput is not the production estimate. Watch for recompilation when a shape
appears for the first time in this process. Record images/s, step-time variation,
data-loading waits, CPU/storage utilization and per-rank GPU memory. The trainer
logs aggregate images/s, not separate loader timing or peak CUDA memory: obtain
those with a profiler or explicit timing/memory instrumentation on the GPU host.
Do not infer peak memory from successful CPU tests or from a single GPU snapshot.

**GPU exit criteria:** CUDA/compile and NCCL checks pass; encoder/VAE outputs and
text reconstructions are acceptable; all buckets pass forward/backward; save/resume
and held-out inference work; and representative throughput/memory fit the target
server. Start full SFT from the intended original checkpoint in a new production
workdir, using the approved training split—not the balanced smoke manifest or its
short-run checkpoint.

## Caption conditioning and optional encoder caches

The supplied SFT configs use 1,024 caption tokens and an embedding width of 2,304.
Changing the limit changes sequence length, text RoPE capacity and the learned
null-caption tensor. Fresh checkpoint initialization preserves the null-caption
prefix and tiles it into new trainable positions; other compatible model weights
are retained. This is an initialization strategy, not a guarantee of long-caption
rendering quality. Validate conditioning after SFT. Set a different limit only for
a new run, and rebuild smoke assertions/cache plans as needed.

**Validate and count tokens for all captions; precomputing all embeddings is not
required.** The current trainer tokenizes captions and runs the frozen T5Gemma
encoder online for each sampled batch.

| Operation | Recommendation |
| --- | --- |
| Validate captions and count tokens | Run once over the entire dataset before training. |
| Cache token IDs and attention masks | Optional; modest storage and usually a smaller speed benefit than caching embeddings. |
| Cache T5Gemma embeddings | Consider only after validation and profiling show that encoder computation is a substantial bottleneck. |

For 200,000 captions padded to 1,024 tokens, BF16 embeddings alone occupy
`200000 * 1024 * 2304 * 2 = 943,718,400,000` bytes: approximately **944 GB**
(879 GiB), excluding masks and metadata. FP32 storage doubles that to approximately
1.89 TB. The current encoder helper returns FP32 tensors, so a future BF16 cache
writer would need an explicit conversion. Shorter captions can reduce storage in
a suitable cache format, but loading must reproduce the expected padding behavior.

The current SFT loader does not consume cached token IDs or embeddings; either
optimization requires additional implementation. A cache must retain the sample
mapping, attention masks, tokenizer/encoder revision, token limit, special-token
settings, padding policy, and a caption hash so stale entries can be detected.
Validate cached outputs against online encoding before using them. Classifier-free
caption dropout must still happen during training, and the trainable text adapter
must remain online rather than being included in the frozen-encoder cache.

Start with online encoding for the smoke test and initial throughput measurement.
Choose caching based on measured time saved versus preprocessing time, disk space,
and storage bandwidth; a frozen encoder makes caching possible, not mandatory.

## Train

Complete the preflight above first. Set `SFT_TRAIN_MANIFEST` to the approved
training split for the selected resolution. Use a fresh production workdir.

```bash
torchrun --standalone --nproc_per_node=8 -m training.main \
  --config "$SFT_CONFIG" --manifest "$SFT_TRAIN_MANIFEST" \
  --init_from "$SFT_INIT" --workdir "$SFT_WORKDIR" --fsdp 8
```

Use the matching config, initialization checkpoint and pixel cache for the selected
512 or 1024 resolution; they are not interchangeable.

The examples assume the 3B architecture selected by the configs. Set `model_size`
to the matching preset when using a different checkpoint architecture. Incompatible
trainable tensor shapes fail loading rather than being silently skipped.

`--init_from` loads inference/EMA weights, starting the optimizer, step counter,
and EMA tracking fresh. `--resume` restores a complete training checkpoint. An
existing `checkpoint.pt` in the workdir takes precedence so interrupted jobs can
restart with their original launch command. Use a new workdir for a new SFT run.
Missing checkpoint paths fail rather than falling back to random initialization.

The 1024 starting config uses one epoch, learning rate `1e-5`, global batch 32, four
accumulation microbatches per rank, EMA decay `0.9995`, logging every 50 steps,
rolling checkpoints every 1,000 steps, retained checkpoint copies every 2,500
steps, and activation checkpointing. After loading and filtering the final manifest,
the trainer resolves the exact stopping step as
`sum(ceil(bucket_count / global_batch))`; this is roughly 6,250 steps for 200,000
images. The 512 config retains its shorter step-based baseline settings. These are
starting settings to select with held-out rendering evaluation, not a validated
dense-text recipe. Global batch must be divisible by data-parallel world size and
the resulting local batch by accumulation count. Use `--batch_size`, `--grad_accum`,
and `--total_steps` for overrides; an explicit `--total_steps` takes precedence
over `num_epochs` for smoke tests and controlled runs.
W&B remains opt-in through `config.wandb.log_wandb`; when enabled, rank zero reports
the resolved batch decomposition, parallel sizes, optimizer/runtime settings,
checkpoint cadence, sampler steps per epoch, and training metrics.

## Resolution and batching behavior

`config.input.buckets` contains `(height, width)` in image pixels. Both dimensions
must be multiples of 16 for the current FLUX.2 VAE and patch size. The defaults are
roughly equal-area shapes, not minimum side lengths of 512 or 1024. Add larger-area
buckets explicitly when the source data and memory budget justify them. Each image
is assigned to its closest eligible aspect ratio, with ties favoring the larger
area. This assignment is fixed; it does not randomly resize each image every step.

The defaults adapt Lumina-Image-2.0's `imgproc.generate_crop_size_list` approach:
generate quantized candidate shapes along a fixed pixel-budget boundary, then
choose by aspect ratio. Here `datasets.image_geometry.generate_buckets` uses a
32-pixel grid at 512, a maximum long/short ratio of 3, and adds the exact 3:2 / 2:3
anchors. The 1024 config doubles these dimensions. There are 25 candidate shapes;
18 receive images in the audited size histogram. All have area at most the base
resolution squared. Empty buckets are not sampled. Lumina's center crop is replaced
by fit-and-pad to preserve captioned edge text. Its mixed-shape attention masks
are not needed for this trainer's existing homogeneous batches.

This changes bucket assignments and sampling relative to the earlier two-bucket
configs: start a new run/workdir. Keep the old config and manifest when resuming
an existing run. The two-bucket gallery in the artifact appendix is historical.

The epoch-aware sampler independently shuffles every nonempty bucket, divides it
into homogeneous global batches, and randomly interleaves those batches across
aspect ratios. It exhausts every eligible image before reshuffling for the next
epoch. An incomplete bucket tail wraps to the beginning of that epoch's shuffled
order so batch size and distributed slicing stay fixed; buckets smaller than the
global batch necessarily repeat examples. One sampler epoch therefore takes
`sum(ceil(bucket_count / global_batch))` optimizer steps. All data-parallel workers
receive disjoint rank slices of the same global batch except when a bucket itself
is smaller than the global batch.

The batch plan is a deterministic function of seed and epoch, and resume derives
the epoch and within-epoch offset from the saved optimizer step. Preserve the
manifest/order, bucket config, seed, world size, and batch size when resuming to
retain the same sample sequence. Noise/dropout RNG state is not checkpointed by
the existing trainer, so resume is not a bitwise-identical continuation of the
entire training computation.

Tensor-parallel workers receive tensor dimensions before allocating each batch.
Compiled blocks use dynamic shapes; set `config.compile=False` for eager execution.
Microbatch size is fixed in this first implementation: choose it for the largest
bucket. Mixed shapes within a microbatch are not supported or needed here.

Spatial sinusoidal embeddings and RoPE use a shared scale
`256 / sqrt(image_height * image_width)`. This exactly extends the original square
scaling while retaining rectangular aspect geometry. The original square tensors
and checkpoint keys remain compatible. Derived geometry is cached outside model
state. The SFT checkpoint records caption length and configured training buckets.
The 512 and 1024 configs retain their respective training timestep shifts (0 and
0.3); custom mixtures spanning much larger areas need their own recipe validation.

## Evaluate the saved checkpoint

From `i1/torch_inference`:

```bash
python generate.py \
  --checkpoint "$SFT_WORKDIR/checkpoint.pt" \
  --height 896 --width 1184 \
  --prompts-file /path/to/held_out_prompts.txt \
  --rewrite-prompt false \
  --outdir "$SFT_WORKDIR/samples"
```

Height and width default to the checkpoint's base square resolution. Caption
length comes from the checkpoint instead of a hard-coded 256-token limit. Inference
also rejects overflow by default; `--caption-overflow truncate` explicitly opts in.
Disable prompt rewriting when evaluating exact intended strings.

Compare the original checkpoint, square SFT, and bucketed SFT on held-out prompts
at matched output shapes. Measure exact text/transcription errors and inspect text
layout, including small text and content near image edges. Successful shape tests
do not establish generation-quality gains.

## Artifact inventory and recorded results (2026-09-16)

These are historical observations, not substitutes for the exit criteria above.
The active pipeline records live state separately from completed smoke reports.

| Artifact root | Contents and interpretation |
| --- | --- |
| `/cephfs/liuxinyu/DenseText-Project/artifacts/gpt_image_200k_sft_audit` | Original full caption/image audits, geometry comparison, and historical galleries. |
| `/cephfs/liuxinyu/DenseText-Project/artifacts/gpt_image_200k_sft_audit/lumina_index_smoke` | Shard-00000 corrected index, transform checks and a 100-example gallery; this gallery skipped tokenizer execution. |
| `/cephfs/liuxinyu/DenseText-Project/artifacts/gpt_image_200k_sft_precompute` | Full pipeline outputs/logs and `status.json`; inspect live status rather than assuming completion. Also contains `cache_smoke_1024`, `cache_smoke_checks.json`, and separate `vae_smoke` diagnostics. |

The original 200,000-caption audit found p50 567, p95 632, p99 657 and maximum
732 tokens, with zero over 1,024. The full image decode checked 200,000 rows in
1,458 seconds: 199,999 decoded and one PNG was truncated,
`sciformula_mathematics_1c47358aa86c1730_codex` (shard 00000, row group 147, row 1).
It found 36,115 incorrect declared sizes: 14,721 reversed orientations and 21,394
images outside exact 3:2/2:3 geometry. Correct the dimensions and keep valid other
aspects; the earlier exact-aspect-only baseline contained 178,605 images.

The original `preprocessing_gallery` used only two exact-aspect buckets and is
historical, not coverage of the current generated set. The complete size histogram
predicts 18 populated shapes among 25 candidates. At 1024, calculated mean padding
falls from 2.568% with two buckets to 0.220%, and maximum padding from 50% to 6.971%.
These are geometric estimates, not measured generation-quality gains.

The real shard-00000 index smoke test retained 1,999 images, corrected 318 sizes
and excluded the known corrupt PNG. Both resolutions passed sample transform checks
across its 11 populated buckets. Its 1024 pixel cache wrote 6,214,569,984 verified
bytes; a separate 43-example parity check covered all 11 buckets and passed. The
18-test CPU regression suite passed, including cache integrity, sampling, duplicate
reporting and concurrent-reader tests. Full pretrained GPU/FSDP training is not
established by these CPU results. Consult the VAE reports/logs for their actual
coverage rather than treating generated inspection images as human approval.
