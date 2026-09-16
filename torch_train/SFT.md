# PyTorch SFT for dense text

The SFT configs initialize from an existing i1 checkpoint, train on original
images grouped into rectangular buckets, and accept captions up to 1,024 tokens.
The VAE and T5Gemma encoder remain frozen; the DiT and its text adapter are trained.
The JSONL SFT path does not require TensorFlow; the existing TFRecord training path
is unchanged.

## Prepare the training environment

The following setup targets Linux and a single server with eight A800 80 GB GPUs.
Use the same environment for caption auditing, training, and inference. These are
setup instructions, not a claim that full pretrained eight-GPU training has been
validated in this environment; the existing regression tests use tiny CPU models.

### 1. Check the host and create an isolated environment

```bash
nvidia-smi
nvidia-smi topo -m
cc --version

python3.11 -m venv ~/envs/i1_sft
source ~/envs/i1_sft/bin/activate
python -m pip install --upgrade pip
```

The host needs an NVIDIA driver compatible with the selected CUDA wheel, Python
3.11 with `venv` support, and a working C/C++ compiler for `torch.compile`. If using
Conda instead, replace the two `venv` commands with
`conda create -n i1_sft python=3.11 -y` and `conda activate i1_sft`.
The topology output should show the expected NVLink connections. In a container,
make all eight allocated GPUs visible and provide enough shared memory for the
data-loader workers.

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
  accelerate safetensors sentencepiece "huggingface_hub>=0.34,<1.0"
python -m pip check
```

PyTorch 2.9.1 is pinned here to avoid accidentally installing a CPU-only or changing
major-version environment. The other version pins follow the repository's training
instructions. The model uses PyTorch scaled-dot-product attention; a separate
`flash-attn` installation is not required. TensorFlow, torchvision, and torchaudio
are not needed for the JSONL SFT path.

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

### 4. Verify imports, CUDA, BF16, and compilation

Run this inside the activated environment on the allocated GPU server:

```bash
python - <<'PY'
import torch
import transformers
import diffusers
from transformers import T5GemmaModel
from diffusers import AutoencoderKL
from torch.distributed.fsdp import fully_shard

print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("Transformers:", transformers.__version__, "Diffusers:", diffusers.__version__)
assert torch.cuda.is_available(), "CUDA is unavailable; check driver and GPU visibility."
assert torch.cuda.device_count() == 8, "Expected eight visible GPUs for this setup."
assert torch.cuda.is_bf16_supported(), "BF16 is unavailable."
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(index, props.name, f"{props.total_memory / 2**30:.1f} GiB")

@torch.compile
def compiled_op(x):
    return torch.nn.functional.silu(x @ x)

x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(compiled_op(x)).all().item()
torch.cuda.synchronize()
print("CUDA/BF16/compile check passed")
PY
```

This performs a small GPU computation without loading pretrained weights. A
compilation failure should be resolved before using the default compiled training
config. `config.compile=False` can be used to diagnose training in eager mode.

### 5. Verify communication between all eight GPUs

This small NCCL check verifies the distributed launch and a collective operation;
it does not benchmark interconnect bandwidth or validate full FSDP training:

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
assert value.item() == world * (world + 1) / 2
torch.cuda.synchronize()
if rank == 0:
    print("Eight-GPU NCCL all-reduce passed")
dist.destroy_process_group()
PY
torchrun --standalone --nproc_per_node=8 "$i1_check_dir/check_nccl.py"
rm "$i1_check_dir/check_nccl.py"
rmdir "$i1_check_dir"
```

Finally, from the repository root, run the lightweight regression tests and record
the resolved environment alongside your experiment files:

```bash
cd /path/to/i1
python -m unittest discover -s torch_train/tests -v
python -m pip freeze > /path/to/experiment/i1-sft-environment.txt
cd torch_train
```

Create `/path/to/experiment` first and replace all placeholder paths. The training
and caption-audit commands below run from `i1/torch_train`. On later logins,
reactivate the environment and restore the cache variables before running them.

## Data

Create a JSONL manifest, one image and caption per line:

```json
{"image_path":"images/poster.png","caption":"A poster with the heading ...","width":1600,"height":1200}
```

`prompt` is also accepted instead of `caption`. Paths are relative to the manifest
unless `config.input.image_root` is set. Dimensions are optional; supplying them
avoids scanning image headers during dataset construction. Dimensions must describe
the image after EXIF orientation is applied, and are checked when loading pixels.
Captions must be nonempty strings. Keep the exact intended text in the captions.

The default transform fits the entire image into its nearest eligible aspect-ratio
bucket and adds white padding. This retains text at the image edges. Padding is
part of the training image and receives the normal image loss. `resize_mode="crop"`
enables resize-and-center-crop instead, but captions must then match the visible
content. No stretching, random cropping, or horizontal flipping is applied.

Source images below the configured pixel area are filtered; upscaling is disabled
by default. `min_image_side` can additionally enforce a minimum short side. Inspect
preprocessed samples for readable characters: pixel area alone does not ensure it.
Square TFRecords cannot recover text already removed by their original crop.

## Check caption lengths first

From `i1/torch_train`:

```bash
python -m datasets.inspect_captions --manifest /path/to/train.jsonl --token_len 1024
```

This loads only the T5Gemma tokenizer and reports token-count percentiles, maximum,
and the number exceeding the limit. It exits nonzero if any caption exceeds the
limit. Model access must already be configured for the Google checkpoint.

Character count is not a reliable token limit. The SFT configs use 1,024 tokens;
the data loader raises on overflow rather than silently dropping text. Choose a
different limit with `--token_len` when initializing a new SFT run. Explicitly set
`config.input.caption_overflow="truncate"` only if truncation is intended.

The embedding width remains 2,304. Increasing the limit changes sequence length,
text RoPE capacity, and the learned null-caption tensor. Fresh checkpoint
initialization preserves the null-caption prefix and tiles it to initialize extra
positions. The new positions are trainable. All other compatible model weights
are retained. This is an initialization strategy, not a guarantee of long-caption
generation quality; evaluate the resulting conditioning after SFT.

## Preflight image and text processing

Before committing to a full run, perform the following checks in order. Audit the
whole dataset for mechanical errors, then inspect representative examples for
readability and image-caption alignment.

1. **Validate every image-caption record.** Decode all images and check their
   EXIF-corrected dimensions against the manifest. Check for missing/corrupt files,
   invalid captions, and unintended duplicates. Report filtered examples and the
   number assigned to each bucket. Merely constructing `BucketedImages` does not
   fully validate image files when width and height are supplied: pixels are read
   in `__getitem__`. Exercise the actual loading/transform path for every eligible
   record before launching distributed training.
2. **Audit all captions with the actual tokenizer.** Run `datasets.inspect_captions`
   above and resolve every overflow. English captions of 1,000–2,500 characters
   still need token counts; character counts do not establish that they fit.
   Inspect tokenized-and-decoded examples containing numbers, punctuation, unusual
   words, and long quoted passages. Preserve exact intended spelling, punctuation,
   and meaningful line breaks. Do not automatically summarize captions just to
   fit the limit, since that can discard the text the model should render.
3. **Inspect 100–200 processed examples across all buckets.** Include small text,
   long documents, extreme aspect ratios, and text near image edges. Display each
   original beside the actual resized/padded training image, with its full caption,
   token count, source dimensions, and assigned bucket. Include enlarged text crops
   and inspect at native pixel size; a small contact-sheet thumbnail can hide lost
   characters. Confirm that the caption belongs to the image and accurately states
   its visible text. White padding prevents geometric cropping, but downsampling
   can still erase characters. Repeat this check for both 512 and 1024 configs.
4. **Inspect VAE reconstructions for 20–50 text-heavy examples.** Compare the
   processed image with its reconstruction from the frozen FLUX.2 VAE, including
   enlarged text crops. Use the latent distribution's mode for a deterministic
   diagnostic; training currently samples latents. If testing normalized latents,
   reverse the training normalization before decoding. This separates losses from
   resizing and VAE compression from denoiser behavior. If characters are already
   badly degraded here, address resolution or data selection before a long SFT.
5. **Run a short training smoke test.** Exercise every populated bucket and both
   short and long captions. Verify finite text embeddings, latents, losses, and
   gradients. For the supplied configs, image batches are NHWC floats in [-1, 1],
   token IDs/masks have shape `(B, 1024)`, encoder outputs `(B, 1024, 2304)`, and
   normalized latents `(B, 32, H/8, W/8)`. Save a checkpoint, resume it, and generate
   with held-out long prompts with rewriting disabled. A decreasing training loss
   alone does not demonstrate improved transcription or layout. After warming up
   every bucket, measure about 200 steady-state steps on the intended eight-GPU
   setup to establish throughput and peak memory.

Keep a held-out evaluation set separate before selecting hyperparameters. Include
different text densities and aspect ratios, and compare exact text/transcription
errors as well as visual layout against the original checkpoint.

Only the caption-length audit is currently provided as a standalone audit command.
The preprocessing galleries, full image validation report, and VAE reconstruction
report described here are recommended preflight tasks, not implemented commands.
The synthetic tests below do not replace these checks on your real data and models.

## Is offline text preprocessing necessary?

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

```bash
# Initialize from the 512 checkpoint, approximately 512^2 pixel-area buckets.
torchrun --nproc_per_node=8 -m training.main \
  --config configs/sft_512.py \
  --manifest /path/to/train.jsonl \
  --init_from /path/to/512_resolution_checkpoint_torch.pt \
  --workdir /path/to/sft_512 --fsdp 8

# Initialize from the 1024 checkpoint, approximately 1024^2 pixel-area buckets.
torchrun --nproc_per_node=8 -m training.main \
  --config configs/sft_1024.py \
  --manifest /path/to/train.jsonl \
  --init_from /path/to/1024_resolution_checkpoint_torch.pt \
  --workdir /path/to/sft_1024 --fsdp 8
```

The examples assume the 3B architecture selected by the configs. Set `model_size`
to the matching preset when using a different checkpoint architecture. Incompatible
trainable tensor shapes fail loading rather than being silently skipped.

`--init_from` loads inference/EMA weights, starting the optimizer, step counter,
and EMA tracking fresh. `--resume` restores a complete training checkpoint. An
existing `checkpoint.pt` in the workdir takes precedence so interrupted jobs can
restart with their original launch command. Use a new workdir for a new SFT run.
Missing checkpoint paths fail rather than falling back to random initialization.

The starting configs use learning rate `1e-5`, 10,000 steps, global batch 32, four
accumulation microbatches per rank, and activation checkpointing. These are starting
settings to tune, not a validated dense-text recipe. Global batch must be divisible
by data-parallel world size and the resulting local batch by accumulation count.
Use `--batch_size`, `--grad_accum`, and `--total_steps` for overrides.

## Resolution and batching behavior

`config.input.buckets` contains `(height, width)` in image pixels. Both dimensions
must be multiples of 16 for the current FLUX.2 VAE and patch size. The defaults are
roughly equal-area shapes, not minimum side lengths of 512 or 1024. Add larger-area
buckets explicitly when the source data and memory budget justify them. Each image
is assigned to its closest eligible aspect ratio, with ties favoring the larger
area. This assignment is fixed; it does not randomly resize each image every step.

Each optimizer step samples one nonempty bucket, weighted by its eligible image
count. All data-parallel workers use the same shape and receive slices of one
global sampled batch. Sampling is without replacement within that batch unless
the bucket contains fewer images than the global batch size. The sampler is a
deterministic function of seed and step. Preserve the manifest/order, bucket config,
seed, world size, and batch size when resuming to retain the same sample sequence.
Noise/dropout RNG state is not checkpointed by the existing trainer, so resume is
not a bitwise-identical continuation of the entire training computation.

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
  --checkpoint /path/to/sft_1024/checkpoint.pt \
  --height 896 --width 1184 \
  --prompts-file /path/to/held_out_prompts.txt \
  --rewrite-prompt false \
  --outdir /path/to/sft_samples
```

Height and width default to the checkpoint's base square resolution. Caption
length comes from the checkpoint instead of a hard-coded 256-token limit. Inference
also rejects overflow by default; `--caption-overflow truncate` explicitly opts in.
Disable prompt rewriting when evaluating exact intended strings.

Compare the original checkpoint, square SFT, and bucketed SFT on held-out prompts
at matched output shapes. Measure exact text/transcription errors and inspect text
layout, including small text and content near image edges. Successful shape tests
do not establish generation-quality gains.

## Lightweight verification

From the repository root:

```bash
python -m unittest discover -s torch_train/tests -v
```

The tests use tiny synthetic models, CPU tensors, and a fake tokenizer. They cover
rectangular shapes, positional consistency, long-caption CFG, checkpoint loading,
optimizer/EMA round trips, training/inference parity, compilation with activation
checkpointing, deterministic sampling, image padding, overflow handling, and a
two-process Gloo broadcast. They do not download pretrained weights or validate a
full CUDA/FSDP training run.
