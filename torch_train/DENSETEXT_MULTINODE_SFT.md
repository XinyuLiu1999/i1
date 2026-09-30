# End-to-end: DenseText merge → i1 multi-node SFT (no precompute)

This is the full path from the `densetext-merge` export to a multi-node SFT run
that reads images directly from the merged Parquet files. There are two
variants, selected only by config (§5):
- **A. 1024**;
- **B. mixed resolution**, with 1024, 1536 and 2048 pixel-budget tiers. Launcher details,
topology rules and communication checks are in `MULTINODE.md`; this guide shows
the order of operations and the settings for this dataset.

Paths below are the ones on the migration machine. Replace them if the training
nodes mount storage elsewhere, but **every node must see the same manifest and
Parquet files at the same paths**.

```bash
export MERGED=/user/lxy8802/DenseText-merged           # densetext-merge output
export I1=/user/lxy8802/i1/torch_train
export SFT_PY=/user/lxy8802/miniforge3/envs/i1_sft/bin/python
export HF_HUB_CACHE=/user/lxy8802/.cache/data_juicer/models
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

## 0. What the pipeline produces and what SFT reads

| File | Used by SFT? |
|---|---|
| `$MERGED/data/<subset>/<job>/part-*.parquet` | Yes, via `parquet_path`/`row_group`/`row_in_group` in the manifest |
| `$MERGED/i1_manifest.jsonl` | Input to the caption-length filter (step 2) — **not** trained on directly |
| `$MERGED/i1_manifest.max1024tok.jsonl` | Input to the transparency filter (step 2) |
| `$MERGED/i1_manifest.max1024tok.opaque.jsonl` | **The training manifest** (`SFT_MANIFEST`) |
| `$MERGED/transparency_scan.parquet` | Per-image alpha measurements from step 2; reused when the threshold changes |
| `manifest.parquet`, `indexes/`, `receipts/`, `audit/`, `_SUCCESS.json` | Merge provenance and verification only |

Each manifest line is
`{"id", "caption", "width", "height", "parquet_path", "row_group", "row_in_group"}`.
`parquet_path` is relative to the manifest's directory. At load time i1 checks
that the Parquet row's `id` and `caption` equal the manifest's. A mismatch is
treated as a stale manifest and raised as an error.

## 1. Finish the merge (CPU machine)

```bash
cd /user/lxy8802/densetext-merge
export LD_LIBRARY_PATH=/user/lxy8802/miniforge3/lib   # sqlite CXXABI fix for this venv
source /tmp/densetext-merge-env/bin/activate
densetext-merge --config config.local.yaml export     # inventory..language are reused
densetext-merge --config config.local.yaml verify
deactivate
```

`export.row_group_rows: 8` keeps each random image read to about 4 MB. At 64 rows
per group, one sample read about 32 MB. Export and filter/language resume per
shard, so if either is interrupted, rerun the same command.

Do not change `config.local.yaml` or the merge code after this point. Either
change alters the run identity, and the merge then refuses the existing
work/output directories. Only proceed once `verify` has written
`$MERGED/_SUCCESS.json`.

Expected size: about 4.31M rows and roughly 2.2 TB of image bytes.

## 2. Filter the training manifest (CPU, about 30 minutes)

Two filters run in order. Each one only drops lines, and the second reads the
first one's output:

```bash
cd $I1
# Captions longer than the SFT token budget (minutes)
$SFT_PY -m datasets.filter_manifest $MERGED/i1_manifest.jsonl --token_len 1024 --workers 16
# → $MERGED/i1_manifest.max1024tok.jsonl
# → $MERGED/i1_manifest.max1024tok.jsonl.report.json          (counts, percentiles, histogram, dropped ids)

# Substantially transparent images (about 30 minutes)
$SFT_PY -m datasets.filter_transparent $MERGED/i1_manifest.max1024tok.jsonl --workers 32
# → $MERGED/transparency_scan.parquet                         (one row per image: format, mode, transparent_frac)
# → $MERGED/i1_manifest.max1024tok.opaque.jsonl               (the training manifest)
# → $MERGED/i1_manifest.max1024tok.opaque.jsonl.report.json   (counts, drops per id prefix, dropped ids)
```

Both filters behave the same way:
- Kept lines are copied byte-for-byte.
- The output must sit next to the input, because relative `parquet_path` values
  resolve from there.
- Writes are atomic, and neither filter overwrites an existing output without
  `--overwrite`.

Do not run the transparency scan at the same time as the merge `verify`. Both
read the whole dataset and would slow each other down.

**Caption length.** SFT runs with `token_len=1024` and
`caption_overflow="error"`, so one over-length caption in a batch would stop
training. About 0.4% of merged captions exceed 1024 T5Gemma tokens (max seen
≈ 2.2K). The filter uses the same tokenizer call as training. Speed is about 10k
captions/s per 16 workers, so roughly 5 minutes for 4.3M.

**Transparency.** The SFT loader flattens every image with `.convert("RGB")`,
which discards alpha. Transparent pixels then show whatever colour is stored
underneath them, often black or an arbitrary solid colour. The caption may say
"white background" while the training image shows black bars. So the filter
drops images where more than 1% of pixels have alpha below 128.

Measured on a random 1% of the merged Parquet files (45k rows):
- **What gets dropped:**
  - about 1.2% of rows, roughly 50k in total;
  - almost all from blip3ocr (2.3% of its rows);
  - 1 row in about 800 from chartgalaxy_real;
  - none from danqing, screenparse, chartgalaxy_synthetic, monet or paper2fig100k.
- **Threshold:** images are mostly either fully opaque or substantially
  transparent. Any threshold from 0.1% to 5% removes about the same rows.
- **Speed:** limited by reading. The scan reads nearly every image, because
  images are read 8 at a time in row groups. Only JPEGs are skipped without
  decoding, recognised from their first bytes. At about 1.3 GB/s the full
  2.1 TB takes about 30 minutes. Allow up to an hour on colder storage.
- **Changing the threshold:** the scan is reused on later runs. A different
  threshold only rewrites the manifest and takes about a minute. For example,
  run `--max_transparent 0.05 --overwrite`. Pass `--rescan` only if the Parquet
  data changed.
- **Missing ids:** the filter fails if a manifest id is missing from the scan.
  That means the scan belongs to other data.

Do not composite transparent images onto white in the loader instead. The
captions were written from the flattened images, so compositing would make
some of them wrong in the other direction.

**Further filtering.** To filter further later (by source, id list, OCR stats,
and so on), write another filtered JSONL **in the same directory**, derived from
the training manifest. Rules:
- Only drop lines. Never edit `caption`/`id`/`parquet_path`/`row_*`.
- Do not modify `i1_manifest.jsonl` itself. Merge `verify` checks it against
  `manifest.parquet` and `_SUCCESS.json`.

`manifest.parquet` has the same rows plus the metadata columns (`source_dataset`,
OCR stats, language, and so on), so it is the convenient place to select ids.

## 3. Sanity check on the CPU machine (optional, no GPU)

```bash
cd $I1
$SFT_PY - <<'EOF'
import json, os
from datasets.data_sources import ImageRecord, open_record_image
m = os.environ["MERGED"] + "/i1_manifest.max1024tok.opaque.jsonl"
with open(m) as f:
    for i, line in zip(range(20), f):
        r = json.loads(line)
        rec = ImageRecord(r["id"], r["caption"], r["width"], r["height"],
                          parquet_path=os.path.join(os.path.dirname(m), r["parquet_path"]),
                          row_group=r["row_group"], row_in_group=r["row_in_group"])
        img = open_record_image(rec)
        assert img.size == (r["width"], r["height"]), r["id"]
print("ok")
EOF
```

## 4. Prepare every training node

Follow `MULTINODE.md` §1:
- the same code and `i1_sft` env (Python 3.11, torch 2.9.1+cu126);
- the complete T5Gemma and FLUX.2 caches under `$HF_HUB_CACHE`;
- the init checkpoint `$HF_HUB_CACHE/i1-3B/1024_resolution_checkpoint_torch.pt`;
- a shared `SFT_WORKDIR`.

Then run the communication check on all nodes:

```bash
bash $I1/run_multinode.sh --check-communication       # expect PASS ... backend=nccl on every rank
```

Resource checks per node:
- **Host RAM.** Each rank loads the bucketed index, about 9 GB for 4.3M rows,
  which takes about 2 minutes at startup. Each rank also runs 4 DataLoader workers,
  so 8 ranks per node can need well over 100 GB. If memory is tight, lower
  `config.input.num_workers` (in `configs/sft_512.py`) or check `/dev/shm`.
  The mixed-resolution config uses 2 workers per rank, because each 2048 sample
  is about 12 MB of RGB passed through shared memory.
- **Checkpoint RAM.** Only global rank 0 holds the full model/EMA/Adam copy in
  host memory while saving.
- **Shared disk.**
  - `checkpoint.pt` (resume state: model, EMA and Adam moments) is tens of GB and
    is rewritten every 1000 steps.
  - A permanent copy `checkpoint.pt-000010000`, `-000020000`, … is kept every
    10000 steps (`keep_ckpt_steps`).
  - Budget roughly 14 kept copies for one epoch.

## 5. Choose a variant, topology and length

Both variants train on the same filtered manifest, start from the same 1024
checkpoint and use the same launcher. They differ only in `SFT_CONFIG`.

| | **A. 1024** | **B. Mixed resolution (1024/1536/2048)** |
|---|---|---|
| `SFT_CONFIG` | `configs/sft_1024_captioned.py` | `configs/sft_multires_captioned.py` |
| Buckets | 45, all at 1024² area | 209: the same 45 plus 1536² and 2048² area tiers |
| Tier split on the merged data (measured, 30.6k-row sample) | 100% at 1024 | 79.2% at 1024, 13.0% at 1536, 7.8% at 2048 |
| Image tokens per sample | 4,096 | 4,096 / 9,216 / 16,384 |
| Estimated GPU compute per epoch | 1× | ≈1.6× (per image: 1536 ≈ 2.6×, 2048 ≈ 5.8×) |
| DataLoader workers per rank | 4 | 2 (set in the config) |
| W&B experiment | `i1-1024-captioned-v4` | `i1-multires-2048-captioned` |

Settings shared by both:
- lr 1e-5 (constant), EMA 0.9995, `train_timestep_shift` 0.3, gradient checkpointing, 1024 caption tokens;
- `drop_remainder=True`, so no image is duplicated as padding;
- 1 epoch by default.

The launcher's default config is `sft_1024.py`, which has square-ish buckets
rather than the captioned set, so always set `SFT_CONFIG` explicitly.

**How B assigns images.** Each image goes to the highest tier whose area is at
or below its own pixel area, then to the closest aspect ratio within that tier.
Images are never upscaled; larger sources are downsampled to the tier. Every
merged row fits a bucket in both variants.

`pos_embed` and RoPE are rebuilt for the current resolution, so the 1024
checkpoint loads under B as well.

B's config sets `fsdp_axis_size=8` and `grad_accum_steps=4` for one node. The
launcher's `FSDP_SIZE`/`GRAD_ACCUM` override both, so use the table below for
either variant.

**B: timestep shift (optional, not yet validated).**
- One `train_timestep_shift` applies to every tier.
- In this code, 0.3 equals an SD3-style shift of about 3.3, which suits 1024².
- The usual rule scales the shift by √(token ratio). That gives about 0.15 for
  the 2048 tier, so 0.3 under-noises it.
- To try a compromise, add a config file and point `SFT_CONFIG` at it. Compare
  2048 samples before committing to it.

```bash
cat > $I1/configs/sft_multires_captioned_shift02.py <<'PY'
from configs.sft_multires_captioned import get_config as _base


def get_config():
    config = _base()
    config.transport.train_timestep_shift = 0.2  # between the 1024 (0.3) and 2048 (~0.15) values
    config.wandb.experiment = "i1-multires-2048-captioned-shift0.2"
    return config
PY
```

**Length.** With a global batch of 32, one epoch is about 4.24M / 32 ≈ **132k
steps** for either variant (after the step 2 filters). B takes longer in wall-clock time, not in steps.
- The exact step count is logged at startup, together with the `drop_remainder`
  count of omitted images.
- To stop earlier, pass `--total_steps N`, the cumulative final step.
- Because the lr is constant, a stopped run can be extended by resuming with a
  larger `--total_steps`.

Batch arithmetic, per `MULTINODE.md` §5 (microbatch per GPU = global / DP ranks / accum):

| Nodes × GPUs | `GLOBAL_BATCH_SIZE` | `GRAD_ACCUM` | microbatch/GPU |
|---|---:|---:|---:|
| 2 × 8 | 32 | 2 | 1 |
| 4 × 8 | 32 | 1 | 1 |
| 4 × 8 | 64 | 2 | 1 |

Keep one image per GPU microbatch for B, because 2048 samples are about 17k
tokens. If you raise the global batch, consider scaling the lr and recomputing
the step count.

## 6. Smoke runs (3 steps, then resume to 4)

Use throwaway workdirs. These runs exercise the real manifest, all ranks, save
and resume. Settings shared by both smoke runs:

```bash
export SFT_MANIFEST=$MERGED/i1_manifest.max1024tok.opaque.jsonl
export GPUS_PER_NODE=8 GLOBAL_BATCH_SIZE=32 GRAD_ACCUM=2
export WANDB_MODE=offline
smoke() {   # usage: smoke <workdir>; uses SFT_CONFIG/SFT_MANIFEST from the environment
  export SFT_WORKDIR=$1; unset SFT_RESUME
  export SFT_INIT=$HF_HUB_CACHE/i1-3B/1024_resolution_checkpoint_torch.pt
  bash $I1/run_multinode.sh --total_steps 3 --log_every 1 --ckpt_steps 3 &&
  unset SFT_INIT && export SFT_RESUME=$SFT_WORKDIR/checkpoint.pt &&
  bash $I1/run_multinode.sh --total_steps 4 --log_every 1 --ckpt_steps 4
}
```

**A (1024):**

```bash
export SFT_CONFIG=$I1/configs/sft_1024_captioned.py
smoke /shared/outputs/densetext_sft_smoke_1024
```

**B (mixed resolution).** Three random steps are unlikely to hit the 2048
tier, which is where peak memory and step time are set. Run the smoke test on a
manifest that contains only images of at least 2048² area. It is written next to
the main manifest so its relative paths still resolve. Every step then uses a
2048 bucket.

```bash
$SFT_PY - <<'PY'
import json, os
d = os.environ["MERGED"]
with open(f"{d}/i1_manifest.max1024tok.opaque.jsonl") as src, open(f"{d}/i1_manifest.smoke2048.jsonl", "w") as dst:
    for line in src:
        r = json.loads(line)
        if r["width"] * r["height"] >= 2048 * 2048:
            dst.write(line)
PY
export SFT_CONFIG=$I1/configs/sft_multires_captioned.py   # or the shift variant from §5
SFT_MANIFEST=$MERGED/i1_manifest.smoke2048.jsonl smoke /shared/outputs/densetext_sft_smoke_multires
```

While B's smoke test runs, watch `nvidia-smi` on one node for peak memory. Also
note the per-step time, which is the worst case for this variant.

Check the node logs in `$SFT_WORKDIR/logs/node_*.log`:
- finite loss;
- the same step count on all ranks;
- a `saved checkpoint` line;
- the resume log line showing step 3 → 4.

Afterwards, delete the smoke directories and `i1_manifest.smoke2048.jsonl`.

## 7. Production launch (platform start script, same on every node)

```bash
#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/user/lxy8802/.cache/huggingface
export HF_HUB_CACHE=/user/lxy8802/.cache/data_juicer/models
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Pick one variant (§5). Use a new, empty, shared workdir per run.
VARIANT=${VARIANT:-1024}   # 1024 | multires
case $VARIANT in
  1024)     export SFT_CONFIG=/user/lxy8802/i1/torch_train/configs/sft_1024_captioned.py ;;
  multires) export SFT_CONFIG=/user/lxy8802/i1/torch_train/configs/sft_multires_captioned.py ;;
  *) echo "unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac
export SFT_WORKDIR=/shared/outputs/densetext_sft_${VARIANT}_run001

export SFT_MANIFEST=/user/lxy8802/DenseText-merged/i1_manifest.max1024tok.opaque.jsonl
export SFT_INIT=$HF_HUB_CACHE/i1-3B/1024_resolution_checkpoint_torch.pt
export GPUS_PER_NODE=8 GLOBAL_BATCH_SIZE=32 GRAD_ACCUM=2      # 2 nodes × 8 GPUs
export WANDB_PROJECT=DenseText-SFT WANDB_MODE=online
# The W&B key is read by the launcher from /user/lxy8802/.bashrc (see below).

exec bash /user/lxy8802/i1/torch_train/run_multinode.sh
# optional extra trainer flags go after the script, e.g. --total_steps 60000
```

To run both variants, launch two separate jobs, one with `VARIANT=1024` and one
with `VARIANT=multires`. They write to different workdirs and W&B experiments.
With the same seed and manifest, their per-step losses are **not** directly
comparable: B's 2048 samples have a different noise/token balance. Compare them
by sample quality (§10), not by loss curves.

The platform supplies `WORLD_SIZE` (node count), `RANK` (node rank),
`MASTER_ADDR` and `MASTER_PORT`. The launcher owns `--config`/`--manifest`/
`--workdir`/`--fsdp`/`--tp`/`--batch_size`/`--grad_accum`/`--init_from`/`--resume`.

What the launcher does:
- runs `training.main` with FSDP over all GPUs (TP=1) and `--no_compile`;
- tiles the 256-token null caption to 1024 tokens on `--init_from`;
- logs to W&B (project DenseText-SFT) from rank 0.

**W&B key.**
- If the platform injects `WANDB_API_KEY`, that key wins.
- Otherwise the launcher reads the `export WANDB_API_KEY=...` line from
  `WANDB_KEY_FILE` (default `/user/lxy8802/.bashrc`). It parses only that line,
  because the file returns early in non-interactive shells, and it never prints
  the key.
- Every node must be able to read that file, and node 0 runs
  `wandb login --verify` before starting.

## 8. Monitoring

- **Logs:**
  - `$SFT_WORKDIR/logs/node_<rank>.log`;
  - W&B loss/lr every 50 steps (`log_training_steps`);
  - the startup lines report the index size, bucket counts, the `drop_remainder`
    summary and the total steps.
- **Stale-data errors** (`id`/`caption` mismatch) mean the manifest does not match
  the Parquet files it points to. Regenerate the filtered manifest from the
  verified merge output; never hand-edit captions.
- **Throughput:**
  - Row groups are 8 images, and each DataLoader worker caches the last group it
    read.
  - If data loading is the bottleneck, check shared-storage read bandwidth first
    (about 0.5 MB per image, random access).
  - Mixed resolution only:
    - Step time varies with the tier drawn. All ranks share a bucket per step,
      so a 2048 step is several times slower than a 1024 step.
    - Judge throughput over hundreds of steps, not from single log lines.
    - CPU decode and resize at 2048 is about 4× the 1024 cost. If GPU
      utilization drops on 2048 steps, check the data loader first.

## 9. Resume after interruption

Use the same data, config, topology, batch, accumulation and seed:

```bash
unset SFT_INIT
export SFT_RESUME=$SFT_WORKDIR/checkpoint.pt
exec bash /user/lxy8802/i1/torch_train/run_multinode.sh   # same extra flags as before
```

The launcher refuses to start in a workdir that already has `checkpoint.pt`
unless `SFT_RESUME` points to it. This prevents accidentally continuing a
different run. The sampler resumes the per-epoch bucket order exactly from the
saved step.

## 10. Evaluate

Kept copies `checkpoint.pt-0000N0000` and `checkpoint.pt` load directly in
inference, which uses the EMA weights. From `i1/torch_inference`:

```bash
$SFT_PY generate.py --checkpoint $SFT_WORKDIR/checkpoint.pt-000010000 \
  --height 832 --width 1248 --prompts-file /path/to/held_out_prompts.txt \
  --rewrite-prompt false --outdir $SFT_WORKDIR/samples/step10000
```

For **B (mixed resolution)**, also generate at the 2048 tier, for example
`--height 2048 --width 2048` or a 3:2 shape from the 2048 buckets. Held-out
prompts at 1024 shapes alone will not show what the high-resolution training
changed. Compare A and B checkpoints at the same step and the same output shapes.

For a side-by-side run against the initialization on CVTG-2K / LongText /
BizGenEval, use `compare_sft_checkpoints.sh`. Set `START_CHECKPOINT`,
`SFT_CHECKPOINT` and `PYTHON_BIN`, because its defaults point at old `/cephfs`
paths.

Judge progress with the following, not by training loss alone:
- held-out prompts at matched shapes and settings;
- exact spelling, numbers, line breaks and small or edge text.
