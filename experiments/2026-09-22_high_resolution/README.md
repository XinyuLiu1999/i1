# High-resolution bucket SFT · 2026-09-22

Train captioned dense-text images at up to **1536² or 2048² pixels** by adding
resolution tiers to the original 1024 captioned SFT buckets. This experiment uses
the shared global flow-MSE trainer and initializes from the original 1024 base
checkpoint by default. Set `SFT_INIT` to another checkpoint for a fresh continuation
run. Regional and perceptual losses from the separate September 21 experiment are
not enabled here.

| Variant | Resolution tiers | Total buckets | Maximum image tokens (16px patches) |
|---|---|---:|---:|
| `config_1536.py` | 1024 + 1536 | 116 | 9,216 |
| `config_2048.py` | 1024 + 1536 + 2048 | 209 | 16,384 |

These are **area budgets**, not maximum side lengths: rectangular buckets can
have a side longer than 1536/2048. Each tier uses a 32px frontier, aspect ratios
up to 3:1, and exact 3:2 / 2:3 anchors. All dimensions are divisible by 16.
The original 45 buckets remain unchanged.

Bucket selection first chooses the highest eligible tier whose squared resolution
does not exceed the source pixel area, then chooses the closest aspect ratio in
that tier. The existing no-upscale check still applies. For example, a 1535×1535
source uses the 1024 tier, 1536×1536 uses 1536, and 2048×2048 uses 2048 when enabled.
Sources larger than the cap are downsampled. Sources below 1024² remain filtered.
Fit-and-pad with white borders preserves edge text. This avoids a lower tier's
exact aspect match defeating the higher resolution.

Defaults: global batch 32, eight FSDP GPUs, TP 1, accumulation 4 (one image per GPU
microbatch), bf16, activation checkpointing, eager execution, 1024 caption tokens,
LR 1e-5, EMA 0.9995, timestep shift 0.3, and one epoch recalculated from populated
buckets. More buckets can increase repeated samples in partially filled batches.
At 2048², image token count is four times the 1024² count; full-model GPU memory
and throughput need a smoke run on the target hardware.

## Prepare a separate cache

Use the existing `i1_sft` environment and model cache from
[the SFT guide](../../torch_train/SFT.md). Exported v4 captions can be reused, but
the 1024 pixel cache cannot recover source detail. Build a **new cache per variant**
from original captioned Parquet images; the tier policy is included in the cache
fingerprint and incompatible caches are rejected.

```bash
export DENSE_PROJECT=/cephfs/liuxinyu/DenseText-Project
export EXP="$DENSE_PROJECT/i1/experiments/2026-09-22_high_resolution"
export TRAIN_PYTHON=/root/miniconda3/envs/i1_sft/bin/python
export RESOLUTION=2048  # or 1536, for both precompute and training
export HIGHRES_CACHE="$DENSE_PROJECT/artifacts/high_resolution_2026-09-22/cache_$RESOLUTION"
export HF_HUB_CACHE=/cephfs/liuxinyu/.cache/data_juicer/models
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

"$TRAIN_PYTHON" "$EXP/precompute.py" \
  --source /nfs_yaoyuan/liuxinyu/textdense_primary_english_captioned_v4 \
  --output-dir "$HIGHRES_CACHE" --resolution "$RESOLUTION" --workers 8
```

Precompute retains the production caption/decode/dimension checks and default
4096-side / 4096² source limits. It writes `summary.json`, `bucket_plan.json`,
`rejected.jsonl`, and `cache.jsonl`; the plan includes estimated RGB cache bytes.
RGB storage can approach 2.25× / 4× the 1024 cache if all sources use the top tier.
Repeat the same command to resume verified parts. Use the same Pillow version for
precompute and training. Alternatively, pass an original-image JSONL manifest to
the trainer for online resizing.

## Train

```bash
export SFT_MANIFEST="$HIGHRES_CACHE/cache.jsonl"
export SFT_INIT=/cephfs/liuxinyu/.cache/data_juicer/models/i1-3B/1024_resolution_checkpoint_torch.pt
export SFT_WORKDIR="$DENSE_PROJECT/artifacts/high_resolution_2026-09-22/train_$RESOLUTION"

# Separate two-step smoke; inspect memory and try populated high-resolution
# shapes as well, since two randomly selected steps need not hit the largest tier.
WANDB_MODE=disabled TRAIN_STEPS=2 SFT_WORKDIR="${SFT_WORKDIR}_smoke" \
  bash "$EXP/run_train.sh"

# Full one-epoch experiment; W&B uses the DenseText-SFT project.
bash "$EXP/run_train.sh"
```

The launcher accepts the shared trainer's extra arguments, including
`--batch_size`, `--grad_accum`, and `--total_steps`. If changing `TRAIN_GPUS`, ensure
the global batch is divisible by GPU count and accumulation; microbatch size is
`batch_size / TRAIN_GPUS / grad_accum`. `TRAIN_STEPS` overrides the epoch budget.
An existing `checkpoint.pt` in the workdir resumes optimizer/EMA/step state;
use a new workdir when changing resolution, cache, or initialization. The launcher
disables automatic VM completion by default; explicitly pass `--completion-config`
if needed. Logs are saved to `train.log`.

Compare against the 1024 captioned baseline using the same source set,
initialization, seed, and evaluation prompts. Record small-text quality, peak GPU
memory, actual source/tier counts, optimizer steps, and GPU time, since resolution
and extra partially filled buckets change the compute cost.

## Local checks

```bash
"$TRAIN_PYTHON" -m unittest discover -s "$EXP" -p 'test_*.py' -v
```

Checks cover tier thresholds, aspect priority, no-upscale behavior, bucket budgets,
legacy cache fingerprints, and exact precompute/online-loader parity. They do not
establish that the full 3B model fits on the production GPUs.
