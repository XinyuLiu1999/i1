# Region-weighted flow SFT · 2026-09-21

Initialize both arms from the **original 1024 base checkpoint**, with fresh
optimizer/EMA/step state. The safe default is ordinary global velocity MSE
(`REGION_WEIGHT=0`), with perceptual loss disabled. Compare a gradient-calibrated
positive `REGION_WEIGHT` against `REGION_WEIGHT=0 PERCEPTUAL_WEIGHT=0`, using the
**same filtered cache, seed, sample order and training budget**. This measures the
incremental effect of the regional velocity loss over ordinary SFT. Perceptual
supervision remains available as an opt-in follow-up after separate calibration.
The existing ~200K-image SFT checkpoint is a historical reference, not the matched
control: the comparison requires the same accepted cache and training budget. A continuation
experiment would require its own matched continuation control to separate the
new loss from extra training; it is not the default here.

## Objective and defaults

For OCR region `j` in image `i`, let `h_ij` be its height after the exact
fit-and-pad transform, measured in training pixels, and set
`w_ij = min(1, h_ref / h_ij)`. The defaults are `h_ref=4` and minimum accepted
height 4 pixels. Regions are rasterized separately at training resolution and
combined by a pixelwise maximum, so overlap is deterministic and never sums
duplicate detections. Averaging each 8×8 cell gives the latent weight map `W_i`.
Let `E_i = mean_channels((predicted_velocity − target_velocity)²)`:

```text
L = mean_i(mean_pixels(E_i))
    + λ · mean_i(mean_pixels(W_i² · E_i))
    + γ · sum_k mean(W_ik² · (F_k(predicted_RGB) − F_k(target_RGB))²)
```

The cache stores `w_ij`, while each loss applies `w_ij²`. Squaring only in the
objective provides the intended inverse-height-squared correction without storing
very small squared values in float16 (for example, a 1000px box stores `4e-3`, not
`1.6e-5`). The mean includes all batch, channel and spatial elements, including
zeros outside text regions. Empty masks contribute zero; equal-sized
accumulation/distributed batches preserve sample weighting. `λ=γ=0` is ordinary
flow MSE.

Defaults are `λ=0` and `γ=0`: no regional term, perceptual decoding or ODM loading
occurs unless explicitly enabled. `run_train.sh` requires `REGION_WEIGHT` to be
set, even for the zero-weight control.

The optional perceptual term uses the frozen pretrained ODM ResNet from the local
`FluxText` checkout, matching its four layer1–layer4 feature losses with a
full-tensor, squared-mask-weighted mean at each scale. `γ=20` appears in the checked-in
FLUX-Text training YAML, but its normalization differs and it is **not tuned for
i1**. Opt in only after inspecting real noisy reconstructions and separately
calibrating its gradient norm across timesteps.

When perceptual supervision is enabled, the reference reconstructs with
`noise - predicted_velocity`. Our flow target
has the opposite sign (`clean_latent - noise`), so we use
`predicted_clean_latent = clean_latent + predicted_velocity - target_velocity`.
We reverse FLUX.2's latent normalization and decode through the frozen VAE.
This matches the reference training reconstruction, not the alternative
`z_t + (1-t)*predicted_velocity`, which would change timestep weighting.
Decoded and target RGB are both in `[-1,1]`, without clamping or resizing. Only
the target feature branch runs under `no_grad`; the predicted branch retains
gradients through the frozen decoder and ODM encoder to the denoiser.

Our weights remain soft 8×8 latent maps; FLUX-Text samples its position mask
with nearest interpolation onto the packed 16px token grid. Perceptual weights
are resized by area to the actual feature shapes, retaining fractional weights
and supporting rectangular buckets. The 4px feature map repeats the cached 8px
weights rather than recovering original pixel boundaries. We retain caption-only
i1 SFT without glyph/editing conditions.
There is no transcription/CTC loss and no added inference component.

When enabled, the perceptual branch runs in FP32, one image at a time by default
(`PERCEPTUAL_CHUNK_SIZE=1`), with non-reentrant activation checkpointing
(`PERCEPTUAL_CHECKPOINTING=1`). It still supervises every masked image, with
empty-mask images contributing zero to the mean over **all** images. Checkpoint
recomputation trades extra compute for lower activation memory. Set
`PERCEPTUAL_CHECKPOINTING=0` to disable it after checking available GPU memory.

The objective is now `region-weighted-flow-v5` and the cache format is v2. Old
union-mask caches are not reusable, and v1–v4 checkpoints are rejected for
resume. The regional/perceptual weights, checkpoint SHA256, encoder source SHA256,
reconstruction, mask policy, precision and memory settings are recorded and
checked on resume. Start matched arms in fresh workdirs from the original base
checkpoint.

Training settings match the recorded run in `artifacts/sft_densetext_captioned_v4_1024`
(W&B run `20260919_174958-9y30rvkx`), including its launch overrides:

| Setting | Default |
|---|---|
| Resolution / geometry | 1024 pixel budget; same 45 aspect buckets; fit + white padding; no upscale |
| Captions / sources | 1024 T5Gemma tokens; Danqing, Monet, Paper2Fig100k, ChartGalaxy-real |
| Batch / parallelism | global 32; FSDP 8; TP 1; accumulation **1**; microbatch 4/GPU |
| Schedule | **1 epoch**, recalculated after filtering as `sum(ceil(bucket_count/32))` |
| Optimizer / EMA | LR `1e-5`; Adam β=(0.9,0.95); clip 1; EMA `0.9995`; seed 0 |
| Execution | bf16 AMP; activation checkpointing; eager (`--no_compile`) |
| Flow / output | timestep shift 0.3; logs every 50; checkpoints every 1000, retained every 2500 plus first/final |
| Regional | Height weight `min(1, 4/h_i)`, max overlap, squared-mask-weighted MSE; λ must be calibrated |
| Perceptual | Disabled (`PERCEPTUAL_WEIGHT=0`); optional ODM layer1–4 branch retained |

## 0. Environment and paths

Use the existing production `qwen` environment for captioning and `i1_sft` for
export, CPU precompute and training. The latter needs PyTorch 2.9.1+cu128,
Transformers, Diffusers, PyArrow, NumPy, Pillow and W&B. Use the **same Pillow
version** for precompute and training (recorded in the cache fingerprint).
The shared HF cache must contain the production caption model,
`google/t5gemma-2b-2b-ul2-it` and the `black-forest-labs/FLUX.2-dev` VAE.

```bash
export DENSE_PROJECT=/cephfs/liuxinyu/DenseText-Project
export EXP="$DENSE_PROJECT/i1/experiments/2026-09-21_region_weighted_flow"
export UNIFIED_SOURCE=/nfs_yaoyuan/liuxinyu/textdense_primary_english_unified_v2
export CAPTION_OUTPUT=/nfs_yaoyuan/liuxinyu/textdense_primary_english_captioned_v4
export REGION_CACHE="$DENSE_PROJECT/artifacts/region_weighted_flow_2026-09-21/cache_1024_all_captions"
export SFT_MANIFEST="$REGION_CACHE/cache.jsonl"
export SFT_INIT=/cephfs/liuxinyu/.cache/data_juicer/models/i1-3B/1024_resolution_checkpoint_torch.pt
export SFT_WORKDIR="$DENSE_PROJECT/artifacts/region_weighted_flow_2026-09-21/base_region_calibrated_p0"
export PERCEPTUAL_WEIGHT=0
export PERCEPTUAL_CHECKPOINT="/cephfs/liuxinyu/.cache/data_juicer/models/models--GD-ML--FLUX-Text/snapshots/dcebeaee2f9fdb2876706a9b803b9413408f1f4f/epoch_100.pt"
export FLUXTEXT_ROOT="$DENSE_PROJECT/FluxText"
export GPU_COMPLETION_CONFIG="$DENSE_PROJECT/local_captioning/completion_config.json"
export TRAIN_PYTHON=/root/miniconda3/envs/i1_sft/bin/python
export HF_HUB_CACHE=/cephfs/liuxinyu/.cache/data_juicer/models
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

The following ODM setup is only needed for opt-in perceptual experiments.
The official pretrained text-feature model is already in the shared HF cache.
The config defaults to the pinned snapshot above under `HF_HUB_CACHE`; override
`PERCEPTUAL_CHECKPOINT` to use another location. Validate the existing file with:

```bash
"$TRAIN_PYTHON" "$EXP/prepare_perceptual.py"
```

This verifies [the official ODM checkpoint](https://huggingface.co/GD-ML/FLUX-Text/blob/main/epoch_100.pt)
(~710 MB) against its published SHA256 and strictly validates all convolutional
and BN weights. An existing file is checked without network access. If the file
is missing, the script downloads it and resumes `.partial` downloads.
Perceptual-enabled training uses the local file without network access and fails
early if it is missing. Keep the local `FluxText/src/loss/ocr_loss/base_model/ODM_encoder.py`
source available at `FLUXTEXT_ROOT`; no FluxText pipeline dependencies are needed.

The completion JSON supplies only `name` and `password`; keep it private
(`chmod 600`) and do not commit it. Both captioning and training set
`vmids=[socket.gethostname().split('.')[0]]` at runtime: this platform uses the
short hostname as the VM ID. Stored JSON `vmids` are ignored, so each job releases
only its own VM. Invalid/localhost hostnames fail instead of using stored targets.
Reserve space for a new RGB cache (~600 GB before filtering), masks
(~6 GB), and several ~42 GB training checkpoints.

## 1. GPU captioning (skip when reusing completed v4 captions)

The existing launcher selects the same four sources and preserves raw OCR in the
export. To recaption with changed settings, use a new `CAPTION_OUTPUT`.

```bash
/root/miniconda3/envs/qwen/bin/python "$DENSE_PROJECT/local_captioning/run_unified_production.py" \
  --stage prepare --dataset "$UNIFIED_SOURCE" --output-dir "$CAPTION_OUTPUT" \
  --sources danqing monet paper2fig100k chartgalaxy_real
GPUS=0,1,2,3,4,5,6,7 /root/miniconda3/envs/qwen/bin/python \
  "$DENSE_PROJECT/local_captioning/run_unified_production.py" \
  --stage caption --dataset "$UNIFIED_SOURCE" --output-dir "$CAPTION_OUTPUT" \
  --sources danqing monet paper2fig100k chartgalaxy_real \
  --completion-config "$GPU_COMPLETION_CONFIG"
```

Resume with identical GPU count/order and settings. Completion follows the merged
`captions.jsonl`; omit `--completion-config` to keep the captioning VM running.
The production launcher never sends completion notifications for its CPU-only
`prepare` or `export` stages, even if `GPU_TASK_COMPLETION_CONFIG` is inherited.

## 2. CPU export, filtering, RGB cache and OCR masks

```bash
test -f "$CAPTION_OUTPUT/captions.jsonl"
"$TRAIN_PYTHON" "$DENSE_PROJECT/local_captioning/run_unified_production.py" \
  --stage export --dataset "$UNIFIED_SOURCE" --output-dir "$CAPTION_OUTPUT" \
  --sources danqing monet paper2fig100k chartgalaxy_real
"$TRAIN_PYTHON" "$EXP/precompute.py" --source "$CAPTION_OUTPUT" \
  --output-dir "$REGION_CACHE" --workers 8 --reference-region-height 4
"$TRAIN_PYTHON" - <<'PY'
import json, os
from pathlib import Path
s = json.loads((Path(os.environ['REGION_CACHE']) / 'summary.json').read_text())
assert s['status'] == 'complete' and s['verified_all_written_bytes']
print('accepted/masked/unmasked:', s['count'], s['masked_images'], s['unmasked_images'])
print('rejected:', s['rejected_counts'])
print('OCR audit:', s['mask_stats'])
print('steps/epoch:', sum((n + 31)//32 for n in s['bucket_counts'].values()))
PY
```

Captions containing **photograph** are accepted regardless of word position or
case. Existing caption status, 1024-token limit (no truncation), image
decode/dimension, ≤4096-side/4096²-area and bucket checks still apply.
Rejections go to `rejected.jsonl`.

Caches built with the former photograph exclusion still omit those images.
Rebuild from the captioned export in a new cache directory (the example above
uses `cache_1024_all_captions`), then recalibrate λ and use fresh training workdirs
for both matched arms. Changing this policy invalidates precompute resume settings;
editing an existing cache's metadata cannot restore excluded samples.

OCR defaults: confidence ≥0.8; nonempty transcript; finite, in-bounds polygon/xyxy
box; height ≥4 training pixels. Each valid region receives
`min(1, 4 / transformed_height)` before overlaps are combined by maximum. The
training-pixel weights are averaged over 8×8 cells to make float16 latent weight
maps. These mark OCR boxes/polygons, not individual glyphs. Override the reference
height with `--reference-region-height`; doing so requires a new cache.
Missing/invalid OCR or source-dimension mismatch produces an audited zero mask;
the image remains eligible for global flow loss. A missing OCR column or zero
usable masks in the entire cache fails precompute.

Rerun the same command to resume verified parts. Changed inputs, filters or mask
thresholds require a **new cache directory**. Train only from the top-level
`cache.jsonl` after its `summary.json` is complete. CPU export/precompute never
calls the task manager. This stage caches RGB and masks, not VAE/text embeddings.

## 3. GPU training and automatic shutdown

Use the same eight-GPU allocation class as the previous run, with working NCCL,
bf16, sufficient `/dev/shm` and checkpoint storage. Launch from `tmux` or a durable
scheduler allocation; authenticate W&B (`wandb login`) for the default online logs.

```bash
test -f "$SFT_INIT" && test -f "$SFT_MANIFEST"
test "$(stat -c '%a' "$GPU_COMPLETION_CONFIG")" = 600
"$TRAIN_PYTHON" -c 'import torch; assert torch.cuda.device_count()==8; assert torch.cuda.is_bf16_supported()'
# First, a separate two-step control smoke: saves a checkpoint but keeps the VM running.
DISABLE_AUTO_SHUTDOWN=1 REGION_WANDB=0 REGION_WEIGHT=0 TRAIN_STEPS=2 \
  SFT_WORKDIR="${SFT_WORKDIR}_smoke" bash "$EXP/run_train.sh"
# Production: substitute the coefficient selected by the calibration and sweep below.
REGION_WEIGHT="$CALIBRATED_REGION_WEIGHT" PERCEPTUAL_WEIGHT=0 bash "$EXP/run_train.sh"
```

### Calibrating λ from real-batch gradients

Do not inherit `λ=1` from FLUX-Text or UniGlyph. Both use different mask and loss
normalizations; UniGlyph's observation that `λ=4` hurts therefore does not define
a bound for this objective. Squaring the soft mask reduces the raw regional
gradient, so the selected λ will likely be several times larger than for the v4
linear-mask objective (roughly 5× is a planning estimate, not a coefficient to
inherit). Calibrate it from the v5 gradients at the initialization checkpoint, before
optimizer clipping or updates, using the exact production loader, model mode,
AMP policy, noise, timestep distribution and trainable parameters:

1. Sample 32–64 real batches across populated aspect buckets and sources. Keep
   zero-region images at their natural frequency, and also report statistics
   conditional on a nonempty regional gradient.
2. On one shared forward graph per batch, compute `L_global` and the unscaled
   `L_region` (`λ=1`). Use `torch.autograd.grad` separately to measure
   `g_g=∇θL_global` and `g_r=∇θL_region`. Measure FP32 norms before gradient
   clipping. With FSDP, all-reduce the sums of squared local shard gradients and
   their dot product before taking square roots.
3. Record `G=||g_g||`, `R=||g_r||`, and `D=dot(g_g,g_r)` for every batch.
   Sum `G²`, `R²` and `D` across sampled batches (and across FSDP shards), while
   also reporting per-batch log-ratios and cosines so outliers remain visible.
4. Define the combined norm
   `T(λ)=sqrt(G² + 2λD + λ²R²)` and the measured regional share
   `q(λ)=λR/T(λ)`. Solve `q(λ_s)=s` for targets
   `s∈{0.30,0.40,0.50}`. The positive solution is
   `λ_s=[s²D+s·sqrt(s²D²+(1-s²)G²R²)]/[(1-s²)R²]`.
   Verify these shares batch-by-batch; strong negative cosine or a wide ratio
   distribution is a reason to stratify by timestep/bucket rather than trust one λ.
5. Run short matched jobs for the zero control and coefficients around the 40%
   point, for example `{0, λ_40/2, λ_40, 2λ_40}`. Select using stability,
   clipping rate, global validation quality and small-text accuracy—not loss
   magnitude alone—then run the full matched control/region pair in fresh workdirs.

Run the measurement with the same `SFT_INIT`, `SFT_MANIFEST` and eight-GPU
allocation as training:

```bash
export CALIBRATION_WORKDIR="$DENSE_PROJECT/artifacts/region_weighted_flow_2026-09-21/calibration_base_v5"
CALIBRATION_BATCHES=64 bash "$EXP/run_calibrate.sh"
"$TRAIN_PYTHON" - "$CALIBRATION_WORKDIR/calibration.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
print('coefficients:', report['coefficients'])
print('matched sweep:', report['sweep'])
print('per-batch shares:', report['all_batches']['shares'])
print('warnings:', report['warnings'])
PY
```

`calibrate.py` reuses the shared trainer's initialization, trainable-parameter
selection, parallelization, encoders, noise/timestep preparation, and the exact
production region iterator. It measures the first 64 shuffled, interleaved
production batches without replacing zero-mask images or rebalancing sources.
Inspect the reported bucket/source coverage; increase `CALIBRATION_BATCHES` if
important groups are absent. `CALIBRATION_BATCHES=2` in a separate workdir is a
GPU smoke check, not a coefficient-selection run. Shared trainer options such as
`--batch_size` and `--grad_accum` can be passed to the launcher; use the production
settings for the actual calibration. Distributed measurement requires TP=1 and
a single FSDP group (`--fsdp` equals the launched world size).

The single-device path uses `torch.autograd.grad` twice. FSDP2 instead uses two
backward passes on the same retained graph, clearing `.grad` between them, so
its parameter-swapping and reduce-scatter hooks produce the global-batch mean
gradients. It then sums the squared FP32 local shard gradients and their dot
products across ranks before taking norms (scalar sums accumulate in FP64).
Accumulation sums each loss's microbatch gradients before measuring norms.
There are no optimizer/EMA allocations, clipping, updates, training checkpoints,
W&B runs or completion notifications, including inherited shutdown credentials.
An existing training checkpoint or calibration output is rejected; use a fresh
`CALIBRATION_WORKDIR`.

`batches.jsonl` preserves each completed batch's `G`, `R`, `D`, squared norms,
log10(G/R), cosine, losses, mask statistics, sample IDs, source counts and all
sampled timesteps. `calibration.json` is written only after successful measurement
and includes checkpoint SHA256, objective/cache provenance, execution settings,
λ30/λ40/λ50, the proposed sweep, per-batch shares, quantiles over all batches and
over nonempty regional gradients, and bucket/batch-mean-timestep summaries.
The timestep summaries group whole batches by their mean timestep; they are
diagnostics, not isolated per-timestep gradients. Undefined ratios/cosines are
JSON `null`; zero-region batches remain in the aggregate global norm. A run with
no regional gradient fails without recommending a coefficient, retaining its
batch measurements for inspection. Warnings flag negative alignment, wide norm
ratios, incomplete coverage and fewer than 32 batches. The measured λ40 is a
sweep center; set `CALIBRATED_REGION_WEIGHT` only after the matched short-job
evaluation in step 5.

Recalibrate after changing the mask rule, `h_ref`, data mixture, timestep policy,
trainable parameter set or initialization checkpoint. The final checkpoint records
both λ and the complete v2 mask settings.

`run_train.sh` defaults `SFT_INIT` to the original 1024 base checkpoint and uses
`--init_from` for a fresh workdir. An existing `checkpoint.pt`
in that workdir takes precedence: it restores raw weights, optimizer, EMA, step
and sampler position. It checks objective/data/topology provenance and rejects
incompatible resumes; stochastic noise RNG restarts as in the shared trainer.
Use a distinct workdir for every arm; do not reuse a continuation workdir for a
base-initialized run. `TRAIN_STEPS=1500` explicitly replaces the
epoch budget; when resuming it is the **final step number**, not additional steps.

Run the required matched control in a **separate job/allocation**, keeping the
same `SFT_INIT`, `SFT_MANIFEST` and budget:

```bash
REGION_WEIGHT=0 PERCEPTUAL_WEIGHT=0 \
  SFT_WORKDIR="$DENSE_PROJECT/artifacts/region_weighted_flow_2026-09-21/base_w0_p0" \
  bash "$EXP/run_train.sh"
```

Evaluate base, `base_w0_p0` and `base_region_calibrated_p0` on identical held-out BizGenEval prompts,
seeds and inference settings, including small-text accuracy. The two SFT arms
isolate the regional-flow contribution. A later perceptual experiment should use
a separate workdir and explicitly positive `PERCEPTUAL_WEIGHT`, holding
the calibrated `REGION_WEIGHT` fixed. Compare both equal training steps and equal GPU time,
and record peak GPU memory. Run a new GPU smoke with perceptual supervision
enabled before that experiment; flow-only memory measurements do not establish
that the extra branch fits.

Rank 0 notifies the task manager only after the final checkpoint is saved, all
ranks finish and W&B flushes. Exceptions/nonfinite loss skip notification; a
failed notification retries three times and exits nonzero with the checkpoint
intact. `DISABLE_AUTO_SHUTDOWN=1` disables notification, including an inherited
completion environment variable. Logs: `train.log`, W&B global/regional losses,
mean regional weight, masked-image fraction, unweighted perceptual loss and weighted
perceptual loss. Checkpoints retain standard inference format; frozen ODM/VAE
parameters and their optimizer states are not added to the denoiser checkpoint.

## 4. Render the short-sweep comparison on multiple GPUs

After all four 500-step arms finish, run:

```bash
export EXP=/cephfs/liuxinyu/DenseText-Project/i1/experiments/2026-09-21_region_weighted_flow
GPU_IDS=0,1,2,3,4,5,6,7 bash "$EXP/run_sweep_eval.sh"
```

The default `SWEEP_ROOT` is
`artifacts/region_weighted_flow_2026-09-21/sweep_500` under the project root.
The launcher evaluates `w0`, `w29.778307248555066`, `w59.55661449711013`, and
`w119.11322899422026`. It prepares **40 category-stratified BizGenEval prompts**
and generates one image per prompt/checkpoint with **50 denoising steps**.
Each checkpoint runs across the selected GPUs, then the next checkpoint starts.
Each GPU loads its own complete inference model; this is prompt parallelism.

All arms use identical prompt partitions and worker seeds (`SEED + worker index`,
default `SEED=0`), 1024×1024 output, CFG 12, CFG rescale 1.0, timestep shift 0.3,
and batch size 1. Prompt rewriting is disabled; each arm uses the same
1024-token truncation policy. The subset is for initial model selection; retain
separate prompts for final evaluation.

```bash
# Inspect commands and validate checkpoints/prompts without loading GPU models.
GPU_IDS=0,1,2,3 bash "$EXP/run_sweep_eval.sh" --dry-run

# Override GPU count, prompt count, or denoising steps.
GPU_IDS=0,1,2,3 NUM_PROMPTS=80 NUM_STEPS=100 bash "$EXP/run_sweep_eval.sh"

# Evaluate a subset of the arms, with an explicit output directory.
GPU_IDS=0,1 REGION_WEIGHTS=0,59.55661449711013 \
  OUTPUT_ROOT="$DENSE_PROJECT/artifacts/region_weighted_flow_2026-09-21/sweep_500/evaluation_control_vs_lambda40" \
  bash "$EXP/run_sweep_eval.sh"
```

`TRAIN_PYTHON`, `SWEEP_ROOT`, `OUTPUT_ROOT`, `BIZGENEVAL_SOURCE`, `SEED`,
`CHECKPOINT_STEP` (default 500), and `GPU_LAUNCH_DELAY` (default 10 seconds)
are also configurable. `GPU_IDS` defaults to GPU 0. CLI equivalents are listed
by `bash "$EXP/run_sweep_eval.sh" --help` and override environment defaults.
Checkpoint loads are staggered to limit simultaneous host memory/storage demand.
If the original BizGenEval source path is unavailable, the launcher uses the
existing 400-prompt copy in `artifacts/bizgeneval_evaluation/inputs/`, then
`artifacts/bizgeneval_start_vs_sft6262/inputs/` as a fallback. It prints the selected
source; `BIZGENEVAL_SOURCE` overrides this lookup.
Every selected arm must have a matching checkpoint-save entry in its `train.log`;
the initial step-1 checkpoint does not qualify.

Outputs default to `$SWEEP_ROOT/evaluation_40prompts_50steps/`:

```text
inputs/                    selected metadata and shared output filenames
evaluation.json            settings, worker seeds/ranges, checkpoint fingerprints
comparison.html            searchable side-by-side gallery for all selected arms
w0/                        control PNGs
w29.778307248555066/        lambda40/2 PNGs
w59.55661449711013/         lambda40 PNGs
w119.11322899422026/        2*lambda40 PNGs
logs/w*/gpu*.log            per-GPU generation logs
```

Rerun the same command to skip existing images and finish missing ones. The
manifest rejects changed checkpoints, prompts, settings, or GPU partitions in
an existing output directory; choose a new `OUTPUT_ROOT` for those changes.
After all images are verified, open `comparison.html` to compare every prompt's
outputs side by side. The page includes expandable prompt text, search by
prompt/category/ID, adjustable image size, and links to the full-resolution PNGs.
It works locally without a web server; keep the HTML beside the image folders
when copying the results. Compare small-text accuracy and overall quality.
This script generates images and the comparison page; it does not run the
external API judge or automatically choose λ.

To add the existing base, full-data SFT, and captioned SFT renders (sets 01, 03,
and 04) to a completed gallery without running inference:

```bash
"$TRAIN_PYTHON" "$EXP/add_sweep_references.py"
```

Use `--output-root` for another completed sweep or `--images-root` for another
`bizgeneval_evaluation/images` directory. The script verifies exact prompt
matches, copies only the selected images into the gallery's `references/`
directory, and rebuilds `comparison.html`. It preserves rectangular image shapes
and labels the references' context/seed/geometry differences. The saved
`comparison_references.json` also retains these columns when the evaluation
launcher rebuilds the HTML later.

## Local checks

```bash
"$TRAIN_PYTHON" -m unittest discover -s "$EXP" -p 'test_*.py' -v
```

Tests cover filtering, mask geometry, loss/gradient arithmetic, multi-process CPU
precompute, cache corruption, rectangular TP broadcasts, and the actual trainer
loop with a tiny DiT and stubbed frozen encoders/completion endpoint. Perceptual
tests check reconstruction sign, frozen parameters/BN statistics, actual input
gradients, full-batch/accumulation equivalence, empty masks, strict ODM loading,
checkpointed gradient equivalence, and resume rejection after weight changes.
Calibration tests check the coefficient solver, empty-region reporting,
microbatch gradient accumulation, two-process CPU FSDP2 measurements against a
full-batch reference, and unchanged model weights/no optimizer/no shutdown in
the actual calibration runner. Production 3B-model FSDP execution must be checked
with the GPU smokes above.
