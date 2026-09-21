# BizGenEval: starting checkpoint vs. SFT step 6262

This pipeline generates all 400 [BizGenEval](https://github.com/microsoft/BizGenEval)
images with both i1 checkpoints, evaluates them with the official Gemini-based
judge in `/cephfs/liuxinyu/BizGenEval`, summarizes each run, and writes direct
SFT-minus-starting-checkpoint comparisons.

The configured checkpoints are:

- Starting checkpoint: `/cephfs/liuxinyu/.cache/data_juicer/models/i1-3B/1024_resolution_checkpoint_torch.pt`
- SFT checkpoint: `/cephfs/liuxinyu/DenseText-Project/artifacts/sft_1024_full_20260917_094936/checkpoint.pt-000006262`

Both paths can be overridden through `START_CHECKPOINT` and `SFT_CHECKPOINT`.

## 1. Install the generation environment

Use the existing `i1_sft` environment, or follow `../../torch_inference/README.md`.
The scripts default to `/root/miniconda3/envs/i1_sft/bin/python`; override it
with `GENERATION_PYTHON` if needed.

## 2. Generate both image sets

From `i1/benchmark_eval/bizgeneval`:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
./run_generation.sh
```

The SFT checkpoint is generated first, followed by the starting checkpoint.
The run is resumable: existing PNGs are skipped. Defaults match the i1 release
inference settings (`250` denoising steps, CFG 12, CFG rescale 1.0, seed 42).
Both checkpoints use the same prompt partition, seeds, and sampling settings.
Keep `GPU_IDS` unchanged when resuming if you need bit-for-bit preservation of
each worker's seeded latent stream.

BizGenEval prompts are used directly and are not rewritten. The starting
checkpoint always generates at 1024x1024 because it was not trained on the
rectangular buckets. The SFT checkpoint uses the closest native 1024-SFT
aspect-ratio bucket (about one megapixel), derived from each row's declared
`aspect_ratio`, then `reference_image_wh`, then square as a fallback. Therefore,
the final comparison measures the deployed configuration of each checkpoint;
it does not isolate weight changes from the SFT model's aspect-ratio training.

The model context is capped at 1,024 tokens for a controlled comparison. The
full dataset currently has 107/400 prompts longer than that under T5Gemma's
tokenizer, so the default `CAPTION_OVERFLOW=truncate` truncates those prompts
for both checkpoints. Set `CAPTION_OVERFLOW=error` to fail instead.

Useful overrides:

```bash
# Quick end-to-end smoke set (first four benchmark rows, fewer steps, one GPU)
LIMIT=4 NUM_STEPS=2 GPU_IDS=0 ./run_generation.sh

# Run or resume only one checkpoint
CHECKPOINT_SET=starting GPU_IDS=0,1,2,3 ./run_generation.sh
CHECKPOINT_SET=sft      GPU_IDS=0,1,2,3 ./run_generation.sh

# Put outputs elsewhere
OUTPUT_ROOT=/path/to/bizgeneval_results GPU_IDS=0,1 ./run_generation.sh
```

Keep `LIMIT` and `OUTPUT_ROOT` identical when generation and evaluation are
separate commands. Do not use a smoke-test output directory for the full run,
because preparation intentionally rewrites the selected dataset manifest.

### Compare 1,024-token truncation with all input tokens

`run_context_comparison.sh` selects all 107 prompts whose untruncated T5Gemma
length exceeds 1,024 tokens and generates them twice with the SFT checkpoint.
The first arm truncates at 1,024. The second expands the model context to the
longest selected prompt and sets caption overflow to `error`, ensuring that it
retains every input token or fails instead of silently truncating.

```bash
GPU_IDS=0,1,2,3,4,5,6,7 ./run_context_comparison.sh
```

The two arms use the same prompt order, aspect-ratio buckets, GPU partition,
seeds, and inference settings. Outputs are written by default to
`artifacts/bizgeneval_sft_context_comparison_107`, including a side-by-side
`comparison.html` viewer and `inputs/token_lengths.tsv`. The all-token arm is
an inference-time context extension beyond the checkpoint's trained 1,024-token
context, so it measures extrapolation rather than a natively long-context SFT.

Use `PREPARE_ONLY=1` to inspect the selected prompts without starting GPU work,
or override `PROMPT_COUNT` to run a smaller subset. With the default count of
107, `SELECTION` has no effect because every qualifying prompt is included.

### Generate the five checkpoint/context settings

`run_five_settings.sh` generates the complete benchmark under these settings:

1. Starting checkpoint truncated at 256 tokens.
2. Starting checkpoint extended to retain every token.
3. Full-data SFT step 6262 truncated at 1,024 tokens.
4. DenseText-captioned SFT step 6245 truncated at 1,024 tokens.
5. DenseText-captioned SFT step 6245 extended to retain every token.

The script measures the longest prompt with the same T5Gemma tokenizer used by
inference, uses `caption-overflow=error` for both all-token arms, and is
resumable through `--skip-existing`. It runs the two DenseText-captioned step
6245 settings first, followed by the starting-checkpoint settings and the
full-data SFT checkpoint.

```bash
GPU_IDS=0,1,2,3,4,5,6,7 ./run_five_settings.sh
```

Use `PREPARE_ONLY=1` to prepare the inputs and inspect the measured token range
without loading a checkpoint. `LIMIT`, `NUM_STEPS`, `OUTPUT_ROOT`, and the other
generation overrides accepted by `run_generation.sh` are also available.

## 3. Install the official evaluator

The judge calls Gemini and therefore requires network/API access:

```bash
cd /cephfs/liuxinyu/BizGenEval
conda create -n bizgeneval python=3.12 -y
conda activate bizgeneval
pip install -r requirements.txt
pip install google-genai pyyaml requests
export GEMINI_API_KEY="your-api-key"
```

The extra packages are used by the repository's `utils/gemini.py`. The default
judge and concurrency come from `config/evaluation_config.yaml` (currently
`gemini-3-flash-preview` and 64 workers). Gemini evaluation incurs API cost.

## 4. Evaluate and compare both checkpoints

```bash
cd /cephfs/liuxinyu/DenseText-Project/i1/benchmark_eval/bizgeneval
conda activate bizgeneval
export GEMINI_API_KEY="your-api-key"
./run_evaluation.sh
```

Evaluation is resumable through BizGenEval's per-image result cache. The script
first verifies that every selected image exists, then runs the official
`evaluation.image_evaluation` and `evaluation.summarize` modules. It refuses to
summarize partial/malformed Gemini responses; rerun the same command to retry
only those incomplete judgments.

The default output root is
`/cephfs/liuxinyu/DenseText-Project/artifacts/bizgeneval_start_vs_sft6262`:

```text
inputs/                         prepared dataset and expected filenames
images/starting_checkpoint/     starting-checkpoint PNGs
images/checkpoint_000006262/    SFT PNGs
eval_results/<checkpoint>/      per-image Gemini judgments
summaries/<checkpoint>/         official domain/dimension CSVs and summary.json
comparison/                     SFT-minus-starting comparison CSVs
logs/<checkpoint>/              per-GPU generation logs
```

To evaluate only one image set, set `CHECKPOINT_SET=starting` or
`CHECKPOINT_SET=sft`. To use another evaluator config, set
`EVALUATION_CONFIG=/path/to/evaluation_config.yaml`.
