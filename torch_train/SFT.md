# PyTorch SFT for dense text

The SFT configs initialize from an existing i1 checkpoint, train on original
images grouped into rectangular buckets, and accept captions up to 1,024 tokens.
The VAE and T5Gemma encoder remain frozen; the DiT and its text adapter are trained.
Use the environment described in README.md. The JSONL SFT path does not require
TensorFlow; the existing TFRecord training path is unchanged.

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
