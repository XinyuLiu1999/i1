"""Captioned SFT with 1024/1536[/2048] pixel-budget tiers."""

from configs.sft_1024_captioned import get_config as base_config
from datasets.image_geometry import generate_buckets


def get_config(max_resolution=2048):
    if max_resolution not in (1536, 2048):
        raise ValueError("max_resolution must be 1536 or 2048.")
    config = base_config()
    config.input.bucket_resolutions = [1024] * len(config.input.buckets)
    for resolution in (1536, 2048):
        if resolution > max_resolution:
            break
        # Exact 3:2 and 2:3 anchors, quantized to the model's 16px multiple.
        short_side = int((resolution ** 2 / 6) ** 0.5) // 16 * 16 * 2
        long_side = short_side * 3 // 2
        buckets = generate_buckets(
            resolution, step=32, max_ratio=3.0,
            extra_shapes=[(short_side, long_side), (long_side, short_side)],
        )
        config.input.buckets.extend(buckets)
        config.input.bucket_resolutions.extend([resolution] * len(buckets))
    config.image_size = max_resolution
    config.fsdp_axis_size = 8
    # Global batch 32 / 8 GPUs / 4 accumulation steps = one image per microbatch.
    config.grad_accum_steps = 4
    config.use_grad_ckpt = True
    config.compile = False
    config.input.num_workers = 2
    config.wandb.experiment = f"i1-multires-{max_resolution}-captioned"
    return config
