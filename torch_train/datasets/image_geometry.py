"""Pixel-budgeted image shapes, following Lumina-Image-2.0/imgproc.py's approach.

Unlike Lumina's center crop, the SFT loader fits and pads to retain edge text.
All shapes in this module use (height, width), not PIL's (width, height).
"""

import math


def select_bucket(height, width, config):
    """Return a bucket index or an exclusion reason under the training policy."""
    resolutions = config.get("bucket_resolutions")
    if resolutions is not None:
        if len(resolutions) != len(config["buckets"]) or any(
            not isinstance(r, int) or r <= 0 or bh * bw > r * r
            for r, (bh, bw) in zip(resolutions, config["buckets"])
        ):
            raise ValueError("bucket_resolutions must give a positive pixel-budget side for every bucket.")
    if width * height < config.get("min_image_area", 0) or min(width, height) < config.get("min_image_side", 0):
        return None, "source_too_small"
    resize_scale = min if config.get("resize_mode", "pad") == "pad" else max
    candidates = []
    for index, (bh, bw) in enumerate(config["buckets"]):
        # Mixed-resolution experiments first choose the largest tier supported
        # by the source area, then minimize padding within that tier. Otherwise
        # an exact-aspect 1024 bucket can win even for a 2048 source image.
        if resolutions is not None and height * width < resolutions[index] ** 2:
            continue
        if not config.get("allow_upscale", False) and resize_scale(bh / height, bw / width) > 1.0:
            continue
        # Prefer matching aspect ratios, then the largest eligible area.
        score = (abs(math.log((bw / bh) / (width / height))), -bh * bw)
        if resolutions is not None:
            score = (-resolutions[index], *score)
        candidates.append((score, index))
    if not candidates:
        return None, "no_bucket_without_upscaling"
    return min(candidates)[1], None


def generate_buckets(resolution, step=32, max_ratio=3.0, extra_shapes=()):
    """Generate quantized shapes along a fixed-area frontier, deterministically.

    `step` controls aspect granularity independently of the model's 16px multiple.
    Include transposes and optional exact-aspect anchors within the same budget.
    """
    if (not isinstance(resolution, int) or resolution <= 0
            or not isinstance(step, int) or step <= 0 or step % 16
            or resolution % step):
        raise ValueError("resolution must be positive and divisible by step; step must be a multiple of 16.")
    if not math.isfinite(max_ratio) or max_ratio < 1:
        raise ValueError("max_ratio must be finite and at least 1.")
    budget = (resolution // step) ** 2
    shapes = set()
    for height in range(1, budget + 1):
        width = budget // height
        if max(width, height) / min(width, height) <= max_ratio:
            shapes.update(((height * step, width * step), (width * step, height * step)))
    for shape in extra_shapes:
        if (len(shape) != 2 or any(not isinstance(x, int) or x <= 0 or x % 16 for x in shape)
                or math.prod(shape) > resolution ** 2 or max(shape) / min(shape) > max_ratio):
            raise ValueError(f"Invalid extra bucket {shape} for the requested pixel/aspect budget.")
        shapes.add(tuple(shape))
    return sorted(shapes, key=lambda shape: (shape[1] / shape[0], shape))
