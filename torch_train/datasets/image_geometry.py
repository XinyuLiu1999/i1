"""Pixel-budgeted image shapes, following Lumina-Image-2.0/imgproc.py's approach.

Unlike Lumina's center crop, the SFT loader fits and pads to retain edge text.
All shapes in this module use (height, width), not PIL's (width, height).
"""

import math


class BucketSelector:
    """Validate a bucket policy once, then select buckets for many source sizes."""

    def __init__(self, config):
        buckets = config["buckets"]
        resolutions = config.get("bucket_resolutions")
        if resolutions is not None:
            if len(resolutions) != len(buckets) or any(
                not isinstance(r, int) or r <= 0 or bh * bw > r * r
                for r, (bh, bw) in zip(resolutions, buckets)
            ):
                raise ValueError("bucket_resolutions must give a positive pixel-budget side for every bucket.")
        self.min_area = config.get("min_image_area", 0)
        self.min_side = config.get("min_image_side", 0)
        self.pad = config.get("resize_mode", "pad") == "pad"
        self.allow_upscale = config.get("allow_upscale", False)
        self.buckets = [(index, bh, bw, bw / bh, -bh * bw,
                         None if resolutions is None else resolutions[index])
                        for index, (bh, bw) in enumerate(buckets)]

    def __call__(self, height, width):
        """Return a bucket index or an exclusion reason under the training policy."""
        if width * height < self.min_area or min(width, height) < self.min_side:
            return None, "source_too_small"
        resize_scale = min if self.pad else max
        aspect = width / height
        best = None
        for index, bh, bw, bucket_aspect, negative_area, resolution in self.buckets:
            # Mixed-resolution experiments first choose the largest tier supported
            # by the source area, then minimize padding within that tier. Otherwise
            # an exact-aspect 1024 bucket can win even for a 2048 source image.
            if resolution is not None and height * width < resolution ** 2:
                continue
            if not self.allow_upscale and resize_scale(bh / height, bw / width) > 1.0:
                continue
            # Prefer matching aspect ratios, then the largest eligible area.
            score = (abs(math.log(bucket_aspect / aspect)), negative_area, index)
            if resolution is not None:
                score = (-resolution, *score)
            if best is None or score < best:
                best = score
        if best is None:
            return None, "no_bucket_without_upscaling"
        return best[-1], None


def select_bucket(height, width, config):
    """Return a bucket index or an exclusion reason under the training policy."""
    return BucketSelector(config)(height, width)


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
