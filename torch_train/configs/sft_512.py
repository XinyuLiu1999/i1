from configs.i1_512 import get_config as _base_config
from utils.config import ConfigDict
from datasets.image_geometry import generate_buckets


def get_config():
    config = _base_config()
    config.init_from = ""  # --init_from /path/to/512_resolution_checkpoint_torch.pt
    config.total_steps = 10_000
    config.lr = 1e-5  # Starting point for SFT; validate on held-out rendering prompts.
    config.token_len = 1024
    config.use_grad_ckpt = True
    config.grad_accum_steps = 4
    config.compile = True
    config.ckpt_steps = 1000
    config.keep_ckpt_steps = 5000
    config.input = ConfigDict(dict(
        type="bucketed",
        manifest="",  # --manifest /path/to/corrected_images.jsonl (see SFT.md)
        image_root="",  # Empty: resolve image paths relative to the manifest.
        # Lumina-style fixed pixel budget; retain exact 3:2 / 2:3 anchors.
        # (height, width), at most 512^2 pixels, with no center cropping.
        buckets=generate_buckets(512, step=32, max_ratio=3.0,
                                 extra_shapes=[(416, 624), (624, 416)]),
        batch_size=32,
        num_workers=4,
        min_image_area=512 * 512,
        min_image_side=0,
        allow_upscale=False,
        resize_mode="pad",  # Preserve all text; white letterbox padding.
        caption_overflow="error",
        # Omit each bucket's partial tail batch instead of filling it with repeats.
        drop_remainder=True,
    ))
    return config
