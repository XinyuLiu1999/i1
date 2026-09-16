from configs.i1_512 import get_config as _base_config
from utils.config import ConfigDict


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
        manifest="/path/to/train.jsonl",
        image_root="",  # Empty: resolve image paths relative to the manifest.
        # (height, width); approximately 512^2 area, not a 512-pixel minimum side.
        buckets=[(512, 512), (448, 592), (592, 448), (416, 640),
                 (640, 416), (368, 720), (720, 368)],
        batch_size=32,
        num_workers=4,
        min_image_area=512 * 512,
        min_image_side=0,
        allow_upscale=False,
        resize_mode="pad",  # Preserve all text; white letterbox padding.
        caption_overflow="error",
    ))
    return config
