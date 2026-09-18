"""1024 SFT config for the captioned dense-text production cache."""

from configs.sft_1024 import get_config as _base_config
from datasets.image_geometry import generate_buckets


def get_config():
    config = _base_config()
    # A native 32px frontier reduced measured mean padding from 2.41% to 1.45%
    # on Danqing + Monet + Paper2Fig100k + ChartGalaxy while retaining all rows.
    config.input.buckets = generate_buckets(
        1024, step=32, max_ratio=3.0,
        extra_shapes=[(832, 1248), (1248, 832)],
    )
    config.wandb.experiment = "i1-1024-captioned-v4"
    return config
