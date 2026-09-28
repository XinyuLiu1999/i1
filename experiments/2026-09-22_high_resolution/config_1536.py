"""1024 + 1536 pixel-budget SFT."""
from configs.sft_multires_captioned import get_config as multires_config


def get_config():
    return multires_config(1536)
