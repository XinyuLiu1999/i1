"""Production-shaped SFT config for the destructive shutdown smoke test."""

from configs.sft_1024_captioned import get_config as _production_config


def get_config():
    config = _production_config()
    # The smoke test validates training, checkpointing, and VM release. Avoid
    # creating a short-lived W&B run that may not flush before the VM powers off.
    config.wandb.log_wandb = False
    return config
