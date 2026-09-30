from configs.sft_512 import get_config as _base_config


def get_config():
    config = _base_config()
    config.image_size = 1024
    # Resolve the exact stopping step after the final manifest has been filtered
    # and bucketed. --total_steps remains an explicit command-line override.
    config.num_epochs = 1
    config.total_steps = None
    config.log_training_steps = 50
    config.ckpt_steps = 1000
    config.keep_ckpt_steps = 10000
    # The pretraining value (0.9999) retains too much of the initialization in a
    # short SFT run. This decay has an effective averaging window near 2K updates.
    config.ema_decay_rate = 0.9995
    config.input.buckets = [(h * 2, w * 2) for h, w in config.input.buckets]
    config.input.min_image_area = 1024 * 1024
    config.transport.train_timestep_shift = 0.3
    config.wandb.log_wandb = True
    config.wandb.project = "DenseText-SFT"
    config.wandb.experiment = "i1-1024-full"
    return config
