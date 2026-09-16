from configs.sft_512 import get_config as _base_config


def get_config():
    config = _base_config()
    config.image_size = 1024
    config.input.buckets = [(h * 2, w * 2) for h, w in config.input.buckets]
    config.input.min_image_area = 1024 * 1024
    config.transport.train_timestep_shift = 0.3
    return config
