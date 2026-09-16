"""Versioned, lossless uint8 RGB cache shared by the exporter and training loader."""

from functools import lru_cache
import hashlib
import json

import numpy as np
import PIL


def transform_spec(config):
    return dict(version=1, buckets=[list(shape) for shape in config['buckets']],
                resize_mode=config.get('resize_mode', 'pad'),
                allow_upscale=config.get('allow_upscale', False),
                min_image_area=config.get('min_image_area', 0),
                min_image_side=config.get('min_image_side', 0),
                resampling='PIL.LANCZOS', padding=[255, 255, 255], pillow=PIL.__version__)


def fingerprint(config):
    return hashlib.sha256(json.dumps(transform_spec(config), sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=8)
def _mapped_pixels(path):
    return np.memmap(path, mode='r', dtype=np.uint8)


def cached_pixels(record):
    pixels = _mapped_pixels(record.cache_path)
    size = record.cache_height * record.cache_width * 3
    start = record.cache_offset
    if start < 0 or start + size > pixels.size:
        raise ValueError(f'Truncated pixel cache for {record.identifier}: {record.cache_path}')
    return pixels[start:start + size].reshape(record.cache_height, record.cache_width, 3)
