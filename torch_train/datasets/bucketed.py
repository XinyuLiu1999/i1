"""Homogeneous-shape batches from original images and a JSONL manifest.

Each optimizer step uses the same bucket on all data-parallel ranks. Sampling
is a pure function of (seed, step), including after checkpoint resume.
"""
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .captions import tokenize_captions


class BucketedImages(Dataset):
    def __init__(self, config, multiple=16):
        self.config = config
        self.buckets = [tuple(shape) for shape in config["buckets"]]  # (height, width)
        if not self.buckets or len(set(self.buckets)) != len(self.buckets):
            raise ValueError("Provide a nonempty list of unique (height, width) buckets.")
        for shape in self.buckets:
            if len(shape) != 2 or any(not isinstance(x, int) or x <= 0 or x % multiple for x in shape):
                raise ValueError(f"Invalid bucket {shape}: dimensions must be positive multiples of {multiple}.")
        self.resize_mode = config.get("resize_mode", "pad")
        if self.resize_mode not in ("pad", "crop"):
            raise ValueError("resize_mode must be 'pad' or 'crop'.")
        self.allow_upscale = config.get("allow_upscale", False)
        self.records = []
        self.groups = [[] for _ in self.buckets]
        self.filtered = Counter()
        manifest = Path(config["manifest"]).expanduser().resolve()
        root = Path(config.get("image_root") or manifest.parent).expanduser().resolve()
        with manifest.open() as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                path = Path(record["image_path"]).expanduser()
                path = path if path.is_absolute() else root / path
                caption = record.get("caption", record.get("prompt"))
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError(f"{manifest}:{line_no}: expected a nonempty caption or prompt string.")
                if "width" in record and "height" in record:
                    width, height = int(record["width"]), int(record["height"])
                else:
                    with Image.open(path) as image:
                        width, height = ImageOps.exif_transpose(image).size
                if width <= 0 or height <= 0:
                    raise ValueError(f"Invalid image dimensions at {manifest}:{line_no}.")
                if width * height < config.get("min_image_area", 0) or min(width, height) < config.get("min_image_side", 0):
                    self.filtered["source_too_small"] += 1
                    continue
                candidates = []
                for idx, (bh, bw) in enumerate(self.buckets):
                    scale = self.resize_scale(height, width, bh, bw)
                    if not self.allow_upscale and scale > 1.0:
                        continue
                    # Prefer matching aspect ratios; among equal ratios use the
                    # largest available bucket that does not require upscaling.
                    score = (abs(math.log((bw / bh) / (width / height))), -bh * bw)
                    candidates.append((score, idx))
                if not candidates:
                    self.filtered["no_bucket_without_upscaling"] += 1
                    continue
                bucket = min(candidates)[1]
                self.groups[bucket].append(len(self.records))
                self.records.append((str(path), caption, height, width, bucket))
        if not self.records:
            raise ValueError(f"No eligible images in {manifest}. Filter counts: {dict(self.filtered)}")

    def resize_scale(self, height, width, bh, bw):
        return (min if self.resize_mode == "pad" else max)(bh / height, bw / width)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, caption, height, width, bucket = self.records[index]
        bh, bw = self.buckets[bucket]
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if image.size != (width, height):
            raise ValueError(f"Stale dimensions for {path}: manifest={(width, height)}, actual={image.size}.")
        scale = self.resize_scale(height, width, bh, bw)
        if self.resize_mode == "pad":
            new_w, new_h = max(1, min(bw, round(width * scale))), max(1, min(bh, round(height * scale)))
        else:
            new_w, new_h = max(bw, math.ceil(width * scale)), max(bh, math.ceil(height * scale))
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        if self.resize_mode == "pad":
            canvas = Image.new("RGB", (bw, bh), (255, 255, 255))
            canvas.paste(image, ((bw - new_w) // 2, (bh - new_h) // 2))
            image = canvas
        else:
            left, top = (new_w - bw) // 2, (new_h - bh) // 2
            image = image.crop((left, top, left + bw, top + bh))
        # Match the existing VAE encoder's NHWC [-1, 1] contract.
        return torch.from_numpy(np.asarray(image, dtype=np.float32) / 127.5 - 1), caption


class BucketBatchSampler(Sampler):
    def __init__(self, groups, global_batch_size, rank, world_size, total_steps, start_step=0, seed=0):
        if not 0 <= rank < world_size or global_batch_size <= 0 or global_batch_size % world_size:
            raise ValueError("Global batch must be positive and divisible by the data-parallel world size.")
        if seed < 0 or not 0 <= start_step <= total_steps:
            raise ValueError("Invalid seed or step range.")
        self.groups = [np.asarray(g, dtype=np.int64) for g in groups if len(g)]
        if not self.groups:
            raise ValueError("No nonempty buckets.")
        counts = np.array([len(g) for g in self.groups], dtype=np.float64)
        self.probabilities = counts / counts.sum()
        self.global_bs = global_batch_size
        self.local_bs = global_batch_size // world_size
        self.rank, self.seed = rank, seed
        self.start_step, self.total_steps = start_step, total_steps

    def __len__(self):
        return self.total_steps - self.start_step

    def __iter__(self):
        for step in range(self.start_step + 1, self.total_steps + 1):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
            group = self.groups[rng.choice(len(self.groups), p=self.probabilities)]
            indices = rng.choice(group, size=self.global_bs, replace=len(group) < self.global_bs)
            start = self.rank * self.local_bs
            yield indices[start:start + self.local_bs].tolist()


def collate_images(samples):
    images, captions = zip(*samples)
    return torch.stack(images), list(captions)


def build_bucket_iterator(dataset, config, tokenizer, token_len, dist_info, total_steps, start_step, seed):
    sampler = BucketBatchSampler(dataset.groups, config["batch_size"], dist_info.dp_rank,
                                 dist_info.dp_world, total_steps, start_step, seed)
    workers = config.get("num_workers", 4)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers, collate_fn=collate_images,
                        pin_memory=dist_info.device.type == "cuda", persistent_workers=workers > 0,
                        generator=torch.Generator().manual_seed(seed + dist_info.dp_rank))
    for images, captions in loader:
        tok = tokenize_captions(tokenizer, captions, token_len, config.get("caption_overflow", "error"))
        yield {"image": images, "input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}
