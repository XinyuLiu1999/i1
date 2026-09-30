"""Homogeneous-shape batches from JSONL images or GPT-Image Parquet shards.

Each optimizer step uses the same bucket on all data-parallel ranks. Epoch batch
plans are pure functions of the seed and dataset order, including after checkpoint
resume.
"""
from collections import Counter
import math

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .captions import tokenize_captions
from .data_sources import iter_image_records, open_record_image
from .image_geometry import BucketSelector
from .pixel_cache import cached_pixels, fingerprint


class BucketedImages(Dataset):
    def __init__(self, config, multiple=16, *, allow_empty=False):
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
        source = config["manifest"]
        if not source:
            raise ValueError("Set --manifest or config.input.manifest; use a corrected index for GPT-Image-200K.")
        expected_fingerprint = fingerprint(config)
        # Source sizes repeat heavily; the policy is a pure function of (height, width).
        select_bucket, selections = BucketSelector(config), {}
        for record in iter_image_records(source, image_root=config.get("image_root")):
            if record.cache_path is not None and record.transform_fingerprint != expected_fingerprint:
                raise ValueError(f"Pixel cache transform mismatch for {record.identifier}; use its original config or rebuild.")
            if record.width is None or record.height is None:
                with open_record_image(record) as image:
                    width, height = image.size
            else:
                width, height = record.width, record.height
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions for {record.identifier}.")
            selection = selections.get((height, width))
            if selection is None:
                selection = selections[height, width] = select_bucket(height, width)
            bucket, exclusion = selection
            if exclusion is not None:
                self.filtered[exclusion] += 1
                continue
            if record.cache_path is not None and (record.cache_height, record.cache_width) != self.buckets[bucket]:
                raise ValueError(f"Pixel cache bucket mismatch for {record.identifier}.")
            self.groups[bucket].append(len(self.records))
            self.records.append((record, height, width, bucket))
        if not self.records and not allow_empty:
            raise ValueError(f"No eligible images in {source}. Filter counts: {dict(self.filtered)}")

    def resize_scale(self, height, width, bh, bw):
        return (min if self.resize_mode == "pad" else max)(bh / height, bw / width)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record, height, width, bucket = self.records[index]
        bh, bw = self.buckets[bucket]
        if record.cache_path is not None:
            return torch.from_numpy(np.asarray(cached_pixels(record), dtype=np.float32) / 127.5 - 1), record.caption
        image = open_record_image(record)
        if image.size != (width, height):
            raise ValueError(
                f"Stale dimensions for {record.identifier}: metadata={(width, height)}, actual={image.size}."
            )
        image = self.process_image(image, (bh, bw))
        # Match the existing VAE encoder's NHWC [-1, 1] contract.
        return torch.from_numpy(np.asarray(image, dtype=np.float32) / 127.5 - 1), record.caption

    def process_image(self, image, bucket):
        """Apply the exact resize/pad or resize/crop transform used for training."""
        bh, bw = bucket
        width, height = image.size
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
        return image


def bucket_steps_per_epoch(groups, global_batch_size, drop_remainder=False):
    if global_batch_size <= 0:
        raise ValueError("Global batch must be positive.")
    if drop_remainder:
        return sum(len(group) // global_batch_size for group in groups)
    return sum((len(group) + global_batch_size - 1) // global_batch_size
               for group in groups if len(group))


def bucket_remainder_counts(groups, global_batch_size):
    """Images left out of each epoch, and images whose bucket never fills a batch."""
    dropped = sum(len(group) % global_batch_size for group in groups)
    never = sum(len(group) for group in groups if len(group) < global_batch_size)
    return dropped, never


class BucketBatchSampler(Sampler):
    def __init__(self, groups, global_batch_size, rank, world_size, total_steps, start_step=0, seed=0,
                 drop_remainder=False):
        if not 0 <= rank < world_size or global_batch_size <= 0 or global_batch_size % world_size:
            raise ValueError("Global batch must be positive and divisible by the data-parallel world size.")
        if seed < 0 or not 0 <= start_step <= total_steps:
            raise ValueError("Invalid seed or step range.")
        # With drop_remainder, each epoch omits a freshly shuffled tail of every
        # bucket instead of filling it with duplicates; the plan stays seed-pure.
        minimum = global_batch_size if drop_remainder else 1
        self.groups = [np.asarray(g, dtype=np.int64) for g in groups if len(g) >= minimum]
        if not self.groups:
            raise ValueError("No bucket can fill a global batch." if drop_remainder else "No nonempty buckets.")
        self.global_bs = global_batch_size
        self.local_bs = global_batch_size // world_size
        self.rank, self.seed = rank, seed
        self.drop_remainder = drop_remainder
        self.start_step, self.total_steps = start_step, total_steps
        self.steps_per_epoch = bucket_steps_per_epoch(self.groups, self.global_bs, drop_remainder)

    def __len__(self):
        return self.total_steps - self.start_step

    def _epoch_batches(self, epoch):
        batches = []
        for group_index, group in enumerate(self.groups):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, group_index]))
            shuffled = group[rng.permutation(len(group))]
            if self.drop_remainder:
                batch_count = len(group) // self.global_bs
                shuffled = shuffled[:batch_count * self.global_bs]
                batches.extend(shuffled.reshape(batch_count, self.global_bs))
                continue
            batch_count = (len(group) + self.global_bs - 1) // self.global_bs
            padded_size = batch_count * self.global_bs
            if padded_size > len(shuffled):
                padding = np.resize(shuffled, padded_size - len(shuffled))
                shuffled = np.concatenate((shuffled, padding))
            batches.extend(shuffled.reshape(batch_count, self.global_bs))

        # Interleave shapes instead of processing every batch for one aspect ratio
        # contiguously. The plan remains a pure function of seed and epoch, so a
        # resumed process can reconstruct it without checkpointing sampler state.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 0xBADC0DE]))
        order = rng.permutation(len(batches))
        return [batches[index] for index in order]

    def __iter__(self):
        epoch, offset = divmod(self.start_step, self.steps_per_epoch)
        step = self.start_step
        while step < self.total_steps:
            batches = self._epoch_batches(epoch)
            for indices in batches[offset:]:
                start = self.rank * self.local_bs
                yield indices[start:start + self.local_bs].tolist()
                step += 1
                if step == self.total_steps:
                    return
            epoch += 1
            offset = 0


def collate_images(samples):
    images, captions = zip(*samples)
    return torch.stack(images), list(captions)


def build_bucket_iterator(dataset, config, tokenizer, token_len, dist_info, total_steps, start_step, seed):
    sampler = BucketBatchSampler(dataset.groups, config["batch_size"], dist_info.dp_rank,
                                 dist_info.dp_world, total_steps, start_step, seed,
                                 drop_remainder=config.get("drop_remainder", False))
    workers = config.get("num_workers", 4)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers, collate_fn=collate_images,
                        pin_memory=dist_info.device.type == "cuda", persistent_workers=workers > 0,
                        generator=torch.Generator().manual_seed(seed + dist_info.dp_rank))
    for images, captions in loader:
        tok = tokenize_captions(tokenizer, captions, token_len, config.get("caption_overflow", "error"))
        yield {"image": images, "input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}
