"""Verified RGB + soft height-weighted OCR batches using the bucket sampler."""
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.bucketed import BucketedImages, BucketBatchSampler
from datasets.captions import tokenize_captions
from datasets.precompute_captioned import file_hash
from region_masks import MASK_VERSION, LATENT_FACTOR


@lru_cache(maxsize=8)
def mapped_masks(path):
    return np.memmap(path, mode="r", dtype="<f2")


class RegionImages(BucketedImages):
    def __init__(self, config, multiple=16):
        path = Path(config["manifest"]).resolve()
        self.root = path.parent
        self.summary = json.loads((self.root / "summary.json").read_text())
        if (self.summary.get("format") != "region-weighted-flow-cache-v2"
                or self.summary.get("status") != "complete"
                or not self.summary.get("verified_all_written_bytes")):
            raise ValueError("Run this experiment's CPU precompute to completion first")
        self.manifest_sha256 = file_hash(path)
        if self.manifest_sha256 != self.summary["cache_manifest_sha256"]:
            raise ValueError("Training manifest checksum mismatch")
        if not self.summary.get("masked_images"):
            raise ValueError("No OCR-supervised images in this cache")
        verified_paths = {str((self.root / item["path"]).resolve())
                          for item in self.summary["mask_files"]}
        for item in self.summary["mask_files"]:
            payload = self.root / item["path"]
            if payload.stat().st_size != item["bytes"]:
                raise ValueError(f"Truncated OCR mask file: {payload}")
            # One rank verifies the full mask payload; torchrun aborts peers if it fails.
            if int(os.environ.get("RANK", "0")) == 0 and file_hash(payload) != item["sha256"]:
                raise ValueError(f"OCR mask checksum mismatch: {payload}")
        super().__init__(config, multiple=multiple)
        if self.filtered or len(self) != self.summary["count"]:
            raise ValueError("Final manifest must be accepted-only with the exact precompute geometry")
        self.mask_records = []
        with path.open() as handle:
            for index, line in enumerate(handle):
                row = json.loads(line)
                record = self.records[index][0]
                if row["id"] != record.identifier:
                    raise ValueError("Invalid ID ordering")
                mask = row["region_mask"]
                if (mask["version"] != MASK_VERSION or mask["dtype"] != "<f2"
                        or mask["height"] * LATENT_FACTOR != record.cache_height
                        or mask["width"] * LATENT_FACTOR != record.cache_width
                        or type(mask["offset"]) is not int or mask["offset"] < 0 or mask["offset"] % 2):
                    raise ValueError(f"Invalid OCR mask contract for {record.identifier}")
                mask_path = str((self.root / mask["path"]).resolve())
                if mask_path not in verified_paths:
                    raise ValueError("OCR mask references an unverified payload")
                self.mask_records.append((mask_path,
                                          mask["offset"] // 2, mask["height"], mask["width"]))

    def __getitem__(self, index):
        image, caption = super().__getitem__(index)
        path, offset, height, width = self.mask_records[index]
        values = mapped_masks(path)[offset:offset + height * width]
        if values.size != height * width or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError(f"Invalid/truncated OCR mask for {self.records[index][0].identifier}")
        mask = torch.from_numpy(values.astype(np.float32).reshape(height, width))
        return image, caption, mask


def collate_regions(samples):
    images, captions, masks = zip(*samples)
    return torch.stack(images), list(captions), torch.stack(masks)


def build_iterator(dataset, config, tokenizer, token_len, dist_info, total_steps, start_step, seed):
    sampler = BucketBatchSampler(dataset.groups, config["batch_size"], dist_info.dp_rank,
                                 dist_info.dp_world, total_steps, start_step, seed)
    workers = config.get("num_workers", 4)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                        collate_fn=collate_regions, pin_memory=dist_info.device.type == "cuda",
                        persistent_workers=workers > 0,
                        generator=torch.Generator().manual_seed(seed + dist_info.dp_rank))
    for images, captions, masks in loader:
        tokens = tokenize_captions(tokenizer, captions, token_len, "error")
        yield dict(image=images, input_ids=tokens["input_ids"],
                   attention_mask=tokens["attention_mask"], region_mask=masks)
