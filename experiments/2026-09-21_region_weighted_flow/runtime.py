"""Experiment hooks; optimization/checkpoint/shutdown remain in the shared trainer."""
import hashlib
import json
import math
from pathlib import Path

import torch

from region_data import RegionImages, build_iterator
from region_loss import region_flow_loss
from region_masks import LATENT_FACTOR
from datasets.precompute_captioned import file_hash
from perceptual_loss import encoder_source, load_odm, RegionPerceptualLoss
from utils.common import log


class RegionFlow:
    dataset_class = RegionImages
    build_iterator = staticmethod(build_iterator)

    def configure(self, config, args):
        if config.input.get("type") != "bucketed" or config.input.get("resize_mode") != "pad":
            raise ValueError("Region flow requires the accepted fit-and-pad cache")
        from vae.vae import VAE_CONFIGS
        if VAE_CONFIGS[config.vae_type]["vae_compression_factor"] != LATENT_FACTOR:
            raise ValueError("OCR mask latent stride does not match the VAE")
        self.weight = float(config.region_weight)
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("Invalid region loss weight")
        if config.input.get("caption_overflow") != "error":
            raise ValueError("Use audited captions without training-time truncation")
        self.perceptual_weight = float(config.perceptual_weight)
        if not math.isfinite(self.perceptual_weight) or self.perceptual_weight < 0:
            raise ValueError("Invalid perceptual loss weight")
        self.perceptual = None
        self.perceptual_metadata = dict(weight=self.perceptual_weight)
        if self.perceptual_weight:
            if type(config.perceptual_chunk_size) is not int or config.perceptual_chunk_size < 1:
                raise ValueError("Perceptual chunk size must be a positive integer")
            for path in (Path(config.perceptual_checkpoint), encoder_source(config.fluxtext_root)):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing perceptual dependency: {path}. "
                                            "See README setup, or set PERCEPTUAL_WEIGHT=0 for the control.")
            self.perceptual_metadata.update(
                checkpoint_sha256=file_hash(config.perceptual_checkpoint),
                encoder_source_sha256=file_hash(encoder_source(config.fluxtext_root)),
                features="odm-resnet50-layer1-layer2-layer3-layer4",
                reconstruction="clean-latent+predicted-velocity-target-velocity",
                normalization="sum-four-full-tensor-squared-mask-weighted-feature-means",
                masks="height-weighted-latent-map-area-resized-to-features",
                precision="float32", checkpointing=config.perceptual_checkpointing,
                chunk_size=config.perceptual_chunk_size,
            )

    def setup(self, config, vae, device):
        if self.perceptual_weight:
            encoder = load_odm(config.perceptual_checkpoint, config.fluxtext_root, device)
            self.perceptual = RegionPerceptualLoss(
                encoder, vae, config.vae_type,
                checkpointing=config.perceptual_checkpointing, chunk_size=config.perceptual_chunk_size)

    def checkpoint_metadata(self, config, dataset, dist_info):
        if dataset.summary["settings"]["token_limit"] != config.token_len:
            raise ValueError("Training token limit differs from CPU precompute")
        metadata = dict(
            name="region-weighted-flow-v5", weight=self.weight,
            normalization="full-tensor-mean-of-squared-mask-weighted-velocity-squared-error",
            perceptual=self.perceptual_metadata,
            manifest_sha256=dataset.manifest_sha256,
            mask_files_sha256=hashlib.sha256(json.dumps(dataset.summary["mask_files"], sort_keys=True).encode()).hexdigest(),
            region_settings=dataset.summary["settings"], seed=config.seed,
            global_batch=config.input.batch_size, grad_accum=config.grad_accum_steps,
            dp_world=dist_info.dp_world, tp=dist_info.model_size, fsdp=dist_info.fsdp_size,
            token_len=config.token_len, lr=config.lr, ema_decay=config.ema_decay_rate,
            transport=dict(config.transport),
        )
        if dist_info.is_main:
            log(f"Region flow: lambda={self.weight:g}, {dataset.summary['masked_images']}/{len(dataset)} "
                f"images with masks; perceptual weight={self.perceptual_weight:g} per feature scale; "
                "images without masks use global flow only")
        return metadata

    @staticmethod
    def validate_resume(checkpoint, expected):
        actual = checkpoint.get("config", {}).get("training_objective")
        if actual != expected:
            raise ValueError("Resume objective/data/training settings differ. Use a new workdir and "
                             "--init_from for a new arm; --resume is only for the same experiment.")

    @staticmethod
    def batch_specs(batch_size, config):
        n = config.image_size // LATENT_FACTOR
        return [("region_mask", (batch_size, n, n), torch.float32)]

    def loss(self, prediction, target, batch, micro_slice, *, latents, images):
        mask = batch["region_mask"][micro_slice]
        loss, metrics = region_flow_loss(prediction, target, mask, self.weight)
        perceptual = prediction.new_zeros((), dtype=torch.float32)
        if self.perceptual_weight:
            if self.perceptual is None:
                raise RuntimeError("Initialize perceptual dependencies before training")
            perceptual = self.perceptual(prediction, target, latents, images, mask)
            loss = loss + self.perceptual_weight * perceptual
        metrics.update(perceptual_loss=perceptual,
                       weighted_perceptual_loss=self.perceptual_weight * perceptual)
        return loss, metrics
