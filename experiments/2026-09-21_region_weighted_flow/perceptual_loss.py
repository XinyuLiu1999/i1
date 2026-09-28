"""Frozen FLUX-Text ODM features with differentiable FLUX.2 VAE decoding."""
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from vae.vae import reverse_scale_latents


def encoder_source(fluxtext_root):
    return Path(fluxtext_root) / "src/loss/ocr_loss/base_model/ODM_encoder.py"


def load_odm(checkpoint_path, fluxtext_root, device):
    """Load every feature-extraction weight; never accept a random partial model."""
    source = encoder_source(fluxtext_root)
    spec = importlib.util.spec_from_file_location("region_flow_odm_encoder", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.ResNet(layers=[3, 4, 6, 3], output_dim=512, heads=32,
                         input_resolution=512, width=64)
    # FLUX-Text does not use attention pooling in its four-scale feature loss.
    del model.attnpool
    # The official checkpoint includes NumPy scalar training metadata. Allow
    # these numeric types without enabling arbitrary pickle object loading.
    with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype,
                                           type(np.dtype("float64")), type(np.dtype("float32"))]):
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = raw.get("state_dict", raw)
    prefix = "module.visual."
    visual = {key[len(prefix):]: value for key, value in state.items()
              if key.startswith(prefix)}
    if not visual:
        raise ValueError("ODM checkpoint must contain module.visual.* feature weights")
    visual = {key: value for key, value in visual.items()
              if not key.startswith(("attnpool.", "fpn_head."))}
    # Strict validation includes BN statistics, not just trainable tensors.
    model.load_state_dict(visual, strict=True)
    return model.to(device=device, dtype=torch.float32).requires_grad_(False).eval()


def reconstruct_clean_latents(prediction, target_velocity, clean_latents):
    """FLUX-Text's noise - velocity, adapted to our data-minus-noise target.

    noise = clean_latents - target_velocity, hence z_hat = noise + prediction.
    This intentionally matches the reference training reconstruction, rather than
    z_t + (1-t)*prediction, which would introduce different timestep weighting.
    """
    return clean_latents.detach().float() + prediction.float() - target_velocity.detach().float()


class RegionPerceptualLoss(nn.Module):
    def __init__(self, encoder, vae, vae_type="flux2", *, checkpointing=True, chunk_size=1):
        super().__init__()
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("Perceptual chunk size must be a positive integer")
        self.encoder = encoder.requires_grad_(False).eval()
        self.vae = vae.requires_grad_(False).eval()
        self.vae_type = vae_type
        self.checkpointing = checkpointing
        self.chunk_size = chunk_size

    def _chunk_loss(self, predicted_latents, images, mask):
        # Keep VAE decoding/features in fp32 even inside the trainer's bf16 AMP.
        # Frozen weights still require input gradients on the predicted branch.
        with torch.autocast(device_type=predicted_latents.device.type, enabled=False):
            unscaled = reverse_scale_latents(predicted_latents.float(), self.vae_type)
            reconstructed = self.vae.decode(unscaled, return_dict=False)[0]
            if reconstructed.shape != images.shape:
                raise ValueError("Perceptual reconstruction must match the training image geometry")
            predicted_features = self.encoder(reconstructed.float())[0][:4]
            with torch.no_grad():
                target_features = self.encoder(images.detach().float())[0][:4]
            losses = []
            for predicted, target in zip(predicted_features, target_features):
                # Preserve soft weights at each feature resolution, including
                # rectangular buckets; unlike FLUX-Text's binary nearest masks.
                region_weight = F.interpolate(
                    mask[:, None].float(), size=predicted.shape[-2:], mode="area")
                losses.append(
                    ((predicted.float() - target.float()).square() * region_weight.square()).mean())
            return torch.stack(losses).sum()

    def forward(self, prediction, target_velocity, clean_latents, images, mask):
        """Sum four feature MSEs, averaged over ALL images (empty masks add zero)."""
        self.encoder.eval()
        self.vae.eval()
        mask = mask.to(device=prediction.device, dtype=torch.float32)
        images = images.detach().permute(0, 3, 1, 2).contiguous()  # NHWC [-1, 1]
        predicted_latents = reconstruct_clean_latents(prediction, target_velocity, clean_latents)
        active = (mask.sum(dim=(1, 2)) > 0).nonzero().flatten()
        result = prediction.float().sum() * 0.0
        for start in range(0, len(active), self.chunk_size):
            indices = active[start:start + self.chunk_size]
            args = (predicted_latents[indices], images[indices], mask[indices])
            if self.checkpointing and torch.is_grad_enabled() and predicted_latents.requires_grad:
                loss = checkpoint(self._chunk_loss, *args, use_reentrant=False)
            else:
                loss = self._chunk_loss(*args)
            result = result + loss * (len(indices) / prediction.shape[0])
        return result
