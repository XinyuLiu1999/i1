"""Matched 1024 SFT from the original base checkpoint."""
import os
from pathlib import Path

from configs.sft_1024_captioned import get_config as base_config


def get_config():
    config = base_config()
    config.init_from = "/cephfs/liuxinyu/.cache/data_juicer/models/i1-3B/1024_resolution_checkpoint_torch.pt"
    # Match the completed run's actual launch (its YAML records accumulation=1,
    # overriding the older SFT.md conservative example with accumulation=4).
    config.grad_accum_steps = 1
    config.fsdp_axis_size = 8
    config.compile = False
    # Zero is the safe default. A positive coefficient must be calibrated from
    # measured global/region gradient norms for this exact normalization.
    config.region_weight = float(os.environ.get("REGION_WEIGHT", "0"))
    project = Path(__file__).resolve().parents[3]
    # Start with global flow MSE. Regional and perceptual supervision are opt-in;
    # its coefficient and noisy reconstructions need calibration for caption-only i1.
    config.perceptual_weight = float(os.environ.get("PERCEPTUAL_WEIGHT", "0"))
    hub_cache = Path(os.environ.get("HF_HUB_CACHE", "/cephfs/liuxinyu/.cache/data_juicer/models"))
    odm_snapshot = hub_cache / "models--GD-ML--FLUX-Text/snapshots/dcebeaee2f9fdb2876706a9b803b9413408f1f4f"
    config.perceptual_checkpoint = os.environ.get(
        "PERCEPTUAL_CHECKPOINT", str(odm_snapshot / "epoch_100.pt"))
    config.fluxtext_root = os.environ.get("FLUXTEXT_ROOT", str(project / "FluxText"))
    config.perceptual_checkpointing = os.environ.get("PERCEPTUAL_CHECKPOINTING", "1") != "0"
    config.perceptual_chunk_size = int(os.environ.get("PERCEPTUAL_CHUNK_SIZE", "1"))
    config.wandb.log_wandb = os.environ.get("REGION_WANDB", "1") != "0"
    config.wandb.experiment = os.environ.get(
        "REGION_RUN_NAME", f"i1-region-flow-2026-09-21-w{config.region_weight:g}-p{config.perceptual_weight:g}")
    return config
