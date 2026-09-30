from __future__ import annotations

import os
import shutil

import torch

from training.parallel import _permute_gate_up, _unpermute_gate_up
from utils.common import log

SKIP_ON_LOAD = ("pos_embed", "rope_embedder")


def _is_dtensor(x) -> bool:
    try:
        from torch.distributed.tensor import DTensor
    except Exception:
        return False
    return isinstance(x, DTensor)


def _gather_full(t, keep=True):
    # full_tensor() is a collective and must run on every rank, but only the
    # writing rank keeps a host copy; the others release each gathered tensor.
    if _is_dtensor(t):
        t = t.full_tensor()
    return t.detach().cpu() if keep else None


def _gather_named(named: dict, tp_size: int, permuted_keys, keep=True) -> dict | None:
    out = {} if keep else None
    for k, v in named.items():
        full = _gather_full(v, keep)
        if not keep:
            continue
        if tp_size > 1 and k in permuted_keys:
            full = _unpermute_gate_up(full, tp_size)
        out[k] = full
    return out


def _distribute_like(name: str, full: torch.Tensor, ref, tp_size: int, permuted_keys):
    full = full.to(dtype=ref.dtype)
    if tp_size > 1 and name in permuted_keys:
        full = _permute_gate_up(full, tp_size)
    if _is_dtensor(ref):
        from torch.distributed.tensor import distribute_tensor
        return distribute_tensor(full, ref.device_mesh, ref.placements)
    return full.to(ref.device)


def _skip(name: str) -> bool:
    return any(s in name for s in SKIP_ON_LOAD)


def resolve_resume_path(save_ckpt_path, config):
    if save_ckpt_path and os.path.exists(save_ckpt_path):
        return save_ckpt_path
    resume = config.get("resume", "")
    if resume:
        if not os.path.isfile(resume):
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume}")
        return resume
    return None


def load_model_weights(model, ckpt, dist_info, initialize=False):
    ts = ckpt.get("train_state", {})
    raw = ckpt["model"] if initialize else ts.get("model_raw", ckpt["model"])
    filtered = {k: v for k, v in raw.items() if not _skip(k)}
    # Fresh SFT starts from inference/EMA weights. Only the learned null-caption
    # length changes; preserve its prefix and tile it to initialize extra slots.
    name = "text_encoder_adapter.learnable_null_caption"
    if initialize and name in filtered:
        value = filtered[name]
        target = model.state_dict()[name]
        if value.shape != target.shape and value.shape[0] == target.shape[0] and value.shape[2] == target.shape[2]:
            repeats = (target.shape[1] + value.shape[1] - 1) // value.shape[1]
            filtered[name] = value.repeat(1, repeats, 1)[:, :target.shape[1]].clone()
            if dist_info.is_main:
                log(f"SFT initialization: null caption {value.shape[1]} -> {target.shape[1]} tokens (tiled prefix)")
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    missing = [m for m in missing if not _skip(m)]
    if missing or unexpected:
        raise RuntimeError(f"resume weight mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    if dist_info.is_main:
        skipped = [k for k in raw if _skip(k)]
        log(f"resume: loaded {len(filtered)} model tensors, rebuilt {len(skipped)} resolution-dependent "
            f"(pos_embed/rope) at current resolution")


def load_train_states(ema, optimizer, ckpt, dist_info, permuted_keys=None):
    tp = dist_info.model_size
    permuted = permuted_keys or set()
    ts = ckpt.get("train_state", {})
    if "opt" not in ts:
        raise ValueError("Checkpoint has no optimizer state. Use --init_from for a fresh SFT run, not --resume.")
    if ema is not None and "model_raw" in ts:
        ema_src = ckpt["model"]
        for name in list(ema.shadow.keys()):
            if _skip(name) or name not in ema_src:
                continue
            ema.shadow[name] = _distribute_like(name, ema_src[name], ema.shadow[name], tp, permuted)
    opt = ts["opt"]
    optimizer.count = int(opt["count"])
    for name in list(optimizer.mu.keys()):
        if _skip(name):
            continue
        if name in opt["mu"]:
            optimizer.mu[name] = _distribute_like(name, opt["mu"][name], optimizer.mu[name], tp, permuted)
        if name in opt["nu"]:
            optimizer.nu[name] = _distribute_like(name, opt["nu"][name], optimizer.nu[name], tp, permuted)
    return int(ts.get("step", 0))


def save_checkpoint(path, model, ema, optimizer, step, ckpt_cfg, dist_info, step_copy=None):
    tp = dist_info.model_size
    permuted = getattr(model, "_tp_permuted_keys", set())
    keep = dist_info.is_main
    model_sd = _gather_named(dict(model.state_dict()), tp, permuted, keep)
    ema_sd = _gather_named(ema.shadow, tp, permuted, keep) if ema is not None else None
    opt_mu = _gather_named(optimizer.mu, tp, permuted, keep)
    opt_nu = _gather_named(optimizer.nu, tp, permuted, keep)
    if not keep:
        return
    inference_sd = ema_sd if ema_sd is not None else model_sd
    train_state = {"step": step, "opt": {"count": optimizer.count, "mu": opt_mu, "nu": opt_nu}}
    if ema_sd is not None:
        train_state["model_raw"] = model_sd
    ckpt = {
        "config": ckpt_cfg,
        "model": {k: v.float() for k, v in inference_sd.items()},
        "train_state": train_state,
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    msg = f"[step {step}] saved checkpoint to {path} (inference: config+model[EMA]; resume: +train_state)"
    if step_copy is not None:
        keep_path = f"{path}-{int(step_copy):09d}"
        shutil.copyfile(path, keep_path)
        msg += f"; kept permanent copy {keep_path}"
    log(msg)
