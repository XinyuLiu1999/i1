"""Gradient measurements and coefficient selection; no optimizer or clipping."""
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from datasets.bucketed import BucketBatchSampler
from datasets.precompute_captioned import file_hash
from diffusion.rectified_flow import RectifiedFlowConfig
from region_loss import region_flow_loss
from runtime import RegionFlow
from training.main import prepare_flow_microbatch
from utils.common import log

TARGETS = (0.30, 0.40, 0.50)


def geometry(g2, r2, dot):
    """Undefined ratios/cosines are JSON null, including empty-region batches."""
    g, r = math.sqrt(g2), math.sqrt(r2)
    return dict(G2=g2, R2=r2, D=dot, G=g, R=r,
                log10_G_over_R=math.log10(g / r) if g and r else None,
                cosine=max(-1., min(1., dot / (g * r))) if g and r else None)


def regional_share(g2, r2, dot, coefficient):
    total2 = g2 + 2 * coefficient * dot + coefficient ** 2 * r2
    return coefficient * math.sqrt(r2 / total2) if total2 > 0 else None


def solve_coefficient(g2, r2, dot, share):
    if not all(math.isfinite(x) for x in (g2, r2, dot, share)):
        raise ValueError("Nonfinite gradient geometry/target")
    if g2 <= 0 or r2 <= 0 or not 0 < share < 1:
        raise ValueError("Calibration needs nonzero global/regional gradients and 0 < share < 1")
    g, r = math.sqrt(g2), math.sqrt(r2)
    cosine = max(-1., min(1., dot / (g * r)))
    radical = math.sqrt(1 - share * share + (share * cosine) ** 2)
    # Rationalized negative-cosine branch avoids cancellation.
    factor = (share / (radical - share * cosine) if cosine < 0 else
              share * (share * cosine + radical) / (1 - share * share))
    return (g / r) * factor


def quantiles(values):
    values = [x for x in values if x is not None]
    if not values:
        return dict(count=0)
    return dict(count=len(values), **dict(zip(
        ("min", "p05", "p25", "median", "p75", "p95", "max"),
        map(float, np.quantile(values, [0, .05, .25, .5, .75, .95, 1])))))


def summarize(rows, coefficients=None):
    g2, r2, dot = (math.fsum(row[key] for row in rows) for key in ("G2", "R2", "D"))
    result = dict(batches=len(rows), **geometry(g2, r2, dot))
    result["aggregate_log10_G_over_R"] = result["log10_G_over_R"]
    result["aggregate_cosine"] = result["cosine"]
    result["log10_G_over_R"] = quantiles(row["log10_G_over_R"] for row in rows)
    result["cosine"] = quantiles(row["cosine"] for row in rows)
    if coefficients is not None:
        result["shares"] = {
            key: dict(aggregate=regional_share(g2, r2, dot, value),
                      per_batch=quantiles(regional_share(row["G2"], row["R2"], row["D"], value)
                                          for row in rows))
            for key, value in coefficients.items()}
    return result


def build_report(rows):
    aggregate = summarize(rows)
    coefficients = {f"lambda_{round(100 * s)}": solve_coefficient(
        aggregate["G2"], aggregate["R2"], aggregate["D"], s) for s in TARGETS}
    for row in rows:
        row["shares"] = {key: regional_share(row["G2"], row["R2"], row["D"], value)
                         for key, value in coefficients.items()}
    nonempty = [row for row in rows if row["R2"] > 0]
    groups = defaultdict(list)
    timestep_groups = defaultdict(list)
    for row in rows:
        groups[row["bucket"]].append(row)
        if "timestep_mean" in row:
            index = min(3, int(row["timestep_mean"] * 4))
            timestep_groups[f"{index / 4:.2f}-{(index + 1) / 4:.2f}"].append(row)
    warnings = []
    if any(row["cosine"] is not None and row["cosine"] < -.5 for row in rows):
        warnings.append("Strong negative cosine (< -0.5): inspect batches before selecting lambda.")
    ratios = aggregate["log10_G_over_R"]
    if ratios["count"] and ratios["p95"] - ratios["p05"] > 1:
        warnings.append("G/R spans over 10x (p05 to p95): inspect bucket/timestep strata.")
    if len(rows) < 32:
        warnings.append("Fewer than 32 batches: smoke measurement only.")
    l40 = coefficients["lambda_40"]
    return dict(coefficients=coefficients, sweep=[0., l40 / 2, l40, 2 * l40],
                all_batches=summarize(rows, coefficients),
                nonempty_region_batches=summarize(nonempty, coefficients),
                by_bucket={key: summarize(value, coefficients) for key, value in groups.items()},
                by_batch_mean_timestep={key: summarize(value, coefficients)
                                       for key, value in timestep_groups.items()},
                warnings=warnings)


def measure_gradients(model, global_loss, regional_loss, *, sharded=False):
    """Two VJPs on the same graph. Returned tensors own their gradient storage.

    FSDP2 swaps in unsharded parameters during forward, and its communication
    hooks consume accumulated .grad. autograd.grad against sharded parameters
    bypasses that path. Use two backwards with cleared .grad for FSDP2 instead.
    Both paths measure the same derivatives without clipping or updates.
    """
    if not sharded:
        params = [p for p in model.parameters() if p.requires_grad]
        global_grads = torch.autograd.grad(global_loss, params, retain_graph=True, allow_unused=True)
        regional_grads = torch.autograd.grad(regional_loss, params, allow_unused=True)
        return [[g.detach().float() if g is not None else None for g in grads]
                for grads in (global_grads, regional_grads)]

    def take_grads():
        grads = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            grad = p.grad
            if grad is not None:
                grad = grad.to_local() if hasattr(grad, "to_local") else grad
                grad = grad.detach().float().clone()
            grads.append(grad)
        model.zero_grad(set_to_none=True)
        return grads

    model.zero_grad(set_to_none=True)
    global_loss.backward(retain_graph=True)
    global_grads = take_grads()
    regional_loss.backward()
    return global_grads, take_grads()


def add_gradients(accumulated, incoming):
    if accumulated is None:
        return incoming
    for index, value in enumerate(incoming):
        if value is not None:
            if accumulated[index] is None:
                accumulated[index] = value
            else:
                accumulated[index].add_(value)
    return accumulated


def gradient_products(global_grads, regional_grads, device, *, distributed=False):
    """Sum unique FP32 shard products, then reduce scalars before square roots.

    Inputs must already be gradients of the global batch mean (FSDP's averaged
    reduce-scatter). Summing norms of unreduced rank-local gradients is incorrect.
    The caller restricts distributed runs to pure FSDP, with no replicated axes.
    """
    products = torch.zeros(3, device=device, dtype=torch.float64)
    for g, r in zip(global_grads, regional_grads):
        if g is not None:
            products[0] += g.square().sum(dtype=torch.float64)
        if r is not None:
            products[1] += r.square().sum(dtype=torch.float64)
        if g is not None and r is not None:
            products[2] += (g * r).sum(dtype=torch.float64)
    if distributed:
        dist.all_reduce(products, op=dist.ReduceOp.SUM)
    if not torch.isfinite(products).all().item():
        raise FloatingPointError("Nonfinite calibration gradients")
    return products.tolist()


class RegionCalibration(RegionFlow):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--calibration-batches", type=int, default=64,
                            help="Number of unmodified production batches to measure (default: 64).")

    def configure(self, config, args):
        if args.calibration_batches < 1:
            raise ValueError("--calibration-batches must be positive")
        if not args.workdir or args.resume or config.get("resume"):
            raise ValueError("Calibration requires a fresh --workdir and initialization, not resume")
        self.output = Path(args.workdir)
        if any((self.output / name).exists() for name in
               ("checkpoint.pt", "calibration.json", "batches.jsonl")):
            raise ValueError("Calibration output already exists; use a fresh --workdir")
        if not config.get("init_from"):
            raise ValueError("Calibration requires --init_from")
        if config.perceptual_weight:
            raise ValueError("Regional calibration requires PERCEPTUAL_WEIGHT=0")
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if config.tensor_parallel_size != 1 or (world > 1 and config.fsdp_axis_size != world):
            raise ValueError("Calibration supports TP=1 and a single FSDP group (fsdp=world size)")
        if (config.input.batch_size < 1 or config.grad_accum_steps < 1
                or config.input.batch_size % (world * config.grad_accum_steps)):
            raise ValueError("Global batch must be divisible by world size * grad_accum")
        # Ignore inherited completion credentials; diagnostics can never release a VM.
        args.completion_config = None
        config.save_ckpt = False
        config.wandb.log_wandb = False
        config.region_weight = 0.
        super().configure(config, args)

    def run_diagnostics(self, config, args, model, dataset, bundle, vae, info, ckpt_cfg):
        count = args.calibration_batches
        per_rank = config.input.batch_size // info.dp_world
        micro_bs = per_rank // config.grad_accum_steps
        iterator = self.build_iterator(dataset, config.input, bundle.tokenizer, bundle.text_token_len,
                                       info, count, 0, config.seed)
        # Replay only the sampler indices to audit sources/IDs without changing the
        # production dataset, collator, padding, or frequency of empty masks.
        audit_sampler = BucketBatchSampler(dataset.groups, config.input.batch_size, 0, 1,
                                           count, 0, config.seed)
        sources = []
        with Path(config.input.manifest).open() as handle:
            for line in handle:
                sources.append(json.loads(line).get("source_dataset", "unknown"))
        provenance = dict(
            initialization=str(Path(config.init_from).resolve()),
            initialization_sha256=file_hash(config.init_from) if info.is_main else None,
            training_objective=ckpt_cfg["training_objective"], torch_version=torch.__version__,
            amp_bfloat16=bool(config.get("amp", True)), model_training=True,
            activation_checkpointing=bool(config.use_grad_ckpt), compiled=bool(config.get("compile", True)),
            freeze_patterns=list(config.freeze_patterns),
            trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
            gradient_method="two-backwards-fsdp2" if info.is_distributed else "autograd.grad",
            sampling="first production batches, shuffled/interleaved buckets; original padding and zero masks",
        )
        if not provenance["trainable_parameter_count"]:
            raise ValueError("No trainable parameters to calibrate")
        model.train()
        torch.manual_seed(config.seed + 1 + info.dp_rank)
        np.random.seed(config.seed + 1 + info.dp_rank)
        rf_cfg = RectifiedFlowConfig.from_config(config.transport)
        rows = []
        source_counts = Counter()
        for step, indices in enumerate(audit_sampler, 1):
            batch = next(iterator)
            global_grads = regional_grads = None
            stats = torch.zeros(4, device=info.device, dtype=torch.float64)
            times = []
            for micro in range(config.grad_accum_steps):
                sl = slice(micro * micro_bs, (micro + 1) * micro_bs)
                images, latents, hidden, attention, xt, ut, t = prepare_flow_microbatch(
                    batch, sl, config, vae, bundle.text_encoder, info.device, rf_cfg)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.get("amp", True)):
                    prediction = model(xt, t, hidden, attention, train=True)
                    _, metrics = region_flow_loss(prediction, ut, batch["region_mask"][sl], 1.)
                losses = torch.stack([metrics["flow_loss"], metrics["region_flow_loss"]])
                finite = torch.isfinite(losses).all().to(torch.int32)
                if info.is_distributed:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError(f"Nonfinite calibration loss at batch {step}")
                gg, gr = measure_gradients(model, losses[0] / config.grad_accum_steps,
                                          losses[1] / config.grad_accum_steps,
                                          sharded=info.is_distributed)
                global_grads = add_gradients(global_grads, gg)
                regional_grads = add_gradients(regional_grads, gr)
                stats += torch.stack([metrics[key].detach() for key in
                                      ("flow_loss", "region_flow_loss", "region_weight_mean",
                                       "region_image_fraction")]).double() / config.grad_accum_steps
                times.extend(t.detach().cpu().tolist())
            g2, r2, dot = gradient_products(global_grads, regional_grads, info.device,
                                            distributed=info.is_distributed)
            del global_grads, regional_grads, gg, gr
            if info.is_distributed:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                stats /= info.world_size
                rank_times = [None] * info.world_size
                dist.all_gather_object(rank_times, times)
                times = [t for values in rank_times for t in values]
            row = dict(batch=step, bucket=f"{batch['image'].shape[1]}x{batch['image'].shape[2]}",
                       **geometry(g2, r2, dot),
                       **dict(zip(("global_loss", "region_loss", "mask_mean", "masked_image_fraction"),
                                  stats.tolist())),
                       timesteps=times, timestep_mean=float(np.mean(times)),
                       source_counts=dict(Counter(sources[i] for i in indices)),
                       sample_ids=[dataset.records[i][0].identifier for i in indices])
            source_counts.update(row["source_counts"])
            rows.append(row)
            if info.is_main:
                with (self.output / "batches.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                log(f"calibration {step}/{count}: G={row['G']:.6g} R={row['R']:.6g} "
                    f"cosine={row['cosine']} bucket={row['bucket']}")
        report = build_report(rows)
        report.update(format="region-flow-gradient-calibration-v1", provenance=provenance,
                      source_counts=dict(source_counts), batch_measurements=rows,
                      masked_image_fraction=float(np.mean([row["masked_image_fraction"] for row in rows])))
        sampled_buckets = {row["bucket"] for row in rows}
        # Dataset bucket tuples are (height, width), matching the tensor geometry.
        populated = {f"{h}x{w}" for (h, w), group in zip(dataset.buckets, dataset.groups) if len(group)}
        report["unsampled_buckets"] = sorted(populated - sampled_buckets)
        report["unsampled_sources"] = sorted(set(sources) - source_counts.keys())
        if report["unsampled_buckets"] or report["unsampled_sources"]:
            report["warnings"].append("Some buckets/sources were not sampled; increase --calibration-batches if needed.")
        if info.is_main:
            temporary = self.output / "calibration.json.tmp"
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(self.output / "calibration.json")
            log(f"Calibration saved to {self.output / 'calibration.json'}; "
                f"lambda_40={report['coefficients']['lambda_40']:.8g}; sweep={report['sweep']}")
