"""Single-node BizGenEval generation and official judging for DenseText SFT."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from prepare_inputs import image_name, select_bucket

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "torch_train" / "datasets"))
from image_geometry import generate_buckets


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    checkpoint = os.environ.get("SFT_CHECKPOINT")
    if not checkpoint and os.environ.get("SFT_WORKDIR"):
        checkpoint = str(Path(os.environ["SFT_WORKDIR"]) / "checkpoint.pt")
    p.add_argument("--checkpoint", type=Path, default=checkpoint)
    p.add_argument("--bizgeneval-root", type=Path,
                   default=os.environ.get("BIZGENEVAL_ROOT", ROOT.parent / "BizGenEval"))
    p.add_argument("--data-path", type=Path)
    p.add_argument("--output-root", type=Path, default=os.environ.get("OUTPUT_ROOT"))
    p.add_argument("--gpu-ids", default=os.environ.get("GPU_IDS"),
                   help="CUDA IDs/UUIDs; default: all GPUs visible to this process")
    p.add_argument("--stage", choices=("all", "prepare", "generate", "evaluate"), default="generate",
                   help="Default: generate images only; use all or evaluate to call the judge")
    p.add_argument("--resolution", type=int, choices=(1024, 1536, 2048), default=1024)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--num-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text-num-tokens", type=int, default=1024)
    p.add_argument("--native-text-context", dest="text_num_tokens", action="store_const", const=None,
                   help="Use checkpoint-native context without overriding text_num_tokens")
    p.add_argument("--caption-overflow", choices=("truncate", "error"), default="truncate")
    p.add_argument("--cfg-scale", type=float, default=12.0)
    p.add_argument("--cfg-rescale", type=float, default=1.0)
    p.add_argument("--inference-timestep-shift", type=float, default=0.3)
    p.add_argument("--diffusion-batch-size", type=int, default=1)
    p.add_argument("--vae-batch-size", type=int, default=1)
    p.add_argument("--evaluation-config", type=Path)
    p.add_argument("--force-rerun", action="store_true", help="Rejudge all images")
    p.add_argument("--dry-run", action="store_true", help="Prepare and print commands without models/API calls")
    args = p.parse_args(argv)
    if args.checkpoint is None:
        p.error("Set --checkpoint, SFT_CHECKPOINT, or SFT_WORKDIR")
    for key in ("num_steps", "text_num_tokens", "diffusion_batch_size", "vae_batch_size"):
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if args.limit < 0 or args.seed < 0:
        p.error("--limit and --seed must be non-negative")
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if not args.checkpoint.is_file():
        p.error("--checkpoint must point to a checkpoint file")
    args.bizgeneval_root = args.bizgeneval_root.expanduser().resolve()
    args.data_path = (args.data_path or args.bizgeneval_root / "assets/bizgeneval.jsonl").expanduser().resolve(strict=True)
    args.evaluation_config = (args.evaluation_config or args.bizgeneval_root / "config/evaluation_config.yaml").expanduser().resolve()
    args.output_root = (args.output_root or args.checkpoint.parent / "bizgeneval" /
                        f"{args.checkpoint.name}_{args.resolution}").expanduser().resolve()
    return args


def gpu_ids(value):
    if value is None:
        value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        import torch
        value = ",".join(str(i) for i in range(torch.cuda.device_count()))
    ids = [s.strip() for s in value.split(",")]
    if (not ids or any(not re.fullmatch(r"(?:[0-9]+|GPU-[\w-]+|MIG-[\w/.-]+)", s) for s in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("No valid unique CUDA GPUs; set --gpu-ids (e.g. 0,1,2,3)")
    return ids


def partitions(count, workers):
    workers = min(count, workers)
    base, extra = divmod(count, workers)
    start = 0
    for index in range(workers):
        end = start + base + (index < extra)
        yield start, end
        start = end


def prepare_data(args):
    rows = [json.loads(line) for line in args.data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("Benchmark is empty")
    short = 832 if args.resolution == 1024 else int((args.resolution ** 2 / 6) ** 0.5) // 16 * 16 * 2
    long = short * 3 // 2
    buckets = generate_buckets(args.resolution, step=32, max_ratio=3.0,
                               extra_shapes=[(short, long), (long, short)])
    names = []
    for row in rows:
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("Every row needs a non-empty prompt")
        if not row.get("questions") or not row.get("eval_tag"):
            raise ValueError("Every row needs the official questions and eval_tag")
        name = image_name(row)
        if Path(name).name != name or not name.lower().endswith(".png"):
            raise ValueError(f"Unsafe or non-PNG filename: {name}")
        names.append(name)
        row["_i1_height"], row["_i1_width"] = select_bucket(row, buckets)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate output filenames")
    return rows, names


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def ensure_identity(path, identity):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"Run settings changed: {path}. Use a new --output-root to avoid mixing results.")
    else:
        save_json(path, identity)


def generation_command(args, start, end, worker):
    out = args.output_root
    return [sys.executable, str(ROOT / "torch_inference/generate.py"),
            "--checkpoint", str(args.checkpoint),
            "--prompts-jsonl", str(out / "inputs/bizgeneval_i1.jsonl"),
            "--output-names-file", str(out / "inputs/output_names.txt"),
            "--jsonl-height-key", "_i1_height", "--jsonl-width-key", "_i1_width",
            "--start-idx", str(start), "--end-idx", str(end),
            "--seed", str(args.seed + worker), "--rewrite-prompt", "false", "--skip-existing",
            "--device", "cuda", "--resolution", "1024",
            "--outdir", str(out / "images")] + [
                value for key in ("num_steps", "text_num_tokens", "caption_overflow", "cfg_scale",
                                  "cfg_rescale", "inference_timestep_shift", "diffusion_batch_size", "vae_batch_size")
                if getattr(args, key) is not None
                for value in ("--" + key.replace("_", "-"), str(getattr(args, key)))]


def run_workers(jobs, log_dir):
    """One model replica per GPU; fail fast and reap all workers on interruption."""
    children = []
    try:
        for index, (gpu, command) in enumerate(jobs):
            log_path = log_dir / f"worker_{index}.log"
            with log_path.open("a", encoding="utf-8") as log:
                child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu,
                                                       "PYTHONUNBUFFERED": "1"}, stdout=log, stderr=subprocess.STDOUT)
            children.append((child, log_path))
        while any(child.poll() is None for child, _ in children):
            for child, log_path in children:
                if child.poll() not in (None, 0):
                    raise RuntimeError(f"Generation failed ({child.returncode}); see {log_path}")
            time.sleep(0.2)
        for child, log_path in children:
            if child.returncode:
                raise RuntimeError(f"Generation failed ({child.returncode}); see {log_path}")
    finally:
        for child, _ in children:
            if child.poll() is None:
                child.terminate()
        for child, _ in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def check_cuda(ids):
    # generate.py otherwise falls back to CPU if a selected device is unavailable.
    subprocess.run([
        sys.executable, "-c",
        "import sys,torch; n=int(sys.argv[1]); "
        "ok=torch.cuda.is_available() and torch.cuda.device_count()==n; "
        "sys.exit(0 if ok else 'Selected GPUs are unavailable in this i1_sft environment')",
        str(len(ids)),
    ], env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(ids)}, check=True)


def execute(args):
    if args.stage in ("all", "evaluate"):
        if not (args.bizgeneval_root / "evaluation/image_evaluation.py").is_file():
            raise FileNotFoundError(f"Official evaluator missing in {args.bizgeneval_root}")
        if not args.evaluation_config.is_file():
            raise FileNotFoundError(f"Evaluator config missing: {args.evaluation_config}")
    rows, names = prepare_data(args)
    out = args.output_root
    stat = args.checkpoint.stat()
    identity = {"version": 1, "checkpoint": str(args.checkpoint), "checkpoint_size": stat.st_size,
                "checkpoint_mtime_ns": stat.st_mtime_ns,
                "data_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                "sampling": {key: getattr(args, key) for key in (
                    "resolution", "seed", "num_steps", "text_num_tokens", "caption_overflow",
                    "cfg_scale", "cfg_rescale", "inference_timestep_shift", "diffusion_batch_size", "vae_batch_size")}}
    if not (out / "run.json").exists() and any(p.name != ".lock" for p in out.iterdir()):
        raise ValueError("Output directory is not empty and has no run.json; use a new --output-root")
    ensure_identity(out / "run.json", identity)
    for name in ("inputs", "images", "logs"):
        (out / name).mkdir(exist_ok=True)
    data = out / "inputs/bizgeneval_i1.jsonl"
    data.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (out / "inputs/output_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"Prepared {len(rows)} prompts at {args.resolution} tier: {out}", flush=True)
    if args.stage == "prepare":
        return
    if args.stage in ("all", "generate"):
        ids = gpu_ids(args.gpu_ids)
        shards = list(partitions(len(rows), len(ids)))
        # Pin shard count for deterministic existing inference RNG streams. GPU IDs can change.
        ensure_identity(out / "workers.json", {"shards": [list(shard) for shard in shards]})
        jobs = [(ids[i], generation_command(args, start, end, i)) for i, (start, end) in enumerate(shards)]
        for i, ((gpu, cmd), (start, end)) in enumerate(zip(jobs, shards)):
            print(f"worker {i}: GPU {gpu}, rows [{start}, {end}), log={out / 'logs' / f'worker_{i}.log'}", flush=True)
            if args.dry_run:
                import shlex
                print(shlex.join(cmd))
        if not args.dry_run:
            check_cuda(ids[:len(shards)])
            run_workers(jobs, out / "logs")
    commands = [[sys.executable, str(HERE / "validate_images.py"), "--names", str(out / "inputs/output_names.txt"),
                 "--image-dir", str(out / "images"), "--data", str(data), "--geometry", "native_buckets"]]
    if args.stage in ("all", "evaluate"):
        judge_identity = {"config_sha256": hashlib.sha256(args.evaluation_config.read_bytes()).hexdigest()}
        if not args.dry_run:
            # A changed judge config must not silently reuse old judgments.
            judge_path = out / "judge.json"
            if not args.force_rerun:
                ensure_identity(judge_path, judge_identity)
            else:
                # Invalidate cached identity until a forced rejudge fully succeeds.
                save_json(judge_path, {"pending_force_rerun": True})
        commands += [
            [sys.executable, "-m", "evaluation.image_evaluation", "--data_path", str(data),
             "--img_dir", str(out / "images"), "--save_dir", str(out / "eval_results"),
             "--config_path", str(args.evaluation_config)] + (["--force_rerun"] if args.force_rerun else []),
            [sys.executable, str(HERE / "validate_results.py"), "--data", str(data),
             "--result-dir", str(out / "eval_results")],
            [sys.executable, "-m", "evaluation.summarize", "--data_path", str(data),
             "--result_dir", str(out / "eval_results"), "--save_dir", str(out / "summaries")]]
    for command in commands:
        if args.dry_run:
            import shlex
            print(shlex.join(command))
        else:
            subprocess.run(command, cwd=args.bizgeneval_root, check=True)
    if not args.dry_run and args.stage in ("all", "evaluate"):
        save_json(out / "judge.json", judge_identity)
    print(f"{'Dry run' if args.dry_run else 'Stage'} complete: {out}", flush=True)


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    with (args.output_root / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another run is using {args.output_root}")
        execute(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
