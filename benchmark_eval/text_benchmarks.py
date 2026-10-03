"""Single-checkpoint, multi-GPU LongText/CVTG generation and optional LongText scoring."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SAMPLING = ("resolution", "seed", "num_steps", "text_num_tokens", "caption_overflow",
            "cfg_scale", "cfg_rescale", "inference_timestep_shift", "diffusion_batch_size", "vae_batch_size")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    checkpoint = os.environ.get("SFT_CHECKPOINT")
    if not checkpoint and os.environ.get("SFT_WORKDIR"):
        checkpoint = str(Path(os.environ["SFT_WORKDIR"]) / "checkpoint.pt")
    p.add_argument("--checkpoint", type=Path, default=checkpoint)
    p.add_argument("--benchmark", required=True, choices=("longtext", "cvtg", "cvtg-2k"))
    p.add_argument("--stage", choices=("prepare", "generate", "evaluate", "all"), default="generate")
    p.add_argument("--output-root", type=Path, default=os.environ.get("OUTPUT_ROOT"))
    p.add_argument("--gpu-ids", default=os.environ.get("GPU_IDS"))
    p.add_argument("--evaluation-python", default=os.environ.get("LONGTEXT_PYTHON", sys.executable),
                   help="Python with LongText scoring dependencies (default: generation Python)")
    p.add_argument("--prompt-variant", choices=("original", "simple_rewrite", "complex_rewrite"), default="original")
    p.add_argument("--limit", type=int, default=0, help="First N prompts; 0 uses the full benchmark")
    p.add_argument("--resolution", type=int, choices=(256, 512, 1024, 1536, 2048), default=1024)
    p.add_argument("--num-steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text-num-tokens", type=int, help="Default: checkpoint-native context; can extend but not shrink")
    p.add_argument("--caption-overflow", choices=("truncate", "error"), default="truncate")
    p.add_argument("--cfg-scale", type=float, default=12.0)
    p.add_argument("--cfg-rescale", type=float, default=1.0)
    p.add_argument("--inference-timestep-shift", type=float, default=0.3)
    p.add_argument("--diffusion-batch-size", type=int, default=1)
    p.add_argument("--vae-batch-size", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", help="Write manifests and print commands without loading models")
    args = p.parse_args(argv)
    if args.benchmark == "cvtg":
        args.benchmark = "cvtg-2k"
    if args.benchmark == "cvtg-2k" and args.stage in ("evaluate", "all"):
        p.error("CVTG only supports --stage prepare or generate; scoring is not implemented")
    if args.checkpoint is None:
        p.error("Set --checkpoint, SFT_CHECKPOINT, or SFT_WORKDIR")
    for key in ("num_steps", "text_num_tokens", "diffusion_batch_size", "vae_batch_size"):
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if args.limit < 0 or args.seed < 0:
        p.error("--limit and --seed must be non-negative")
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if not args.checkpoint.is_file():
        p.error("--checkpoint must be a file")
    args.output_root = (args.output_root or args.checkpoint.parent / args.benchmark /
                        f"{args.checkpoint.name}_{args.resolution}_{args.prompt_variant}").expanduser().resolve()
    return args


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_data(args):
    suffix = "" if args.prompt_variant == "original" else "_" + args.prompt_variant
    prompt_dir = ROOT / "jax/inference/prompts"
    samples = []
    if args.benchmark == "longtext":
        official = read_jsonl(HERE / "longtext/text_prompts.jsonl")
        selected = read_jsonl(prompt_dir / f"longtext{suffix}.jsonl")
        if [r["prompt_id"] for r in selected] != [r["prompt_id"] for r in official]:
            raise ValueError("LongText prompt IDs do not match the official benchmark")
        for row, prompt in list(zip(official, selected))[:args.limit or None]:
            for repeat in range(4):
                samples.append({"prompt": prompt["prompt"], "name": f"{row['prompt_id']}_{repeat}.png",
                                "metadata": row})
    else:
        official = json.loads((prompt_dir / "CVTG-2K.json").read_text(encoding="utf-8"))
        selected = json.loads((prompt_dir / f"CVTG-2K{suffix}.json").read_text(encoding="utf-8"))
        if [r[0] for r in selected] != [r[0] for r in official]:
            raise ValueError("CVTG prompt IDs do not match the official benchmark")
        for index, ((key, original), (_, prompt)) in enumerate(list(zip(official, selected))[:args.limit or None]):
            samples.append({"prompt": prompt, "name": f"{index:05d}.png",
                            "metadata": {"id": key, "prompt": original}})
    if not samples or any(not isinstance(r["prompt"], str) or not r["prompt"].strip() for r in samples):
        raise ValueError("Benchmark prompts must be non-empty strings")
    names = [r["name"] for r in samples]
    if len(set(names)) != len(names) or any(Path(n).name != n for n in names):
        raise ValueError("Duplicate or unsafe image names")
    return samples


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def ensure_identity(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Run settings changed: {path}. Use a new --output-root.")
    else:
        save_json(path, value)


def gpu_ids(value, python):
    if value is None:
        value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        count = int(subprocess.check_output([python, "-c", "import torch; print(torch.cuda.device_count())"], text=True))
        value = ",".join(map(str, range(count)))
    ids = [s.strip() for s in value.split(",")]
    if not ids or len(set(ids)) != len(ids) or any(
            not re.fullmatch(r"(?:[0-9]+|GPU-[\w-]+|MIG-[\w/.-]+)", s) for s in ids):
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


def check_cuda(ids, python):
    subprocess.run([python, "-c", "import sys,torch; "
                    "ok=torch.cuda.is_available() and torch.cuda.device_count()==int(sys.argv[1]); "
                    "sys.exit(0 if ok else 'Selected CUDA GPUs are unavailable')", str(len(ids))],
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(ids)}, check=True)


def run_jobs(jobs, log_dir, cwd=ROOT):
    """Fail fast and terminate process groups, including torchrun children."""
    children = []
    try:
        for index, (gpu, command) in enumerate(jobs):
            log_path = log_dir / f"worker_{index}.log"
            print(f"GPU {gpu}: {shlex.join(command)}\n  log: {log_path}", flush=True)
            with log_path.open("a", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=cwd, start_new_session=True,
                                         env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"},
                                         stdout=log, stderr=subprocess.STDOUT)
            children.append((child, log_path))
        while True:
            codes = [child.poll() for child, _ in children]
            for code, (_, log_path) in zip(codes, children):
                if code not in (None, 0):
                    raise RuntimeError(f"Worker failed ({code}); see {log_path}")
            if all(code is not None for code in codes):
                break
            time.sleep(0.2)
    finally:
        # A failed launcher may leave descendants alive after its own exit.
        for child, _ in children:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for child, _ in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def generation_command(args, start, end, worker):
    command = [sys.executable, str(ROOT / "torch_inference/generate.py"),
               "--checkpoint", str(args.checkpoint),
               "--prompts-jsonl", str(args.output_root / "inputs/samples.jsonl"),
               "--output-names-file", str(args.output_root / "inputs/output_names.txt"),
               "--outdir", str(args.output_root / "images"),
               "--start-idx", str(start), "--end-idx", str(end),
               "--seed", str(args.seed + worker), "--rewrite-prompt", "false", "--skip-existing",
               "--device", "cuda", "--height", str(args.resolution), "--width", str(args.resolution)]
    for key in SAMPLING:
        if key not in ("resolution", "seed") and getattr(args, key) is not None:
            command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    return command


def validate_images(args, samples, allow_missing=False):
    from PIL import Image
    expected = {r["name"] for r in samples}
    actual = {p.name for p in (args.output_root / "images").glob("*.png")}
    if actual - expected or (not allow_missing and actual != expected):
        raise ValueError(f"Image set mismatch: {len(expected-actual)} missing, {len(actual-expected)} unexpected")
    for name in sorted(actual):
        with Image.open(args.output_root / "images" / name) as image:
            if image.size != (args.resolution, args.resolution):
                raise ValueError(f"Wrong image size: {name}: {image.size}")
            image.load()


def merge_results(directory, samples, image_dir, workers):
    expected = {str((image_dir / row["name"]).resolve()): row["metadata"] for row in samples}
    records = {}
    for rank in range(workers):
        for row in read_jsonl(directory / f"results_chunk{rank}.jsonl"):
            key = str(Path(row["image"]).resolve())
            if key not in expected or key in records:
                raise ValueError(f"Unexpected or duplicate judgment: {key}")
            if (not isinstance(row.get("ocr_results"), str) or row.get("ocr_gt") != expected[key]["text"]
                    or row.get("prompt") != expected[key]["prompt"]):
                raise ValueError(f"Invalid judgment: {key}")
            records[key] = row
    if records.keys() != expected.keys():
        raise ValueError("Incomplete LongText scoring; summary was not published")
    path = directory / "results.jsonl"
    path.write_text("".join(json.dumps(records[key], ensure_ascii=False) + "\n" for key in expected), encoding="utf-8")
    return path


def evaluate(args, samples):
    out = args.output_root
    ids = gpu_ids(args.gpu_ids, args.evaluation_python)[:len(samples)]
    base = [args.evaluation_python, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc_per_node={len(ids)}", str(HERE / "longtext/evaluate_text_reward.py"),
            "--sample_dir", str(out / "images"), "--mode", "en", "--global_seed", str(args.seed),
            "--prompt_file", str(out / "inputs/text_prompts.jsonl")]
    if args.dry_run:
        print(shlex.join(base + ["--output_dir", "<temporary-score-directory>"]))
        print(shlex.join([args.evaluation_python, str(HERE / "longtext/summary_scores.py"),
                          "<temporary-score-directory>/results.jsonl", "--mode", "en"]))
        return
    check_cuda(ids, args.evaluation_python)
    log_dir = out / "logs/evaluation"
    log_dir.mkdir(exist_ok=True)
    # Fresh chunks on every evaluation prevent mixing old world sizes or partial runs.
    with tempfile.TemporaryDirectory(prefix=".longtext-score-", dir=out) as temporary:
        directory = Path(temporary)
        run_jobs([(",".join(ids), base + ["--output_dir", str(directory)])], log_dir)
        result = merge_results(directory, samples, out / "images", len(ids))
        subprocess.run([args.evaluation_python, str(HERE / "longtext/summary_scores.py"),
                        str(result), "--mode", "en"], cwd=HERE / "longtext", check=True)
        scored = read_jsonl(result)
        total = sum(row["gt_word_count"] for row in scored)
        summary = {"benchmark": "longtext", "checkpoint": str(args.checkpoint),
                   "prompt_count": len(samples) // 4, "image_count": len(samples),
                   "prompt_variant": args.prompt_variant,
                   "text_score": sum(row["match_word_count"] for row in scored) / total,
                   "judge_model": "Qwen/Qwen2.5-VL-7B-Instruct"}
        save_json(directory / "summary.json", summary)
        destination = out / "eval_results"
        destination.mkdir(exist_ok=True)
        for name in ("results.jsonl", "scores.txt", "summary.json"):
            (directory / name).replace(destination / name)
        print(f"LongText Text Score: {summary['text_score']:.4f}; {destination}", flush=True)


def execute(args):
    out = args.output_root
    samples = prepare_data(args)
    stat = args.checkpoint.stat()
    identity = {"version": 1, "benchmark": args.benchmark, "checkpoint": str(args.checkpoint),
                "checkpoint_size": stat.st_size, "checkpoint_mtime_ns": stat.st_mtime_ns,
                "prompt_variant": args.prompt_variant,
                "data_sha256": hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest(),
                "sampling": {key: getattr(args, key) for key in SAMPLING}}
    if not (out / "run.json").exists() and any(p.name != ".lock" for p in out.iterdir()):
        raise ValueError("Output directory is not empty and has no run.json; use a new --output-root")
    ensure_identity(out / "run.json", identity)
    for name in ("inputs", "images", "logs"):
        (out / name).mkdir(exist_ok=True)
    (out / "inputs/samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples), encoding="utf-8")
    (out / "inputs/output_names.txt").write_text("".join(row["name"] + "\n" for row in samples), encoding="utf-8")
    if args.benchmark == "longtext":
        (out / "inputs/text_prompts.jsonl").write_text(
            "".join(json.dumps(row["metadata"], ensure_ascii=False) + "\n" for row in samples[::4]), encoding="utf-8")
    print(f"Prepared {args.benchmark}: {len(samples)} images -> {out}", flush=True)
    if args.stage == "prepare":
        return
    if args.stage in ("generate", "all"):
        ids = gpu_ids(args.gpu_ids, sys.executable)
        repeats = 4 if args.benchmark == "longtext" else 1
        shards = [(a * repeats, b * repeats) for a, b in partitions(len(samples) // repeats, len(ids))]
        ensure_identity(out / "workers.json", {"shards": [list(shard) for shard in shards]})
        jobs = [(ids[i], generation_command(args, start, end, i)) for i, (start, end) in enumerate(shards)]
        if args.dry_run:
            for gpu, command in jobs:
                print(f"CUDA_VISIBLE_DEVICES={gpu} {shlex.join(command)}")
        else:
            validate_images(args, samples, allow_missing=True)
            check_cuda(ids[:len(shards)], sys.executable)
            run_jobs(jobs, out / "logs")
    if not args.dry_run:
        validate_images(args, samples)
    if args.stage in ("evaluate", "all"):
        evaluate(args, samples)
    print(f"{'Dry run' if args.dry_run else 'Stage'} complete: {args.stage}", flush=True)


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
