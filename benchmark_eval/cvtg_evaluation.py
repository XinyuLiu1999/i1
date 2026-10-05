"""CPU coordinator for CVTG scoring; one complete evaluator per selected GPU."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import tempfile

from text_benchmarks import check_cuda, gpu_ids, partitions, read_jsonl, run_jobs, save_json

HERE = Path(__file__).resolve().parent
CVTG = HERE / "cvtg-2k"
METRICS = ("clipscore", "vqascore", "aesthetic_score")


def configure_cache(cache_dir, use_hf_mirror=False):
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    # Set before importing transformers, huggingface_hub, OpenCLIP, or t2v_metrics.
    for name in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME"):
        os.environ[name] = str(cache)
    if use_hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    else:
        os.environ.pop("HF_ENDPOINT", None)
    return cache


def official_prompts(benchmark_dir):
    prompts = {}
    for category in ("CVTG", "CVTG-Style"):
        for area in (2, 3, 4, 5):
            data = json.loads((Path(benchmark_dir) / category / f"{area}.json").read_text())
            for row in data["data_list"]:
                key = f"{category}_{area}_{row['index']}"
                if key in prompts:
                    raise ValueError(f"Duplicate official ID: {key}")
                prompts[key] = {"id": key, "benchmark_type": category, "area": area, "prompt": row["prompt"]}
    return prompts


def load_samples(benchmark_dir, result_dir, manifest=None):
    """Use official prompts even when images were generated using rewritten prompts."""
    official = official_prompts(benchmark_dir)
    result_dir = Path(result_dir).resolve()
    samples = []
    if manifest is not None:
        seen = set()
        for row in read_jsonl(Path(manifest)):
            name, metadata = row["name"], row["metadata"]
            if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
                raise ValueError(f"Unsafe image name: {name}")
            key = metadata["id"]
            if key not in official or key in seen or metadata["prompt"] != official[key]["prompt"]:
                raise ValueError(f"Invalid/duplicate CVTG metadata: {key}")
            seen.add(key)
            samples.append({**official[key], "image": str(result_dir / name)})
    else:
        # Legacy process.py layout: missing/unknown images are errors, not partial scores.
        expected = {}
        for key, row in official.items():
            expected[(row["benchmark_type"], str(row["area"]), key.rsplit("_", 1)[1])] = row
        for path in sorted(result_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                relative = path.relative_to(result_dir)
                key = (*relative.parts[:-1], path.stem)
                if key not in expected:
                    raise ValueError(f"Unknown image in legacy CVTG layout: {relative}")
                samples.append({**expected[key], "image": str(path)})
        if {r["id"] for r in samples} != set(official):
            raise ValueError("Incomplete legacy CVTG image set; use --samples-file for an explicit subset")
    if not samples or len({r["image"] for r in samples}) != len(samples) or len({r["id"] for r in samples}) != len(samples):
        raise ValueError("Empty or duplicate CVTG samples")
    return samples


def validate_samples(samples):
    from PIL import Image
    for row in samples:
        with Image.open(row["image"]) as image:
            image.load()


def validate_record(row, expected):
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"Mismatched CVTG judgment {expected['id']}: {key}")
    words = [w for match in re.findall(r"'(.*?)'", expected["prompt"]) for w in match.lower().split()]
    total, correct = row.get("total_words"), row.get("correct_words")
    if type(total) is not int or type(correct) is not int or total != len(words) or not 0 <= correct <= total:
        raise ValueError(f"Invalid OCR counts: {expected['id']}")
    distances = row.get("ned_word_data")
    if not isinstance(distances, list) or len(distances) != total:
        raise ValueError(f"Invalid word-level NED: {expected['id']}")
    values = distances + [row.get(k) for k in METRICS]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError(f"Non-finite or missing CVTG score: {expected['id']}")
    if (any(not 0 <= v <= 1 for v in distances) or not 0 <= row["vqascore"] <= 1
            or not 0 <= row["clipscore"] <= 2.5001):
        raise ValueError(f"Out-of-range CVTG score: {expected['id']}")


def aggregate(rows):
    total = sum(r["total_words"] for r in rows)
    correct = sum(r["correct_words"] for r in rows)
    distances = [v for r in rows for v in r["ned_word_data"]]
    return {"word_accuracy": correct / max(total, 1),
            "ned": math.fsum(distances) / max(len(distances), 1),
            **{metric: math.fsum(r[metric] for r in rows) / len(rows) for metric in METRICS},
            "total_images": len(rows), "total_words": total, "correct_words": correct}


def merge_results(directory, samples, shards):
    ordered = []
    for rank, (start, end) in enumerate(shards):
        expected = {r["id"]: r for r in samples[start:end]}
        records = {}
        for row in read_jsonl(directory / f"results_chunk{rank}.jsonl"):
            key = row.get("id")
            if key not in expected or key in records:
                raise ValueError(f"Unexpected or duplicate CVTG judgment: {key}")
            validate_record(row, expected[key])
            records[key] = row
        if records.keys() != expected.keys():
            raise ValueError("Incomplete CVTG scoring; results were not published")
        ordered.extend(records[key] for key in expected)
    areas = []
    for category in ("CVTG", "CVTG-Style"):
        for area in (2, 3, 4, 5):
            rows = [r for r in ordered if r["benchmark_type"] == category and r["area"] == area]
            if rows:
                areas.append({"benchmark_type": category, "area": area, **aggregate(rows),
                              "ned_word_data": [v for r in rows for v in r["ned_word_data"]]})
    return ordered, {"overall_results": aggregate(ordered), "area_results": areas}


def worker_command(python, samples_file, output, start, end, cache_dir, batch_size, mirror, device):
    command = [python, str(Path(__file__).resolve()), "--worker", "--samples-file", str(samples_file),
               "--output_file", str(output), "--start", str(start), "--end", str(end),
               "--cache_dir", str(cache_dir), "--clip-batch-size", str(batch_size), "--device", device]
    return command + (["--use_hf_mirror"] if mirror else ["--no_hf_mirror"])


def _evaluate_dataset(samples, output_file, ids, python=sys.executable, cache_dir=None, batch_size=16,
                     mirror=False, device="cuda", dry_run=False, summary_metadata=None):
    import shlex
    output_file = Path(output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = str(Path(cache_dir or Path.home() / ".cache/cvtg").expanduser().resolve())
    shards = list(partitions(len(samples), len(ids)))
    if dry_run:
        for rank, (start, end) in enumerate(shards):
            command = worker_command(python, "<temporary>/samples.jsonl", f"<temporary>/results_chunk{rank}.jsonl",
                                     start, end, cache_dir, batch_size, mirror, device)
            print(f"CUDA_VISIBLE_DEVICES={ids[rank]} {shlex.join(command)}")
        return
    validate_samples(samples)
    if device == "cuda":
        check_cuda(ids[:len(shards)], python)
    log_dir = output_file.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cvtg-score-", dir=output_file.parent) as temporary:
        directory = Path(temporary)
        manifest = directory / "samples.jsonl"
        manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in samples), encoding="utf-8")
        jobs = [(ids[rank], worker_command(python, manifest, directory / f"results_chunk{rank}.jsonl",
                                          start, end, cache_dir, batch_size, mirror, device))
                for rank, (start, end) in enumerate(shards)]
        run_jobs(jobs, log_dir)
        records, results = merge_results(directory, samples, shards)
        summary = {**(summary_metadata or {}), "benchmark": "cvtg-2k", "judge_model": "clip-flant5-xxl",
                   "image_count": len(samples), "prompt_count": len(samples),
                   "clip_batch_size": batch_size, "gpu_count": len(shards) if device == "cuda" else 0,
                   **results["overall_results"]}
        # All validation completes before replacing any previously successful result.
        save_json(directory / "results.json", results)
        save_json(directory / "summary.json", summary)
        (directory / "results.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records), encoding="utf-8")
        (directory / "results.jsonl").replace(output_file.with_suffix(".jsonl"))
        (directory / "summary.json").replace(output_file.with_name(output_file.stem + "_summary.json"))
        (directory / "results.json").replace(output_file)
    print(f"CVTG results: {output_file}\n{json.dumps(results['overall_results'], indent=2)}", flush=True)


def evaluate_dataset(samples, output_file, ids, **kwargs):
    output_file = Path(output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Shared by the standalone CLI and benchmark runner, even with different callers.
    with output_file.with_suffix(".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another evaluation is using {output_file}")
        return _evaluate_dataset(samples, output_file, ids, **kwargs)


def evaluate_from_runner(args):
    normalized = load_samples(CVTG / "prompts", args.output_root / "images",
                              args.output_root / "inputs/samples.jsonl")
    ids = gpu_ids(args.gpu_ids, args.evaluation_python)
    evaluate_dataset(normalized, args.output_root / "eval_results/results.json", ids,
                     python=args.evaluation_python, cache_dir=args.evaluation_cache_dir,
                     batch_size=args.clip_batch_size, mirror=args.use_hf_mirror, dry_run=args.dry_run,
                     summary_metadata={"checkpoint": str(args.checkpoint), "prompt_variant": args.prompt_variant})


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark_dir", "--benchmark-dir", type=Path, default=CVTG / "prompts")
    p.add_argument("--result_dir", "--result-dir", type=Path)
    p.add_argument("--samples-file", type=Path, help="Runner inputs/samples.jsonl; images stay flat")
    p.add_argument("--output_file", "--output-file", type=Path, required=True)
    p.add_argument("--cache_dir", "--cache-dir", default=os.environ.get("CVTG_CACHE_DIR", str(Path.home() / ".cache/cvtg")))
    p.add_argument("--gpu-ids", default=os.environ.get("GPU_IDS"))
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--clip-batch-size", type=int, default=16)
    p.add_argument("--use_hf_mirror", action="store_true", default=True)
    p.add_argument("--no_hf_mirror", dest="use_hf_mirror", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--start", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--end", type=int, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.clip_batch_size <= 0:
        p.error("--clip-batch-size must be positive")
    if args.output_file.suffix != ".json" and not args.worker:
        p.error("--output_file must end in .json")
    if args.worker:
        if args.samples_file is None:
            p.error("Workers require --samples-file")
    elif args.result_dir is None:
        p.error("--result_dir is required")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        cache = configure_cache(args.cache_dir, args.use_hf_mirror)
        sys.path.insert(0, str(CVTG))
        from unified_metrics_eval import UnifiedMetricsEvaluator
        samples = read_jsonl(args.samples_file)[args.start:args.end]
        if not samples:
            raise ValueError("Empty worker shard")
        # Paddle's downloader is not process-safe; serialize cold-cache initialization.
        # Each GPU keeps its own model after releasing the shared cache lock.
        print(f"Initializing models on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}; "
              f"shared cache: {cache}", flush=True)
        with (cache / ".cvtg-models.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            evaluator = UnifiedMetricsEvaluator(device=args.device, cache_dir=cache)
        print(f"Models loaded; scoring {len(samples)} images", flush=True)
        with args.output_file.open("w", encoding="utf-8") as output:
            for index, row in enumerate(evaluator.score_samples(samples, args.clip_batch_size)):
                validate_record(row, samples[index])
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                print(f"Scored {index + 1}/{len(samples)}: {row['id']}", flush=True)
        return
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    samples = load_samples(args.benchmark_dir, args.result_dir, args.samples_file)
    device = "cpu" if args.device == "cpu" else "cuda"
    ids = [""] if device == "cpu" else gpu_ids(args.gpu_ids, sys.executable)
    evaluate_dataset(samples, args.output_file, ids, cache_dir=args.cache_dir,
                     batch_size=args.clip_batch_size, mirror=args.use_hf_mirror,
                     device=device, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
