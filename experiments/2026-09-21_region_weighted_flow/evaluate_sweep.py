#!/usr/bin/env python3
"""Render matched BizGenEval subsets across regional-loss sweep checkpoints."""
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote

PROJECT = Path(__file__).resolve().parents[3]
INFERENCE = PROJECT / 'i1/torch_inference'
DEFAULT_WEIGHTS = '0,29.778307248555066,59.55661449711013,119.11322899422026'


def default_source():
    candidates = [Path('/cephfs/liuxinyu/BizGenEval/assets/bizgeneval.jsonl'),
                  PROJECT / 'artifacts/bizgeneval_evaluation/inputs/bizgeneval_i1.jsonl',
                  PROJECT / 'artifacts/bizgeneval_start_vs_sft6262/inputs/bizgeneval_i1.jsonl']
    return next((path for path in candidates if path.is_file()), candidates[0])


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep-root', type=Path, default=os.environ.get(
        'SWEEP_ROOT', str(PROJECT / 'artifacts/region_weighted_flow_2026-09-21/sweep_500')))
    parser.add_argument('--output-root', type=Path, default=os.environ.get('OUTPUT_ROOT'))
    parser.add_argument('--gpus', default=os.environ.get('GPU_IDS', '0'), help='Comma-separated GPU IDs; default: 0')
    parser.add_argument('--weights', default=os.environ.get('REGION_WEIGHTS', DEFAULT_WEIGHTS))
    parser.add_argument('--num-prompts', type=positive, default=os.environ.get('NUM_PROMPTS', '40'))
    parser.add_argument('--num-steps', type=positive, default=os.environ.get('NUM_STEPS', '50'))
    parser.add_argument('--seed', type=int, default=os.environ.get('SEED', '0'))
    parser.add_argument('--checkpoint-step', type=positive, default=os.environ.get('CHECKPOINT_STEP', '500'),
                        help='Require a saved checkpoint at this training step in each train.log')
    parser.add_argument('--source', type=Path, default=os.environ.get(
        'BIZGENEVAL_SOURCE', str(default_source())))
    parser.add_argument('--launch-delay', type=float, default=os.environ.get('GPU_LAUNCH_DELAY', '10'),
                        help='Seconds between GPU worker launches to stagger checkpoint loading')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and print commands without loading models')
    args = parser.parse_args()
    args.gpus = [gpu.strip() for gpu in args.gpus.split(',')]
    if (not all(gpu.isdecimal() for gpu in args.gpus)
            or len(set(map(int, args.gpus))) != len(args.gpus)):
        parser.error('--gpus must contain distinct nonnegative integer IDs')
    args.weights = [weight.strip() for weight in args.weights.split(',')]
    try:
        valid = all(math.isfinite(float(w)) and float(w) >= 0 and '/' not in w and '\\' not in w
                    for w in args.weights)
    except ValueError:
        valid = False
    if not valid or len(set(args.weights)) != len(args.weights):
        parser.error('--weights must contain distinct finite nonnegative coefficients')
    if args.seed < 0 or not math.isfinite(args.launch_delay) or args.launch_delay < 0:
        parser.error('--seed and --launch-delay must be nonnegative')
    args.sweep_root = args.sweep_root.expanduser().resolve()
    args.source = args.source.expanduser().resolve()
    args.output_root = (args.output_root or args.sweep_root /
                        f'evaluation_{args.num_prompts}prompts_{args.num_steps}steps').expanduser().resolve()
    return args


def checkpoint_info(args, weight):
    directory = args.sweep_root / f'w{weight}'
    checkpoint = directory / 'checkpoint.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    # A training workdir already has a step-1 checkpoint while it is running.
    # Require the final save marker, rather than mistaking file existence for completion.
    log = (directory / 'train.log').read_text()
    saves = [line for line in log.splitlines()
             if 'saved checkpoint to ' in line and '/checkpoint.pt ' in line]
    if not saves or f'[step {args.checkpoint_step}] saved checkpoint' not in saves[-1]:
        raise ValueError(f'{directory}: no latest saved checkpoint at step {args.checkpoint_step}; '
                         'finish the matched training arms before evaluation')
    stat = checkpoint.stat()
    return dict(path=str(checkpoint), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def worker_plan(args):
    count = min(len(args.gpus), args.num_prompts)
    base, extra = divmod(args.num_prompts, count)
    start = 0
    workers = []
    for index, gpu in enumerate(args.gpus[:count]):
        end = start + base + (index < extra)
        workers.append(dict(gpu=gpu, start=start, end=end, seed=args.seed + index))
        start = end
    return workers


def command(args, checkpoint, label, worker):
    return [sys.executable, '-u', str(INFERENCE / 'generate.py'),
            '--checkpoint', checkpoint,
            '--prompts-jsonl', str(args.output_root / 'inputs/metadata.jsonl'),
            '--output-names-file', str(args.output_root / 'inputs/output_names.txt'),
            '--start-idx', str(worker['start']), '--end-idx', str(worker['end']),
            '--seed', str(worker['seed']), '--skip-existing',
            '--rewrite-prompt', 'false', '--text-num-tokens', '1024',
            '--caption-overflow', 'truncate', '--resolution', '1024',
            '--height', '1024', '--width', '1024', '--num-steps', str(args.num_steps),
            '--cfg-scale', '12', '--cfg-rescale', '1.0', '--inference-timestep-shift', '0.3',
            '--diffusion-batch-size', '1', '--vae-batch-size', '1', '--device', 'cuda',
            '--outdir', str(args.output_root / label)]


def run_workers(args, checkpoint, label, workers):
    processes = []
    try:
        for index, worker in enumerate(workers):
            logfile = args.output_root / 'logs' / label / f"gpu{worker['gpu']}.log"
            logfile.parent.mkdir(parents=True, exist_ok=True)
            print(f"{label}: GPU {worker['gpu']}, prompts [{worker['start']}, {worker['end']}), "
                  f"seed {worker['seed']}; log: {logfile}", flush=True)
            with logfile.open('a') as handle:
                process = subprocess.Popen(command(args, checkpoint, label, worker),
                                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': worker['gpu']},
                                           stdout=handle, stderr=subprocess.STDOUT)
            processes.append(process)
            if index + 1 < len(workers) and args.launch_delay:
                time.sleep(args.launch_delay)
        while True:
            codes = [process.poll() for process in processes]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f'{label}: generation worker failed; see {args.output_root / "logs" / label}')
            if all(code == 0 for code in codes):
                break
            time.sleep(1)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def write_comparison(args, labels, inputs):
    rows = [json.loads(line) for line in inputs['metadata.jsonl'].decode().splitlines()]
    names = inputs['output_names.txt'].decode().splitlines()
    columns = [dict(directory=label, title=f'λ = {label[1:]}',
                    detail=f'Step {args.checkpoint_step} · 1024 tokens · square') for label in labels]
    references_path = args.output_root / 'comparison_references.json'
    references = json.loads(references_path.read_text()) if references_path.is_file() else None
    if references:
        columns.extend(references['columns'])
    headings = ''.join(f'<th>{html.escape(column["title"])}<br>'
                       f'<small>{html.escape(column["detail"])}</small></th>' for column in columns)
    sections = []
    for index, (row, name) in enumerate(zip(rows, names), 1):
        title = f"{index:02d} · {row.get('domain', '')} / {row.get('dimension', '')} · ID {row.get('id', '')}"
        prompt = row['prompt']
        cards = []
        for column in columns:
            url = quote(f'{column["directory"]}/{name}', safe='/')
            alt = html.escape(f"{title}, {column['title']}", quote=True)
            width, height = column.get('image_sizes', {}).get(name, [1024, 1024])
            cards.append(f'<td><a href="{url}" target="_blank" rel="noopener">'
                         f'<img src="{url}" loading="lazy" width="{width}" height="{height}" alt="{alt}"></a></td>')
        sections.append(
            f'<tbody data-search="{html.escape(title + " " + prompt, quote=True)}">'
            f'<tr><th class="prompt" colspan="{len(columns)}">{html.escape(title)}'
            f'<details><summary>Show prompt</summary><p>{html.escape(prompt)}</p></details></th></tr>'
            f'<tr>{"".join(cards)}</tr></tbody>')
    page = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Regional loss · BizGenEval comparison</title>
<style>
:root{--tile:320px;color-scheme:light;font:15px system-ui,sans-serif;color:#20282d;background:#f3f5f6}
body{margin:24px}h1{font-size:24px;margin:0 0 8px}header p{color:#53616a}
.controls{display:flex;gap:24px;align-items:center;flex-wrap:wrap;margin:20px 0}
input[type=search]{padding:10px 12px;width:320px;max-width:80vw;border:1px solid #bdc8cd;border-radius:6px}
label{display:flex;align-items:center;gap:10px}.table-wrap{overflow:auto;max-height:80vh;border:1px solid #d6dfe3;border-radius:8px;background:white}
table{border-collapse:separate;border-spacing:0}thead th{position:sticky;top:0;background:#e4edf0;z-index:1;white-space:nowrap;padding:14px}
td{padding:8px;border-bottom:1px solid #d6dfe3;vertical-align:top}img{display:block;width:var(--tile);height:auto;max-width:none}
.prompt{text-align:left;padding:14px;background:#f8fafb;border-top:1px solid #d6dfe3}
details{font-weight:normal;margin-top:6px}summary{cursor:pointer;color:#236779}
details p{max-width:1000px;white-space:pre-wrap;font-size:14px;line-height:1.6}tbody[hidden]{display:none}
small{color:#53616a}a:focus-visible,summary:focus-visible{outline:3px solid #236779}
</style>
<header><h1>Regional loss · BizGenEval comparison</h1>
<p>__DESCRIPTION__</p>__REFERENCE_NOTE__<small>Compare matching rows. Click an image to open the original full-resolution output.</small></header>
<div class="controls"><input id="search" type="search" aria-label="Filter prompts" placeholder="Filter by prompt, category or ID">
<label>Image size <input id="size" type="range" min="160" max="1024" value="320" step="16"></label>
<span id="count" aria-live="polite"></span></div>
<div class="table-wrap"><table><thead><tr>__HEADINGS__</tr></thead>__SECTIONS__</table></div>
<script>
const rows=[...document.querySelectorAll('tbody[data-search]')];
const search=document.getElementById('search');
function filter(){const query=search.value.toLowerCase();let count=0;
rows.forEach(row=>{row.hidden=!row.dataset.search.toLowerCase().includes(query);if(!row.hidden)count++;});
document.getElementById('count').textContent=`${count} / ${rows.length} prompts`;}
search.addEventListener('input',filter);filter();
document.getElementById('size').addEventListener('input',event=>{
document.documentElement.style.setProperty('--tile',event.target.value+'px');});
</script></html>
'''
    description = (f'{len(labels)} sweep checkpoints · training step {args.checkpoint_step} · '
                   f'{args.num_prompts} stratified prompts · {args.num_steps} denoising steps · '
                   '1024-token truncation · matched worker seeds within the sweep')
    reference_note = f'<p>{html.escape(references["note"])}</p>' if references else ''
    page = (page.replace('__DESCRIPTION__', html.escape(description))
            .replace('__REFERENCE_NOTE__', reference_note)
            .replace('__HEADINGS__', headings).replace('__SECTIONS__', ''.join(sections)))
    path = args.output_root / 'comparison.html'
    temporary = path.with_suffix('.html.tmp')
    temporary.write_text(page, encoding='utf-8')
    temporary.replace(path)
    return path


def main():
    args = arguments()
    checkpoints = {f'w{weight}': checkpoint_info(args, weight) for weight in args.weights}
    print(f'Prompt source: {args.source}', flush=True)
    workers = worker_plan(args)
    # Reuse the existing category-stratified selector and filename contract.
    with tempfile.TemporaryDirectory(prefix='region-sweep-prompts-') as temporary:
        subprocess.run([sys.executable, str(INFERENCE / 'prepare_bizgeneval_subset.py'),
                        '--input', str(args.source), '--output-dir', temporary,
                        '--limit', str(args.num_prompts), '--selection', 'stratified'],
                       check=True, stdout=subprocess.DEVNULL)
        inputs = {name: (Path(temporary) / name).read_bytes()
                  for name in ('metadata.jsonl', 'output_names.txt', 'prompts.txt')}
    names = inputs['output_names.txt'].decode().splitlines()
    if len(set(names)) != args.num_prompts or any(Path(name).name != name for name in names):
        raise ValueError('Expected unique, safe output names for every selected prompt')
    manifest = dict(format='region-sweep-evaluation-v1', checkpoints=checkpoints,
                    checkpoint_step=args.checkpoint_step, num_prompts=args.num_prompts,
                    num_steps=args.num_steps, workers=workers, resolution=[1024, 1024],
                    text_num_tokens=1024, caption_overflow='truncate', rewrite_prompt=False,
                    cfg_scale=12, cfg_rescale=1., inference_timestep_shift=.3,
                    diffusion_batch_size=1, vae_batch_size=1,
                    inputs_sha256={name: hashlib.sha256(value).hexdigest() for name, value in inputs.items()})
    manifest_path = args.output_root / 'evaluation.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Evaluation settings, prompts, GPU partition, or checkpoints changed; '
                             'use a new --output-root to avoid mixing images')
        for name, value in inputs.items():
            if (args.output_root / 'inputs' / name).read_bytes() != value:
                raise ValueError(f'Saved evaluation input changed: {name}')
    elif args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError('Nonempty output directory has no evaluation manifest; use a new --output-root')
    print(f'{len(checkpoints)} checkpoints x {args.num_prompts} prompts; {args.num_steps} denoising steps; '
          f'{len(workers)} GPUs per checkpoint; output: {args.output_root}', flush=True)
    if args.dry_run:
        import shlex
        for label, info in checkpoints.items():
            for worker in workers:
                print(f"CUDA_VISIBLE_DEVICES={worker['gpu']} " + shlex.join(command(args, info['path'], label, worker)))
        return
    (args.output_root / 'inputs').mkdir(parents=True, exist_ok=True)
    for name, value in inputs.items():
        (args.output_root / 'inputs' / name).write_bytes(value)
    temporary = manifest_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(manifest_path)
    from PIL import Image
    for label, info in checkpoints.items():
        if checkpoint_info(args, label[1:]) != info:
            raise ValueError(f'{label}: checkpoint changed after preflight; stop training before evaluation')
        run_workers(args, info['path'], label, workers)
        for name in names:
            with Image.open(args.output_root / label / name) as image:
                image.load()
                if image.size != (1024, 1024):
                    raise ValueError(f'{label}/{name}: unexpected image dimensions')
        print(f'{label}: verified {len(names)} images', flush=True)
    page = write_comparison(args, list(checkpoints), inputs)
    print(f'Done. Open {page}', flush=True)


def interrupted(signum, frame):
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
