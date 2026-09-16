"""Run corrected-index, pixel-cache and CPU audit stages with persistent logs/status."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run_pipeline(source, output, workers=8, resolution=1024, existing_index_pid=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    index = output / 'corrected_images.jsonl'
    cache = output / f'cache_{resolution}'
    state = dict(pid=os.getpid(), status='running', resolution=resolution, workers=workers, stages=[])

    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temporary = output / 'status.json.partial'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(output / 'status.json')

    def step(name, module, arguments):
        entry = dict(name=name, status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                     log=str(output / f'{name}.log'))
        state['stages'].append(entry)
        state['current_stage'] = name
        save()
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        command = [sys.executable, '-u', '-m', module, *map(str, arguments)]
        with open(entry['log'], 'w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            entry['pid'] = process.pid
            save()
            result = process.wait()
        entry.update(status='complete' if result == 0 else 'failed', exit_code=result,
                     finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        if result:
            raise RuntimeError(f'{name} exited with code {result}; see {entry["log"]}')

    try:
        if existing_index_pid is not None:
            state['current_stage'] = 'waiting_for_running_index'
            state['index_builder_pid'] = existing_index_pid
            save()
            while not (index.is_file() and index.with_suffix('.report.json').is_file()):
                process = Path(f'/proc/{existing_index_pid}/status')
                if not process.exists() or '\nState:\tZ' in process.read_text():
                    raise RuntimeError('Index builder stopped without publishing a complete index; see index_build.log.')
                time.sleep(5)
            report = json.loads(index.with_suffix('.report.json').read_text())
            if Path(report['source']).resolve() != Path(source).resolve():
                raise ValueError('Existing corrected index belongs to another source.')
            state['stages'].append(dict(name='index_build', status='complete', indexed=report['indexed'],
                                        log=str(output / 'index_build.log')))
        else:
            step('index_build', 'datasets.build_image_index',
                 ['--manifest', source, '--output', index, '--workers', workers])
        step('caption_audit', 'datasets.inspect_captions',
             ['--manifest', index, '--report', output / 'caption_audit.json'])
        step('pixel_cache', 'datasets.precompute_images',
             ['--manifest', index, '--output_dir', cache, '--resolution', resolution, '--workers', workers])
        step('cache_checks', 'datasets.check_precompute',
             ['--manifest', cache / 'cache.jsonl', '--resolution', resolution,
              '--report', output / 'cache_checks.json'])
        step('io_benchmark', 'datasets.benchmark_images',
             ['--source_manifest', index, '--cache_manifest', cache / 'cache.jsonl',
              '--resolution', resolution, '--report', output / 'io_benchmark.json'])
        step('gallery', 'datasets.build_inspection_gallery',
             ['--manifest', index, '--output_dir', output / 'gallery', '--count', 150])
        state['status'] = 'complete'
        state['current_stage'] = None
        state['training_manifest'] = str(cache / 'cache.jsonl')
        save()
    except Exception as error:
        state['status'] = 'failed'
        state['error'] = f'{type(error).__name__}: {error}'
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resolution', type=int, choices=[512, 1024], default=1024)
    parser.add_argument('--existing_index_pid', type=int,
                        help='Wait for an index builder already writing corrected_images.jsonl in output_dir.')
    args = parser.parse_args()
    run_pipeline(args.source, args.output_dir, args.workers, args.resolution, args.existing_index_pid)


if __name__ == '__main__':
    main()
