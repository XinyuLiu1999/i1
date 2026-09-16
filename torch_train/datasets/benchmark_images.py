"""Compare source and cached CPU loading without large shared-memory IPC queues.

Each process consumes its rank's deterministic sampled images locally. This tests
concurrent storage/decode/transform work, not DataLoader IPC, tokenization or GPUs.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .bucketed import BucketedImages, BucketBatchSampler


def _reader(manifest, config, rank, world, steps, warmup):
    import torch
    torch.set_num_threads(1)
    dataset = BucketedImages(dict(config, manifest=manifest))
    sampler = BucketBatchSampler(dataset.groups, 32, rank, world, steps + warmup, seed=20260916)
    times = []
    first = None
    count = 0
    for step, indices in enumerate(sampler):
        start = time.time()
        if step == warmup:
            first = start
        for index in indices:
            pixels, caption = dataset[index]
            if not caption or pixels.shape[-1] != 3:
                raise ValueError('Invalid batch sample.')
        if step >= warmup:
            times.append(time.time() - start)
            count += len(indices)
    return dict(start=first, end=time.time(), count=count,
                p50_batch_seconds=float(np.percentile(times, 50)),
                p95_batch_seconds=float(np.percentile(times, 95)))


def benchmark(source, cached, config, output, readers=(1, 4, 8), steps=16, warmup=2):
    if steps < 1 or warmup < 0 or any(n < 1 or 32 % n for n in readers):
        raise ValueError('Positive steps/readers required; readers must divide global batch 32.')
    results = []
    for label, manifest in [('source', source), ('cache', cached)]:
        for count in readers:
            with ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context('spawn')) as executor:
                futures = [executor.submit(_reader, str(manifest), dict(config), rank, count, steps, warmup)
                           for rank in range(count)]
                ranks = [future.result() for future in futures]
            elapsed = max(row['end'] for row in ranks) - min(row['start'] for row in ranks)
            result = dict(input=label, readers=count, measured_images=sum(row['count'] for row in ranks),
                          elapsed_seconds=elapsed,
                          images_per_second=sum(row['count'] for row in ranks) / elapsed,
                          ranks=ranks)
            results.append(result)
            print(json.dumps({key: value for key, value in result.items() if key != 'ranks'}), flush=True)
    report = dict(scope='CPU concurrent readers; includes filesystem cache effects and rank startup skew; '
                        'excludes DataLoader IPC/prefetch, GPU transfer, tokenization, and model compute.',
                  global_batch=32, warmup_batches=warmup, measured_batches=steps, results=results)
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_manifest', required=True)
    parser.add_argument('--cache_manifest', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--resolution', type=int, choices=[512, 1024], default=1024)
    parser.add_argument('--steps', type=int, default=16)
    args = parser.parse_args()
    from configs.sft_512 import get_config as config512
    from configs.sft_1024 import get_config as config1024
    config = (config512 if args.resolution == 512 else config1024)().input
    benchmark(args.source_manifest, args.cache_manifest, config, args.report, steps=args.steps)


if __name__ == '__main__':
    main()
