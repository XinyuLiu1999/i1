"""Small torchrun connectivity check; no model weights or dataset required."""
from __future__ import annotations

import argparse
from datetime import timedelta
import os
import socket

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cpu")
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError(f"GPU {local_rank} does not support native BF16")
    dist.init_process_group(args.backend, timeout=timedelta(seconds=args.timeout))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        value = torch.tensor([rank + 1.0], device=device)
        dist.all_reduce(value)
        expected = world * (world + 1) / 2
        if value.item() != expected:
            raise RuntimeError(f"all_reduce returned {value.item()}, expected {expected}")
        # Exercise a collective also used for sharded model weights.
        shards = [torch.empty_like(value) for _ in range(world)]
        dist.all_gather(shards, torch.tensor([float(rank)], device=device))
        if [t.item() for t in shards] != list(range(world)):
            raise RuntimeError("all_gather returned unexpected ranks")
        print(f"PASS host={socket.gethostname()} rank={rank}/{world} "
              f"local_rank={local_rank} backend={args.backend} device={device}", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
