"""Inference-cost measurement: NFE, latency, samples/s, peak GPU memory. Fixed protocol (prereg §4):
batch 64, fp32, no torch.compile, CUDA events + synchronize, 5 warm-up + N timed repeats, report median."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch


@torch.no_grad()
def benchmark(sample_fn: Callable[[], object], n_warmup: int = 5, n_repeats: int = 20, batch_size: int | None = None, device: str = "cuda") -> dict:
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    for _ in range(n_warmup):
        sample_fn()
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(n_repeats):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        sample_fn()
        if is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    out = {"latency_ms_median": med * 1000, "latency_ms_mean": float(np.mean(times)) * 1000, "latency_ms_std": float(np.std(times)) * 1000, "n_repeats": n_repeats}
    if batch_size:
        out["samples_per_s"] = batch_size / med
    if is_cuda:
        out["peak_mem_MiB"] = torch.cuda.max_memory_allocated() / 2**20
    return out
