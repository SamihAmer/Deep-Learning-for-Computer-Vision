"""
Lightweight training profiler for comparing computational cost across loss variants.

Records wall-clock time, GPU memory, and per-epoch timing breakdown.
All data is appended to the existing training_log.json with no extra files.

Usage:
    profiler = TrainingProfiler(device)

    for epoch in range(epochs):
        profiler.start_epoch()

        # ... training loop ...
        profiler.log_forward()    # call after forward pass
        profiler.log_backward()   # call after backward pass

        profiler.end_epoch(train_loss=loss_val)

    profiler.summary()  # prints final comparison-ready stats
"""

import time
from typing import Dict, Optional

import torch


class TrainingProfiler:
    """
    Tracks per-epoch and aggregate training costs.

    Designed for the ablation study: run this with each loss variant,
    then compare the summary stats to see the real overhead of
    topology-aware losses on CTA data.
    """

    def __init__(self, device: torch.device, loss_name: str = "unknown"):
        self.device = device
        self.loss_name = loss_name
        self.use_cuda = device.type == "cuda"

        # Per-epoch accumulators
        self._epoch_start = 0.0
        self._forward_times = []
        self._backward_times = []
        self._t_mark = 0.0

        # Aggregate tracking
        self.epoch_records = []
        self.peak_memory_bytes = 0
        self.total_train_time = 0.0

    def start_epoch(self):
        """Call at the beginning of each training epoch."""
        if self.use_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        self._epoch_start = time.perf_counter()
        self._forward_times = []
        self._backward_times = []

    def mark(self):
        """Set a timestamp for measuring intervals."""
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        self._t_mark = time.perf_counter()

    def log_forward(self):
        """Call immediately after the forward pass + loss computation."""
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        if self._t_mark > 0:
            self._forward_times.append(now - self._t_mark)
        self._t_mark = now

    def log_backward(self):
        """Call immediately after backward + optimizer step."""
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        if self._t_mark > 0:
            self._backward_times.append(now - self._t_mark)
        self._t_mark = now

    def end_epoch(self, train_loss: float = 0.0) -> Dict:
        """
        Call at the end of each epoch. Returns a record dict that
        can be merged into the training log.
        """
        if self.use_cuda:
            torch.cuda.synchronize(self.device)

        epoch_time = time.perf_counter() - self._epoch_start

        # Peak GPU memory for this epoch
        if self.use_cuda:
            peak_mem = torch.cuda.max_memory_allocated(self.device)
            self.peak_memory_bytes = max(self.peak_memory_bytes, peak_mem)
        else:
            peak_mem = 0

        # Average step times
        avg_forward = (
            sum(self._forward_times) / len(self._forward_times)
            if self._forward_times else 0.0
        )
        avg_backward = (
            sum(self._backward_times) / len(self._backward_times)
            if self._backward_times else 0.0
        )

        self.total_train_time += epoch_time

        record = {
            "epoch_time_sec": round(epoch_time, 2),
            "avg_forward_ms": round(avg_forward * 1000, 1),
            "avg_backward_ms": round(avg_backward * 1000, 1),
            "peak_gpu_memory_gb": round(peak_mem / 1e9, 2),
            "n_steps": len(self._forward_times),
        }
        self.epoch_records.append(record)
        return record

    def summary(self) -> Dict:
        """
        Produce a summary suitable for the results table.

        Returns dict with:
            - total_hours: total wall-clock training time
            - avg_epoch_sec: mean epoch duration
            - avg_forward_ms: mean forward pass time per step
            - avg_backward_ms: mean backward pass time per step
            - peak_gpu_gb: maximum GPU memory across all epochs
        """
        n = len(self.epoch_records)
        if n == 0:
            return {}

        summary = {
            "loss_name": self.loss_name,
            "total_epochs": n,
            "total_hours": round(self.total_train_time / 3600, 2),
            "avg_epoch_sec": round(
                sum(r["epoch_time_sec"] for r in self.epoch_records) / n, 1
            ),
            "avg_forward_ms": round(
                sum(r["avg_forward_ms"] for r in self.epoch_records) / n, 1
            ),
            "avg_backward_ms": round(
                sum(r["avg_backward_ms"] for r in self.epoch_records) / n, 1
            ),
            "peak_gpu_gb": round(self.peak_memory_bytes / 1e9, 2),
        }
        return summary

    def format_summary(self) -> str:
        """Pretty-print the summary for terminal output."""
        s = self.summary()
        if not s:
            return "No profiling data recorded."

        return (
            f"\n  Profiling summary for [{s['loss_name']}]:\n"
            f"    Total training time: {s['total_hours']:.2f} hours ({s['total_epochs']} epochs)\n"
            f"    Avg epoch:           {s['avg_epoch_sec']:.1f} sec\n"
            f"    Avg forward pass:    {s['avg_forward_ms']:.1f} ms/step\n"
            f"    Avg backward pass:   {s['avg_backward_ms']:.1f} ms/step\n"
            f"    Peak GPU memory:     {s['peak_gpu_gb']:.2f} GB\n"
        )
