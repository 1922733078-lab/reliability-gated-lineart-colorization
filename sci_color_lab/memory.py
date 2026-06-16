from __future__ import annotations

import os
import threading
import time
from typing import Any

import torch


def _read_current_rss_gb() -> float | None:
    status_path = "/proc/self/status"
    try:
        with open(status_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("VmRSS:"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    return None
                rss_kb = float(parts[1])
                return round(rss_kb / (1024.0 * 1024.0), 4)
    except Exception:
        return None
    return None


class PeakMemoryMonitor:
    def __init__(self, device: str | torch.device | None = None, poll_interval_s: float = 0.05) -> None:
        self.poll_interval_s = max(float(poll_interval_s), 0.01)
        self.device = None
        if device is not None:
            normalized = torch.device(device)
            if normalized.type == "cuda" and torch.cuda.is_available():
                self.device = normalized
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._started = False
        self._stopped = False
        self._result: dict[str, Any] | None = None
        self.cpu_memory_peak_gb: float | None = None
        self.gpu_memory_peak_gb: float | None = None
        self.gpu_memory_reserved_peak_gb: float | None = None

    def _sample_once(self) -> None:
        cpu_rss_gb = _read_current_rss_gb()
        gpu_allocated_gb = None
        gpu_reserved_gb = None
        if self.device is not None:
            try:
                gpu_allocated_gb = round(torch.cuda.memory_allocated(self.device) / (1024.0**3), 4)
                gpu_reserved_gb = round(torch.cuda.memory_reserved(self.device) / (1024.0**3), 4)
            except Exception:
                gpu_allocated_gb = None
                gpu_reserved_gb = None

        with self._lock:
            if cpu_rss_gb is not None:
                if self.cpu_memory_peak_gb is None or cpu_rss_gb > self.cpu_memory_peak_gb:
                    self.cpu_memory_peak_gb = cpu_rss_gb
            if gpu_allocated_gb is not None:
                if self.gpu_memory_peak_gb is None or gpu_allocated_gb > self.gpu_memory_peak_gb:
                    self.gpu_memory_peak_gb = gpu_allocated_gb
            if gpu_reserved_gb is not None:
                if self.gpu_memory_reserved_peak_gb is None or gpu_reserved_gb > self.gpu_memory_reserved_peak_gb:
                    self.gpu_memory_reserved_peak_gb = gpu_reserved_gb

    def _poll(self) -> None:
        while not self._stop_event.wait(self.poll_interval_s):
            self._sample_once()

    def start(self) -> "PeakMemoryMonitor":
        if self._started:
            return self
        self._started = True
        self._sample_once()
        self._thread = threading.Thread(target=self._poll, name="peak-memory-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        if self._stopped and self._result is not None:
            return dict(self._result)
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(1.0, self.poll_interval_s * 4.0))
        self._sample_once()
        with self._lock:
            self._result = {
                "cpu_memory_peak_gb": self.cpu_memory_peak_gb,
                "gpu_memory_peak_gb": self.gpu_memory_peak_gb,
                "gpu_memory_reserved_peak_gb": self.gpu_memory_reserved_peak_gb,
                "memory_unit": "GB",
            }
        self._stopped = True
        return dict(self._result)
