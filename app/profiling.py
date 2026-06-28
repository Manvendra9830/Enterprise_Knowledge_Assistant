"""Lightweight latency profiling helpers for request hot paths."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


@contextmanager
def timed_step(timings: dict[str, float], name: str) -> Iterator[None]:
    """Record elapsed milliseconds for a named step."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = (time.perf_counter() - start) * 1000


def log_latency_breakdown(operation: str, timings: dict[str, float]) -> None:
    """Log a compact latency breakdown table as a single structured line."""
    total_ms = sum(timings.values())
    parts = " | ".join(f"{name}={elapsed:.1f}ms" for name, elapsed in timings.items())
    logger.info("latency_breakdown operation=%s total=%.1fms | %s", operation, total_ms, parts)
