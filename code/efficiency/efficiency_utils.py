import json
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def cuda_sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_cuda_peak_stats() -> None:
    if not torch.cuda.is_available():
        return
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)


def get_cuda_peak_gb() -> Dict[str, float]:
    """Returns summed peak allocated/reserved GB across all visible CUDA devices."""
    if not torch.cuda.is_available():
        return {"peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0}

    peak_allocated = 0
    peak_reserved = 0
    for device_index in range(torch.cuda.device_count()):
        peak_allocated += int(torch.cuda.max_memory_allocated(device_index))
        peak_reserved += int(torch.cuda.max_memory_reserved(device_index))

    gb = 1024 ** 3
    return {
        "peak_allocated_gb": peak_allocated / gb,
        "peak_reserved_gb": peak_reserved / gb,
    }


def try_get_process_rss_gb() -> Optional[float]:
    """Best-effort CPU RSS; used when CUDA is unavailable."""
    try:
        import psutil  # type: ignore

        gb = 1024 ** 3
        return psutil.Process(os.getpid()).memory_info().rss / gb
    except Exception:
        return None


@dataclass
class SampleEfficiency:
    sample_idx: int
    time_s: float
    peak_allocated_gb: float
    peak_reserved_gb: float
    human_time_s: Optional[float] = None
    llm_time_s: Optional[float] = None
    # For text-level accounting (one text == one sample)
    source: Optional[str] = None  # "human" | "llm" | None
    group_idx: Optional[int] = None  # original dataset index when applicable


def _safe_quantiles(values: List[float], qs: Tuple[float, ...]) -> Dict[str, float]:
    if not values:
        return {f"p{int(q * 100)}": 0.0 for q in qs}
    values_sorted = sorted(values)
    out: Dict[str, float] = {}
    for q in qs:
        k = (len(values_sorted) - 1) * q
        f = int(k)
        c = min(f + 1, len(values_sorted) - 1)
        if f == c:
            v = values_sorted[f]
        else:
            v = values_sorted[f] + (values_sorted[c] - values_sorted[f]) * (k - f)
        out[f"p{int(q * 100)}"] = float(v)
    return out


def summarize_efficiency(rows: List[SampleEfficiency]) -> Dict[str, Any]:
    times = [r.time_s for r in rows]
    alloc = [r.peak_allocated_gb for r in rows]
    reserv = [r.peak_reserved_gb for r in rows]

    def _summary(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0, **_safe_quantiles(vals, (0.5, 0.9, 0.95, 0.99))}
        return {
            "mean": float(statistics.mean(vals)),
            "stdev": float(statistics.pstdev(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
            **_safe_quantiles(vals, (0.5, 0.9, 0.95, 0.99)),
        }

    return {
        "time_s_per_sample": _summary(times),
        "peak_allocated_gb_per_sample": _summary(alloc),
        "peak_reserved_gb_per_sample": _summary(reserv),
    }


def measure_one_sample(
    sample_idx: int,
    human_fn: Callable[[], Any],
    llm_fn: Callable[[], Any],
) -> SampleEfficiency:
    """Measure one dataset entry (human text + llm text) as a single 'sample'."""
    rss_before = try_get_process_rss_gb() if not torch.cuda.is_available() else None

    reset_cuda_peak_stats()
    cuda_sync_if_needed()

    t0 = time.perf_counter()
    h0 = time.perf_counter()
    human_fn()
    cuda_sync_if_needed()
    h1 = time.perf_counter()

    l0 = time.perf_counter()
    llm_fn()
    cuda_sync_if_needed()
    l1 = time.perf_counter()

    t1 = time.perf_counter()

    peaks = get_cuda_peak_gb()

    # CPU-only fallback: approximate 'space' as RSS delta
    if not torch.cuda.is_available():
        rss_after = try_get_process_rss_gb()
        if rss_before is not None and rss_after is not None:
            delta = max(0.0, rss_after - rss_before)
            peaks = {"peak_allocated_gb": delta, "peak_reserved_gb": delta}

    return SampleEfficiency(
        sample_idx=sample_idx,
        time_s=float(t1 - t0),
        peak_allocated_gb=float(peaks["peak_allocated_gb"]),
        peak_reserved_gb=float(peaks["peak_reserved_gb"]),
        human_time_s=float(h1 - h0),
        llm_time_s=float(l1 - l0),
    )


def measure_one_text(
    sample_idx: int,
    text_fn: Callable[[], Any],
    *,
    source: Optional[str] = None,
    group_idx: Optional[int] = None,
) -> SampleEfficiency:
    """Measure one text as a single 'sample'.

    Args:
        sample_idx: Running index in the flattened stream.
        text_fn: Callable that runs the method on a single text (usually a batch of size 1).
        source: Optional tag (e.g., "human" or "llm").
        group_idx: Optional original dataset index.
    """
    rss_before = try_get_process_rss_gb() if not torch.cuda.is_available() else None

    reset_cuda_peak_stats()
    cuda_sync_if_needed()

    t0 = time.perf_counter()
    text_fn()
    cuda_sync_if_needed()
    t1 = time.perf_counter()

    peaks = get_cuda_peak_gb()

    if not torch.cuda.is_available():
        rss_after = try_get_process_rss_gb()
        if rss_before is not None and rss_after is not None:
            delta = max(0.0, rss_after - rss_before)
            peaks = {"peak_allocated_gb": delta, "peak_reserved_gb": delta}

    return SampleEfficiency(
        sample_idx=sample_idx,
        time_s=float(t1 - t0),
        peak_allocated_gb=float(peaks["peak_allocated_gb"]),
        peak_reserved_gb=float(peaks["peak_reserved_gb"]),
        source=source,
        group_idx=group_idx,
    )


def save_efficiency_json(
    out_path: str,
    payload: Dict[str, Any],
) -> None:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
