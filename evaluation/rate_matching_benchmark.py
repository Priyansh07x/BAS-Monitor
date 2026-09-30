"""
evaluation/rate_matching_benchmark.py
Gate B8.2 — Rate-Matching Strategy Benchmark
ISRO SIH26174 BAS Experiment Monitor — Workstream B

PURPOSE
=======
Executes a reproducible, deterministic benchmark comparing frame-rate consumption
strategies (OPPORTUNISTIC_LATEST, FIXED_10FPS, FIXED_15FPS, TIME_DECIMATED)
using the non-blocking latest-frame architecture (B7) and monotonic telemetry (B8.1).

EPISTEMIC SEPARATION & LIMITATIONS
===================================
1. SYNTHETIC SCHEDULING BENCHMARK ONLY:
   All neural model inference durations in this benchmark (10 ms, 33 ms, 66 ms, 100 ms, 150 ms)
   are CONTROLLED SYNTHETIC WORKLOADS to measure rate-control, scheduling, frame replacement,
   frame age, and concurrency dynamics.
   They do NOT represent real trained model weights or physical NPU execution.

2. BLOCKED REAL-MODEL METRICS (Explicitly labeled):
   - Minimum AI FPS required for EXP-001 action recognition
   - Action miss rate / temporal boundary precision
   - Macro-F1 / precision / recall degradation at reduced FPS
   - Real neural network inference latency on physical edge hardware (Raspberry Pi 5 / Hailo-8L)
   These metrics require the future EXP-001 dataset and trained models.

3. ARCHITECTURAL NEUTRALITY:
   This benchmark provides empirical measurements and objective trade-off data.
   It does NOT make the final architectural decision (deferred to Gate B8.3).
   Production strategy remains OPPORTUNISTIC_LATEST (UNCHANGED).
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
from dataclasses import asdict, dataclass
import datetime
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Try importing psutil for CPU utilization metrics
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# Path setup
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.ai.inference_worker import InferenceWorker, LatestFrameBuffer, RateStrategy
from backend.ai.result_adapter import AIResultAdapter


logger = logging.getLogger("BAS_RateMatchingBenchmark")


# ---------------------------------------------------------------------------
# Constants & Configuration Matrix
# ---------------------------------------------------------------------------
BENCHMARK_SEED: int = 42
DEFAULT_PRODUCER_RATES: List[float] = [10.0, 15.0, 30.0]
DEFAULT_STRATEGIES: List[str] = [
    RateStrategy.OPPORTUNISTIC_LATEST.value,
    RateStrategy.FIXED_10FPS.value,
    RateStrategy.FIXED_15FPS.value,
    RateStrategy.TIME_DECIMATED.value,
]
DEFAULT_SYNTHETIC_WORKLOADS_MS: List[float] = [10.0, 33.0, 66.0, 100.0, 150.0]


# ---------------------------------------------------------------------------
# Synthetic Inference Pipeline (Controlled Workloads)
# ---------------------------------------------------------------------------
class SyntheticInferencePipeline:
    """
    Deterministic simulated perception pipeline for scheduler/rate-control benchmarking.
    Simulates a controlled inference workload duration without executing real neural models.
    """

    def __init__(self, simulated_duration_sec: float = 0.033):
        self.simulated_duration_sec = simulated_duration_sec
        self.call_count = 0

    def process_frame_public(
        self,
        frame: np.ndarray,
        annotate: bool = True,
        timestamp: Any = None,
        metadata: Any = None,
        expected_step: Any = None,
        fsm_status: Any = None,
        next_step: Any = None,
        rectifier: Any = None,
    ) -> Dict[str, Any]:
        """Simulates deterministic inference duration and returns valid public contract."""
        self.call_count += 1
        if self.simulated_duration_sec > 0:
            time.sleep(self.simulated_duration_sec)

        return AIResultAdapter.adapt(
            action="PICK_RED",
            confidence=0.95,
            timestamp=timestamp or "2026-09-24T12:00:00",
            expected_step=expected_step or "S1",
            detected_step="S1",
            fsm_status=fsm_status or "VALID",
            next_step=next_step or "S2",
        )

    def reset(self) -> None:
        self.call_count = 0

    def release(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Benchmark Data Models
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkRunResult:
    """Detailed telemetry result from a single benchmark configuration run."""
    run_id: str
    producer_fps: float
    strategy: str
    target_ai_fps: Optional[float]
    min_ai_interval_sec: float
    synthetic_workload_ms: float
    duration_sec: float
    frames_submitted: int
    frames_processed: int
    frames_replaced: int
    effective_producer_fps: float
    effective_ai_fps: float
    frame_drop_rate: float
    mean_frame_age_ms: float
    p95_frame_age_ms: float
    mean_inference_ms: float
    p95_inference_ms: float
    mean_dispatch_ms: float
    p95_dispatch_ms: float
    mean_end_to_end_ms: float
    p95_end_to_end_ms: float
    worker_utilization_pct: float
    strategy_overhead_ms: float
    cpu_utilization_pct: Optional[float]
    workload_type: str = "SYNTHETIC_SIMULATION"


# ---------------------------------------------------------------------------
# Benchmark Execution Engine
# ---------------------------------------------------------------------------
def run_single_benchmark_run(
    producer_fps: float,
    strategy: Union[RateStrategy, str],
    synthetic_workload_ms: float,
    duration_sec: float = 0.3,
    min_ai_interval_sec: Optional[float] = None,
    seed: int = BENCHMARK_SEED,
) -> BenchmarkRunResult:
    """
    Executes a single deterministic benchmark run for a specific combination of
    producer rate, rate-matching strategy, and synthetic workload duration.
    """
    np.random.seed(seed)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    simulated_sec = synthetic_workload_ms / 1000.0

    strat_enum = RateStrategy(strategy) if isinstance(strategy, str) else strategy
    pipeline = SyntheticInferencePipeline(simulated_duration_sec=simulated_sec)

    worker = InferenceWorker(
        pipeline=pipeline,
        rate_strategy=strat_enum,
        min_ai_interval_sec=min_ai_interval_sec,
    )
    worker.start()

    # Determine frame count and submission pacing
    frame_interval = 1.0 / producer_fps
    num_frames = max(3, int(round(producer_fps * duration_sec)))

    process = psutil.Process(os.getpid()) if _HAS_PSUTIL else None
    cpu_start = process.cpu_percent(interval=None) if process else None
    proc_time_start = time.process_time()

    t_run_start = time.monotonic()

    # Deterministic frame submission pacing loop
    for i in range(num_frames):
        t_target = t_run_start + (i * frame_interval)
        t_now = time.monotonic()
        sleep_dur = t_target - t_now
        if sleep_dur > 0.0005:
            time.sleep(sleep_dur)

        cap_time = time.monotonic()
        worker.submit_frame(
            dummy_frame,
            metadata={"capture_monotonic": cap_time, "frame_index": i},
        )

    # Allow worker to finish processing any in-flight or pending frame
    # Timeout is proportional to synthetic workload duration
    drain_timeout = max(0.05, simulated_sec * 1.5 + (worker.min_ai_interval_sec or 0.0))
    time.sleep(drain_timeout)

    t_run_end = time.monotonic()
    actual_duration = max(0.001, t_run_end - t_run_start)

    proc_time_end = time.process_time()
    cpu_end = process.cpu_percent(interval=None) if process else None

    # Retrieve worker telemetry
    telemetry = worker.get_telemetry()
    worker.shutdown()

    submitted = telemetry["frames_submitted"]
    processed = telemetry["frames_processed"]
    replaced = telemetry["frames_replaced"]

    eff_prod_fps = round(submitted / actual_duration, 2)
    eff_ai_fps = round(processed / actual_duration, 2)
    drop_rate = round(replaced / submitted, 4) if submitted > 0 else 0.0

    mean_inf = telemetry["mean_inference_ms"]
    mean_age = telemetry["mean_frame_age_ms"]
    mean_disp = telemetry["mean_dispatch_latency_ms"]

    # Worker utilization: active inference time / total duration
    total_active_inf_sec = (processed * mean_inf) / 1000.0
    worker_util_pct = min(100.0, round((total_active_inf_sec / actual_duration) * 100.0, 2))

    # Strategy scheduling overhead = total end-to-end latency minus pure inference duration
    overhead_ms = max(0.0, round(telemetry["mean_end_to_end_ms"] - mean_inf, 3))

    cpu_pct: Optional[float] = None
    if _HAS_PSUTIL and cpu_end is not None and cpu_end > 0:
        cpu_pct = round(cpu_end, 1)
    else:
        # Fallback process-time CPU approximation
        cpu_proc_sec = proc_time_end - proc_time_start
        cpu_pct = round(min(100.0, (cpu_proc_sec / actual_duration) * 100.0), 1)

    target_ai = worker.target_ai_fps
    run_id = f"P{int(producer_fps)}_{strat_enum.value}_W{int(synthetic_workload_ms)}ms"

    return BenchmarkRunResult(
        run_id=run_id,
        producer_fps=producer_fps,
        strategy=strat_enum.value,
        target_ai_fps=target_ai,
        min_ai_interval_sec=worker.min_ai_interval_sec,
        synthetic_workload_ms=synthetic_workload_ms,
        duration_sec=round(actual_duration, 3),
        frames_submitted=submitted,
        frames_processed=processed,
        frames_replaced=replaced,
        effective_producer_fps=eff_prod_fps,
        effective_ai_fps=eff_ai_fps,
        frame_drop_rate=drop_rate,
        mean_frame_age_ms=telemetry["mean_frame_age_ms"],
        p95_frame_age_ms=telemetry["p95_frame_age_ms"],
        mean_inference_ms=mean_inf,
        p95_inference_ms=telemetry["p95_inference_ms"],
        mean_dispatch_ms=mean_disp,
        p95_dispatch_ms=telemetry["p95_dispatch_latency_ms"],
        mean_end_to_end_ms=telemetry["mean_end_to_end_ms"],
        p95_end_to_end_ms=telemetry["p95_end_to_end_ms"],
        worker_utilization_pct=worker_util_pct,
        strategy_overhead_ms=overhead_ms,
        cpu_utilization_pct=cpu_pct,
        workload_type="SYNTHETIC_SIMULATION",
    )


def run_rate_matching_benchmark(
    producer_rates: Optional[List[float]] = None,
    strategies: Optional[List[str]] = None,
    synthetic_workloads_ms: Optional[List[float]] = None,
    duration_per_run_sec: float = 0.25,
    seed: int = BENCHMARK_SEED,
    output_path: Optional[Union[str, Path]] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """
    Executes the full Workstream B Gate B8.2 Rate-Matching Strategy Benchmark across all
    matrix combinations and generates structured machine-readable results.
    """
    p_rates = producer_rates or DEFAULT_PRODUCER_RATES
    strats = strategies or DEFAULT_STRATEGIES
    workloads = synthetic_workloads_ms or DEFAULT_SYNTHETIC_WORKLOADS_MS

    if not quiet:
        print("=" * 80)
        print("Workstream B Gate B8.2 — Rate-Matching Strategy Benchmark")
        print("=" * 80)
        print(f"Producer Rates (FPS) : {p_rates}")
        print(f"Strategies Tested    : {strats}")
        print(f"Synthetic Workloads  : {workloads} ms")
        print(f"Duration per run     : {duration_per_run_sec} s")
        print(f"Total Configurations : {len(p_rates) * len(strats) * len(workloads)}")
        print("=" * 80)

    results_list: List[BenchmarkRunResult] = []
    total_runs = len(p_rates) * len(strats) * len(workloads)
    current_run = 0

    t_bench_start = time.monotonic()

    for p_rate in p_rates:
        for strat in strats:
            for w_ms in workloads:
                current_run += 1
                if not quiet:
                    print(
                        f"[{current_run:02d}/{total_runs:02d}] "
                        f"Producer: {p_rate:4.1f} FPS | Strategy: {strat:20s} | Workload: {w_ms:5.1f} ms ...",
                        end="",
                        flush=True,
                    )

                res = run_single_benchmark_run(
                    producer_fps=p_rate,
                    strategy=strat,
                    synthetic_workload_ms=w_ms,
                    duration_sec=duration_per_run_sec,
                    seed=seed,
                )
                results_list.append(res)

                if not quiet:
                    print(
                        f" -> AI FPS: {res.effective_ai_fps:4.1f} | Drop: {res.frame_drop_rate*100:5.1f}% | "
                        f"Age: {res.mean_frame_age_ms:5.1f} ms"
                    )

    t_bench_end = time.monotonic()
    total_benchmark_duration_sec = round(t_bench_end - t_bench_start, 2)

    # -------------------------------------------------------------------------
    # Comparative Summary Analysis Tables
    # -------------------------------------------------------------------------
    # 1. Standard 30 FPS Camera Ingestion Comparison (Workload = 33 ms & 66 ms & 100 ms)
    summary_30fps_camera: List[Dict[str, Any]] = []
    for r in results_list:
        if r.producer_fps == 30.0 and r.synthetic_workload_ms in [33.0, 66.0, 100.0]:
            summary_30fps_camera.append({
                "strategy": r.strategy,
                "synthetic_workload_ms": r.synthetic_workload_ms,
                "target_ai_fps": r.target_ai_fps,
                "effective_ai_fps": r.effective_ai_fps,
                "frames_submitted": r.frames_submitted,
                "frames_processed": r.frames_processed,
                "frames_replaced": r.frames_replaced,
                "drop_rate_pct": round(r.frame_drop_rate * 100, 1),
                "mean_frame_age_ms": r.mean_frame_age_ms,
                "p95_frame_age_ms": r.p95_frame_age_ms,
                "worker_utilization_pct": r.worker_utilization_pct,
                "strategy_overhead_ms": r.strategy_overhead_ms,
            })

    # 2. Fast vs Slow Workload Analysis (10 ms vs 150 ms at 30 FPS)
    workload_extremes: List[Dict[str, Any]] = []
    for r in results_list:
        if r.producer_fps == 30.0 and r.synthetic_workload_ms in [10.0, 150.0]:
            workload_extremes.append({
                "strategy": r.strategy,
                "synthetic_workload_ms": r.synthetic_workload_ms,
                "effective_ai_fps": r.effective_ai_fps,
                "drop_rate_pct": round(r.frame_drop_rate * 100, 1),
                "mean_frame_age_ms": r.mean_frame_age_ms,
                "worker_utilization_pct": r.worker_utilization_pct,
            })

    # Compile structured payload
    payload: Dict[str, Any] = {
        "benchmark_metadata": {
            "gate": "B8.2",
            "gate_title": "Rate-Matching Strategy Benchmark",
            "workstream": "Workstream B (AI & Procedure Intelligence)",
            "timestamp": datetime.datetime.now().isoformat(),
            "benchmark_seed": seed,
            "total_configurations_tested": len(results_list),
            "total_benchmark_duration_sec": total_benchmark_duration_sec,
            "duration_per_run_sec": duration_per_run_sec,
            "environment": {
                "os": platform.system(),
                "os_release": platform.release(),
                "python_version": platform.python_version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "has_psutil": _HAS_PSUTIL,
            },
        },
        "epistemic_notice": {
            "workload_type": "SYNTHETIC_SIMULATION",
            "statement": (
                "All inference durations in this benchmark (10 ms, 33 ms, 66 ms, 100 ms, 150 ms) "
                "are controlled deterministic synthetic workloads designed to benchmark rate-control, "
                "pacing, drop rates, and queue/buffer behavior. They do NOT represent real model accuracy "
                "or physical NPU execution."
            ),
            "real_model_status": "BLOCKED",
            "blocked_metrics": {
                "recognition_accuracy": "BLOCKED — EXP-001 dataset not yet collected",
                "action_miss_rate": "BLOCKED — Requires temporal ground truth labels",
                "macro_f1_score": "BLOCKED — Model weights pending Part-1 Colab export",
                "edge_npu_latency": "BLOCKED — Physical Hailo-8L / RPi5 hardware execution pending",
            },
            "production_strategy_status": "OPPORTUNISTIC_LATEST (UNCHANGED)",
            "architectural_decision": "DEFERRED to Gate B8.3",
        },
        "benchmark_parameters": {
            "producer_rates_fps": p_rates,
            "strategies_tested": strats,
            "synthetic_workloads_ms": workloads,
            "buffer_architecture": "LatestFrameBuffer (single-slot replacement)",
        },
        "summary_analysis": {
            "camera_30fps_ingestion_summary": summary_30fps_camera,
            "workload_extremes_summary": workload_extremes,
        },
        "runs": [asdict(r) for r in results_list],
    }

    # Write output file
    if output_path is None:
        output_path = _REPO_ROOT / "evaluation" / "results" / "b8_2_rate_matching_benchmark.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if not quiet:
        print("=" * 80)
        print(f"Benchmark completed successfully in {total_benchmark_duration_sec}s.")
        print(f"Structured results written to: {output_path}")
        print("=" * 80)

    return payload


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Workstream B Gate B8.2 — Rate-Matching Strategy Benchmark"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON result file (default: evaluation/results/b8_2_rate_matching_benchmark.json)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.25,
        help="Duration in seconds for each configuration run (default: 0.25s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BENCHMARK_SEED,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console progress output",
    )
    args = parser.parse_args()

    run_rate_matching_benchmark(
        duration_per_run_sec=args.duration,
        seed=args.seed,
        output_path=args.output,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
