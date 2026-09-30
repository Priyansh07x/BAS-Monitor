"""
test_b8_2_rate_matching_benchmark.py — Test Suite for Gate B8.2 Rate-Matching Strategy Benchmark
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B8.2)

Validates:
1. Strategy configuration validation (enum and string names).
2. Opportunistic strategy behavior (unthrottled consumption when idle).
3. Fixed 10 FPS configuration (100 ms target interval).
4. Fixed 15 FPS configuration (~66.67 ms target interval).
5. Time-decimation configuration (custom monotonic interval).
6. Latest-frame semantics strictly preserved across all strategies.
7. No unbounded input queue (buffer remains strictly single-slot).
8. Producer remains non-blocking across all strategies under heavy load.
9. Deterministic synthetic workload simulation.
10. FPS calculation accuracy.
11. Frame drop/replacement counter calculation.
12. Frame-age calculation at pickup.
13. Mean and p95 percentile latency metrics.
14. Invalid strategy handling (rejects unknown strategy with ValueError).
15. Benchmark result schema and output generation.
16. Synthetic-vs-real labeling (explicit BLOCKED status for real-model metrics).
"""

import json
import time
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pytest

from backend.ai.inference_worker import InferenceWorker, LatestFrameBuffer, RateStrategy
from evaluation.rate_matching_benchmark import (
    SyntheticInferencePipeline,
    run_single_benchmark_run,
    run_rate_matching_benchmark,
    BenchmarkRunResult,
)


@pytest.fixture
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


# =============================================================================
# 1. Strategy Configuration Validation
# =============================================================================

def test_strategy_configuration_validation():
    """Verifies valid strategy modes can be set via Enum or string names."""
    worker = InferenceWorker()
    assert worker.rate_strategy == RateStrategy.OPPORTUNISTIC_LATEST
    assert worker.min_ai_interval_sec == 0.0
    assert worker.target_ai_fps is None

    # FIXED_10FPS
    worker.set_rate_strategy(RateStrategy.FIXED_10FPS)
    assert worker.rate_strategy == RateStrategy.FIXED_10FPS
    assert abs(worker.min_ai_interval_sec - 0.100) < 1e-5
    assert worker.target_ai_fps == 10.0

    # FIXED_15FPS (string)
    worker.set_rate_strategy("FIXED_15FPS")
    assert worker.rate_strategy == RateStrategy.FIXED_15FPS
    assert abs(worker.min_ai_interval_sec - (1.0 / 15.0)) < 1e-5
    assert worker.target_ai_fps == 15.0

    # TIME_DECIMATED (custom interval)
    worker.set_rate_strategy("TIME_DECIMATED", min_ai_interval_sec=0.050)
    assert worker.rate_strategy == RateStrategy.TIME_DECIMATED
    assert abs(worker.min_ai_interval_sec - 0.050) < 1e-5
    assert worker.target_ai_fps == 20.0

    # OPPORTUNISTIC_LATEST (string lowercase / uppercase normalization)
    worker.set_rate_strategy("opportunistic_latest")
    assert worker.rate_strategy == RateStrategy.OPPORTUNISTIC_LATEST
    assert worker.min_ai_interval_sec == 0.0
    assert worker.target_ai_fps is None


# =============================================================================
# 2. Invalid Strategy Handling
# =============================================================================

def test_invalid_strategy_handling():
    """Verifies that invalid strategy names or types raise ValueError."""
    worker = InferenceWorker()

    with pytest.raises(ValueError, match="Invalid rate strategy"):
        worker.set_rate_strategy("INVALID_UNKNOWN_STRATEGY")

    with pytest.raises(ValueError, match="Invalid rate strategy type"):
        worker.set_rate_strategy(12345)  # type: ignore

    with pytest.raises(ValueError, match="min_ai_interval_sec must be a positive float"):
        worker.set_rate_strategy(RateStrategy.TIME_DECIMATED, min_ai_interval_sec=-0.5)

    with pytest.raises(ValueError, match="min_ai_interval_sec must be a positive float"):
        worker.set_rate_strategy(RateStrategy.TIME_DECIMATED, min_ai_interval_sec=0.0)


# =============================================================================
# 3. Opportunistic Strategy Behavior
# =============================================================================

def test_opportunistic_strategy_behavior(dummy_frame):
    """Verifies opportunistic strategy processes frames immediately without artificial throttling."""
    pipe = SyntheticInferencePipeline(simulated_duration_sec=0.005)
    worker = InferenceWorker(pipeline=pipe, rate_strategy=RateStrategy.OPPORTUNISTIC_LATEST)
    worker.start()

    # Submit 3 fast frames sequentially with slight delay
    for _ in range(3):
        worker.submit_frame(dummy_frame)
        time.sleep(0.01)

    time.sleep(0.03)
    worker.stop()

    # Opportunistic should process all 3 without interval-induced pacing drops
    assert worker.frames_processed == 3
    assert worker.frames_replaced == 0
    assert worker.rate_strategy == RateStrategy.OPPORTUNISTIC_LATEST
    worker.shutdown()


# =============================================================================
# 4. Fixed 10 FPS Configuration & Pacing
# =============================================================================

def test_fixed_10fps_configuration_and_pacing(dummy_frame):
    """Verifies FIXED_10FPS paces processing to ~100 ms intervals."""
    pipe = SyntheticInferencePipeline(simulated_duration_sec=0.005)
    worker = InferenceWorker(pipeline=pipe, rate_strategy=RateStrategy.FIXED_10FPS)
    worker.start()

    assert worker.target_ai_fps == 10.0
    assert abs(worker.min_ai_interval_sec - 0.100) < 1e-4

    # Rapidly submit 4 frames within 50 ms
    for _ in range(4):
        worker.submit_frame(dummy_frame)
        time.sleep(0.01)

    time.sleep(0.03)
    worker.stop()

    # First frame processed; remaining 3 submitted during 100 ms window should result in replacements
    assert worker.frames_submitted == 4
    assert worker.frames_processed >= 1
    assert worker.frames_replaced >= 1
    worker.shutdown()


# =============================================================================
# 5. Fixed 15 FPS Configuration
# =============================================================================

def test_fixed_15fps_configuration(dummy_frame):
    """Verifies FIXED_15FPS configures ~66.67 ms interval and 15 FPS target."""
    worker = InferenceWorker(rate_strategy=RateStrategy.FIXED_15FPS)
    assert worker.target_ai_fps == 15.0
    assert abs(worker.min_ai_interval_sec - (1.0 / 15.0)) < 1e-4


# =============================================================================
# 6. Time-Decimation Configuration
# =============================================================================

def test_time_decimation_configuration():
    """Verifies TIME_DECIMATED strategy accepts custom monotonic interval."""
    worker = InferenceWorker(
        rate_strategy=RateStrategy.TIME_DECIMATED,
        min_ai_interval_sec=0.200,
    )
    assert worker.rate_strategy == RateStrategy.TIME_DECIMATED
    assert worker.min_ai_interval_sec == 0.200
    assert worker.target_ai_fps == 5.0


# =============================================================================
# 7. Latest-Frame Semantics Strictly Preserved
# =============================================================================

def test_latest_frame_semantics_preserved_under_pacing(dummy_frame):
    """Verifies that while worker waits for pacing interval, newest frame replaces older pending frame."""
    processed_indices = []

    class TrackingPipeline:
        def process_frame_public(self, frame, annotate=True, timestamp=None, metadata=None, **kwargs):
            if metadata and "index" in metadata:
                processed_indices.append(metadata["index"])
            return {"action": "IDLE", "status": "VALID"}
        def reset(self): pass
        def release(self): pass

    worker = InferenceWorker(
        pipeline=TrackingPipeline(),
        rate_strategy=RateStrategy.FIXED_10FPS,  # 100 ms interval
    )
    worker.start()

    # Submit frame 1
    worker.submit_frame(dummy_frame, metadata={"index": 1})
    time.sleep(0.02)  # Worker processes frame 1

    # During the 100 ms interval cooldown, submit frame 2, then frame 3, then frame 4
    worker.submit_frame(dummy_frame, metadata={"index": 2})
    worker.submit_frame(dummy_frame, metadata={"index": 3})
    worker.submit_frame(dummy_frame, metadata={"index": 4})

    # Wait for interval to elapse and worker to process next opportunity
    time.sleep(0.12)
    worker.stop()

    # Frame 1 was processed first. Next processed frame MUST be frame 4 (freshest), skipping 2 and 3!
    assert processed_indices[0] == 1
    assert 4 in processed_indices
    assert 2 not in processed_indices
    assert 3 not in processed_indices
    assert worker.frames_replaced == 2
    worker.shutdown()


# =============================================================================
# 8. No Unbounded Queue & Buffer Remains Single-Slot
# =============================================================================

def test_no_unbounded_queue_buffer_remains_single_slot(dummy_frame):
    """Verifies buffer stores exactly 0 or 1 item at all times regardless of strategy."""
    worker = InferenceWorker(rate_strategy=RateStrategy.FIXED_10FPS)
    for i in range(20):
        worker._buffer.put(dummy_frame, metadata={"id": i})

    assert worker._buffer.submission_count == 20
    assert worker._buffer.replacement_count == 19
    assert worker._buffer.has_pending()

    # Exactly 1 pop drains the entire buffer
    data = worker._buffer.get(timeout=0.05)
    assert data is not None
    _, _, meta, _ = data
    assert meta["id"] == 19
    assert not worker._buffer.has_pending()


# =============================================================================
# 9. Producer Remains Non-Blocking
# =============================================================================

def test_producer_remains_non_blocking_under_all_strategies(dummy_frame):
    """Verifies that frame submissions complete in < 1 ms across all strategy modes."""
    strategies = [
        RateStrategy.OPPORTUNISTIC_LATEST,
        RateStrategy.FIXED_10FPS,
        RateStrategy.FIXED_15FPS,
        RateStrategy.TIME_DECIMATED,
    ]

    for strat in strategies:
        pipe = SyntheticInferencePipeline(simulated_duration_sec=0.05)  # 50 ms slow pipeline
        worker = InferenceWorker(pipeline=pipe, rate_strategy=strat)
        worker.start()

        # Measure 10 rapid submissions
        t0 = time.monotonic()
        for _ in range(10):
            worker.submit_frame(dummy_frame)
        t_elapsed = time.monotonic() - t0

        # 10 submissions must take << 10 ms (non-blocking)
        assert t_elapsed < 0.02, f"Submission blocked on strategy {strat}: {t_elapsed}s"
        worker.shutdown()


# =============================================================================
# 10. Deterministic Synthetic Workload
# =============================================================================

def test_deterministic_synthetic_workload_simulation(dummy_frame):
    """Verifies SyntheticInferencePipeline executes with accurate controlled latency."""
    durations = [0.010, 0.033, 0.066]
    for dur in durations:
        pipe = SyntheticInferencePipeline(simulated_duration_sec=dur)
        t0 = time.monotonic()
        res = pipe.process_frame_public(dummy_frame)
        t_elapsed = time.monotonic() - t0

        assert abs(t_elapsed - dur) < 0.015  # Tolerance for scheduler jitter
        assert res["action"] == "PICK_RED"
        assert res["status"] == "VALID"


# =============================================================================
# 11. Drop & Replacement Rate Calculation
# =============================================================================

def test_drop_and_replacement_calculation(dummy_frame):
    """Verifies drop rate calculation formula."""
    buffer = LatestFrameBuffer()
    assert buffer.submission_count == 0
    assert buffer.replacement_count == 0

    # 10 submissions without reading -> 9 replacements
    for _ in range(10):
        buffer.put(dummy_frame)

    assert buffer.submission_count == 10
    assert buffer.replacement_count == 9

    drop_rate = buffer.replacement_count / buffer.submission_count
    assert abs(drop_rate - 0.9) < 1e-6


# =============================================================================
# 12. Frame Age Calculation
# =============================================================================

def test_frame_age_calculation(dummy_frame):
    """Verifies worker records frame age at pickup in telemetry."""
    pipe = SyntheticInferencePipeline(simulated_duration_sec=0.005)
    worker = InferenceWorker(pipeline=pipe)
    worker.start()

    t_cap = time.monotonic() - 0.030  # 30 ms ago
    worker.submit_frame(dummy_frame, metadata={"capture_monotonic": t_cap})
    time.sleep(0.04)

    telem = worker.get_telemetry()
    assert telem["mean_frame_age_ms"] >= 25.0
    worker.shutdown()


# =============================================================================
# 13. Mean and p95 Latency Percentiles Calculation
# =============================================================================

def test_mean_and_p95_percentiles_calculation():
    """Verifies numpy-based mean and p95 calculations."""
    samples = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    mean_val = InferenceWorker._compute_mean(samples)
    p95_val = InferenceWorker._compute_p95(samples)

    assert mean_val == 55.0
    assert abs(p95_val - float(np.percentile(samples, 95))) < 1e-6


# =============================================================================
# 14. Single Benchmark Run Result Model
# =============================================================================

def test_single_benchmark_run_execution():
    """Verifies run_single_benchmark_run returns valid BenchmarkRunResult dataclass."""
    res = run_single_benchmark_run(
        producer_fps=30.0,
        strategy=RateStrategy.FIXED_10FPS,
        synthetic_workload_ms=10.0,
        duration_sec=0.15,
        seed=42,
    )

    assert isinstance(res, BenchmarkRunResult)
    assert res.producer_fps == 30.0
    assert res.strategy == "FIXED_10FPS"
    assert res.target_ai_fps == 10.0
    assert res.synthetic_workload_ms == 10.0
    assert res.frames_submitted >= 3
    assert res.frames_processed >= 1
    assert res.workload_type == "SYNTHETIC_SIMULATION"


# =============================================================================
# 15. Benchmark Result Schema and Structure
# =============================================================================

def test_benchmark_result_schema_and_epistemic_notice(tmp_path):
    """Verifies that the benchmark runner outputs conforming JSON with required metadata and labels."""
    output_file = tmp_path / "test_benchmark_results.json"
    payload = run_rate_matching_benchmark(
        producer_rates=[30.0],
        strategies=["OPPORTUNISTIC_LATEST", "FIXED_10FPS"],
        synthetic_workloads_ms=[10.0, 33.0],
        duration_per_run_sec=0.1,
        seed=42,
        output_path=output_file,
        quiet=True,
    )

    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    # Validate Schema
    assert loaded["benchmark_metadata"]["gate"] == "B8.2"
    assert loaded["benchmark_metadata"]["workstream"] == "Workstream B (AI & Procedure Intelligence)"
    assert loaded["epistemic_notice"]["workload_type"] == "SYNTHETIC_SIMULATION"
    assert loaded["epistemic_notice"]["real_model_status"] == "BLOCKED"
    assert "recognition_accuracy" in loaded["epistemic_notice"]["blocked_metrics"]
    assert "macro_f1_score" in loaded["epistemic_notice"]["blocked_metrics"]
    assert loaded["epistemic_notice"]["production_strategy_status"] == "OPPORTUNISTIC_LATEST (UNCHANGED)"
    assert len(loaded["runs"]) == 4  # 1 producer rate * 2 strategies * 2 workloads


# =============================================================================
# 16. Synthetic-vs-Real Labeling Invariance
# =============================================================================

def test_synthetic_vs_real_labeling_invariance():
    """Verifies all runs are explicitly tagged as SYNTHETIC_SIMULATION."""
    res = run_single_benchmark_run(
        producer_fps=15.0,
        strategy=RateStrategy.OPPORTUNISTIC_LATEST,
        synthetic_workload_ms=33.0,
        duration_sec=0.1,
        seed=42,
    )
    assert res.workload_type == "SYNTHETIC_SIMULATION"
