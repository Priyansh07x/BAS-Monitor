"""
Workstream B — Gate B17.1 Test Suite
Runtime Performance & System Resource Profiling Suite

Verifies:
1. Benchmark discovery and execution of 'runtime_resource_profiling'.
2. Latency percentiles calculation (Mean, P50, P95, P99, Min, Max, Std) and monotonicity invariants.
3. Stage A–F latency breakdowns (Rectification, Detection, Consistency, Temporal, FSM, Adapter).
4. AI / pipeline throughput FPS calculation with explicit warmup frame exclusion.
5. Process RAM RSS footprint (initial, peak, final, growth) and CPU utilization via psutil.
6. Memory growth boundedness (<= 50 MB growth under sustained 100-frame run).
7. TelemetryDiagnosticAggregator P50/P99 latency metric population and snapshot exposure.
8. Strict preservation of the frozen 8-field public AI contract (docs/architecture.md Section 2).
9. Benchmark export to JSON and text summaries.
10. Standalone CLI benchmark execution.
"""

import os
import sys
import tempfile
import numpy as np
import pytest

from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.telemetry_aggregator import TelemetryDiagnosticAggregator, PipelineTelemetrySection
from backend.ai.result_adapter import AIResultAdapter, PUBLIC_CONTRACT_KEYS
from evaluation.benchmark_harness import BenchmarkHarness, BenchmarkResult, main as benchmark_main


def test_runtime_resource_profiling_benchmark_execution():
    """Verify B17.1 runtime resource profiling executes cleanly and returns passed BenchmarkResult."""
    result = BenchmarkHarness.run_benchmark("runtime_resource_profiling", num_frames=30, warmup_frames=3)
    assert isinstance(result, BenchmarkResult)
    assert result.benchmark_name == "runtime_resource_profiling"
    assert result.category == "RUNTIME_RESOURCE_PROFILING"
    assert result.passed is True
    assert result.duration_seconds > 0.0
    assert result.iterations == 30


def test_latency_percentiles_calculation_and_monotonicity():
    """Verify latency percentiles (Mean, P50, P95, P99, Min, Max, Std) and mathematical invariants."""
    result = BenchmarkHarness.run_benchmark("runtime_resource_profiling", num_frames=40, warmup_frames=5)
    lat = result.metrics["latencies_ms"]

    for key in ["mean", "p50", "p95", "p99", "min", "max", "std"]:
        assert key in lat, f"Missing latency metric: {key}"
        assert isinstance(lat[key], (int, float))
        assert lat[key] >= 0.0

    # Monotonicity & bounding invariants
    assert lat["min"] <= lat["p50"], f"Min ({lat['min']}) > P50 ({lat['p50']})"
    assert lat["p50"] <= lat["p95"], f"P50 ({lat['p50']}) > P95 ({lat['p95']})"
    assert lat["p95"] <= lat["p99"], f"P95 ({lat['p95']}) > P99 ({lat['p99']})"
    assert lat["p99"] <= lat["max"], f"P99 ({lat['p99']}) > Max ({lat['max']})"
    assert lat["min"] <= lat["mean"] <= lat["max"], f"Mean ({lat['mean']}) outside [min, max]"


def test_stage_a_to_f_latency_breakdowns():
    """Verify Stage A–F latency breakdowns are reported with complete percentile distributions."""
    result = BenchmarkHarness.run_benchmark("runtime_resource_profiling", num_frames=30, warmup_frames=2)
    stage_lat = result.metrics["stage_latencies_ms"]

    expected_stages = [
        "stage_a_rectification",
        "stage_b_detection",
        "stage_c_consistency",
        "stage_d_temporal",
        "stage_e_fsm",
        "stage_f_adapter",
        "total_pipeline",
    ]

    for stage in expected_stages:
        assert stage in stage_lat, f"Missing stage breakdown: {stage}"
        s_stats = stage_lat[stage]
        for p_key in ["mean", "p50", "p95", "p99"]:
            assert p_key in s_stats, f"Missing {p_key} in {stage}"
            assert isinstance(s_stats[p_key], (int, float))
            assert s_stats[p_key] >= 0.0

    # Stage sum validation: sum of stage means should be approximately total pipeline mean
    sum_stage_means = (
        stage_lat["stage_a_rectification"]["mean"]
        + stage_lat["stage_b_detection"]["mean"]
        + stage_lat["stage_c_consistency"]["mean"]
        + stage_lat["stage_d_temporal"]["mean"]
        + stage_lat["stage_e_fsm"]["mean"]
        + stage_lat["stage_f_adapter"]["mean"]
    )
    tot_mean = stage_lat["total_pipeline"]["mean"]
    assert abs(sum_stage_means - tot_mean) < 5.0, f"Stage sum {sum_stage_means} deviates from total {tot_mean}"


def test_fps_and_throughput_computation():
    """Verify throughput FPS measurement and explicit warmup isolation."""
    result = BenchmarkHarness.run_benchmark("runtime_resource_profiling", num_frames=50, warmup_frames=5)
    fps_metrics = result.metrics["fps"]

    assert fps_metrics["frames_processed"] == 50
    assert fps_metrics["warmup_frames_excluded"] == 5
    assert fps_metrics["measurement_duration_seconds"] > 0.0
    assert fps_metrics["throughput_fps"] > 0.0

    # Verify calculated throughput equals frames / measurement_duration within rounding tolerance
    expected_fps = round(50 / fps_metrics["measurement_duration_seconds"], 2)
    assert abs(fps_metrics["throughput_fps"] - expected_fps) <= 2.0


def test_process_ram_rss_and_cpu_profiling():
    """Verify OS process RAM RSS (initial, peak, final, growth) and CPU % tracking."""
    result = BenchmarkHarness.run_benchmark("runtime_resource_profiling", num_frames=50, warmup_frames=5)
    res = result.metrics["resources"]

    assert "ram_initial_rss_mb" in res
    assert "ram_peak_rss_mb" in res
    assert "ram_final_rss_mb" in res
    assert "ram_growth_mb" in res
    assert "cpu_utilization_percent" in res
    assert "has_psutil" in res

    assert res["ram_initial_rss_mb"] > 0.0
    assert res["ram_peak_rss_mb"] >= res["ram_initial_rss_mb"]
    assert res["ram_final_rss_mb"] > 0.0
    assert isinstance(res["ram_growth_mb"], (int, float))
    assert res["ram_growth_mb"] <= 50.0  # Safe memory footprint threshold
    assert res["cpu_utilization_percent"] >= 0.0


def test_telemetry_diagnostic_aggregator_percentile_snapshot():
    """Verify TelemetryDiagnosticAggregator accurately tracks P50 and P99 latencies in live snapshots."""
    agg = TelemetryDiagnosticAggregator(window_size=100)

    # Ingest synthetic latency values
    for lat in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
        agg.record_worker_timings(
            process_end_mono=1000.0 + lat,
            infer_dur_ms=lat,
            e2e_ms=lat * 1.5,
        )

    snap = agg.get_snapshot()
    pipe = snap.pipeline

    assert pipe.mean_inference_ms > 0.0
    assert pipe.p50_inference_ms == 55.0  # median of [10..100]
    assert pipe.p95_inference_ms >= 90.0
    assert pipe.p99_inference_ms >= 95.0
    assert pipe.p50_end_to_end_ms == 55.0 * 1.5
    assert pipe.p99_end_to_end_ms >= 95.0 * 1.5


def test_frozen_8_field_public_ai_contract_invariance():
    """Verify that internal Stage A-F telemetry instrumentations do NOT leak into public adapter output."""
    pipeline = InferencePipeline(telemetry_aggregator=TelemetryDiagnosticAggregator())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    public_result = pipeline.process_frame_public(frame)

    # Must be standard dict
    assert isinstance(public_result, dict)
    assert set(public_result.keys()) == set(PUBLIC_CONTRACT_KEYS)

    # Verify no private stage duration keys leaked
    for key in public_result.keys():
        assert not key.startswith("_stage_")
        assert not key.startswith("stage_")
        assert not key.endswith("_ms")


def test_benchmark_export_json_and_text():
    """Verify benchmark harness export serializes runtime profiling metrics to JSON and text."""
    result = BenchmarkHarness.run_benchmark("runtime_resource_profiling", num_frames=20, warmup_frames=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = BenchmarkHarness.export_results(result, tmpdir, base_name="b17_1_test_export")
        json_path = paths["json_path"]
        txt_path = paths["txt_path"]

        assert os.path.exists(json_path)
        assert os.path.exists(txt_path)

        with open(json_path, "r", encoding="utf-8") as f:
            import json
            data = json.load(f)
            assert data["benchmark_name"] == "runtime_resource_profiling"
            assert "latencies_ms" in data["metrics"]
            assert "stage_latencies_ms" in data["metrics"]
            assert "resources" in data["metrics"]

        with open(txt_path, "r", encoding="utf-8") as f:
            txt_content = f.read()
            assert "BENCHMARK REPORT: RUNTIME_RESOURCE_PROFILING" in txt_content
            assert "latencies_ms" in txt_content
            assert "stage_latencies_ms" in txt_content


def test_benchmark_harness_cli_runtime_profiling():
    """Verify CLI main entrypoint executes runtime_resource_profiling benchmark cleanly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exit_code = benchmark_main(["--benchmark", "runtime_resource_profiling", "--output-dir", tmpdir, "--quiet"])
        assert exit_code == 0
