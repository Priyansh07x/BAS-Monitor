"""
tests/test_available_components_benchmark.py
Gate B5.3 — Benchmark Available Components Tests
ISRO SIH26174 BAS Experiment Monitor — Workstream B

Validates:
1. Dynamic environment detection works and captures system capabilities without crashing.
2. Spatial feature vector extraction produces exact 111-dim vector.
3. Temporal sliding window buffering produces exact (1, 30, 111) tensor.
4. Frame preprocessing outputs correct detector (640x640) and pose (256x256) dimensions.
5. Rectification preprocessing measures baseline and 90-degree overhead.
6. Multi-angle orientation preprocessing covers all 7 canonical angles (0° - 270°).
7. Candidate A video tensor preparation produces exact 5D shapes (1, 3, 16, 112, 112) and (1, 3, 32, 224, 224).
8. JSON benchmark report matches expected schema and contains all required sections.
9. Blocked model metrics (accuracy, F1, inference latency) are explicitly null / BLOCKED.
10. No fake or hardcoded timing numbers are asserted.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from evaluation.available_components_benchmark import (
    BENCHMARK_SEED,
    CANONICAL_ANGLES,
    TCN_FEATURE_DIM,
    TCN_WINDOW_SIZE,
    detect_runtime_environment,
    benchmark_feature_extraction,
    benchmark_temporal_buffering,
    benchmark_frame_preprocessing,
    benchmark_rectification_preprocessing,
    benchmark_orientation_preprocessing,
    benchmark_candidate_tensors,
    run_available_components_benchmark,
)


def test_runtime_environment_detection():
    """Verify runtime environment detector captures platform details without throwing exceptions."""
    env = detect_runtime_environment()

    assert "platform" in env and len(env["platform"]) > 0
    assert "python_version" in env and len(env["python_version"]) > 0
    assert "cpu_count" in env and env["cpu_count"] >= 1
    assert "numpy_version" in env
    assert isinstance(env["opencv_installed"], bool)
    assert isinstance(env["pytorch_installed"], bool)
    assert isinstance(env["tflite_installed"], bool)


def test_spatial_feature_extraction_benchmark():
    """Verify feature extraction benchmark outputs 111-dim vector and valid stats."""
    res = benchmark_feature_extraction(iterations=3, warmup=1)

    assert res["vector_dimension"] == TCN_FEATURE_DIM
    assert res["vector_memory_bytes"] == TCN_FEATURE_DIM * 4
    assert res["vector_dtype"] == "float32"

    timing = res["timing"]
    assert timing["iterations"] == 3
    assert timing["mean_ms"] >= 0.0
    assert timing["throughput"] > 0.0


def test_temporal_buffering_benchmark():
    """Verify temporal buffering benchmark outputs (1, 30, 111) tensor and valid stats."""
    res = benchmark_temporal_buffering(iterations=3, warmup=1)

    assert res["window_size"] == TCN_WINDOW_SIZE
    assert res["stacked_tensor_shape"] == [1, TCN_WINDOW_SIZE, TCN_FEATURE_DIM]
    assert res["buffer_memory_bytes"] == TCN_WINDOW_SIZE * TCN_FEATURE_DIM * 4

    timing = res["timing"]
    assert timing["iterations"] == 3
    assert timing["mean_ms"] >= 0.0


def test_frame_preprocessing_benchmark():
    """Verify detector (640x640) and pose (256x256) preprocessing dimensions."""
    res = benchmark_frame_preprocessing(iterations=2, warmup=1)

    assert res["detector_input_shape"] == [640, 640, 3]
    assert res["pose_input_shape"] == [256, 256, 3]
    assert res["detector_preprocessing_640x640"]["iterations"] == 2
    assert res["pose_preprocessing_256x256"]["iterations"] == 2


def test_rectification_preprocessing_benchmark():
    """Verify camera rectification benchmark outputs baseline, rectified, and overhead timings."""
    res = benchmark_rectification_preprocessing(iterations=2, warmup=1)

    assert "baseline_pipeline" in res
    assert "rectified_pipeline_90deg" in res
    assert "overhead" in res
    assert res["overhead"]["mean_ms"] is not None


def test_multi_angle_orientation_preprocessing():
    """Verify all 7 canonical angles are evaluated in orientation preprocessing."""
    res = benchmark_orientation_preprocessing(iterations=2, warmup=1)

    for angle in CANONICAL_ANGLES:
        key = f"{angle}_deg"
        assert key in res, f"Canonical angle {angle}° missing from benchmark"
        assert res[key]["angle_degrees"] == angle
        assert res[key]["timing"]["mean_ms"] >= 0.0
        assert "preprocessing" in res[key]["note"].lower()


def test_candidate_tensor_preparation():
    """Verify Candidate A (16x112 and 32x224) and Candidate B (30x111) tensor prep."""
    res = benchmark_candidate_tensors(iterations=2, warmup=1)

    a16 = res["candidate_a_standard_16x112"]
    a32 = res["candidate_a_highres_32x224"]
    b30 = res["candidate_b_window_30x111"]

    assert a16["tensor_shape"] == [1, 3, 16, 112, 112]
    assert a16["memory_bytes"] == 1 * 3 * 16 * 112 * 112 * 4

    assert a32["tensor_shape"] == [1, 3, 32, 224, 224]
    assert a32["memory_bytes"] == 1 * 3 * 32 * 224 * 224 * 4

    assert b30["tensor_shape"] == [1, 30, 111]
    assert b30["memory_bytes"] == 1 * 30 * 111 * 4


def test_benchmark_json_schema(tmp_path):
    """Verify the generated JSON benchmark output conforms to expected structure."""
    out = tmp_path / "b5_3_schema_test.json"
    report = run_available_components_benchmark(
        iterations=2, output_path=out, seed=BENCHMARK_SEED, verbose=False
    )

    assert out.exists()
    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["gate"] == "B5.3"
    assert data["status"] == "PARTIALLY_PASSED"
    assert "environment" in data
    assert "feature_extraction" in data
    assert "temporal_buffering" in data
    assert "frame_preprocessing" in data
    assert "rectification_preprocessing" in data
    assert "orientation_preprocessing" in data
    assert "tensor_preparation" in data
    assert "blocked_model_metrics" in data


def test_blocked_model_metrics_are_explicitly_null():
    """Verify that model-level metrics are explicitly reported as BLOCKED with null values."""
    report = run_available_components_benchmark(
        iterations=1, output_path=None, seed=BENCHMARK_SEED, verbose=False
    )
    blocked = report["blocked_model_metrics"]

    assert blocked["candidate_a_inference_latency"]["status"] == "BLOCKED"
    assert blocked["candidate_a_inference_latency"]["metric_value"] is None

    assert blocked["candidate_b_inference_latency"]["status"] == "BLOCKED"
    assert blocked["candidate_b_inference_latency"]["metric_value"] is None

    assert blocked["model_classification_accuracy"]["status"] == "BLOCKED"
    assert blocked["model_classification_accuracy"]["metric_value"] is None

    assert blocked["model_precision_recall_f1"]["status"] == "BLOCKED"
    assert blocked["model_precision_recall_f1"]["metric_value"] is None

    assert blocked["orientation_f1_degradation"]["status"] == "BLOCKED"
    assert blocked["orientation_f1_degradation"]["metric_value"] is None
