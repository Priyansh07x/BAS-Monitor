"""
tests/test_architecture_benchmark.py
Gate B5.2 — Deterministic Architecture Benchmark Harness Tests
ISRO SIH26174 BAS Experiment Monitor — Workstream B

Validates:
1. Benchmark harness imports cleanly without heavy ML dependencies.
2. Checkpoint inspection correctly detects absent model weights without exceptions.
3. Dataset inspection correctly detects absent EXP-001 procedure dataset.
4. Candidate B 111-dimensional feature vector is extracted and shaped correctly.
5. Candidate B 30-frame temporal window stacks into exact (1, 30, 111) tensor.
6. Candidate A 5D video tensor is prepared with exact (1, 3, 16, 112, 112) shape.
7. Candidate A 5D high-res tensor is prepared with exact (1, 3, 32, 224, 224) shape.
8. Benchmark execution is reproducible with fixed seed.
9. JSON benchmark report matches expected schema and includes all required fields.
10. Blocked metrics (accuracy, F1, model inference latency) are explicitly null / BLOCKED.
11. No fake or fabricated metric values are asserted.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from evaluation.architecture_benchmark import (
    BENCHMARK_SEED,
    TCN_FEATURE_DIM,
    TCN_WINDOW_SIZE,
    R2PLUS1D_SHAPES,
    DEFAULT_OUTPUT_PATH,
    inspect_model_artifacts,
    benchmark_candidate_b_pipeline,
    benchmark_candidate_a_pipeline,
    run_architecture_benchmark,
    _prepare_r2plus1d_tensor_numpy,
    _generate_synthetic_frames,
    _generate_sample_landmarks,
)


def test_artifact_inspection_handles_absent_weights():
    """Verify that absent model weights and runtimes are gracefully detected as BLOCKED."""
    status_a, status_b, dataset_info = inspect_model_artifacts()

    # Candidate A
    assert status_a.model_name.startswith("R(2+1)D-18")
    assert status_a.checkpoint_present is False
    assert status_a.status == "BLOCKED"
    assert len(status_a.blocker_reason) > 0

    # Candidate B
    assert status_b.model_name.startswith("1D-TCN")
    assert status_b.checkpoint_present is False
    assert status_b.status == "BLOCKED"
    assert len(status_b.blocker_reason) > 0

    # Dataset
    assert dataset_info["exp001_procedure_dataset"] == "NOT_COLLECTED"
    assert "BLOCKED" in dataset_info["exp001_status"]


def test_candidate_b_feature_vector_dimension():
    """Verify Candidate B spatial feature vector extraction produces exact 111 dimensions."""
    pose_landmarks, object_boxes = _generate_sample_landmarks()
    from backend.video.frame_processor import FrameProcessor
    vec = FrameProcessor.extract_keypoint_vector(pose_landmarks, object_boxes)

    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.shape == (TCN_FEATURE_DIM,), f"Expected shape ({TCN_FEATURE_DIM},), got {vec.shape}"
    assert vec.nbytes == 111 * 4  # 444 bytes


def test_candidate_b_buffer_and_tensor_shape():
    """Verify Candidate B temporal buffer stacks into exact (1, 30, 111) tensor."""
    meas_b = benchmark_candidate_b_pipeline(iterations=5, seed=BENCHMARK_SEED)

    assert meas_b.feature_dimension == 111
    assert meas_b.temporal_window_size == 30
    assert meas_b.tensor_shape == [1, 30, 111]
    assert meas_b.single_vector_bytes == 444
    assert meas_b.buffer_window_bytes == 444 * 30
    assert meas_b.tensor_memory_bytes == 444 * 30

    # Verify timing stats exist and are non-negative
    fe = meas_b.feature_extraction_timing
    assert fe["iterations"] == 5
    assert fe["mean_ms"] >= 0.0
    assert fe["throughput_fps"] > 0.0


def test_candidate_a_tensor_shapes():
    """Verify Candidate A pure NumPy tensor preparation produces exact 5D video shapes."""
    frames = _generate_synthetic_frames(count=32, height=120, width=160, seed=BENCHMARK_SEED)

    # 1. Standard shape: (1, 3, 16, 112, 112)
    t16 = _prepare_r2plus1d_tensor_numpy(frames, num_frames=16, spatial_size=112)
    assert t16.shape == (1, 3, 16, 112, 112)
    assert t16.dtype == np.float32
    assert 0.0 <= float(t16.min()) and float(t16.max()) <= 1.0

    # 2. High-res shape: (1, 3, 32, 224, 224)
    t32 = _prepare_r2plus1d_tensor_numpy(frames, num_frames=32, spatial_size=224)
    assert t32.shape == (1, 3, 32, 224, 224)
    assert t32.dtype == np.float32
    assert 0.0 <= float(t32.min()) and float(t32.max()) <= 1.0


def test_candidate_a_pipeline_measurements():
    """Verify Candidate A pipeline benchmark executes and records timing statistics."""
    meas_a = benchmark_candidate_a_pipeline(iterations=3, seed=BENCHMARK_SEED)

    assert meas_a.standard_shape == [1, 3, 16, 112, 112]
    assert meas_a.highres_shape == [1, 3, 32, 224, 224]
    assert meas_a.tensor_16x112_bytes == 1 * 3 * 16 * 112 * 112 * 4
    assert meas_a.tensor_32x224_bytes == 1 * 3 * 32 * 224 * 224 * 4

    t16 = meas_a.tensor_prep_16x112_timing
    assert t16["iterations"] == 3
    assert t16["mean_ms"] > 0.0


def test_benchmark_reproducibility(tmp_path):
    """Verify running the benchmark with the same seed produces consistent results."""
    out1 = tmp_path / "bench1.json"
    out2 = tmp_path / "bench2.json"

    rep1 = run_architecture_benchmark(iterations=5, output_path=out1, seed=42, verbose=False)
    rep2 = run_architecture_benchmark(iterations=5, output_path=out2, seed=42, verbose=False)

    assert rep1.seed == rep2.seed
    assert rep1.iterations == rep2.iterations
    assert rep1.candidate_b_measurements["tensor_shape"] == rep2.candidate_b_measurements["tensor_shape"]
    assert rep1.candidate_a_measurements["standard_shape"] == rep2.candidate_a_measurements["standard_shape"]
    assert rep1.status == rep2.status == "PARTIALLY_PASSED"


def test_benchmark_result_json_schema(tmp_path):
    """Verify the generated JSON benchmark output conforms to expected structure."""
    out = tmp_path / "schema_test.json"
    report = run_architecture_benchmark(iterations=3, output_path=out, seed=42, verbose=False)

    assert out.exists()
    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["gate"] == "B5.2"
    assert data["status"] == "PARTIALLY_PASSED"
    assert "system_info" in data
    assert "candidate_a_status" in data
    assert "candidate_b_status" in data
    assert "dataset_status" in data
    assert "candidate_b_measurements" in data
    assert "candidate_a_measurements" in data
    assert "theoretical_estimates" in data
    assert "blocked_metrics" in data


def test_blocked_metrics_are_explicitly_null_and_explained():
    """Verify all unavailable model-level metrics are marked BLOCKED with metric_value=null."""
    report = run_architecture_benchmark(iterations=2, output_path=None, seed=42, verbose=False)
    blocked = report.blocked_metrics

    # Model inference latency
    assert blocked["candidate_a_inference_latency"]["status"] == "BLOCKED"
    assert blocked["candidate_a_inference_latency"]["metric_value"] is None
    assert len(blocked["candidate_a_inference_latency"]["reason"]) > 0

    assert blocked["candidate_b_inference_latency"]["status"] == "BLOCKED"
    assert blocked["candidate_b_inference_latency"]["metric_value"] is None
    assert len(blocked["candidate_b_inference_latency"]["reason"]) > 0

    # Accuracy / F1
    assert blocked["candidate_a_accuracy_f1"]["status"] == "BLOCKED"
    assert blocked["candidate_a_accuracy_f1"]["metric_value"] is None

    assert blocked["candidate_b_accuracy_f1"]["status"] == "BLOCKED"
    assert blocked["candidate_b_accuracy_f1"]["metric_value"] is None

    # Orientation drop delta
    assert blocked["orientation_robustness_drop_delta"]["status"] == "BLOCKED"
    assert blocked["orientation_robustness_drop_delta"]["metric_value"] is None
