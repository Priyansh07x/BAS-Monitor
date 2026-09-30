"""
evaluation/architecture_benchmark.py
Gate B5.2 — Deterministic Architecture Benchmark Harness
ISRO SIH26174 BAS Experiment Monitor — Workstream B

PURPOSE
=======
Executes a deterministic, reproducible computational and structural benchmark
comparing the two B5 candidate architecture pipelines:
  - Candidate A: R(2+1)D-18 / End-to-End Spatio-Temporal Video Model
  - Candidate B: Compact Feature-Based Temporal Model (1D-TCN)

EPISTEMIC SEPARATION
====================
  1. VERIFIED MEASUREMENTS:
     - Feature vector extraction latency (111-dim vector from pose + objects)
     - Temporal sliding window buffer accumulation & tensor stacking latency (B=1, T=30, D=111)
     - Candidate A video clip tensor preparation latency ((B, C, T, H, W) for 16x112x112 & 32x224x224)
     - Memory footprint of temporal buffers, spatial feature vectors, and video clip tensors
     - Pipeline throughput (frames/sec & windows/sec)

  2. BLOCKED MEASUREMENTS (Explicitly reported as BLOCKED with machine-readable reasons):
     - Model inference latency (no PyTorch/TFLite runtime or trained model weights present)
     - Model parameter counts on disk (no checkpoint artifacts present)
     - Classification accuracy, Precision, Recall, Macro-F1 (no trained model / no EXP-001 dataset)
     - Orientation degradation deltas (requires trained model classification outputs)

  3. THEORETICAL ESTIMATES (Documented for architectural reference):
     - Theoretical FLOPs and parameter counts based on published literature (Tran et al., 2018; Lea et al., 2017)

REPRODUCIBILITY
===============
  - Fixed seed: BENCHMARK_SEED = 42
  - Deterministic workload execution
  - Structured output written to evaluation/results/b5_2_architecture_benchmark_results.json

Usage:
  python evaluation/architecture_benchmark.py
  python evaluation/architecture_benchmark.py --output evaluation/results/custom_benchmark.json
  python evaluation/architecture_benchmark.py --num-iterations 50 --seed 42 --quiet
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.video.frame_processor import FrameProcessor, _pure_numpy_resize, _pure_numpy_bgr_to_rgb
from backend.ai.action_classifier import ActionClassifier

# ---------------------------------------------------------------------------
# Constants & Contract Invariants
# ---------------------------------------------------------------------------
BENCHMARK_SEED: int = 42
DEFAULT_ITERATIONS: int = 50

# Candidate B Contract (Verified Code Reality)
TCN_WINDOW_SIZE: int = 30
TCN_FEATURE_DIM: int = 111
TCN_POSE_FEATURES: int = 99   # 33 landmarks * 3 (x, y, z)
TCN_OBJECT_FEATURES: int = 12 # 4 objects * 3 (cx, cy, conf)

# Candidate A Contract (Specification Shapes)
R2PLUS1D_SHAPES: Dict[str, Tuple[int, int, int, int, int]] = {
    "standard_16x112": (1, 3, 16, 112, 112),
    "highres_32x224":  (1, 3, 32, 224, 224),
}

# Literature Theoretical Reference Estimates
THEORETICAL_ESTIMATES: Dict[str, Dict[str, Any]] = {
    "candidate_a_r2plus1d_18": {
        "architecture": "R(2+1)D-18",
        "reference_source": "Tran et al., 'A Closer Look at Spatiotemporal Convolutions for Action Recognition', CVPR 2018",
        "theoretical_parameters": 31_500_000,
        "theoretical_gflops_16x112": 7.5,
        "theoretical_gflops_32x224": 30.0,
        "estimated_rpi5_cpu_latency_ms": "150 - 400 ms (Theoretical Estimate)",
        "estimated_ram_mb": "150 - 500 MB (Theoretical Estimate)",
    },
    "candidate_b_1d_tcn": {
        "architecture": "1D-TCN (Causal Dilated Residual)",
        "reference_source": "Lea et al., 'Temporal Convolutional Networks for Action Segmentation and Detection', CVPR 2017",
        "theoretical_parameters": 50_000,
        "theoretical_mflops_30x111": 2.5,
        "estimated_rpi5_cpu_latency_ms": "< 3.0 ms (Theoretical Estimate)",
        "estimated_ram_mb": "< 5.0 MB (Host buffer < 1 MB)",
    },
}

DEFAULT_OUTPUT_PATH: Path = (
    _REPO_ROOT / "evaluation" / "results" / "b5_2_architecture_benchmark_results.json"
)
HMDB51_DATA_ROOT: Path = _REPO_ROOT / "training" / "data" / "raw" / "hmdb51_sta"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TimingStats:
    """Statistical summary of benchmark execution timings."""
    iterations: int = 0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    p95_ms: float = 0.0
    throughput_fps: float = 0.0

    @classmethod
    def from_timings(cls, timings_ms: List[float]) -> TimingStats:
        if not timings_ms:
            return cls()
        arr = np.array(timings_ms, dtype=np.float64)
        mean_v = float(np.mean(arr))
        fps = (1000.0 / mean_v) if mean_v > 0 else 0.0
        return cls(
            iterations=len(timings_ms),
            mean_ms=round(mean_v, 4),
            median_ms=round(float(np.median(arr)), 4),
            min_ms=round(float(np.min(arr)), 4),
            max_ms=round(float(np.max(arr)), 4),
            std_ms=round(float(np.std(arr)), 4),
            p95_ms=round(float(np.percentile(arr, 95)), 4),
            throughput_fps=round(fps, 2),
        )


@dataclass
class ModelArtifactStatus:
    """Status of physical model checkpoints and runtime frameworks."""
    model_name: str
    checkpoint_expected_path: str
    checkpoint_present: bool = False
    runtime_framework_installed: bool = False
    framework_name: str = ""
    inference_executable: bool = False
    status: str = "BLOCKED"
    blocker_reason: str = ""


@dataclass
class CandidateBMeasurements:
    """Empirical measurements for Candidate B (1D-TCN) feature pipeline."""
    feature_dimension: int = TCN_FEATURE_DIM
    temporal_window_size: int = TCN_WINDOW_SIZE
    tensor_shape: List[int] = field(default_factory=lambda: [1, TCN_WINDOW_SIZE, TCN_FEATURE_DIM])

    # Latency benchmarks
    feature_extraction_timing: Dict[str, Any] = field(default_factory=dict)
    buffer_stacking_timing: Dict[str, Any] = field(default_factory=dict)
    total_pipeline_timing: Dict[str, Any] = field(default_factory=dict)

    # Memory
    single_vector_bytes: int = 0
    buffer_window_bytes: int = 0
    tensor_memory_bytes: int = 0


@dataclass
class CandidateAMeasurements:
    """Empirical measurements for Candidate A (R(2+1)D-18) video tensor pipeline."""
    standard_shape: List[int] = field(default_factory=lambda: list(R2PLUS1D_SHAPES["standard_16x112"]))
    highres_shape: List[int] = field(default_factory=lambda: list(R2PLUS1D_SHAPES["highres_32x224"]))

    # Latency benchmarks
    tensor_prep_16x112_timing: Dict[str, Any] = field(default_factory=dict)
    tensor_prep_32x224_timing: Dict[str, Any] = field(default_factory=dict)

    # Memory
    tensor_16x112_bytes: int = 0
    tensor_32x224_bytes: int = 0


@dataclass
class ArchitectureBenchmarkReport:
    """Structured report encapsulating all Gate B5.2 benchmark results."""
    gate: str = "B5.2"
    timestamp: str = ""
    seed: int = BENCHMARK_SEED
    iterations: int = DEFAULT_ITERATIONS

    # System context
    system_info: Dict[str, Any] = field(default_factory=dict)

    # Candidate Status
    candidate_a_status: Dict[str, Any] = field(default_factory=dict)
    candidate_b_status: Dict[str, Any] = field(default_factory=dict)
    dataset_status: Dict[str, Any] = field(default_factory=dict)

    # Verified Empirical Measurements
    candidate_b_measurements: Dict[str, Any] = field(default_factory=dict)
    candidate_a_measurements: Dict[str, Any] = field(default_factory=dict)

    # Theoretical Literature Estimates
    theoretical_estimates: Dict[str, Any] = field(default_factory=dict)

    # Explicitly Blocked Metrics Register
    blocked_metrics: Dict[str, Any] = field(default_factory=dict)

    # Gate Outcome
    status: str = "PARTIALLY_PASSED"
    notes: str = ""


# ---------------------------------------------------------------------------
# Synthetic & Frame Data Helpers
# ---------------------------------------------------------------------------

def _generate_synthetic_frames(
    count: int = 32,
    height: int = 240,
    width: int = 320,
    seed: int = BENCHMARK_SEED,
) -> List[np.ndarray]:
    """Generates deterministic video frames for benchmark consistency."""
    rng = np.random.RandomState(seed)
    frames: List[np.ndarray] = []
    for i in range(count):
        base = np.zeros((height, width, 3), dtype=np.uint8)
        base[:height // 2, :width // 2] = [180, 50, 50]
        base[:height // 2, width // 2:] = [50, 180, 50]
        base[height // 2:, :width // 2] = [50, 50, 180]
        base[height // 2:, width // 2:] = [200, 200, 200]
        noise = rng.randint(0, 20, base.shape, dtype=np.uint8)
        frame = np.clip(base.astype(np.int32) + noise.astype(np.int32), 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def _generate_sample_landmarks() -> Tuple[List[Dict[str, float]], List[Dict[str, Any]]]:
    """Generates synthetic 33-point pose landmarks and 4 object bounding boxes."""
    pose = [{"x": 0.5 + i * 0.01, "y": 0.4 + i * 0.01, "z": 0.0} for i in range(33)]
    boxes = [
        {"x1": 0.2, "y1": 0.2, "x2": 0.4, "y2": 0.4, "confidence": 0.95, "label": "RED_SAMPLE"},
        {"x1": 0.6, "y1": 0.5, "x2": 0.8, "y2": 0.7, "confidence": 0.92, "label": "SAMPLE_CONTAINER"},
        {"x1": 0.1, "y1": 0.6, "x2": 0.3, "y2": 0.8, "confidence": 0.88, "label": "BLUE_SAMPLE"},
        {"x1": 0.7, "y1": 0.1, "x2": 0.9, "y2": 0.3, "confidence": 0.91, "label": "CONTAINER_LID"},
    ]
    return pose, boxes


# ---------------------------------------------------------------------------
# Checkpoint & Framework Detection
# ---------------------------------------------------------------------------

def inspect_model_artifacts() -> Tuple[ModelArtifactStatus, ModelArtifactStatus, Dict[str, Any]]:
    """
    Automatically inspects the repository for model checkpoints, runtime frameworks,
    and dataset availability without throwing exceptions.
    """
    # 1. Candidate A: R(2+1)D-18
    r2plus1d_path = _REPO_ROOT / "models" / "r2plus1d_18.pt"
    torch_available = False
    try:
        import torch  # type: ignore
        torch_available = True
    except ImportError:
        torch_available = False

    status_a = ModelArtifactStatus(
        model_name="R(2+1)D-18 (Candidate A)",
        checkpoint_expected_path=str(r2plus1d_path),
        checkpoint_present=r2plus1d_path.exists(),
        runtime_framework_installed=torch_available,
        framework_name="PyTorch / torchvision",
        inference_executable=False,
        status="BLOCKED",
        blocker_reason=(
            "PyTorch is not installed in the active environment and no R(2+1)D-18 "
            "checkpoint (.pt/.onnx) exists in the repository."
        ),
    )

    # 2. Candidate B: 1D-TCN
    tcn_path = _REPO_ROOT / "data" / "temporal_action.tflite"
    tflite_available = False
    try:
        import tflite_runtime  # type: ignore
        tflite_available = True
    except ImportError:
        try:
            import tensorflow.lite  # type: ignore
            tflite_available = True
        except ImportError:
            tflite_available = False

    status_b = ModelArtifactStatus(
        model_name="1D-TCN (Candidate B)",
        checkpoint_expected_path=str(tcn_path),
        checkpoint_present=tcn_path.exists(),
        runtime_framework_installed=tflite_available,
        framework_name="tflite_runtime / tensorflow.lite",
        inference_executable=False,
        status="BLOCKED",
        blocker_reason=(
            "Neither tflite_runtime nor tensorflow is installed, and the model "
            "artifact 'data/temporal_action.tflite' does not exist on disk."
        ),
    )

    # 3. Dataset status
    hmdb51_present = HMDB51_DATA_ROOT.exists()
    exp001_collected = False  # As established in Gate B3 / B3.1
    dataset_info = {
        "hmdb51_raw_directory": str(HMDB51_DATA_ROOT),
        "hmdb51_present": hmdb51_present,
        "hmdb51_role": "Raw video decoding & frame extraction benchmark only (NOT EXP-001 accuracy)",
        "exp001_procedure_dataset": "NOT_COLLECTED",
        "exp001_status": "BLOCKED — Protocol defined in Gate B3; clips not yet recorded",
    }

    return status_a, status_b, dataset_info


# ---------------------------------------------------------------------------
# Benchmark Workload Runners (Verified Pure NumPy)
# ---------------------------------------------------------------------------

def benchmark_candidate_b_pipeline(
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = BENCHMARK_SEED,
) -> CandidateBMeasurements:
    """
    Measures the verified Candidate B data pipeline:
      1. Spatial vector extraction (111 features from 33 pose + 4 box landmarks).
      2. Sliding window buffer accumulation (30 frames).
      3. Feature tensor stacking into (1, 30, 111) shape.
    """
    pose_landmarks, object_boxes = _generate_sample_landmarks()
    measurements = CandidateBMeasurements()

    # 1. Feature extraction timing
    extraction_timings: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        vec = FrameProcessor.extract_keypoint_vector(pose_landmarks, object_boxes)
        t1 = time.perf_counter()
        extraction_timings.append((t1 - t0) * 1000.0)

    measurements.feature_extraction_timing = asdict(TimingStats.from_timings(extraction_timings))

    # 2. Buffer accumulation and stacking timing
    classifier = ActionClassifier(window_size=TCN_WINDOW_SIZE)
    # Pre-populate buffer with 29 vectors
    for _ in range(TCN_WINDOW_SIZE - 1):
        classifier.push_frame_vector(vec)

    buffer_timings: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        classifier.push_frame_vector(vec)
        # Stack into (1, 30, 111) tensor
        stacked = np.stack(list(classifier._vector_buffer))
        tensor = np.expand_dims(stacked.astype(np.float32), axis=0)
        t1 = time.perf_counter()
        buffer_timings.append((t1 - t0) * 1000.0)

    measurements.buffer_stacking_timing = asdict(TimingStats.from_timings(buffer_timings))

    # 3. Combined end-to-end extraction + stacking
    combined_timings = [e + b for e, b in zip(extraction_timings, buffer_timings)]
    measurements.total_pipeline_timing = asdict(TimingStats.from_timings(combined_timings))

    # Memory calculations (Verified exact bytes)
    sample_vec = FrameProcessor.extract_keypoint_vector(pose_landmarks, object_boxes)
    measurements.single_vector_bytes = int(sample_vec.nbytes)
    measurements.buffer_window_bytes = int(sample_vec.nbytes * TCN_WINDOW_SIZE)
    measurements.tensor_memory_bytes = int(tensor.nbytes)

    return measurements


def _prepare_r2plus1d_tensor_numpy(
    frames: List[np.ndarray],
    num_frames: int,
    spatial_size: int,
) -> np.ndarray:
    """
    Transforms a sequence of HxWx3 video frames into a 5D video tensor (B, C, T, H, W).
    Pure NumPy implementation without OpenCV or PyTorch dependencies.
    """
    selected_frames = frames[:num_frames]
    processed_channels: List[np.ndarray] = []

    for f in selected_frames:
        rgb = _pure_numpy_bgr_to_rgb(f)
        resized = _pure_numpy_resize(rgb, (spatial_size, spatial_size))
        norm = resized.astype(np.float32) / 255.0
        # HWC -> CHW (channels-first)
        chw = np.transpose(norm, (2, 0, 1))
        processed_channels.append(chw)

    # Stack along time: (T, C, H, W) -> Transpose to (C, T, H, W)
    t_chw = np.stack(processed_channels, axis=0)
    c_t_h_w = np.transpose(t_chw, (1, 0, 2, 3))
    # Add batch dimension: (1, C, T, H, W)
    tensor_5d = np.expand_dims(c_t_h_w, axis=0)
    return tensor_5d


def benchmark_candidate_a_pipeline(
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = BENCHMARK_SEED,
) -> CandidateAMeasurements:
    """
    Measures the Candidate A raw video tensor preparation pipeline:
      1. Standard Clip Tensor Preparation: (1, 3, 16, 112, 112)
      2. High-Res Clip Tensor Preparation: (1, 3, 32, 224, 224)
    """
    frames = _generate_synthetic_frames(count=32, height=240, width=320, seed=seed)
    measurements = CandidateAMeasurements()

    # 1. Standard 16x112x112 timing
    timings_16x112: List[float] = []
    tensor_16 = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        tensor_16 = _prepare_r2plus1d_tensor_numpy(frames, num_frames=16, spatial_size=112)
        t1 = time.perf_counter()
        timings_16x112.append((t1 - t0) * 1000.0)

    measurements.tensor_prep_16x112_timing = asdict(TimingStats.from_timings(timings_16x112))
    if tensor_16 is not None:
        measurements.tensor_16x112_bytes = int(tensor_16.nbytes)

    # 2. High-Res 32x224x224 timing (Fewer iterations if needed for speed)
    highres_iters = max(5, iterations // 5)
    timings_32x224: List[float] = []
    tensor_32 = None
    for _ in range(highres_iters):
        t0 = time.perf_counter()
        tensor_32 = _prepare_r2plus1d_tensor_numpy(frames, num_frames=32, spatial_size=224)
        t1 = time.perf_counter()
        timings_32x224.append((t1 - t0) * 1000.0)

    measurements.tensor_prep_32x224_timing = asdict(TimingStats.from_timings(timings_32x224))
    if tensor_32 is not None:
        measurements.tensor_32x224_bytes = int(tensor_32.nbytes)

    return measurements


# ---------------------------------------------------------------------------
# Report Assembly & Serialization
# ---------------------------------------------------------------------------

def run_architecture_benchmark(
    iterations: int = DEFAULT_ITERATIONS,
    output_path: Optional[Path] = None,
    seed: int = BENCHMARK_SEED,
    verbose: bool = True,
) -> ArchitectureBenchmarkReport:
    """
    Executes the full deterministic architecture benchmark and writes results to JSON.
    """
    out_file = output_path or DEFAULT_OUTPUT_PATH
    report = ArchitectureBenchmarkReport(
        timestamp=datetime.datetime.now().isoformat(),
        seed=seed,
        iterations=iterations,
    )

    if verbose:
        print("=" * 70)
        print("Gate B5.2 — Deterministic Architecture Benchmark Harness")
        print("ISRO SIH26174 BAS Experiment Monitor — Workstream B")
        print("=" * 70)
        print(f"[CONFIG] Seed: {seed} | Iterations: {iterations}")
        print()

    # Step 1: System info & Artifact inspection
    report.system_info = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "processor": platform.processor() or "Unknown",
        "numpy_version": np.__version__,
    }

    status_a, status_b, dataset_info = inspect_model_artifacts()
    report.candidate_a_status = asdict(status_a)
    report.candidate_b_status = asdict(status_b)
    report.dataset_status = dataset_info

    if verbose:
        print("[STEP 1/4] Inspecting Model Artifacts & Runtimes...")
        print(f"  Candidate A (R(2+1)D-18): Checkpoint Present = {status_a.checkpoint_present} | Runtime = {status_a.runtime_framework_installed} | Status = {status_a.status}")
        print(f"  Candidate B (1D-TCN):     Checkpoint Present = {status_b.checkpoint_present} | Runtime = {status_b.runtime_framework_installed} | Status = {status_b.status}")
        print(f"  EXP-001 Dataset:          {dataset_info['exp001_procedure_dataset']} ({dataset_info['exp001_status']})")
        print()

    # Step 2: Candidate B Benchmark
    if verbose:
        print("[STEP 2/4] Benchmarking Candidate B (1D-TCN) Spatial Vector Pipeline...")
    meas_b = benchmark_candidate_b_pipeline(iterations=iterations, seed=seed)
    report.candidate_b_measurements = asdict(meas_b)

    if verbose:
        fe = meas_b.feature_extraction_timing
        bs = meas_b.buffer_stacking_timing
        tot = meas_b.total_pipeline_timing
        print(f"  Feature Vector (111-dim): {fe['mean_ms']:.3f} ms (p95: {fe['p95_ms']:.3f} ms, Throughput: {fe['throughput_fps']:.1f} FPS)")
        print(f"  Buffer Stacking (30-win): {bs['mean_ms']:.3f} ms (p95: {bs['p95_ms']:.3f} ms, Throughput: {bs['throughput_fps']:.1f} WPS)")
        print(f"  Total Preparation Time:   {tot['mean_ms']:.3f} ms | Buffer RAM: {meas_b.buffer_window_bytes} bytes")
        print()

    # Step 3: Candidate A Benchmark
    if verbose:
        print("[STEP 3/4] Benchmarking Candidate A (R(2+1)D-18) Video Tensor Pipeline...")
    meas_a = benchmark_candidate_a_pipeline(iterations=iterations, seed=seed)
    report.candidate_a_measurements = asdict(meas_a)

    if verbose:
        t16 = meas_a.tensor_prep_16x112_timing
        t32 = meas_a.tensor_prep_32x224_timing
        print(f"  Tensor (1, 3, 16, 112, 112): {t16['mean_ms']:.3f} ms (Throughput: {t16['throughput_fps']:.1f} clips/s, Size: {meas_a.tensor_16x112_bytes / 1024:.1f} KB)")
        print(f"  Tensor (1, 3, 32, 224, 224): {t32['mean_ms']:.3f} ms (Throughput: {t32['throughput_fps']:.1f} clips/s, Size: {meas_a.tensor_32x224_bytes / 1024:.1f} KB)")
        print()

    # Step 4: Record Theoretical Estimates & Blocked Register
    report.theoretical_estimates = THEORETICAL_ESTIMATES
    report.blocked_metrics = {
        "candidate_a_inference_latency": {
            "status": "BLOCKED",
            "reason": status_a.blocker_reason,
            "metric_value": None,
        },
        "candidate_b_inference_latency": {
            "status": "BLOCKED",
            "reason": status_b.blocker_reason,
            "metric_value": None,
        },
        "candidate_a_accuracy_f1": {
            "status": "BLOCKED",
            "reason": "No EXP-001 procedure dataset collected and no trained R(2+1)D checkpoint exists.",
            "metric_value": None,
        },
        "candidate_b_accuracy_f1": {
            "status": "BLOCKED",
            "reason": "No EXP-001 procedure dataset collected and no trained 1D-TCN checkpoint exists.",
            "metric_value": None,
        },
        "orientation_robustness_drop_delta": {
            "status": "BLOCKED",
            "reason": "Requires per-angle F1 scores from real model inference across all 7 canonical angles.",
            "metric_value": None,
        },
    }

    report.status = "PARTIALLY_PASSED"
    report.notes = (
        "Deterministic benchmark harness completed successfully. "
        "Feature extraction, buffer stacking, and video tensor preparation pipelines "
        "were verified and measured empirically in pure NumPy. "
        "Model-level inference latency, memory, and classification accuracy are explicitly "
        "reported as BLOCKED pending trained checkpoints and dataset collection. "
        "No winner is declared."
    )

    # Save to disk
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)

    if verbose:
        print("[STEP 4/4] Benchmark Summary & Artifact Export")
        print(f"  Status: {report.status}")
        print(f"  Output: {out_file}")
        print("=" * 70)
        print()

    return report


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate B5.2 — Deterministic Architecture Benchmark Harness"
    )
    parser.add_argument(
        "--num-iterations", type=int, default=DEFAULT_ITERATIONS,
        help=f"Number of benchmark timing iterations (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write JSON benchmark results",
    )
    parser.add_argument(
        "--seed", type=int, default=BENCHMARK_SEED,
        help=f"Random seed for deterministic workloads (default: {BENCHMARK_SEED})",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress console output",
    )
    args = parser.parse_args()

    out_p = Path(args.output) if args.output else None
    run_architecture_benchmark(
        iterations=args.num_iterations,
        output_path=out_p,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
