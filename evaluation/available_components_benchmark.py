"""
evaluation/available_components_benchmark.py
Gate B5.3 — Benchmark Available Components
ISRO SIH26174 BAS Experiment Monitor — Workstream B

PURPOSE
=======
Executes a comprehensive, deterministic, multi-iteration benchmark profiling
ALL components genuinely available in the repository:
  1. Spatial Feature Extraction (111-dimensional vector from 33 pose + 4 object entries)
  2. Temporal Sliding Window Buffering (T=30 deque buffer -> (1, 30, 111) tensor)
  3. Combined Spatial Vector Extraction + Buffer Stacking
  4. Frame Preprocessing (Detector Letterboxing & Pose Normalization)
  5. Camera Rectification Preprocessing (Baseline vs Rectified Overhead)
  6. Multi-Angle Orientation Preprocessing (All 7 Canonical Angles: 0° - 270°)
  7. Candidate A Raw Video Tensor Preparation ((1, 3, 16, 112, 112) & (1, 3, 32, 224, 224))
  8. Candidate B Feature Tensor Preparation ((1, 30, 111))
  9. Actual In-Memory Footprint Analysis (Buffer, Vector, and Tensor memory)

EPISTEMIC SEPARATION
====================
  - VERIFIED MEASUREMENTS: Genuinely executed and measured on available code with warm-up.
  - BLOCKED METRICS: Model inference latency, parameter counts on disk, accuracy, precision,
    recall, F1, confusion matrix, and action-recognition orientation degradation.
  - THEORETICAL ESTIMATES: Literature references for model architecture parameters and FLOPs.
  - FUTURE REQUIRED MEASUREMENTS: Full end-to-end evaluation upon model checkpoint and dataset delivery.

REPRODUCIBILITY
===============
  - Fixed seed: BENCHMARK_SEED = 42
  - Warmup iterations before timing measurements
  - Structured output written to evaluation/results/b5_3_available_components.json
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

from backend.video.frame_processor import (
    FrameProcessor,
    _pure_numpy_resize,
    _pure_numpy_bgr_to_rgb,
)
from backend.video.camera_rectification import (
    CameraRectifier,
    RectificationConfig,
)
from backend.ai.action_classifier import ActionClassifier
from backend.ai.augmentation import AugmentationEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BENCHMARK_SEED: int = 42
DEFAULT_ITERATIONS: int = 50
WARMUP_ITERATIONS: int = 5

CANONICAL_ANGLES: List[int] = [0, 45, 90, 135, 180, 225, 270]
TCN_WINDOW_SIZE: int = 30
TCN_FEATURE_DIM: int = 111

R2PLUS1D_SHAPES: Dict[str, Tuple[int, int, int, int, int]] = {
    "standard_16x112": (1, 3, 16, 112, 112),
    "highres_32x224": (1, 3, 32, 224, 224),
}

DEFAULT_OUTPUT_PATH: Path = (
    _REPO_ROOT / "evaluation" / "results" / "b5_3_available_components.json"
)
HMDB51_DATA_ROOT: Path = _REPO_ROOT / "training" / "data" / "raw" / "hmdb51_sta"


# ---------------------------------------------------------------------------
# Timing Helper
# ---------------------------------------------------------------------------

@dataclass
class ComponentTimingStats:
    """Detailed statistical profile of execution latencies."""
    iterations: int = 0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    p95_ms: float = 0.0
    throughput: float = 0.0
    throughput_unit: str = "items/s"

    @classmethod
    def from_timings(
        cls, timings_ms: List[float], unit: str = "items/s"
    ) -> ComponentTimingStats:
        if not timings_ms:
            return cls(throughput_unit=unit)
        arr = np.array(timings_ms, dtype=np.float64)
        mean_v = float(np.mean(arr))
        tp = (1000.0 / mean_v) if mean_v > 0 else 0.0
        return cls(
            iterations=len(timings_ms),
            mean_ms=round(mean_v, 4),
            median_ms=round(float(np.median(arr)), 4),
            min_ms=round(float(np.min(arr)), 4),
            max_ms=round(float(np.max(arr)), 4),
            std_ms=round(float(np.std(arr)), 4),
            p95_ms=round(float(np.percentile(arr, 95)), 4),
            throughput=round(tp, 2),
            throughput_unit=unit,
        )


# ---------------------------------------------------------------------------
# Dynamic Environment Detector
# ---------------------------------------------------------------------------

def detect_runtime_environment() -> Dict[str, Any]:
    """Detects and reports actual system runtime capabilities without hardcoding."""
    # RAM detection via psutil if available
    ram_gb = "Unknown"
    try:
        import psutil  # type: ignore
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        pass

    # Framework detection
    opencv_installed = False
    try:
        import cv2  # type: ignore
        opencv_installed = True
    except ImportError:
        opencv_installed = False

    torch_installed = False
    try:
        import torch  # type: ignore
        torch_installed = True
    except ImportError:
        torch_installed = False

    tflite_installed = False
    try:
        import tflite_runtime  # type: ignore
        tflite_installed = True
    except ImportError:
        try:
            import tensorflow.lite  # type: ignore
            tflite_installed = True
        except ImportError:
            tflite_installed = False

    return {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "processor": platform.processor() or "Unknown",
        "cpu_count": os.cpu_count() or 1,
        "total_ram_gb": ram_gb,
        "numpy_version": np.__version__,
        "opencv_installed": opencv_installed,
        "pytorch_installed": torch_installed,
        "tflite_installed": tflite_installed,
    }


# ---------------------------------------------------------------------------
# Synthetic Benchmark Data
# ---------------------------------------------------------------------------

def _generate_benchmark_frames(
    count: int = 32,
    height: int = 480,
    width: int = 640,
    seed: int = BENCHMARK_SEED,
) -> List[np.ndarray]:
    """Generates deterministic realistic video frames for preprocessing benchmarks."""
    rng = np.random.RandomState(seed)
    frames: List[np.ndarray] = []
    for _ in range(count):
        base = np.zeros((height, width, 3), dtype=np.uint8)
        # Quadrant coloring
        base[:height // 2, :width // 2] = [200, 70, 70]
        base[:height // 2, width // 2:] = [70, 200, 70]
        base[height // 2:, :width // 2] = [70, 70, 200]
        base[height // 2:, width // 2:] = [180, 180, 180]
        noise = rng.randint(0, 15, base.shape, dtype=np.uint8)
        frame = np.clip(base.astype(np.int32) + noise.astype(np.int32), 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def _generate_benchmark_landmarks() -> Tuple[List[Dict[str, float]], List[Dict[str, Any]]]:
    """Generates realistic 33-joint pose landmarks and 4 object bounding boxes."""
    pose = [{"x": 0.45 + i * 0.005, "y": 0.35 + i * 0.005, "z": -0.05 + i * 0.002} for i in range(33)]
    boxes = [
        {"x1": 0.15, "y1": 0.20, "x2": 0.35, "y2": 0.45, "confidence": 0.96, "label": "RED_SAMPLE"},
        {"x1": 0.55, "y1": 0.50, "x2": 0.75, "y2": 0.70, "confidence": 0.94, "label": "SAMPLE_CONTAINER"},
        {"x1": 0.10, "y1": 0.60, "x2": 0.30, "y2": 0.80, "confidence": 0.89, "label": "BLUE_SAMPLE"},
        {"x1": 0.65, "y1": 0.15, "x2": 0.85, "y2": 0.35, "confidence": 0.92, "label": "CONTAINER_LID"},
    ]
    return pose, boxes


# ---------------------------------------------------------------------------
# Individual Component Benchmarks
# ---------------------------------------------------------------------------

def benchmark_feature_extraction(
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = WARMUP_ITERATIONS,
) -> Dict[str, Any]:
    """Measures FrameProcessor.extract_keypoint_vector latency (111-dim vector)."""
    pose, boxes = _generate_benchmark_landmarks()

    # Warmup
    for _ in range(warmup):
        _ = FrameProcessor.extract_keypoint_vector(pose, boxes)

    timings: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        vec = FrameProcessor.extract_keypoint_vector(pose, boxes)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    stats = ComponentTimingStats.from_timings(timings, unit="frames/s")
    return {
        "timing": asdict(stats),
        "vector_dimension": int(vec.shape[0]),
        "vector_memory_bytes": int(vec.nbytes),
        "vector_dtype": str(vec.dtype),
    }


def benchmark_temporal_buffering(
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = WARMUP_ITERATIONS,
) -> Dict[str, Any]:
    """Measures ActionClassifier 30-frame sliding window accumulation and tensor stacking."""
    pose, boxes = _generate_benchmark_landmarks()
    vec = FrameProcessor.extract_keypoint_vector(pose, boxes)

    classifier = ActionClassifier(window_size=TCN_WINDOW_SIZE)
    for _ in range(TCN_WINDOW_SIZE):
        classifier.push_frame_vector(vec)

    # Warmup
    for _ in range(warmup):
        classifier.push_frame_vector(vec)
        _ = np.expand_dims(np.stack(list(classifier._vector_buffer)).astype(np.float32), axis=0)

    timings: List[float] = []
    stacked_tensor = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        classifier.push_frame_vector(vec)
        stacked = np.stack(list(classifier._vector_buffer))
        stacked_tensor = np.expand_dims(stacked.astype(np.float32), axis=0)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    stats = ComponentTimingStats.from_timings(timings, unit="windows/s")
    return {
        "timing": asdict(stats),
        "window_size": TCN_WINDOW_SIZE,
        "stacked_tensor_shape": list(stacked_tensor.shape) if stacked_tensor is not None else [],
        "buffer_memory_bytes": int(stacked_tensor.nbytes) if stacked_tensor is not None else 0,
    }


def benchmark_frame_preprocessing(
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = WARMUP_ITERATIONS,
) -> Dict[str, Any]:
    """Measures FrameProcessor detector letterboxing and pose normalization."""
    frame = _generate_benchmark_frames(count=1, height=480, width=640)[0]

    # Warmup
    for _ in range(warmup):
        _ = FrameProcessor.preprocess_for_detector(frame, target_size=(640, 640))
        _ = FrameProcessor.preprocess_for_pose(frame, target_size=(256, 256))

    det_timings: List[float] = []
    pose_timings: List[float] = []
    for _ in range(iterations):
        # Detector preprocessing (640x640 letterbox)
        t0 = time.perf_counter()
        det_img, _, _ = FrameProcessor.preprocess_for_detector(frame, target_size=(640, 640))
        t1 = time.perf_counter()
        det_timings.append((t1 - t0) * 1000.0)

        # Pose preprocessing (256x256 normalization)
        t2 = time.perf_counter()
        pose_img = FrameProcessor.preprocess_for_pose(frame, target_size=(256, 256))
        t3 = time.perf_counter()
        pose_timings.append((t3 - t2) * 1000.0)

    return {
        "detector_preprocessing_640x640": asdict(
            ComponentTimingStats.from_timings(det_timings, unit="frames/s")
        ),
        "pose_preprocessing_256x256": asdict(
            ComponentTimingStats.from_timings(pose_timings, unit="frames/s")
        ),
        "detector_input_shape": list(det_img.shape),
        "pose_input_shape": list(pose_img.shape),
    }


def benchmark_rectification_preprocessing(
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = WARMUP_ITERATIONS,
) -> Dict[str, Any]:
    """Measures CameraRectifier execution latency and overhead over baseline."""
    frame = _generate_benchmark_frames(count=1, height=480, width=640)[0]
    rectifier_90 = CameraRectifier(config=RectificationConfig(enabled=True, rotation_deg=90.0))

    # Warmup
    for _ in range(warmup):
        _ = FrameProcessor.process_frame_pipeline(frame, rectifier=None)
        _ = FrameProcessor.process_frame_pipeline(frame, rectifier=rectifier_90)

    base_timings: List[float] = []
    rect_timings: List[float] = []
    for _ in range(iterations):
        # Baseline (Identity bypass)
        t0 = time.perf_counter()
        _ = FrameProcessor.process_frame_pipeline(frame, rectifier=None)
        t1 = time.perf_counter()
        base_timings.append((t1 - t0) * 1000.0)

        # Rectified (90° camera rotation calibration)
        t2 = time.perf_counter()
        _ = FrameProcessor.process_frame_pipeline(frame, rectifier=rectifier_90)
        t3 = time.perf_counter()
        rect_timings.append((t3 - t2) * 1000.0)

    overhead_timings = [r - b for r, b in zip(rect_timings, base_timings)]

    return {
        "baseline_pipeline": asdict(
            ComponentTimingStats.from_timings(base_timings, unit="frames/s")
        ),
        "rectified_pipeline_90deg": asdict(
            ComponentTimingStats.from_timings(rect_timings, unit="frames/s")
        ),
        "overhead": asdict(
            ComponentTimingStats.from_timings(overhead_timings, unit="overhead_ms")
        ),
    }


def benchmark_orientation_preprocessing(
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = WARMUP_ITERATIONS,
) -> Dict[str, Any]:
    """
    Measures frame rotation preprocessing latency across all 7 canonical angles.
    EXPLICIT NOTE: Measures geometric image rotation latency only; does NOT imply
    model action-recognition robustness.
    """
    frame = _generate_benchmark_frames(count=1, height=480, width=640)[0]
    aug_engine = AugmentationEngine()
    angle_results: Dict[str, Any] = {}

    for angle in CANONICAL_ANGLES:
        # Warmup
        for _ in range(warmup):
            _ = aug_engine.rotate(frame, float(angle))

        timings: List[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = aug_engine.rotate(frame, float(angle))
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

        stats = ComponentTimingStats.from_timings(timings, unit="frames/s")
        angle_results[f"{angle}_deg"] = {
            "angle_degrees": angle,
            "timing": asdict(stats),
            "note": "Geometric frame rotation latency only (preprocessing stage)",
        }

    return angle_results


def benchmark_candidate_tensors(
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = WARMUP_ITERATIONS,
) -> Dict[str, Any]:
    """Measures tensor preparation latency and memory for Candidate A and Candidate B."""
    frames = _generate_benchmark_frames(count=32, height=240, width=320)
    pose, boxes = _generate_benchmark_landmarks()
    vec = FrameProcessor.extract_keypoint_vector(pose, boxes)

    # Candidate A: 16x112x112
    # Warmup
    for _ in range(warmup):
        selected = frames[:16]
        _ = np.expand_dims(
            np.transpose(
                np.stack([
                    np.transpose(_pure_numpy_resize(_pure_numpy_bgr_to_rgb(f), (112, 112)).astype(np.float32) / 255.0, (2, 0, 1))
                    for f in selected
                ], axis=0),
                (1, 0, 2, 3),
            ),
            axis=0,
        )

    timings_16: List[float] = []
    t_16 = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        selected = frames[:16]
        t_16 = np.expand_dims(
            np.transpose(
                np.stack([
                    np.transpose(_pure_numpy_resize(_pure_numpy_bgr_to_rgb(f), (112, 112)).astype(np.float32) / 255.0, (2, 0, 1))
                    for f in selected
                ], axis=0),
                (1, 0, 2, 3),
            ),
            axis=0,
        )
        t1 = time.perf_counter()
        timings_16.append((t1 - t0) * 1000.0)

    # Candidate A: 32x224x224 (fewer iters due to CPU load)
    highres_iters = max(5, iterations // 5)
    timings_32: List[float] = []
    t_32 = None
    for _ in range(highres_iters):
        t0 = time.perf_counter()
        selected = frames[:32]
        t_32 = np.expand_dims(
            np.transpose(
                np.stack([
                    np.transpose(_pure_numpy_resize(_pure_numpy_bgr_to_rgb(f), (224, 224)).astype(np.float32) / 255.0, (2, 0, 1))
                    for f in selected
                ], axis=0),
                (1, 0, 2, 3),
            ),
            axis=0,
        )
        t1 = time.perf_counter()
        timings_32.append((t1 - t0) * 1000.0)

    # Candidate B: (1, 30, 111)
    classifier = ActionClassifier(window_size=TCN_WINDOW_SIZE)
    for _ in range(TCN_WINDOW_SIZE):
        classifier.push_frame_vector(vec)

    timings_b: List[float] = []
    t_b = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        classifier.push_frame_vector(vec)
        stacked = np.stack(list(classifier._vector_buffer))
        t_b = np.expand_dims(stacked.astype(np.float32), axis=0)
        t1 = time.perf_counter()
        timings_b.append((t1 - t0) * 1000.0)

    return {
        "candidate_a_standard_16x112": {
            "tensor_shape": list(t_16.shape) if t_16 is not None else [],
            "timing": asdict(ComponentTimingStats.from_timings(timings_16, unit="clips/s")),
            "memory_bytes": int(t_16.nbytes) if t_16 is not None else 0,
            "memory_kb": round((t_16.nbytes / 1024), 2) if t_16 is not None else 0.0,
        },
        "candidate_a_highres_32x224": {
            "tensor_shape": list(t_32.shape) if t_32 is not None else [],
            "timing": asdict(ComponentTimingStats.from_timings(timings_32, unit="clips/s")),
            "memory_bytes": int(t_32.nbytes) if t_32 is not None else 0,
            "memory_kb": round((t_32.nbytes / 1024), 2) if t_32 is not None else 0.0,
        },
        "candidate_b_window_30x111": {
            "tensor_shape": list(t_b.shape) if t_b is not None else [],
            "timing": asdict(ComponentTimingStats.from_timings(timings_b, unit="windows/s")),
            "memory_bytes": int(t_b.nbytes) if t_b is not None else 0,
            "memory_kb": round((t_b.nbytes / 1024), 2) if t_b is not None else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Master Benchmark Coordinator
# ---------------------------------------------------------------------------

def run_available_components_benchmark(
    iterations: int = DEFAULT_ITERATIONS,
    output_path: Optional[Path] = None,
    seed: int = BENCHMARK_SEED,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Runs all component benchmarks and produces the structured JSON report."""
    out_file = output_path or DEFAULT_OUTPUT_PATH

    if verbose:
        print("=" * 75)
        print("Gate B5.3 — Benchmark Available Components")
        print("ISRO SIH26174 BAS Experiment Monitor — Workstream B")
        print("=" * 75)
        print(f"[CONFIG] Seed: {seed} | Iterations: {iterations} | Warmup: {WARMUP_ITERATIONS}")
        print()

    # 1. Environment Detection
    env = detect_runtime_environment()
    if verbose:
        print("[STEP 1/7] Detecting Runtime Environment...")
        print(f"  OS: {env['platform']} | Python: {env['python_version']} | CPUs: {env['cpu_count']} | RAM: {env['total_ram_gb']} GB")
        print(f"  OpenCV: {env['opencv_installed']} | PyTorch: {env['pytorch_installed']} | TFLite: {env['tflite_installed']}")
        print()

    # 2. Feature Extraction & Temporal Buffering
    if verbose:
        print("[STEP 2/7] Benchmarking Spatial Feature Extraction & Temporal Buffering...")
    fe_res = benchmark_feature_extraction(iterations=iterations, warmup=WARMUP_ITERATIONS)
    tb_res = benchmark_temporal_buffering(iterations=iterations, warmup=WARMUP_ITERATIONS)
    if verbose:
        print(f"  Feature Vector (111-dim): {fe_res['timing']['mean_ms']:.4f} ms (p95: {fe_res['timing']['p95_ms']:.4f} ms, {fe_res['timing']['throughput']:.1f} FPS)")
        print(f"  Buffer Stacking (30-win): {tb_res['timing']['mean_ms']:.4f} ms (p95: {tb_res['timing']['p95_ms']:.4f} ms, {tb_res['timing']['throughput']:.1f} WPS)")
        print()

    # 3. Frame Preprocessing
    if verbose:
        print("[STEP 3/7] Benchmarking Frame Preprocessing...")
    prep_res = benchmark_frame_preprocessing(iterations=iterations, warmup=WARMUP_ITERATIONS)
    if verbose:
        det_t = prep_res["detector_preprocessing_640x640"]
        pose_t = prep_res["pose_preprocessing_256x256"]
        print(f"  Detector Letterbox (640x640): {det_t['mean_ms']:.3f} ms (Throughput: {det_t['throughput']:.1f} FPS)")
        print(f"  Pose Normalization (256x256): {pose_t['mean_ms']:.3f} ms (Throughput: {pose_t['throughput']:.1f} FPS)")
        print()

    # 4. Rectification Preprocessing
    if verbose:
        print("[STEP 4/7] Benchmarking Camera Rectification Preprocessing...")
    rect_res = benchmark_rectification_preprocessing(iterations=iterations, warmup=WARMUP_ITERATIONS)
    if verbose:
        oh = rect_res["overhead"]
        print(f"  Baseline Pipeline: {rect_res['baseline_pipeline']['mean_ms']:.3f} ms")
        print(f"  Rectified (90°):   {rect_res['rectified_pipeline_90deg']['mean_ms']:.3f} ms")
        print(f"  Rect Overhead:     {oh['mean_ms']:.3f} ms (p95: {oh['p95_ms']:.3f} ms)")
        print()

    # 5. Multi-Angle Orientation Preprocessing
    if verbose:
        print("[STEP 5/7] Benchmarking Multi-Angle Orientation Preprocessing (All 7 Angles)...")
    orient_res = benchmark_orientation_preprocessing(iterations=iterations, warmup=WARMUP_ITERATIONS)
    if verbose:
        for k, v in orient_res.items():
            print(f"  Angle {v['angle_degrees']:>3}°: {v['timing']['mean_ms']:.3f} ms (Throughput: {v['timing']['throughput']:.1f} FPS)")
        print()

    # 6. Candidate Tensor Preparation
    if verbose:
        print("[STEP 6/7] Benchmarking Candidate Tensor Preparation...")
    tensor_res = benchmark_candidate_tensors(iterations=iterations, warmup=WARMUP_ITERATIONS)
    if verbose:
        a16 = tensor_res["candidate_a_standard_16x112"]
        a32 = tensor_res["candidate_a_highres_32x224"]
        b30 = tensor_res["candidate_b_window_30x111"]
        print(f"  Candidate A (16x112): {a16['timing']['mean_ms']:.3f} ms ({a16['memory_kb']} KB)")
        print(f"  Candidate A (32x224): {a32['timing']['mean_ms']:.3f} ms ({a32['memory_kb']} KB)")
        print(f"  Candidate B (30x111): {b30['timing']['mean_ms']:.4f} ms ({b30['memory_kb']} KB)")
        print()

    # 7. Blocked Register & Master JSON Assembly
    blocked_metrics = {
        "candidate_a_inference_latency": {
            "status": "BLOCKED",
            "reason": "PyTorch is not installed in the active environment and no R(2+1)D-18 checkpoint exists.",
            "metric_value": None,
        },
        "candidate_b_inference_latency": {
            "status": "BLOCKED",
            "reason": "Neither tflite_runtime nor tensorflow is installed, and data/temporal_action.tflite is absent.",
            "metric_value": None,
        },
        "model_classification_accuracy": {
            "status": "BLOCKED",
            "reason": "EXP-001 procedure dataset is NOT COLLECTED; no trained 5-action checkpoints exist.",
            "metric_value": None,
        },
        "model_precision_recall_f1": {
            "status": "BLOCKED",
            "reason": "Requires real predictions on held-out EXP-001 test clips.",
            "metric_value": None,
        },
        "orientation_f1_degradation": {
            "status": "BLOCKED",
            "reason": "Requires per-angle F1 scores from real model inference across all 7 canonical angles.",
            "metric_value": None,
        },
    }

    report = {
        "gate": "B5.3",
        "timestamp": datetime.datetime.now().isoformat(),
        "seed": seed,
        "iterations": iterations,
        "warmup_iterations": WARMUP_ITERATIONS,
        "environment": env,
        "feature_extraction": fe_res,
        "temporal_buffering": tb_res,
        "combined_feature_and_buffer_prep_ms": round(
            fe_res["timing"]["mean_ms"] + tb_res["timing"]["mean_ms"], 4
        ),
        "frame_preprocessing": prep_res,
        "rectification_preprocessing": rect_res,
        "orientation_preprocessing": orient_res,
        "tensor_preparation": tensor_res,
        "blocked_model_metrics": blocked_metrics,
        "status": "PARTIALLY_PASSED",
        "notes": (
            "All available preprocessing, spatial vector extraction, temporal buffering, "
            "and tensor preparation components successfully benchmarked with warm-up. "
            "Model-level inference and accuracy metrics remain explicitly BLOCKED. "
            "No winner is declared."
        ),
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if verbose:
        print("[STEP 7/7] Master Results Exported")
        print(f"  Artifact: {out_file}")
        print("=" * 75)
        print()

    return report


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate B5.3 — Benchmark Available Components"
    )
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS,
        help=f"Number of timing iterations (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write JSON benchmark results",
    )
    parser.add_argument(
        "--seed", type=int, default=BENCHMARK_SEED,
        help=f"Random seed (default: {BENCHMARK_SEED})",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress console output",
    )
    args = parser.parse_args()

    out_p = Path(args.output) if args.output else None
    run_available_components_benchmark(
        iterations=args.iterations,
        output_path=out_p,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
