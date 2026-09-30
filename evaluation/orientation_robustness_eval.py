"""
evaluation/orientation_robustness_eval.py
Gate B4.2 — Real-Data Orientation & Robustness Evaluation
ISRO SIH26174 BAS Experiment Monitor — Workstream B

EVALUATION STATUS
=================
  MODEL EVALUATION:            BLOCKED — No trained checkpoint available in the repository.
                                The Part-1 binary model (catch / not-catch) is under active
                                revision in Colab and has not been exported or committed.
                                No EXP-001-specific dataset has been collected (Gate B3 protocol).
                                Therefore, precision / recall / F1 metrics are NOT computable
                                and are NOT fabricated. They are explicitly marked BLOCKED.

  PREPROCESSING EVALUATION:    EXECUTED — Uses real HMDB51 video frames (the only real data
                                available in the repository) as frame samples. Evaluates:
                                  1. Orientation condition generation across all 7 canonical angles.
                                  2. Frame integrity after augmentation.
                                  3. Annotation synchronization fidelity.
                                  4. Per-angle augmentation timing.
                                  5. Rectification preprocessing overhead (B4.1c.2 hook).
                                  6. All 7 robustness dimensions (rotation, scale, translation,
                                     brightness, blur, occlusion, perspective).

EVALUATION MODE
===============
  SYNTHETIC_STRESS_TEST — As explicitly authorized by docs/workstream-b-orientation-robustness.md
  Section 6 ("Synthetic Stress-Testing as Intermediate Check"):
  "In the absence of full multi-angle physical datasets, synthetic rotations on held-out TEST
  split clips serve as an intermediate diagnostic tool, clearly marked as SYNTHETIC_STRESS_TEST."

DATASET SPLIT POLICY
====================
  Subject-level leakage isolation is maintained by design: HMDB51 has no subject identity
  metadata. All clips used here are strictly for EVALUATION ONLY — no training is performed.
  No transformed variant is injected into any training split.

REPRODUCIBILITY
===============
  Fixed seed: EVAL_SEED = 42
  Configuration recorded in results JSON.
  Rerunnable without external dependencies (pure NumPy + optional OpenCV).

Usage:
  python evaluation/orientation_robustness_eval.py
  python evaluation/orientation_robustness_eval.py --output results/b4_2_results.json
  python evaluation/orientation_robustness_eval.py --max-frames 20 --output my_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or from evaluation/ directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.ai.augmentation import (
    AugmentationConfig,
    AugmentationEngine,
    AugmentationResult,
)
from backend.video.camera_rectification import (
    CameraRectifier,
    RectificationConfig,
    RectificationResult,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVAL_SEED: int = 42
CANONICAL_ANGLES: List[int] = [0, 45, 90, 135, 180, 225, 270]
ORIENTATION_CATEGORIES: Dict[int, str] = {
    0: "NORMAL",
    45: "MODERATE_ROTATION",
    90: "EXTREME_ROTATION",
    135: "MODERATE_ROTATION",
    180: "EXTREME_ROTATION",
    225: "MODERATE_ROTATION",
    270: "EXTREME_ROTATION",
}

HMDB51_DATA_ROOT: Path = _REPO_ROOT / "training" / "data" / "raw" / "hmdb51_sta"
# Use manipulation-adjacent classes from HMDB51 for physical realism
PREFERRED_CLASSES: List[str] = ["catch", "pick", "pour", "throw"]

DEFAULT_OUTPUT_PATH: Path = (
    _REPO_ROOT / "evaluation" / "results" / "b4_2_orientation_robustness_results.json"
)

# Proposed target thresholds (from orientation_robustness_spec.json) — PROPOSED TARGETS only
PROPOSED_LATENCY_OVERHEAD_MS: float = 2.0  # <= 2.0 ms per frame overhead
# F1 drop threshold is not evaluable without model.


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class FrameSample:
    """A single video frame loaded for evaluation."""
    source_class: str
    source_file: str
    frame_index: int
    width: int
    height: int
    image: np.ndarray

    def to_meta(self) -> Dict[str, Any]:
        return {
            "source_class": self.source_class,
            "source_file": self.source_file,
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class AngleConditionResult:
    """Evaluation result for one orientation angle."""
    angle_deg: int
    category: str
    augmentation_type: str  # "SYNTHETIC_STRESS_TEST"

    # Frame integrity checks
    output_shape_valid: bool = False
    pixel_range_valid: bool = False  # [0, 255]
    frame_not_all_black: bool = False  # content survived rotation

    # Annotation synchronization
    annotations_evaluated: bool = False
    box_coords_in_bounds: bool = False
    keypoints_in_bounds: bool = False

    # Timing
    augmentation_time_ms: float = 0.0

    # Errors
    errors: List[str] = field(default_factory=list)


@dataclass
class RobustnessDimensionResult:
    """Evaluation result for one robustness dimension."""
    dimension: str
    augmentation_time_ms: float = 0.0
    output_shape_valid: bool = False
    pixel_range_valid: bool = False
    frame_not_all_black: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class RectificationComparisonResult:
    """Comparison of preprocessing with and without rectification."""
    angle_deg: int
    # Without rectification
    baseline_time_ms: float = 0.0
    baseline_shape_valid: bool = False
    # With rectification
    rectified_time_ms: float = 0.0
    rectified_shape_valid: bool = False
    # Delta
    overhead_ms: float = 0.0
    overhead_within_proposed_target: bool = False
    proposed_target_ms: float = PROPOSED_LATENCY_OVERHEAD_MS
    errors: List[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Complete B4.2 gate evaluation report."""
    gate: str = "B4.2"
    date: str = ""
    seed: int = EVAL_SEED
    evaluation_mode: str = "SYNTHETIC_STRESS_TEST"

    # Data availability
    dataset_used: str = ""
    dataset_status: str = ""
    model_checkpoint_status: str = "BLOCKED — Not present in repository"
    model_evaluation_status: str = "BLOCKED"
    model_evaluation_reason: str = (
        "No trained Part-1 model checkpoint exists in the repository. "
        "The binary catch/not-catch model is under active revision in Colab "
        "and has not been exported or committed. No EXP-001-specific dataset "
        "has been collected under the Gate B3 protocol. "
        "Precision / Recall / F1 metrics are therefore NOT computable."
    )

    # Frames loaded
    frames_attempted: int = 0
    frames_successfully_loaded: int = 0
    frame_source_classes: List[str] = field(default_factory=list)

    # Orientation evaluation
    orientation_results: List[Dict[str, Any]] = field(default_factory=list)
    all_orientations_evaluated: bool = False

    # Robustness dimensions
    robustness_dimension_results: List[Dict[str, Any]] = field(default_factory=list)

    # Rectification comparison
    rectification_comparison: List[Dict[str, Any]] = field(default_factory=list)

    # Aggregate timing
    mean_augmentation_time_ms: float = 0.0
    max_augmentation_time_ms: float = 0.0
    mean_rectification_overhead_ms: float = 0.0
    latency_overhead_proposed_target_ms: float = PROPOSED_LATENCY_OVERHEAD_MS

    # Gate outcome
    preprocessing_eval_passed: bool = False
    model_eval_passed: bool = False  # Always False — no model
    gate_status: str = "PARTIALLY_EVALUATED"
    gate_notes: str = ""

    # Blocked metrics — explicitly set to None
    per_angle_precision: None = None
    per_angle_recall: None = None
    per_angle_f1: None = None
    macro_f1: None = None
    max_orientation_drop: None = None
    confusion_matrix: None = None


# ---------------------------------------------------------------------------
# Frame Loading
# ---------------------------------------------------------------------------

def _load_frame_from_avi(video_path: Path, frame_index: int = 0) -> Optional[np.ndarray]:
    """
    Attempts to load a frame from an AVI video file.
    Returns None if loading fails or cv2 is unavailable.
    """
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        for _ in range(frame_index + 1):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None
        cap.release()
        return frame  # BGR numpy array
    except ImportError:
        return None
    except Exception:
        return None


def _generate_synthetic_frame(width: int = 120, height: int = 90, seed: int = 0) -> np.ndarray:
    """
    Generates a deterministic synthetic frame for environments where cv2 is unavailable.
    Creates a realistic-looking quadrant-color test image with gradient noise.
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Quadrant structure (simulates object-in-scene patterns)
    img[:height // 2, :width // 2] = [180, 40, 40]    # Red-ish top-left
    img[:height // 2, width // 2:] = [40, 40, 180]    # Blue-ish top-right
    img[height // 2:, :width // 2] = [40, 160, 40]    # Green-ish bottom-left
    img[height // 2:, width // 2:] = [220, 220, 220]  # Light bottom-right
    # Add noise to simulate real texture
    noise = rng.randint(0, 30, img.shape, dtype=np.uint8)
    img = np.clip(img.astype(np.int32) + noise.astype(np.int32), 0, 255).astype(np.uint8)
    return img


def load_evaluation_frames(
    max_frames: int = 30,
    seed: int = EVAL_SEED,
) -> Tuple[List[FrameSample], str, str]:
    """
    Loads real video frames from the HMDB51 dataset for evaluation.
    Falls back to synthetic frames if cv2 is unavailable or videos cannot be decoded.

    Returns:
        (frame_list, dataset_used_description, dataset_status)
    """
    rng = np.random.RandomState(seed)
    frames: List[FrameSample] = []
    dataset_used = ""
    dataset_status = ""

    if not HMDB51_DATA_ROOT.exists():
        dataset_status = "ABSENT — training/data/raw/hmdb51_sta/ not found"
        dataset_used = "SYNTHETIC_FALLBACK"
        for i in range(min(max_frames, 10)):
            img = _generate_synthetic_frame(seed=seed + i)
            frames.append(FrameSample(
                source_class="SYNTHETIC",
                source_file=f"synthetic_frame_{i:03d}",
                frame_index=i,
                width=img.shape[1],
                height=img.shape[0],
                image=img,
            ))
        return frames, dataset_used, dataset_status

    # Try to load real HMDB51 frames
    available_classes = [
        d.name for d in HMDB51_DATA_ROOT.iterdir()
        if d.is_dir() and d.name in PREFERRED_CLASSES
    ]
    if not available_classes:
        available_classes = [
            d.name for d in HMDB51_DATA_ROOT.iterdir() if d.is_dir()
        ][:4]

    all_avi_files: List[Tuple[str, Path]] = []
    for cls in available_classes:
        cls_dir = HMDB51_DATA_ROOT / cls
        avi_files = sorted(cls_dir.glob("*.avi"))
        for f in avi_files:
            all_avi_files.append((cls, f))

    if not all_avi_files:
        dataset_status = "PRESENT — directories exist but no .avi files found"
        dataset_used = "SYNTHETIC_FALLBACK"
        for i in range(min(max_frames, 10)):
            img = _generate_synthetic_frame(seed=seed + i)
            frames.append(FrameSample(
                source_class="SYNTHETIC",
                source_file=f"synthetic_frame_{i:03d}",
                frame_index=i,
                width=img.shape[1],
                height=img.shape[0],
                image=img,
            ))
        return frames, dataset_used, dataset_status

    # Shuffle deterministically and attempt to load up to max_frames
    idx_order = rng.permutation(len(all_avi_files)).tolist()
    cv2_available = False
    synthetic_fallback_count = 0
    loaded_real = 0

    for idx in idx_order:
        if len(frames) >= max_frames:
            break
        cls_name, avi_path = all_avi_files[idx]
        frame_img = _load_frame_from_avi(avi_path, frame_index=5)

        if frame_img is not None:
            cv2_available = True
            loaded_real += 1
            frames.append(FrameSample(
                source_class=cls_name,
                source_file=avi_path.name,
                frame_index=5,
                width=frame_img.shape[1],
                height=frame_img.shape[0],
                image=frame_img,
            ))
        else:
            # cv2 may be absent; use synthetic fallback for this slot
            img = _generate_synthetic_frame(seed=seed + len(frames))
            synthetic_fallback_count += 1
            frames.append(FrameSample(
                source_class=cls_name + "_SYNTHETIC_FALLBACK",
                source_file=avi_path.name,
                frame_index=0,
                width=img.shape[1],
                height=img.shape[0],
                image=img,
            ))

    if cv2_available and loaded_real > 0:
        dataset_used = f"HMDB51 ({', '.join(available_classes)})"
        dataset_status = (
            f"PRESENT — {len(all_avi_files)} .avi files across "
            f"{len(available_classes)} classes. "
            f"{loaded_real} real frames loaded. "
            f"{synthetic_fallback_count} frames used synthetic fallback. "
            f"Classes used: {', '.join(available_classes)}. "
            "NOTE: HMDB51 generic labels (catch, pick) are NOT EXP-001 "
            "procedure actions. Used for frame-level preprocessing evaluation only."
        )
    else:
        dataset_used = "SYNTHETIC_FALLBACK"
        dataset_status = (
            "HMDB51 directories present but cv2 (OpenCV) unavailable. "
            "All frames generated synthetically for preprocessing evaluation. "
            "Real HMDB51 frames cannot be decoded without cv2."
        )

    return frames, dataset_used, dataset_status


# ---------------------------------------------------------------------------
# Orientation Evaluation
# ---------------------------------------------------------------------------

def _make_sample_annotations(width: int, height: int) -> Tuple[List[Dict], List[Dict]]:
    """Creates representative normalized bounding boxes and keypoints for a frame."""
    boxes = [
        {"x1": 0.20, "y1": 0.15, "x2": 0.45, "y2": 0.60, "label": "RED_SAMPLE"},
        {"x1": 0.55, "y1": 0.40, "x2": 0.80, "y2": 0.75, "label": "SAMPLE_CONTAINER"},
    ]
    keypoints = [
        {"x": 0.50, "y": 0.30, "z": 0.0, "label": "wrist"},
        {"x": 0.45, "y": 0.35, "z": 0.1, "label": "index_finger"},
        {"x": 0.55, "y": 0.35, "z": 0.1, "label": "thumb"},
    ]
    return boxes, keypoints


def evaluate_orientation_conditions(
    frames: List[FrameSample],
    seed: int = EVAL_SEED,
) -> Tuple[List[AngleConditionResult], float, float]:
    """
    Evaluates all 7 canonical orientation angles across the frame set.
    Returns (results_per_angle, mean_aug_time_ms, max_aug_time_ms).
    """
    engine = AugmentationEngine()
    results: List[AngleConditionResult] = []
    all_times: List[float] = []

    for angle in CANONICAL_ANGLES:
        category = ORIENTATION_CATEGORIES[angle]
        result = AngleConditionResult(
            angle_deg=angle,
            category=category,
            augmentation_type="SYNTHETIC_STRESS_TEST",
        )

        angle_times: List[float] = []

        for frame_sample in frames:
            img = frame_sample.image
            boxes, kps = _make_sample_annotations(frame_sample.width, frame_sample.height)

            try:
                t_start = time.perf_counter()
                aug_result = engine.rotate(img, float(angle), boxes, kps)
                t_end = time.perf_counter()
                elapsed_ms = (t_end - t_start) * 1000.0
                angle_times.append(elapsed_ms)

                # Frame integrity
                out_img = aug_result.image
                shape_valid = (
                    out_img.ndim in (2, 3)
                    and out_img.shape[0] == img.shape[0]
                    and out_img.shape[1] == img.shape[1]
                )
                result.output_shape_valid = shape_valid

                pixel_valid = (
                    int(out_img.min()) >= 0 and int(out_img.max()) <= 255
                )
                result.pixel_range_valid = pixel_valid

                # For 0° (identity), full content survives; for non-zero, check not all-black
                if angle == 0:
                    result.frame_not_all_black = bool(out_img.sum() > 0)
                else:
                    # After rotation, corners will be black — check center content
                    cy, cx = out_img.shape[0] // 2, out_img.shape[1] // 2
                    center_region = out_img[
                        max(0, cy - 10): cy + 10,
                        max(0, cx - 10): cx + 10,
                    ]
                    result.frame_not_all_black = bool(center_region.sum() > 0)

                # Annotation synchronization
                result.annotations_evaluated = True
                boxes_ok = all(
                    0.0 <= b["x1"] <= 1.0
                    and 0.0 <= b["y1"] <= 1.0
                    and 0.0 <= b["x2"] <= 1.0
                    and 0.0 <= b["y2"] <= 1.0
                    for b in aug_result.bounding_boxes
                )
                result.box_coords_in_bounds = boxes_ok

                kps_ok = all(
                    0.0 <= k["x"] <= 1.0
                    and 0.0 <= k["y"] <= 1.0
                    for k in aug_result.keypoints
                )
                result.keypoints_in_bounds = kps_ok

            except Exception as exc:
                result.errors.append(f"angle={angle}deg frame={frame_sample.source_file}: {exc}")

        if angle_times:
            result.augmentation_time_ms = float(np.mean(angle_times))
            all_times.extend(angle_times)

        results.append(result)

    mean_ms = float(np.mean(all_times)) if all_times else 0.0
    max_ms = float(np.max(all_times)) if all_times else 0.0
    return results, mean_ms, max_ms


# ---------------------------------------------------------------------------
# Robustness Dimensions Evaluation
# ---------------------------------------------------------------------------

def evaluate_robustness_dimensions(
    frames: List[FrameSample],
    seed: int = EVAL_SEED,
) -> List[RobustnessDimensionResult]:
    """
    Evaluates all 7 robustness dimensions (from orientation_robustness_spec.json) on real frames.
    Uses the first frame for representative timing measurements.
    """
    engine = AugmentationEngine()
    results: List[RobustnessDimensionResult] = []

    representative_frame = frames[0] if frames else None
    if representative_frame is None:
        return results

    img = representative_frame.image
    boxes, kps = _make_sample_annotations(representative_frame.width, representative_frame.height)

    dimension_configs = [
        ("rotation",    lambda: engine.rotate(img, 90.0, boxes, kps)),
        ("scale",       lambda: engine.scale(img, 0.9, boxes, kps)),
        ("translation", lambda: engine.translate(img, 0.05, -0.05, boxes, kps)),
        ("brightness",  lambda: engine.adjust_brightness(img, gain=0.85, bias=-15.0, bounding_boxes=boxes, keypoints=kps)),
        ("blur",        lambda: engine.apply_blur(img, kernel_size=5, bounding_boxes=boxes, keypoints=kps)),
        ("occlusion",   lambda: engine.apply_occlusion(img, box_fraction=0.12, seed=seed, bounding_boxes=boxes, keypoints=kps)),
        ("perspective", lambda: engine.apply_perspective(img, distortion=0.05, seed=seed, bounding_boxes=boxes, keypoints=kps)),
    ]

    for dim_name, aug_fn in dimension_configs:
        dim_result = RobustnessDimensionResult(dimension=dim_name)
        try:
            # Warm up + time over multiple runs for stable measurement
            timings: List[float] = []
            for _ in range(5):
                t0 = time.perf_counter()
                aug_out = aug_fn()
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000.0)

            dim_result.augmentation_time_ms = float(np.median(timings))
            out_img = aug_out.image
            dim_result.output_shape_valid = (
                out_img.ndim in (2, 3)
                and out_img.shape[:2] == img.shape[:2]
            )
            dim_result.pixel_range_valid = (
                int(out_img.min()) >= 0 and int(out_img.max()) <= 255
            )
            dim_result.frame_not_all_black = bool(out_img.sum() > 0)

        except Exception as exc:
            dim_result.errors.append(str(exc))

        results.append(dim_result)

    return results


# ---------------------------------------------------------------------------
# Rectification Comparison
# ---------------------------------------------------------------------------

def evaluate_rectification_comparison(
    frames: List[FrameSample],
    seed: int = EVAL_SEED,
) -> Tuple[List[RectificationComparisonResult], float]:
    """
    Compares per-angle preprocessing with and without the B4.1c.2 rectification hook.
    Measures latency overhead to validate against the proposed 2.0 ms target.

    Returns (results_per_angle, mean_overhead_ms).
    """
    engine = AugmentationEngine()
    comparison_results: List[RectificationComparisonResult] = []
    all_overheads: List[float] = []

    representative_frame = frames[0] if frames else None
    if representative_frame is None:
        return comparison_results, 0.0

    img = representative_frame.image
    boxes, kps = _make_sample_annotations(representative_frame.width, representative_frame.height)

    for angle in CANONICAL_ANGLES:
        comp = RectificationComparisonResult(angle_deg=angle)

        try:
            # --- Baseline: augmentation WITHOUT rectification ---
            baseline_times: List[float] = []
            for _ in range(10):
                t0 = time.perf_counter()
                aug_result = engine.rotate(img, float(angle), boxes, kps)
                t1 = time.perf_counter()
                baseline_times.append((t1 - t0) * 1000.0)
            comp.baseline_time_ms = float(np.median(baseline_times))
            comp.baseline_shape_valid = (
                aug_result.image.shape[:2] == img.shape[:2]
            )

            # --- Rectified: augmentation THEN rectification hook ---
            rectifier_cfg = RectificationConfig(
                enabled=True,
                rotation_deg=float(angle),
                scale=1.0,
                translation_x=0.0,
                translation_y=0.0,
            )
            rectifier = CameraRectifier(rectifier_cfg)

            rectified_times: List[float] = []
            for _ in range(10):
                t0 = time.perf_counter()
                aug_result_r = engine.rotate(img, float(angle), boxes, kps)
                rect_result = rectifier.rectify(
                    aug_result_r.image,
                    bounding_boxes=aug_result_r.bounding_boxes,
                    keypoints=aug_result_r.keypoints,
                )
                t1 = time.perf_counter()
                rectified_times.append((t1 - t0) * 1000.0)

            comp.rectified_time_ms = float(np.median(rectified_times))
            comp.rectified_shape_valid = (
                rect_result.image.shape[:2] == img.shape[:2]
            )

            overhead = comp.rectified_time_ms - comp.baseline_time_ms
            comp.overhead_ms = max(0.0, overhead)
            comp.overhead_within_proposed_target = (
                comp.overhead_ms <= PROPOSED_LATENCY_OVERHEAD_MS
            )
            all_overheads.append(comp.overhead_ms)

        except Exception as exc:
            comp.errors.append(str(exc))

        comparison_results.append(comp)

    mean_overhead = float(np.mean(all_overheads)) if all_overheads else 0.0
    return comparison_results, mean_overhead


# ---------------------------------------------------------------------------
# Report Serialization
# ---------------------------------------------------------------------------

def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively converts dataclass instances and numpy types to JSON-safe dicts."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def save_report(report: EvaluationReport, output_path: Path) -> None:
    """Saves the evaluation report as structured JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = _dataclass_to_dict(report)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[B4.2] Report saved: {output_path}")


# ---------------------------------------------------------------------------
# Main Evaluation Runner
# ---------------------------------------------------------------------------

def run_evaluation(
    max_frames: int = 30,
    output_path: Optional[Path] = None,
    seed: int = EVAL_SEED,
    verbose: bool = True,
) -> EvaluationReport:
    """
    Executes the full B4.2 gate evaluation and returns a structured report.

    Args:
        max_frames: Maximum number of real video frames to load.
        output_path: Path to write JSON results. If None, uses default.
        seed: Random seed for reproducibility.
        verbose: Whether to print progress to stdout.

    Returns:
        EvaluationReport — structured results with all measurable metrics.
    """
    import datetime

    out_path = output_path or DEFAULT_OUTPUT_PATH
    report = EvaluationReport(
        date=datetime.datetime.now().isoformat(),
        seed=seed,
        latency_overhead_proposed_target_ms=PROPOSED_LATENCY_OVERHEAD_MS,
    )

    if verbose:
        print("=" * 70)
        print("B4.2 — Real-Data Orientation & Robustness Evaluation")
        print("ISRO SIH26174 BAS Experiment Monitor — Workstream B")
        print("=" * 70)
        print(f"[CONFIG] Seed: {seed} | Max frames: {max_frames}")
        print(f"[CONFIG] Evaluation mode: SYNTHETIC_STRESS_TEST")
        print()

    # -----------------------------------------------------------------------
    # Step 1: Load evaluation frames
    # -----------------------------------------------------------------------
    if verbose:
        print("[STEP 1/4] Loading evaluation frames...")
    frames, dataset_used, dataset_status = load_evaluation_frames(
        max_frames=max_frames, seed=seed
    )

    report.dataset_used = dataset_used
    report.dataset_status = dataset_status
    report.frames_attempted = max_frames
    report.frames_successfully_loaded = len(frames)
    report.frame_source_classes = sorted(
        list(set(f.source_class.replace("_SYNTHETIC_FALLBACK", "") for f in frames))
    )

    if verbose:
        print(f"  Dataset: {dataset_used}")
        print(f"  Frames loaded: {len(frames)}")
        print(f"  Source classes: {report.frame_source_classes}")
        print()

    # -----------------------------------------------------------------------
    # Step 2: Orientation condition evaluation (all 7 angles)
    # -----------------------------------------------------------------------
    if verbose:
        print("[STEP 2/4] Evaluating orientation conditions (all 7 canonical angles)...")

    orientation_results, mean_aug_ms, max_aug_ms = evaluate_orientation_conditions(
        frames, seed=seed
    )
    report.orientation_results = [
        {
            "angle_deg": r.angle_deg,
            "category": r.category,
            "augmentation_type": r.augmentation_type,
            "output_shape_valid": r.output_shape_valid,
            "pixel_range_valid": r.pixel_range_valid,
            "frame_not_all_black": r.frame_not_all_black,
            "annotations_evaluated": r.annotations_evaluated,
            "box_coords_in_bounds": r.box_coords_in_bounds,
            "keypoints_in_bounds": r.keypoints_in_bounds,
            "augmentation_time_ms": round(r.augmentation_time_ms, 4),
            "errors": r.errors,
            # Explicitly blocked metrics
            "precision": None,
            "recall": None,
            "f1": None,
            "metric_status": "BLOCKED — no trained model or labeled EXP-001 dataset",
        }
        for r in orientation_results
    ]
    report.all_orientations_evaluated = len(orientation_results) == 7
    report.mean_augmentation_time_ms = round(mean_aug_ms, 4)
    report.max_augmentation_time_ms = round(max_aug_ms, 4)

    if verbose:
        for r in orientation_results:
            status = "OK" if (r.output_shape_valid and r.pixel_range_valid and r.frame_not_all_black) else "WARN"
            print(
                f"  [{status}] {r.angle_deg:>3}° ({r.category:<20}) "
                f"shape={r.output_shape_valid} "
                f"range={r.pixel_range_valid} "
                f"content={r.frame_not_all_black} "
                f"boxes={r.box_coords_in_bounds} "
                f"kps={r.keypoints_in_bounds} "
                f"time={r.augmentation_time_ms:.2f}ms"
            )
        print(f"  Mean aug time: {mean_aug_ms:.3f} ms | Max: {max_aug_ms:.3f} ms")
        print()

    # -----------------------------------------------------------------------
    # Step 3: Robustness dimensions evaluation
    # -----------------------------------------------------------------------
    if verbose:
        print("[STEP 3/4] Evaluating all 7 robustness dimensions...")

    dim_results = evaluate_robustness_dimensions(frames, seed=seed)
    report.robustness_dimension_results = [
        {
            "dimension": r.dimension,
            "augmentation_time_ms": round(r.augmentation_time_ms, 4),
            "output_shape_valid": r.output_shape_valid,
            "pixel_range_valid": r.pixel_range_valid,
            "frame_not_all_black": r.frame_not_all_black,
            "errors": r.errors,
        }
        for r in dim_results
    ]

    if verbose:
        for r in dim_results:
            status = "OK" if (r.output_shape_valid and r.pixel_range_valid) else "WARN"
            print(
                f"  [{status}] {r.dimension:<15} "
                f"shape={r.output_shape_valid} "
                f"range={r.pixel_range_valid} "
                f"content={r.frame_not_all_black} "
                f"time={r.augmentation_time_ms:.3f}ms"
            )
        print()

    # -----------------------------------------------------------------------
    # Step 4: Rectification comparison
    # -----------------------------------------------------------------------
    if verbose:
        print("[STEP 4/4] Evaluating rectification preprocessing overhead...")

    rect_results, mean_overhead = evaluate_rectification_comparison(frames, seed=seed)
    report.rectification_comparison = [
        {
            "angle_deg": r.angle_deg,
            "baseline_time_ms": round(r.baseline_time_ms, 4),
            "rectified_time_ms": round(r.rectified_time_ms, 4),
            "overhead_ms": round(r.overhead_ms, 4),
            "overhead_within_proposed_target": r.overhead_within_proposed_target,
            "proposed_target_ms": r.proposed_target_ms,
            "errors": r.errors,
        }
        for r in rect_results
    ]
    report.mean_rectification_overhead_ms = round(mean_overhead, 4)

    if verbose:
        for r in rect_results:
            target_ok = "<= target" if r.overhead_within_proposed_target else "> PROPOSED TARGET"
            print(
                f"  {r.angle_deg:>3}° baseline={r.baseline_time_ms:.3f}ms "
                f"rectified={r.rectified_time_ms:.3f}ms "
                f"overhead={r.overhead_ms:.3f}ms [{target_ok}]"
            )
        print(f"  Mean rectification overhead: {mean_overhead:.3f} ms")
        print()

    # -----------------------------------------------------------------------
    # Gate outcome determination
    # -----------------------------------------------------------------------
    preprocessing_checks = [
        report.all_orientations_evaluated,
        all(r["output_shape_valid"] for r in report.orientation_results),
        all(r["pixel_range_valid"] for r in report.orientation_results),
        all(r["frame_not_all_black"] for r in report.orientation_results),
        all(r["output_shape_valid"] for r in report.robustness_dimension_results),
        len(report.robustness_dimension_results) == 7,
    ]
    report.preprocessing_eval_passed = all(preprocessing_checks)
    report.model_eval_passed = False  # Structurally blocked

    if report.preprocessing_eval_passed:
        report.gate_status = "PARTIALLY_EVALUATED"
        report.gate_notes = (
            "Preprocessing evaluation PASSED: "
            "All 7 orientation angles validated, all 7 robustness dimensions validated, "
            "rectification overhead measured. "
            "Model evaluation BLOCKED: No Part-1 trained checkpoint present in repository. "
            "Precision / Recall / F1 metrics require real model and labeled EXP-001 dataset. "
            "Gate B4.2 is PARTIALLY_EVALUATED pending model and data availability."
        )
    else:
        report.gate_status = "PREPROCESSING_FAILED"
        report.gate_notes = (
            "One or more preprocessing validation checks failed. "
            "Model evaluation also BLOCKED. See individual result entries for details."
        )

    # -----------------------------------------------------------------------
    # Print final summary
    # -----------------------------------------------------------------------
    if verbose:
        print("=" * 70)
        print(f"[SUMMARY] Gate B4.2 Status: {report.gate_status}")
        print(f"[SUMMARY] Preprocessing eval: {'PASSED' if report.preprocessing_eval_passed else 'FAILED'}")
        print(f"[SUMMARY] Model eval: BLOCKED — no checkpoint / no EXP-001 dataset")
        print(f"[SUMMARY] All 7 angles evaluated: {report.all_orientations_evaluated}")
        print(f"[SUMMARY] Mean aug time: {report.mean_augmentation_time_ms:.3f} ms")
        print(f"[SUMMARY] Mean rectification overhead: {report.mean_rectification_overhead_ms:.3f} ms")
        print(f"[SUMMARY] Per-angle F1: None (BLOCKED)")
        print(f"[SUMMARY] Macro F1: None (BLOCKED)")
        print("=" * 70)
        print()

    save_report(report, out_path)
    return report


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="B4.2 — Orientation & Robustness Evaluation Harness"
    )
    parser.add_argument(
        "--max-frames", type=int, default=30,
        help="Maximum number of real frames to load for evaluation (default: 30)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write JSON results (default: evaluation/results/b4_2_orientation_robustness_results.json)",
    )
    parser.add_argument(
        "--seed", type=int, default=EVAL_SEED,
        help=f"Random seed for reproducibility (default: {EVAL_SEED})",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else None
    run_evaluation(
        max_frames=args.max_frames,
        output_path=out_path,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
