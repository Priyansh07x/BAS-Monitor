# B4.1a — Augmentation Core

**Date:** 2026-09-24  
**Gate:** B4.1a — Orientation & Robustness Augmentation Core  
**Author:** Workstream B (AI / Procedure Intelligence)  
**Status:** IMPLEMENTED — RUNTIME INTEGRATION PENDING (DATA/TRAINING CORE ONLY)

---

## Purpose

This document establishes the architecture, API, and synchronization contract for the reusable **Augmentation Core Engine** ([`backend/ai/augmentation.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/augmentation.py)). 

The module provides deterministic, multi-modal synthetic transformations across the 7 environmental disturbance dimensions established in Gate B4.0:
1. **Rotation** ($0^\circ, 45^\circ, 90^\circ, 135^\circ, 180^\circ, 225^\circ, 270^\circ$, and arbitrary continuous angles)
2. **Scale** (zoom in / zoom out)
3. **Translation** (horizontal and vertical spatial shift)
4. **Brightness** (gain and additive intensity shift)
5. **Blur** (spatial box and Gaussian blurring)
6. **Occlusion** (deterministic Cutout / masking block)
7. **Perspective** (trapezoidal homography / viewpoint tilt)

> [!IMPORTANT]
> **Operational Boundary:**
> - This module supports future offline dataset expansion and Part-1 model training augmentation.
> - It does **NOT** run during live inference in [`backend/ai/inference_pipeline.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/inference_pipeline.py).
> - It does **NOT** prove or validate physical orientation robustness on real microgravity video.
> - Real orientation evaluation remains pending in Gate B4.2.

---

## Ownership

- **Domain:** Data Preprocessing & Training-Time Synthetic Augmentation.
- **Module Location:** [`backend/ai/augmentation.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/augmentation.py)
- **Architectural Rationale:** Sits alongside AI models in `backend/ai/` to provide common data transformation utilities for training scripts, offline dataset preparation, and synthetic test suites without polluting runtime inference pipelines or FSM state machines.

---

## Supported Transformations

| Transformation | Method | Input Parameters | Geometric Synchronization |
|---|---|---|:---:|
| **Rotation** | `rotate()` | `angle_deg` ($0^\circ\text{--}360^\circ$) | **Yes** (Boxes + Keypoints) |
| **Scale** | `scale()` | `scale_factor` ($0.8\times\text{--}1.2\times$) | **Yes** (Boxes + Keypoints) |
| **Translation** | `translate()` | `dx_fraction`, `dy_fraction` ($\pm 10\%$) | **Yes** (Boxes + Keypoints) |
| **Brightness** | `adjust_brightness()` | `gain` ($0.8\text{--}1.2$), `bias` ($\pm 30$) | **No** (Image only) |
| **Blur** | `apply_blur()` | `kernel_size` (odd integer $\ge 3$) | **No** (Image only) |
| **Occlusion** | `apply_occlusion()` | `box_fraction` ($0.15$), `center`, `seed` | **No** (Image only) |
| **Perspective** | `apply_perspective()` | `distortion` ($0.05$), `seed` | **Yes** (Boxes + Keypoints) |
| **Composite** | `apply()` | `AugmentationConfig`, `seed` | **Yes** (Full pipeline) |

*Note: All numerical ranges above represent `PROPOSED DEFAULTS — REQUIRES VALIDATION`.*

---

## Annotation Synchronization

When geometric operations (Rotation, Scale, Translation, Perspective) modify image pixel coordinates, all corresponding annotations are transformed synchronously:

1. **Bounding Boxes (`{"x1", "y1", "x2", "y2", ...}`):**
   - All 4 corners of the source box $(x_1, y_1), (x_2, y_1), (x_2, y_2), (x_1, y_2)$ are projected through the 2D affine matrix $\mathbf{M}_{\text{forward}}$:
     $$\mathbf{p}' = \mathbf{M}_{\text{forward}} \mathbf{p}$$
   - The new bounding box is the axis-aligned minimum bounding envelope:
     $$x_1' = \min(x'_i), \quad y_1' = \min(y'_i), \quad x_2' = \max(x'_i), \quad y_2' = \max(y'_i)$$
   - Coordinates are clipped to $[0.0, 1.0]$.
   - Auxiliary dictionary keys (`label`, `confidence`, `track_id`) are preserved intact.

2. **Landmark Keypoints (`{"x", "y", "z", "visibility", ...}`):**
   - The $(x, y)$ coordinate pair is projected through $\mathbf{M}_{\text{forward}}$.
   - Depth coordinates ($z$), visibility scores, joint names, and handedness tags are preserved intact.

---

## Coordinate Convention

In strict alignment with existing repository perception modules ([`ObjectDetector`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/object_detector.py), [`PoseDetector`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/pose_detector.py), [`HandDetector`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/hand_detector.py)):

- **Normalized Coordinates:** All bounding box edges ($x_1, y_1, x_2, y_2$) and keypoints ($x, y$) operate in normalized floating-point range $[0.0, 1.0]$ relative to image width $W$ and height $H$.
- **Origin:** Top-left corner $(0.0, 0.0)$, with positive $X$ rightward and positive $Y$ downward.
- **Rotation Center:** Center of the frame $(\frac{W}{2}, \frac{H}{2})$ or $(0.50, 0.50)$ normalized.

---

## Determinism

When `seed` (integer) is provided in `AugmentationConfig` or passed to `apply()`, all pseudo-random selections (occlusion patch position, perspective distortion displacement, noise values) use an isolated `np.random.RandomState(seed)` instance.

This guarantees:
- $100\%$ bitwise reproducible test executions.
- Deterministic synthetic dataset batch generation.
- No side-effects on global Python or NumPy random states.

---

## Configuration

The module is driven by `AugmentationConfig`, directly reflecting parameters defined in `config/orientation_robustness_spec.json`:

```python
from backend.ai.augmentation import AugmentationConfig, AugmentationEngine

config = AugmentationConfig(
    enable_rotation=True,
    rotation_angle_deg=90.0,
    enable_scale=True,
    scale_factor=1.1,
    enable_brightness=True,
    brightness_gain=1.1,
    enable_blur=True,
    blur_kernel_size=3,
    enable_occlusion=True,
    occlusion_size_fraction=0.10,
    seed=42,
)
```

---

## Current Limitations

1. **Axis-Aligned Bounding Box Dilation:** Under non-orthogonal rotations ($45^\circ, 135^\circ, 225^\circ, 315^\circ$), converting transformed corners into axis-aligned envelopes expands the box area. Oriented Bounding Box (OBB) support will be addressed in Gate B4.1b.
2. **Zero ML Invariance Claims:** Synthetic rotation does not simulate variable lighting direction or true 3D limb occlusions.
3. **Pure CPU Execution:** NumPy matrix inverse mapping is vectorized and sub-millisecond for small/medium frames; OpenCV C++ acceleration is used automatically when `cv2` is available.

---

## B4.1b Dependencies

Gate B4.1b will build on this core to implement:
- **Orientation-Robust Interaction Logic:** Enhancing [`InteractionEngine`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/experiment/interaction_logic.py) to maintain accurate hand-object contact evaluation under diagonal rotations.
- **Scale-Invariant Centroid Distance:** Preventing IoU breakdown under bounding box dilation.

---

## Usage Example

```python
import numpy as np
from backend.ai.augmentation import AugmentationEngine, AugmentationConfig

engine = AugmentationEngine()

# Input frame and annotations
frame = np.zeros((480, 640, 3), dtype=np.uint8)
boxes = [{"label": "RED_SAMPLE", "x1": 0.20, "y1": 0.20, "x2": 0.40, "y2": 0.40}]
keypoints = [{"x": 0.30, "y": 0.30, "z": 0.0, "name": "RIGHT_WRIST"}]

# Apply 90-degree clockwise rotation
result = engine.rotate(frame, angle_deg=90.0, bounding_boxes=boxes, keypoints=keypoints)

print("Transformed Box:", result.bounding_boxes[0])
print("Transformed Keypoint:", result.keypoints[0])
```

---

## Acceptance Criteria

- [x] Augmentation core module implemented in `backend/ai/augmentation.py`.
- [x] All 7 environmental dimensions supported (Rotation, Scale, Translation, Brightness, Blur, Occlusion, Perspective).
- [x] All 7 canonical orientation angles supported ($0^\circ, 45^\circ, 90^\circ, 135^\circ, 180^\circ, 225^\circ, 270^\circ$).
- [x] Synchronized transformation of bounding boxes and landmark keypoints verified.
- [x] Normalized coordinate convention $[0.0, 1.0]$ preserved.
- [x] Deterministic seed control verified.
- [x] Zero external ML framework dependencies (pure Python + NumPy, optional OpenCV acceleration).
- [x] Zero runtime inference modification (inference pipeline untouched).
- [x] Automated test suite passing ($18/18$ tests).
