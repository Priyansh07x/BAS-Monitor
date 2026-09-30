# B4.1c.1 — Camera Rectification Core

> **Gate:** B4.1c.1 — Orientation-Aware Preprocessing / Camera Rectification Core  
> **Status:** IMPLEMENTED  
> **Date:** 2026-09-24  
> **Module:** [`backend/video/camera_rectification.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/video/camera_rectification.py)  
> **Test Suite:** [`tests/test_camera_rectification.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_camera_rectification.py) (26/26 PASS)  

---

## Purpose

The **Camera Rectification Core** provides a deterministic, reusable geometric transformation component designed to correct for known camera mounting tilt, optical roll, and perspective distortion before downstream perception models execute.

```text
Known camera/view calibration parameters (Rotation, Scale, Translation, Homography)
                                    ↓
      Deterministic 3x3 geometric forward / inverse transform
                                    ↓
   Rectified frame + Synchronized Annotations (Bounding Boxes & Keypoints)
```

This component establishes a pure mathematical rectification pipeline that executes independently of machine learning model weights, ensuring deterministic reproducibility.

---

## Camera Calibration vs Scene Orientation

A critical architectural distinction is maintained across Workstream B:

| Dimension | **Camera Calibration / Mounting Tilt (B4.1c.1)** | **Scene / Astronaut Orientation (B4 / B15)** |
|---|---|---|
| **Definition** | Fixed, static physical misalignment between camera sensor plane and payload work surface. | Dynamic, variable physical orientation of the astronaut or sample specimens in microgravity. |
| **Source** | Hardware bracket mounting tolerances, physical rig tilt ($0^\circ, 45^\circ, 90^\circ, 180^\circ$, etc.). | Free-floating movement, arbitrary astronaut vantage points in zero-G. |
| **Resolution Method** | Deterministic static geometric calibration transform (affine/homography). | Orientation-invariant interaction logic (B4.1b), robust feature representations (B5/B6), and 3D coordinate reasoning (B15). |
| **Automation** | Configured via calibration parameters (`settings.json` / calibration config). | **NEVER** automatically rotates frames based on speculative human posture heuristics. |

> [!IMPORTANT]
> **This component corrects known camera calibration/mounting transforms. It does NOT infer astronaut orientation, and it does NOT prove microgravity orientation robustness.**

---

## Existing Preprocessing

The repository provides [`backend/video/frame_processor.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/video/frame_processor.py) for runtime computer vision preprocessing:
- **`letterbox()`**: Aspect-ratio preserving resize and stride-multiple border padding.
- **`preprocess_for_detector()`**: BGR $\to$ RGB conversion and 640x640 letterbox for YOLO detectors.
- **`preprocess_for_pose()`**: Resizing and $[0.0, 1.0]$ float32 normalization for skeletal detectors.
- **`extract_keypoint_vector()`**: 1D feature flattening (33 3D joints + 4 object centroids).
- **`draw_annotations()`**: Sci-fi HUD bounding box and skeleton visualization.

The new `CameraRectifier` module complements `FrameProcessor` by executing **prior** to detector letterboxing and feature extraction, transforming raw sensor frames into upright rectified frames.

---

## Rectification Transform

The mathematical transformation is parameterized by a $3 \times 3$ homogeneous matrix $M \in \mathbb{R}^{3 \times 3}$ mapping source pixel coordinates $(x_s, y_s)$ to destination pixel coordinates $(x_d, y_d)$:

$$\begin{bmatrix} x_d \\ y_d \\ 1 \end{bmatrix} \sim M \begin{bmatrix} x_s \\ y_s \\ 1 \end{bmatrix}$$

### Composition of Canonical Transformation:
1. **Source Center Shift ($T_{\text{neg}}$):** Translate source center $(c_{x,s}, c_{y,s})$ to origin.
2. **Rotation & Scale ($R$):** Apply rotation angle $\theta$ and scale factor $s$:
   $$R = \begin{bmatrix} s \cos\theta & -s \sin\theta & 0 \\ s \sin\theta & s \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$
3. **Destination Center Shift & Translation ($T_{\text{pos}}$):** Translate to destination center $(c_{x,d}, c_{y,d})$ plus normalized offset $(t_x \cdot W_d, t_y \cdot H_d)$.
4. **Forward Composite Matrix:** $M = T_{\text{pos}} R T_{\text{neg}}$.
5. **Inverse Mapping:** Pixel resampling computes $M^{-1}$ using bilinear or nearest-neighbor interpolation to prevent sampling voids.

Dual execution engine:
- **OpenCV Engine:** Fast C++ hardware acceleration via `cv2.warpAffine` / `cv2.warpPerspective` when OpenCV is present.
- **Pure NumPy Engine:** Vectorized inverse grid mapping with boundary masks and border fill when running in lightweight environments without OpenCV.

---

## Supported Angles

The rectification component provides full support for the 7 canonical evaluation angles specified in [`docs/workstream-b-orientation-robustness.md`](file:///E:/Technical_Projects/2026-SIH/SIH26174/docs/workstream-b-orientation-robustness.md) and [`config/orientation_robustness_spec.json`](file:///E:/Technical_Projects/2026-SIH/SIH26174/config/orientation_robustness_spec.json):

| Angle | Category | Mathematical Transformation Behavior |
|:---:|:---:|---|
| $0^\circ$ | `NORMAL` | Identity no-op bypass. |
| $45^\circ$ | `MODERATE_ROTATION` | $45^\circ$ planar counter-rotation with diagonal bounding envelope projection. |
| $90^\circ$ | `EXTREME_ROTATION` | Orthogonal $90^\circ$ clockwise rotation (top-left $\to$ top-right). |
| $135^\circ$ | `EXTREME_ROTATION` | $135^\circ$ diagonal counter-rotation with coordinate normalization. |
| $180^\circ$ | `EXTREME_ROTATION` | Inverted camera view inversion (top-left $\to$ bottom-right). |
| $225^\circ$ | `EXTREME_ROTATION` | $225^\circ$ diagonal counter-rotation. |
| $270^\circ$ | `EXTREME_ROTATION` | Orthogonal $270^\circ$ clockwise ($90^\circ$ counter-clockwise) rotation. |

---

## Annotation Synchronization

When a rectified frame is produced, any pre-existing annotations (such as synthetic dataset labels or pre-detection proposals) are transformed synchronously:

### 1. Bounding Boxes (`[x1, y1, x2, y2]` in $[0.0, 1.0]$)
Each bounding box's 4 normalized corners are projected through $M$:
$$(x'_i, y'_i) = \text{Project}(x_i, y_i, M)$$
The new axis-aligned bounding envelope is computed and clamped to $[0.0, 1.0]$:
$$x'_1 = \min(x'_i), \quad y'_1 = \min(y'_i), \quad x'_2 = \max(x'_i), \quad y'_2 = \max(y'_i)$$

### 2. Landmark Keypoints (`[{'x': ..., 'y': ..., 'z': ...}]`)
Normalized $x, y$ coordinates are projected through $M$ and clamped to $[0.0, 1.0]$. The depth / relative elevation coordinate $z$, detection confidence, and visibility attributes are **strictly preserved** without distortion.

---

## Configuration

Configuration is managed via `RectificationConfig` dataclass and JSON serializable dictionaries:

```json
{
  "rectification": {
    "enabled": false,
    "rotation_deg": 0.0,
    "scale": 1.0,
    "translation_x": 0.0,
    "translation_y": 0.0,
    "affine_matrix": null,
    "homography_matrix": null,
    "output_size": null,
    "border_mode": "constant",
    "border_value": [0, 0, 0],
    "interpolation": "bilinear",
    "calibration_status": "UNSPECIFIED",
    "calibration_notes": "PROPOSED DEFAULT — REQUIRES EXPERIMENTAL CALIBRATION"
  }
}
```

---

## Default / No-op Behavior

- When `enabled: false` (the default setting), the rectifier activates an instant identity bypass.
- `rectify()` returns exact unmodified image frames and annotation lists with zero computation overhead.
- `is_identity()` returns `True`, ensuring that uncalibrated setups introduce zero latency and zero pixel degradation.

---

## Current Calibration Limitations

1. **No Physical Calibration Parameters Measured:** Flight and test rig mounting angles have not yet been physically surveyed. All non-zero rotation angles remain `PROPOSED DEFAULT — REQUIRES EXPERIMENTAL CALIBRATION`.
2. **Radial / Tangential Lens Distortion:** The core currently provides affine and homography rectification. Non-linear lens distortion coefficients ($k_1, k_2, p_1, p_2$) will be incorporated when specific camera lenses (e.g. wide-angle fisheye) are selected.
3. **Model Integration:** Model integration remains decoupled until the Part-1 model revision in Colab is completed.

---

## B4.1c.2 Dependency

- **Gate B4.1c.2 (Preprocessing Integration Hook):** Will provide an optional, non-breaking hook in `FrameProcessor` to apply `CameraRectifier` before detector letterboxing, strictly enabled only when `rectification.enabled == true`.

---

## B4.2 Dependency

- **Gate B4.2 (Real-Data Robustness Evaluation):** Will benchmark multi-angle orientation performance across all 7 canonical angles on real recorded EXP-001 video sequences once collected under the B3 protocol.

---

## Acceptance Criteria

- [x] Dedicated module in `backend/video/camera_rectification.py`.
- [x] Zero ML framework dependencies; operates on pure NumPy with optional OpenCV acceleration.
- [x] Support for all 7 canonical angles ($0^\circ, 45^\circ, 90^\circ, 135^\circ, 180^\circ, 225^\circ, 270^\circ$).
- [x] Synchronized bounding box and keypoint annotation mapping.
- [x] Identity / no-op default when unconfigured or disabled.
- [x] Explicit disclaimer that physical calibration values are uncalibrated and astronaut orientation is not dynamically estimated.
- [x] Complete unit test suite with 26/26 tests passing in `tests/test_camera_rectification.py`.
