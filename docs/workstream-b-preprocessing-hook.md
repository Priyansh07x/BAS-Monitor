# B4.1c.2 — Preprocessing Integration Hook

> **Gate:** B4.1c.2 — Preprocessing Integration Hook  
> **Status:** IMPLEMENTED  
> **Date:** 2026-09-24  
> **Module:** [`backend/video/frame_processor.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/video/frame_processor.py)  
> **Test Suite:** [`tests/test_preprocessing_rectification_hook.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_preprocessing_rectification_hook.py) (26/26 PASS)  

---

## Purpose

The **Preprocessing Integration Hook** integrates the deterministic camera mounting calibration core ([`backend/video/camera_rectification.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/video/camera_rectification.py)) into the standard vision preprocessing pipeline ([`backend/video/frame_processor.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/video/frame_processor.py)).

It establishes a clean, optional, non-breaking hook for converting raw physical camera frames into rectified upright representations prior to downstream feature extraction and object/pose detection.

> [!IMPORTANT]
> **This hook enables optional camera rectification in preprocessing. It does NOT:**
> - Prove orientation robustness on real physical data.
> - Infer astronaut posture or dynamically rotate frames based on human orientation.
> - Implement 3D payload-relative coordinate reasoning (deferred to Gate B15).
> - Replace real-data evaluation (deferred to Gate B4.2).
> - Integrate the Part-1 binary classification model.

---

## Existing Frame Processing Flow

Prior to Gate B4.1c.2, frame preprocessing was strictly direct:

```text
Raw Camera Frame (BGR)
         ↓
  BGR -> RGB Conversion
         ↓
  Letterboxing / Resizing / Normalization
         ↓
  Downstream Perception (YOLO / Pose / 1D-TCN)
```

In this baseline flow, any fixed physical mounting tilt or camera roll directly skewed the coordinate space fed into the neural network backbones.

---

## New Optional Rectification Stage

With Gate B4.1c.2, an optional camera rectification hook is introduced **before** color conversion and model-specific letterbox/resizing:

```text
Raw Camera Frame (BGR)
         ↓
Optional Camera Rectification Hook (CameraRectifier)
  ├── Enabled == false (Default): Identity bypass (zero modification, zero overhead)
  └── Enabled == true: Deterministic 3x3 geometric rectification (mounting tilt/roll correction)
         ↓
Rectified Frame (BGR)
         ↓
FrameProcessor Operations:
  ├── preprocess_for_detector(): BGR -> RGB + Letterbox (640x640)
  ├── preprocess_for_pose(): BGR -> RGB + Resize (256x256) + Float32 Normalization [0, 1]
  └── process_frame_pipeline(): Unified coordination + Synchronized annotation mapping
         ↓
Downstream Perception
```

---

## Configuration

The rectification hook reads configuration from the `"rectification"` section in [`config/settings.json`](file:///E:/Technical_Projects/2026-SIH/SIH26174/config/settings.json):

```json
{
  "camera": {
    "source": 0,
    "width": 1280,
    "height": 720,
    "fps": 30.0,
    "buffer_size": 1
  },
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

The hook accepts `rectifier` as:
1. `CameraRectifier` instance.
2. `RectificationConfig` dataclass.
3. `dict` configuration mapping.
4. `None` (default).

---

## Disabled / Identity Behavior

When `rectifier` is `None`, or when `rectification.enabled == false`, the hook performs an instant identity bypass:
- `FrameProcessor.rectify_frame(frame, rectifier=None)` returns `frame` directly.
- `FrameProcessor.preprocess_for_detector(frame)` and `FrameProcessor.preprocess_for_pose(frame)` produce identical outputs to pre-B4.1c implementations down to the exact byte.
- Zero extra memory allocations or pixel interpolation calculations occur.

---

## Annotation Handling

Coordinate consistency across preprocessing is strictly preserved:

1. **Pre-Rectification Annotations:** If bounding boxes or keypoints are supplied before rectification (e.g. synthetic test annotations or pre-calibrated proposals), `process_frame_pipeline()` transforms them through the exact same $3 \times 3$ matrix $M$, clipping bounding box envelopes to $[0.0, 1.0]$ and preserving keypoint depth $z$.
2. **Post-Rectification Detections:** Downstream detectors (YOLO, PoseDetector, HandDetector) operating on rectified frames naturally produce detections directly in rectified coordinates.
3. **Double-Transformation Prevention:** Annotations are never transformed twice.

---

## Error Handling

- **Type Validation:** Passing invalid non-rectifier objects (e.g. integers or incompatible objects) to `FrameProcessor` methods raises a descriptive `TypeError`.
- **Singular Matrix / Invalid Calibration:** If a custom homography or affine matrix is mathematically non-invertible, `CameraRectifier` falls back safely to returning the original frame without crashing runtime threads.

---

## Backward Compatibility

1. **Method Signatures:** `preprocess_for_detector`, `preprocess_for_pose`, `letterbox`, `extract_keypoint_vector`, and `draw_annotations` retain complete backward compatibility. The `rectifier` parameter is strictly optional and defaults to `None`.
2. **Zero ML Dependencies:** Pure NumPy fallback functions (`_pure_numpy_resize`, `_pure_numpy_bgr_to_rgb`) ensure `FrameProcessor` imports and executes in minimal environments where OpenCV (`cv2`) is absent.

---

## Current Limitations

1. **Uncalibrated Rig Parameters:** Physical camera mounting angles on the flight/testing rig are currently unspecified (`calibration_status: "UNSPECIFIED"`).
2. **Decoupled AI Models:** Real YOLO weights and Part-1 1D-TCN classifiers remain decoupled while Part 1 is in active revision in Colab.

---

## B4.2 Dependency

- **Gate B4.2 (Real-Data Orientation & Robustness Evaluation):** Will utilize this preprocessing hook when evaluating multi-angle performance across all 7 canonical angles ($0^\circ, 45^\circ, 90^\circ, 135^\circ, 180^\circ, 225^\circ, 270^\circ$) on real recorded EXP-001 video sequences once collected under the B3 protocol.

---

## Acceptance Criteria

- [x] Clean, optional rectification hook in `backend/video/frame_processor.py`.
- [x] Zero change to default preprocessing behavior when disabled (`rectification.enabled == false`).
- [x] Configurable via `config/settings.json` `"rectification"` block.
- [x] Support for all 7 canonical angles ($0^\circ, 45^\circ, 90^\circ, 135^\circ, 180^\circ, 225^\circ, 270^\circ$).
- [x] Unified pipeline method (`process_frame_pipeline`) with synchronized annotation mapping.
- [x] Full backward compatibility for existing callers.
- [x] 26/26 unit tests passing in `tests/test_preprocessing_rectification_hook.py`.
