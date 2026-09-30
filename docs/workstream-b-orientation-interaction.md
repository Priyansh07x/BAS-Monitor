# B4.1b — Orientation-Robust Interaction Logic

**Date:** 2026-09-24  
**Gate:** B4.1b — Orientation-Robust Interaction Logic  
**Author:** Workstream B (AI / Procedure Intelligence)  
**Status:** IMPLEMENTED — RUNTIME INTEGRATED (GEOMETRIC TOLERANCE ENHANCED)

---

## Purpose

This document specifies the upgraded mathematical and spatial logic in [`backend/experiment/interaction_logic.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/experiment/interaction_logic.py). It addresses the geometric degradation of 2D axis-aligned bounding box Intersection over Union (IoU) under diagonal camera roll ($45^\circ, 135^\circ, 225^\circ$) and camera distance scaling, establishing an orientation-tolerant spatial interaction criterion.

> [!IMPORTANT]
> **Operational Scope & Physical Reality:**
> - This implementation improves the **geometric tolerance** of the interaction engine against 2D coordinate rotations.
> - It does **NOT** establish or prove real-world orientation robustness on physical microgravity video.
> - Real orientation evaluation remains pending in Gate B4.2.

---

## Existing Problem

In the baseline implementation of [`backend/experiment/interaction_logic.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/experiment/interaction_logic.py):
1. **Axis-Aligned Bounding Box Dilation:** 2D bounding boxes $(x_1, y_1, x_2, y_2)$ are axis-aligned. When an object or hand is rotated diagonally ($45^\circ, 135^\circ, 225^\circ$), the axis-aligned envelope artificially expands in area by up to $\sqrt{2}\approx 1.41\times$.
2. **IoU Collapse on Angled Contact:** When an astronaut's hand grasps an elongated or rotated sample container from an oblique angle, the intersection area drops relative to the inflated union area, causing $\text{IoU} < 0.05$ even during valid physical grasp.
3. **Fixed Absolute Distance Threshold:** The proximity threshold ($0.150$) was an absolute normalized coordinate constant, failing to scale adaptively when the camera moves closer or farther from the holding rack.

---

## Existing Annotation Representation

Per the existing perception architecture ([`ObjectDetector`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/object_detector.py), [`HandDetector`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/hand_detector.py)):
- **Hand Boxes:** `List[Dict[str, Any]]` with normalized coordinates `x1, y1, x2, y2` in $[0.0, 1.0]$.
- **Object Boxes:** `List[Dict[str, Any]]` with normalized coordinates `x1, y1, x2, y2` and `label` (`RED_SAMPLE`, `BLUE_SAMPLE`, `SAMPLE_CONTAINER`, `CONTAINER_LID`).
- **Hand Landmarks (Optional):** `List[Dict[str, float]]` with normalized coordinates `x, y, z`.

*Note: Detector outputs in the active repository are strictly axis-aligned 2D boxes; oriented bounding box (OBB) angle parameters are not present in upstream detector models.*

---

## Chosen Interaction Evidence

The upgraded interaction engine combines three complementary sources of geometric evidence:

1. **Primary Evidence — Rotation-Invariant Euclidean Centroid Distance:**
   $$d_{\text{centroid}} = \sqrt{(h_x - o_x)^2 + (h_y - o_y)^2}$$
   In $\mathbb{R}^2$, the Euclidean distance between hand center $(h_x, h_y)$ and object center $(o_x, o_y)$ is **strictly invariant under arbitrary 2D rotations**.

2. **Characteristic Scale Normalization:**
   To account for object size and camera distance, each bounding box defines a characteristic radius (half-diagonal):
   $$r_{\text{hand}} = \frac{1}{2}\sqrt{w_{\text{hand}}^2 + h_{\text{hand}}^2}, \quad r_{\text{obj}} = \frac{1}{2}\sqrt{w_{\text{obj}}^2 + h_{\text{obj}}^2}$$
   The scale-normalized proximity ratio is:
   $$S_{\text{contact}} = \frac{d_{\text{centroid}}}{\max(0.01, r_{\text{hand}} + r_{\text{obj}})}$$
   When $S_{\text{contact}} \le 0.85$, the hand is physically within the grasping envelope of the object regardless of orientation angle.

3. **Secondary Evidence — Axis-Aligned IoU:**
   $$\text{IoU} = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$$
   Retained for fast direct overlap confirmation in standard upright scenes.

4. **Landmark Fingertip Refinement:**
   When MediaPipe/Hailo hand landmark arrays are provided, the minimum Euclidean distance from any fingertip keypoint $(l_x, l_y)$ to the object center provides direct physical contact evidence.

---

## Normalized Proximity Strategy

- **Characteristic Reach Envelope:**
  $$d_{\text{edge}} = \max(0.0, d_{\text{centroid}} - (r_{\text{hand}} + r_{\text{obj}}))$$
- **Thresholds (Labeled as PROPOSED DEFAULTS):**
  - `contact_scale_threshold = 0.85` (`PROPOSED DEFAULT — REQUIRES EXPERIMENTAL VALIDATION`)
  - `reach_scale_threshold = 2.00` (`PROPOSED DEFAULT — REQUIRES EXPERIMENTAL VALIDATION`)
  - `proximity_threshold = 0.150` (Preserved baseline)
  - `iou_contact_threshold = 0.050` (Preserved baseline)

---

## IoU Role

Axis-aligned IoU is **not** treated as an orientation-invariant metric. Instead:
- In $0^\circ$ upright scenarios, $\text{IoU} \ge 0.05$ acts as an immediate positive trigger for `HOLDING`.
- In rotated scenarios ($45^\circ, 135^\circ, 225^\circ$), when IoU drops below threshold due to bounding envelope expansion, the scale-normalized centroid proximity $S_{\text{contact}} \le 0.85$ sustains the true `HOLDING` classification without false dropouts.

---

## Decision Logic

```
                    ┌───────────────────────────────┐
                    │ Compute Centroid Distance (d) │
                    │ & Scale Proximity (S_contact) │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                     /─────────────────────────────\
                    <   S_contact <= 0.85           >
                    <   OR d < 0.075                >
                    <   OR IoU >= 0.05              >
                     \─────────────────────────────/
                                    │
                         YES ┌──────┴──────┐ NO
                             │             │
                             ▼             ▼
                      ┌─────────────┐ /─────────────────────────────\
                      │   HOLDING   │<   S_contact <= 2.00           >
                      └─────────────┘<   OR d_edge <= 0.150          >
                                      \─────────────────────────────/
                                                    │
                                         YES ┌──────┴──────┐ NO
                                             │             │
                                             ▼             ▼
                                      ┌─────────────┐ ┌─────────────┐
                                      │ APPROACHING │ │    IDLE     │
                                      └─────────────┘ └─────────────┘
```

---

## Ambiguous Cases & Fail-Safe Behavior

- When a hand is near an object boundary but $S_{\text{contact}} > 0.85$, the engine strictly classifies the state as `APPROACHING` rather than prematurely jumping to `HOLDING`.
- When no hands or objects are detected, the engine defaults safely to `IDLE` with `confidence = 0.0`.
- Hand-object association is generic and applies uniformly across all canonical objects (`RED_SAMPLE`, `BLUE_SAMPLE`, `SAMPLE_CONTAINER`, `CONTAINER_LID`) without object-specific hardcoding.

---

## Backward Compatibility

The public API and return dictionary of `InteractionEngine` remain $100\%$ backward-compatible with [`backend/ai/inference_pipeline.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/inference_pipeline.py):

```python
{
    "state": "HOLDING",
    "target_object": "RED_SAMPLE",
    "confidence": 0.95,
    "closest_distance": 0.042,
    "max_iou": 0.12,
    "normalized_scale_proximity": 0.450
}
```

---

## Synthetic Test Coverage

The test suite in [`tests/test_interaction_orientation.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_interaction_orientation.py) verifies:
- **7 Canonical Angles ($0^\circ, 45^\circ, 90^\circ, 135^\circ, 180^\circ, 225^\circ, 270^\circ$):** Grasps and approaching states maintain correct classification under synthetic scene rotations.
- **Scale Variations ($0.8\times\text{--}1.2\times$):** Zooming maintains stable state transitions.
- **Translations ($\pm 10\%$):** Position offsets preserve relative proximity.
- **Separation & Ambiguity:** Distant items remain `IDLE`; ambiguous bounds remain `APPROACHING`.
- **Landmark Refinement:** Direct fingertip contact upgrades state to `HOLDING`.

---

## Current Limitations

1. **2D Projection Only:** Does not account for true 3D depth separation along optical $Z$-axis (requires 3D pose mesh in Gate B14/B15).
2. **Upstream Detector Dependency:** If upstream YOLO / MediaPipe fails to detect a rotated hand, the interaction engine cannot evaluate proximity.

---

## B4.1c Dependency

Gate B4.1c will implement runtime camera rectification / orientation-aware preprocessing in [`backend/video/frame_processor.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/video/frame_processor.py) to rotate incoming video frames to canonical upright coordinates prior to detector inference when mounting angles are configured.

---

## B4.2 Dependency

Gate B4.2 will benchmark end-to-end multi-angle procedure monitoring against held-out real test videos.

---

## Acceptance Criteria

- [x] Orientation-tolerant interaction engine implemented in `backend/experiment/interaction_logic.py`.
- [x] Euclidean centroid distance utilized as primary rotation-invariant evidence.
- [x] Scale-normalized reach and contact ratios formulated and integrated.
- [x] Standard IoU preserved as secondary evidence for full backward compatibility.
- [x] Conservative fail-safe behavior verified (ambiguous geometry defaults to `APPROACHING`/`IDLE`).
- [x] All 7 canonical orientation angles verified ($0^\circ, 45^\circ, 90^\circ, 135^\circ, 180^\circ, 225^\circ, 270^\circ$).
- [x] Execution flow mapped in `docs/flow.md`.
- [x] Automated test suite passing ($28/28$ tests in `tests/test_interaction_orientation.py`).
