# Workstream B B4.0 — Orientation & Robustness Strategy Specification

**Date:** 2026-09-24  
**Gate:** B4.0 — Orientation & Robustness Strategy Specification  
**Author:** Workstream B (AI / Procedure Intelligence)  
**Status:** SPECIFICATION APPROVED — ACTUAL ROBUSTNESS EVALUATION PENDING (EXP-001 DATASET & MODEL NOT YET COLLECTED)

---

## 1. Scope

This specification establishes the **Orientation & Environmental Robustness Strategy** for the BAS Monitor Workstream B AI action-recognition system. It defines:

1. The exact **orientation evaluation angles** (0°, 45°, 90°, 135°, 180°, 225°, 270°) and categories (`NORMAL`, `MODERATE_ROTATION`, `EXTREME_ROTATION`).
2. The operational distinction between **data collection orientation**, **training-time synthetic augmentation**, and **held-out real evaluation data**.
3. Robustness strategies across all seven core environmental dimensions: **Rotation, Scale, Translation, Brightness, Blur, Occlusion, and Perspective**.
4. The **strict subject-level data leakage prevention policy**.
5. Formal **robustness evaluation metrics** (Precision, Recall, F1 per angle, macro-averages, relative degradation deltas from 0°, and preprocessing latency overhead).
6. The conceptual distinction between **2D camera/image orientation robustness** (Gate B4) and **3D payload-relative coordinate reasoning** (Gate B15).
7. An audit of **current repository implementation gaps** and forward requirements for Gates B4.1 (Implementation) and B4.2 (Evaluation).

> [!IMPORTANT]
> **No robustness evaluation has been performed in this gate.**
> As established in Gates B3 and B3.1, no EXP-001-specific video dataset has been collected, and the Part-1 model is currently decoupled and under modification. Therefore, no numerical benchmark results or performance claims are made in this specification.

---

## 2. Current Repository Reality

An audit of the active perception pipeline and video processing codebase reveals the following baseline reality:

| Component / File | Current Orientation Handling | Reality / Limitation |
|---|---|---|
| **Frame Preprocessor** ([`backend/video/frame_processor.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/video/frame_processor.py)) | Fixed Letterbox & Resizing | Converts BGR $\to$ RGB and scales to $(640, 640)$ or $(256, 256)$. Has **no rotation, affine, or orientation-aware transformations**. |
| **Object Detector** ([`backend/ai/object_detector.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/object_detector.py)) | Axis-Aligned 2D Bounding Boxes | YOLOv8/Hailo interface assumes standard upright coordinates. Simulated fallback hardcodes container at center $(0.50, 0.55)$ and sample vial at $(0.42, 0.43)$. |
| **Pose Detector** ([`backend/ai/pose_detector.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/pose_detector.py)) | Upright Body Pose Assumption | MediaPipe Pose assumes upright human anatomy. Synthetic fallback places joints around $(0.50, 0.45)$ upright. |
| **Hand Detector** ([`backend/ai/hand_detector.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/hand_detector.py)) | Upright Hand Landmarks | MediaPipe Hands detects articulations in image coordinates. Fallback assumes hands near $(0.51, 0.56)$. |
| **Interaction Engine** ([`backend/experiment/interaction_logic.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/experiment/interaction_logic.py)) | Axis-Aligned 2D IoU & Centroid Euclidean Distance | Standard axis-aligned IoU degrades when bounding boxes rotate diagonally ($45^\circ, 135^\circ$), introducing geometric distortion. |
| **Action Classifier** ([`backend/ai/action_classifier.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/action_classifier.py)) | Coordinate-Dependent Feature Vector | Flattens 33 3D keypoints + 4 object centroids into a 111-dim 1D vector. Rotations rotate $(x,y)$ coordinate values, altering the temporal distribution input to 1D-TCN. |
| **Augmentation Tools** | None Present | No training-time or evaluation-time video augmentation pipelines currently exist in the repository. |

---

## 3. Orientation Categories

In strict alignment with `config/dataset_spec.json`, orientation space is partitioned into three standardized categories:

| Category | Defined Degrees | Operational Description & Physical Rationale |
|---|:---:|---|
| `NORMAL` | $0^\circ$ | Standard upright camera orientation relative to payload holding rack and astronaut workspace. |
| `MODERATE_ROTATION` | $45^\circ, 135^\circ, 225^\circ$ | Diagonal camera mounting or oblique operator perspective relative to the rack axis. |
| `EXTREME_ROTATION` | $90^\circ, 180^\circ, 270^\circ$ | Sideways (transverse $90^\circ/270^\circ$) or fully inverted ($180^\circ$) camera or astronaut orientation in microgravity. |

---

## 4. Required Evaluation Angles

The system must be evaluated against exactly seven canonical angles specified by Workstream B:

```
                  0° (Normal Upright)
                   │
         315°      │      45° (Moderate Diagonal)
            \      │      /
             \     │     /
   270° ───────┼─────── 90° (Extreme Transverse)
 (Extreme     /    │     \  (Extreme Transverse)
Transverse)  /     │      \
           225°    │      135° (Moderate Diagonal)
                   │
                 180° (Extreme Inverted)
```

| Angle | Category | Test Scenario Representation |
|:---:|:---:|---|
| **$0^\circ$** | `NORMAL` | Baseline standard upright workspace capture. |
| **$45^\circ$** | `MODERATE_ROTATION` | Oblique upper-right diagonal viewpoint. |
| **$90^\circ$** | `EXTREME_ROTATION` | Clockwise sideways perspective ($90^\circ$ roll). |
| **$135^\circ$** | `MODERATE_ROTATION` | Lower-right diagonal viewpoint. |
| **$180^\circ$** | `EXTREME_ROTATION` | Fully inverted ceiling-mount or inverted operator view ($180^\circ$ roll). |
| **$225^\circ$** | `MODERATE_ROTATION` | Lower-left diagonal viewpoint. |
| **$270^\circ$** | `EXTREME_ROTATION` | Counter-clockwise sideways perspective ($270^\circ$ roll / $-90^\circ$). |

---

## 5. Training-Time Augmentation Strategy

Synthetic training-time augmentations expand model invariance during neural training without requiring exhaustive physical recording sessions for every possible variation.

### Principles

1. **Stochastic Online Transformation:** Applied dynamically per batch during Part-1 training.
2. **Keypoint & Bounding Box Synchronization:** When an image frame is rotated or translated, all corresponding 2D bounding boxes and 3D pose/hand keypoints must undergo identical spatial affine matrix transformations:
   $$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \mathbf{M}_{\text{affine}} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
3. **Temporal Consistency Across Window:** When a 30-frame temporal clip is augmented, the **exact same transformation parameters** (angle, scale factor, lighting shift) must be applied across all 30 contiguous frames to preserve coherent physical motion vectors.

---

## 6. Real-Data Evaluation Strategy

> [!CAUTION]
> **Synthetic Augmentation $\neq$ Real Robustness Validation:**
> Applying mathematical $90^\circ$ rotation to an upright video does not replicate true microgravity physics, real camera perspective changes, or physical shadows cast by re-oriented light sources.

### Evaluation Protocol

1. **Held-Out Real Recordings:** Final robustness verification (Gate B4.2) requires real video recordings captured under distinct physical angles ($0^\circ, 90^\circ, 180^\circ$, etc.).
2. **Synthetic Stress-Testing as Intermediate Check:** In the absence of full multi-angle physical datasets, synthetic rotations on held-out `TEST` split clips serve as an intermediate diagnostic tool, clearly marked as `SYNTHETIC_STRESS_TEST`.
3. **Zero Contamination:** Test split clips must never be seen during training in either raw or synthetically rotated forms.

---

## 7. Other Robustness Dimensions

Beyond 2D rotation, the Workstream B perception system must withstand six additional real-world microgravity and operational disturbances:

| Dimension | Purpose | Training Augmentation | Evaluation Presence | Risks of Unrealistic Augmentation | Validation Method |
|---|---|:---:|:---:|---|---|
| **Rotation** | Invariance to camera roll and operator tilt | Proposed ($0^\circ \text{--} 360^\circ$) | Yes ($0^\circ, 45^\circ, 90^\circ, 135^\circ, 180^\circ, 225^\circ, 270^\circ$) | Black border artifacts from unpadded corners. | Evaluated across 7 discrete canonical angles. |
| **Scale** | Robustness to camera distance ($<1\text{m}$ to $>2\text{m}$) | Proposed ($0.8\times \text{--} 1.2\times$) | Yes (Close, Medium, Far) | Unnatural resolution loss or cropping of critical hand-object contact. | Scale-stratified recall metrics. |
| **Translation** | Invariance to workspace shift within camera FOV | Proposed ($\pm 10\%$ shift) | Yes (Center, Off-center) | Truncating target objects out of frame boundary. | Boundary-proximity detection tests. |
| **Brightness** | Robustness to payload bay illumination shifts | Proposed ($\pm 20\%$ gain/gamma) | Yes (`BRIGHT`, `STANDARD`, `DIM`, `SHADOW`) | Over-saturation causing loss of sample color discrimination (`RED` vs `BLUE`). | Color-confusability matrix under dim lighting. |
| **Blur** | Invariance to rapid astronaut hand motion & camera jitter | Proposed (Gaussian blur $\sigma \in [0.5, 1.5]$) | Yes (`EVAL_NORMAL` vs motion blur) | Extreme blur obliterating fine finger-latch interactions. | Fast-motion action recall benchmarks. |
| **Occlusion** | Robustness to hand occluding sample or container opening | Proposed (Cutout / Random Erasing $\le 15\%$ area) | Yes (`NONE`, `PARTIAL`, `FULL`) | Occluding $100\%$ of object makes classification physically impossible. | Occlusion-level degradation curve. |
| **Perspective** | Invariance to oblique/overhead camera mount pitch | Proposed (Perspective warp tilt $\le 15^\circ$) | Yes (`FRONT`, `SIDE`, `OVERHEAD`, `OBLIQUE`) | Unrealistic keystone distortions destroying geometric proportions. | Viewpoint-specific F1 evaluation. |

*Note: All numerical ranges above are `PROPOSED — REQUIRES VALIDATION` during implementation.*

---

## 8. Evaluation Subsets

To ensure targeted diagnostics, the future held-out `TEST` set is partitioned into dedicated evaluation subsets matching `config/dataset_spec.json`:

| Evaluation Subset | Target Condition Tested | Specific Measurement Goal |
|---|---|---|
| `EVAL_NORMAL` | Standard $0^\circ$ upright, standard lighting, no occlusion. | Establish the **unperturbed baseline accuracy** ($F1_{\text{baseline}}$). |
| `EVAL_ORIENTATION` | Discrete angles ($45^\circ, 90^\circ, 135^\circ, 180^\circ, 225^\circ, 270^\circ$). | Measure **angular degradation delta** ($\Delta F1_\theta = F1_\theta - F1_0$). |
| `EVAL_OCCLUSION` | Partial and full hand-object visual occlusions. | Measure **contact-retention robustness** during grasp/placement. |
| `EVAL_LIGHTING` | Bright, Dim, and Shadow conditions. | Measure **color-separation stability** (`RED_SAMPLE` vs `BLUE_SAMPLE`). |
| `EVAL_VIEWPOINT` | Side, Overhead, and Oblique camera angles. | Measure **geometric perspective invariance**. |
| `EVAL_FAILURE` | Procedural error sequences (`FAILURE_WRONG_ACTION`, etc.). | Verify FSM anomaly detection sensitivity under disturbance. |
| `EVAL_INCOMPLETE` | Partial/aborted actions (`INCOMPLETE_APPROACH`, etc.). | Verify uncertainty gating and false-positive suppression. |

---

## 9. Leakage Prevention Policy

To maintain scientific validity, the **subject-level split strategy** defined in Gate B3 must be strictly enforced:

```
[Physical Subject X] ────► Assigned to ONE Split (e.g., TEST)
        │
        ├── Raw Recording 0° (TEST)
        ├── Real Recording 90° (TEST)
        └── Synthetic Augmentations (TEST ONLY)
```

### Core Leakage Invariants

1. **Source Subject Confinement:** All recordings, camera views, and takes from a given `subject_id` belong exclusively to exactly one split (`TRAIN`, `VALIDATION`, or `TEST`).
2. **Derivative Split Inheritance:** Any synthetically rotated, scaled, blurred, or transformed clip inherits the split of its parent source clip.
3. **Strict Ban on Cross-Split Derivatives:** It is strictly prohibited to place an unrotated clip in `TRAIN` and its synthetically rotated derivative in `TEST`.

---

## 10. Metrics

Future robustness evaluation (Gate B4.2) will report the following standardized metrics:

### 1. Per-Angle Performance

For each angle $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ, 180^\circ, 225^\circ, 270^\circ\}$ across all 5 procedure actions:
- **Precision ($P_\theta$):** $\frac{\text{TP}_\theta}{\text{TP}_\theta + \text{FP}_\theta}$
- **Recall ($R_\theta$):** $\frac{\text{TP}_\theta}{\text{TP}_\theta + \text{FN}_\theta}$
- **F1 Score ($F1_\theta$):** $2 \cdot \frac{P_\theta \cdot R_\theta}{P_\theta + R_\theta}$

### 2. Macro Robustness Metrics

- **Macro-Average F1 ($F1_{\text{macro}}$):**
  $$F1_{\text{macro}} = \frac{1}{7} \sum_{\theta} F1_\theta$$
- **Relative Angular Degradation ($\Delta F1_\theta$):**
  $$\Delta F1_\theta = F1_\theta - F1_{0^\circ}$$
- **Maximum Orientation Drop ($\text{Drop}_{\max}$):**
  $$\text{Drop}_{\max} = F1_{0^\circ} - \min_{\theta}(F1_\theta)$$

### 3. Procedural Robustness Under Perturbation

- **False Trigger Rate Under Incomplete Action ($FTR_{\text{incomplete}}$):** Percentage of incomplete actions incorrectly triggering valid FSM step completion.
- **Anomaly Detection Recall Under Rotation ($ADR_\theta$):** Ability of FSM to flag out-of-sequence errors when video is rotated.

### 4. Edge Preprocessing Latency

- **Latency Overhead ($\Delta t_{\text{prep}}$):** Added milliseconds per frame if runtime orientation normalization or affine rectification is performed before neural inference on edge CPU / Hailo NPU.

*Note: Benchmark pass thresholds (e.g. $\text{Drop}_{\max} \le 15\%$) are `PROPOSED TARGETS — REQUIRES EXPERIMENTAL VALIDATION`.*

---

## 11. Microgravity vs Camera Orientation

It is essential to distinguish between optical camera orientation and 3D microgravity spatial orientation:

| Attribute | 2D Camera Orientation Robustness (Gate B4) | 3D Payload-Relative Reasoning (Gate B15) |
|---|---|---|
| **Domain** | Optical sensor roll / image plane rotation ($0^\circ \text{--} 270^\circ$). | Full $SE(3)$ spatial pose of astronaut limbs relative to payload rack. |
| **Mechanism** | Invariant 2D feature extractors, rotation data augmentation, spatial normalization. | 3D human mesh recovery (HMR), joint angle kinematics, payload coordinate anchoring. |
| **Physics** | Pure visual image transformation. | Microgravity body float dynamics, foot restraint anchoring, multi-axis reach envelopes. |
| **Timing** | Gate B4 (Perception baseline). | Gate B15 (Advanced spatial reasoning extension). |

Gate B4 addresses perception-level visual invariance so the 2D detector and 1D-TCN do not fail simply because a camera is mounted sideways. Gate B15 will subsequently handle full 3D spatial kinematics relative to the payload frame.

---

## 12. Current Implementation Gaps

Before Gate B4.1 implementation can begin, the following technical gaps must be resolved:

1. **No Augmentation Engine:** No Python utility currently exists in `backend/` or `training/` to perform synchronized frame-and-keypoint rotation.
2. **Axis-Aligned IoU Limitation:** `InteractionEngine.compute_iou` uses axis-aligned bounding boxes. When objects rotate diagonally, axis-aligned boxes expand artificially, corrupting IoU calculations.
3. **Keypoint Coordinate Dependency:** `FrameProcessor.extract_keypoint_vector` passes raw $(x,y)$ values to 1D-TCN without canonical orientation normalization.
4. **MediaPipe Upright Bias:** MediaPipe Pose and Hands have degraded tracking accuracy on inverted ($180^\circ$) images unless rotation pre-rectification is applied.

---

## 13. B4.1 Implementation Requirements

Gate B4.1 will implement the following modular components:

1. **`backend/ai/augmentation.py`:** Standalone transformation module capable of rotating, scaling, translating, and shifting frames while synchronously updating bounding box coordinates and landmark keypoint arrays.
2. **Orientation Normalization Hook:** Preprocessing option in `FrameProcessor` to rotate frames to canonical upright orientation prior to MediaPipe / YOLO inference when camera orientation metadata is provided.
3. **OBB / Centroid Interaction Logic:** Enhancing `InteractionEngine` to support oriented bounding box overlap or scale-invariant centroid distances that remain invariant under diagonal rotation.
4. **Unit Tests for Transformation Invariants:** Automated tests verifying that keypoint coordinates rotate faithfully with the image frame.

---

## 14. B4.2 Evaluation Requirements

Gate B4.2 will execute the formal evaluation protocol:

1. **Test Split Evaluation Harness:** Batch runner evaluating the trained model across all 7 discrete angles.
2. **Tabular Metrics Generation:** Populating precision, recall, F1, and $\Delta F1$ tables without synthetic fabrication.
3. **Degradation Report:** Generating formal Workstream B robustness scorecard.

---

## 15. Open Parameters

The following parameters remain open and are designated as `PROPOSED TARGET — REQUIRES VALIDATION`:

| Parameter | Proposed Target | Resolution Path |
|---|:---:|---|
| Maximum allowable F1 degradation under extreme rotation ($\text{Drop}_{\max}$) | $\le 15\%$ drop | Validate on real multi-angle test clips (B4.2). |
| Maximum allowable preprocessing latency overhead | $\le 2.0\text{ ms}$ per frame | Benchmark on Raspberry Pi 5 CPU (B8). |
| Optimal training augmentation rotation probability | $p = 0.50$ | Hyperparameter sweep during Part-1 training. |
| Cutout occlusion area threshold | $\le 15\%$ bounding area | Validate object visibility limits. |

---

## 16. Acceptance Criteria

- [x] Orientation robustness specification document created (`docs/workstream-b-orientation-robustness.md`).
- [x] Machine-readable specification created (`config/orientation_robustness_spec.json`).
- [x] All 7 canonical evaluation angles defined ($0^\circ, 45^\circ, 90^\circ, 135^\circ, 180^\circ, 225^\circ, 270^\circ$).
- [x] 3 orientation categories integrated (`NORMAL`, `MODERATE_ROTATION`, `EXTREME_ROTATION`).
- [x] 7 core robustness dimensions taxonomized (Rotation, Scale, Translation, Brightness, Blur, Occlusion, Perspective).
- [x] 7 evaluation subsets mapped to dataset spec (`EVAL_NORMAL`, `EVAL_ORIENTATION`, etc.).
- [x] Subject-level leakage prevention policy explicitly formulated.
- [x] Evaluation metrics defined (Precision, Recall, F1, macro-average, relative drop, latency).
- [x] Conceptual boundary between 2D image orientation (B4) and 3D payload reasoning (B15) established.
- [x] Current implementation reality and gaps audited across `backend/ai/` and `backend/video/`.
- [x] Forward requirements for B4.1 (Implementation) and B4.2 (Evaluation) defined.
- [x] No fabricated evaluation results or numerical performance benchmarks included.
- [x] Automated test suite verifying spec integrity created and passing.
