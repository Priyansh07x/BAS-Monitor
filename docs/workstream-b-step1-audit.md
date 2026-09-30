# Workstream B Step 1 Audit: B1 — Experiment Definition + AI Contract Freeze

**Date:** 2026-09-23  
**Gate:** B1 — Experiment Definition + AI Contract Freeze  
**Author:** Workstream B (AI / Procedure Intelligence)

---

## Repository Reality

A deep inspection of the current repository reveals the actual state across all perception, procedure, data, and configuration modules:

1. **AI Perception Modules (`backend/ai/`):**
   - [`backend/ai/object_detector.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/object_detector.py): Implements an object detector with YOLO INT8 structure and graceful CPU heuristic fallback.
   - [`backend/ai/pose_detector.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/pose_detector.py): Implements 3D skeletal landmark recovery (33 joints) with graceful synthetic fallback when MediaPipe is unavailable.
   - [`backend/ai/hand_detector.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/hand_detector.py): Implements 21-landmark hand and bounding box tracking with fallback.
   - [`backend/ai/action_classifier.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/action_classifier.py): 1D-TCN sliding window classifier (window size 30 frames) with TFLite runtime support and fallback heuristic based on geometric interaction state (`HOLDING`, `APPROACHING`, `OPERATING`).
   - [`backend/ai/action_recognizer.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/action_recognizer.py): Facade wrapping `ActionClassifier`.
   - [`backend/ai/hailo_inference.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/hailo_inference.py): HailoRT NPU wrapper with automatic host fallback.
   - [`backend/ai/inference_pipeline.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/inference_pipeline.py): Master orchestrator coordinating object detection, 3D pose, hands, geometric interaction, vector extraction, temporal action recognition, and HUD overlay rendering. Currently returns an internal dictionary with `objects`, `pose`, `hands`, `interaction`, `action`, `confidence`, and `annotated_frame`.

2. **Procedure & Sequence Validation (`backend/experiment/`):**
   - [`backend/experiment/procedure_manager.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/experiment/procedure_manager.py): Loads and normalizes JSON procedure step definitions, indexing by step ID, step number, and expected action string.
   - [`backend/experiment/sequence_validator.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/experiment/sequence_validator.py): Deterministic FSM evaluating detected actions against expected procedure steps; triggers voice alerts and logs anomalies/steps.
   - [`backend/experiment/interaction_logic.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/experiment/interaction_logic.py): 2D IoU and euclidean proximity between hand bounding boxes and target objects.

3. **Data and Configuration (`config/`, `data/`):**
   - [`config/experiment.json`](file:///E:/Technical_Projects/2026-SIH/SIH26174/config/experiment.json): Standard procedure config defining ordered steps `S1` through `S5`.
   - [`data/generate_sample_run.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/data/generate_sample_run.py): Generates sample pickle run simulating `architecture.md` AI results.
   - [`data/experiments.json`](file:///E:/Technical_Projects/2026-SIH/SIH26174/data/experiments.json): Experiment storage database manipulated via UI CRUD operations.

4. **Frontend & Simulation (`frontend/`):**
   - [`frontend/assets/js/app.js`](file:///E:/Technical_Projects/2026-SIH/SIH26174/frontend/assets/js/app.js): Currently runs a prototype simulation loop with hardcoded steps (`PICK_CONTAINER`, `PIPETTE_TRANSFER`, etc.) generating randomized confidence.

---

## Canonical Experiment

The canonical experiment definition is established in [`config/experiment.json`](file:///E:/Technical_Projects/2026-SIH/SIH26174/config/experiment.json).

- **Experiment ID:** `EXP-001`
- **Version:** `1.0.0`
- **Name:** Microgravity Sample Transfer and Containment Procedure
- **Purpose:** Standardized protocol for astronaut sample acquisition, container deposition, and vessel sealing in microgravity to validate sequence tracking and AI procedure monitoring.
- **Objects:**
  - `RED_SAMPLE`
  - `BLUE_SAMPLE`
  - `SAMPLE_CONTAINER`
  - `CONTAINER_LID`
- **Actions:**
  - `PICK_RED`
  - `PLACE_RED`
  - `PICK_BLUE`
  - `PLACE_BLUE`
  - `CLOSE_LID`
- **Preconditions:**
  - Workstation camera calibrated and unobstructed
  - Sample tray with red and blue samples in initial rest position
  - Container open with lid accessible
  - AI perception pipeline initialized and in RUNNING state
- **Completion Conditions:**
  - Red sample placed in container
  - Blue sample placed in container
  - Container lid securely closed
  - FSM transitions to COMPLETED state
- **Failure Conditions:**
  - Step executed out of sequence (e.g., S3 before S1)
  - Step skipped without execution
  - Action duration exceeds timeout threshold
  - Action classification confidence below minimum threshold (0.70)
- **Ordered Step Sequence:**

| Step ID | Step Number | Action | Target Object | Description | Timeout | Confidence Thresh | Recovery Instruction |
|:---:|:---:|:---|:---|:---|:---:|:---:|:---|
| **S1** | 1 | `PICK_RED` | `RED_SAMPLE` | Pick up red sample | 30s | 0.70 | Return hand to starting position and re-acquire the red sample. |
| **S2** | 2 | `PLACE_RED` | `RED_SAMPLE` | Place red sample in container | 30s | 0.70 | Retrieve red sample if dislodged and place firmly inside the container. |
| **S3** | 3 | `PICK_BLUE` | `BLUE_SAMPLE` | Pick up blue sample | 30s | 0.70 | Return hand to starting position and re-acquire the blue sample. |
| **S4** | 4 | `PLACE_BLUE` | `BLUE_SAMPLE` | Place blue sample in container | 30s | 0.70 | Retrieve blue sample if dislodged and place firmly inside the container. |
| **S5** | 5 | `CLOSE_LID` | `CONTAINER_LID` | Close the container lid | 25s | 0.70 | Re-align container lid and press firmly until locked. |

---

## Existing Duplicate / Conflicting Definitions

Three distinct experiment definitions were identified across the repository:

1. **Definition 1 (Architecture Contract & Canonical Config):**
   - **Source:** [`docs/architecture.md`](file:///E:/Technical_Projects/2026-SIH/SIH26174/docs/architecture.md), [`config/experiment.json`](file:///E:/Technical_Projects/2026-SIH/SIH26174/config/experiment.json), [`data/generate_sample_run.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/data/generate_sample_run.py)
   - **Actions:** `PICK_RED`, `PLACE_RED`, `PICK_BLUE`, `PLACE_BLUE`, `CLOSE_LID`
   - **Step IDs:** `S1`, `S2`, `S3`, `S4`, `S5`
   - **Role:** Authoritative contract baseline between Workstream B and Workstream A.

2. **Definition 2 (Frontend Simulation Prototype):**
   - **Source:** [`frontend/assets/js/app.js`](file:///E:/Technical_Projects/2026-SIH/SIH26174/frontend/assets/js/app.js) (lines 579–615), [`backend/ai/action_classifier.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/action_classifier.py) (lines 24–28)
   - **Actions:** `PICK_CONTAINER`, `PIPETTE_TRANSFER`, `INSERT_ANALYZER`, `INITIATE_SCAN`, `SEAL_CONTAINER`
   - **Step IDs:** Numeric `1, 2, 3, 4, 5`
   - **Role:** Hardcoded UI simulation for demonstration purposes (documented in [`docs/decision.md`](file:///E:/Technical_Projects/2026-SIH/SIH26174/docs/decision.md) as [DECISION-006]).

3. **Definition 3 (CRUD Storage Database):**
   - **Source:** [`data/experiments.json`](file:///E:/Technical_Projects/2026-SIH/SIH26174/data/experiments.json)
   - **Actions:** Blank (`""`), instructions: "Collect sample", "Transfer sample", "Analyze sample", "Store sample"
   - **Step IDs:** Numeric `1, 2, 3, 4`
   - **Role:** Saved UI-created experiment records.

**Resolution:** Definition 1 is preserved and enriched in `config/experiment.json` as the canonical source. Definitions 2 and 3 are preserved as legacy/simulated/user-created data and documented without deletion.

---

## AI Contract

The public AI result contract between Workstream B and Workstream A is defined authoritatively in [`docs/architecture.md`](file:///E:/Technical_Projects/2026-SIH/SIH26174/docs/architecture.md) §2:

```json
{
  "timestamp": "2026-08-29T10:00:05",
  "action": "PICK_RED",
  "object": "RED_SAMPLE",
  "confidence": 0.94,
  "expected_step": "S1",
  "detected_step": "S1",
  "status": "VALID",
  "next_step": "S2"
}
```

### Key Field Specifications:
- `timestamp`: ISO 8601 string (`YYYY-MM-DDTHH:MM:SS`)
- `action`: String identifying the recognized action (e.g., `PICK_RED`)
- `object`: String identifying the manipulated object (e.g., `RED_SAMPLE`)
- `confidence`: Float between `0.0` and `1.0`
- `expected_step`: String matching canonical step ID (e.g., `"S1"`) or `null`
- `detected_step`: String matching canonical step ID (e.g., `"S1"`)
- `status`: String enum indicating procedural validation status:
  - `VALID`
  - `SKIPPED`
  - `OUT_OF_SEQUENCE`
- `next_step`: String matching subsequent step ID (e.g., `"S2"`) or `null`

Workstream B internally maintains richer evidence (bounding boxes, 3D pose coordinates, hand landmarks, interaction states, model telemetry), but the boundary facing Workstream A must expose the fields above.

---

## Contract Conflicts

The following conflicts between documents and code implementations were identified:

### 1. Status Values Mismatch
- **`docs/architecture.md` defines:**
  - `VALID`
  - `SKIPPED`
  - `OUT_OF_SEQUENCE`
- **`docs/BAS-Monitor_Implementation-Workstream-B.md` & `docs/BAS-Monitor_Implementation-Workstream-A.md` mention:**
  - `VALID`
  - `UNCERTAIN`
  - `OUT_OF_ORDER`
  - `SKIPPED`
  - `UNRECOGNIZED`
- **`backend/experiment/sequence_validator.py` produces:**
  - `VALID`
  - `LOW_CONFIDENCE`
  - `SKIPPED`
  - `OUT_OF_ORDER`
  - `UNRECOGNIZED`
  - `IGNORED`
  - `COMPLETED`
  under the dictionary key `validation_status`.

### 2. Output Shape Mismatch in `inference_pipeline.py`
- `docs/architecture.md` specifies that `inference_pipeline.py` returns the evaluated public contract dict (`expected_step`, `detected_step`, `status`, `next_step`).
- `backend/ai/inference_pipeline.py` currently returns raw multimodal perception dictionary:
  `{ "frame_index", "objects", "pose", "hands", "interaction", "action", "confidence", "annotated_frame" }`.
  The procedural validation is performed downstream by `SequenceValidatorFSM`.

### 3. Step ID Data Types
- `docs/architecture.md` and `config/experiment.json` require string IDs: `"S1"`, `"S2"`, `"S3"`, `"S4"`, `"S5"`.
- `frontend/assets/js/app.js` and `data/experiments.json` use integer step IDs: `1, 2, 3, 4, 5`.

### Strategy for Resolution:
- Maintain `docs/architecture.md` as the authoritative frozen contract during Step 1.
- In later phases, provide an adapter/bridge layer in the inference runtime that maps internal states (`OUT_OF_ORDER` → `OUT_OF_SEQUENCE`, `LOW_CONFIDENCE`/`UNCERTAIN` → `VALID` with low confidence or mapped status if contract is updated) and attaches procedure state to form the exact public payload.

---

## Current Limitations

| Component | Implemented | Simulated | Placeholder | Missing | Notes |
|---|:---:|:---:|:---:|:---:|---|
| **YOLO Object Detector** | | ✅ | | | Code implemented with heuristic bbox fallback; no physical `.pt` model weights |
| **MediaPipe Pose / Hands** | ✅ | | | | MediaPipe detectors active; fallback synthetic data when lib unavailable |
| **1D-TCN Action Classifier** | | ✅ | | | TFLite inference code present; fallback heuristic based on interaction state |
| **Inference Pipeline** | ✅ | | | | Coordinates perception submodules into single frame evaluation |
| **Deterministic FSM** | ✅ | | | | `procedure_manager.py` and `sequence_validator.py` fully operational |
| **Sample Run Simulation** | | | ✅ | | `data/sample_run.pkl` contains hardcoded pickle sequence |
| **Frontend Simulation Loop** | | ✅ | | | `app.js` runs `setInterval(3000ms)` with random confidence |
| **Hailo-8L NPU Runtime** | ✅ | | | | Interface ready; falls back to host CPU without PCIe accelerator |
| **3D HMR (SMPL Mesh)** | | | | ❌ | Future extension (Phase B14/B15); explicitly out of scope for B1 |
| **Offline Voice Alerts** | ✅ | | | | Background thread with macOS `say` and `pyttsx3` fallback |
| **Experiment Telemetry Logger**| ✅ | | | | Logs structured session and anomaly data to `data/logs/` |

---

## Step 1 Acceptance Checklist

- [x] All project reference documents inspected (`architecture.md`, `Workstream-B.md`, `Workstream-A.md`, `GUIDE.md`, `flow.md`, `decision.md`, `implementation-status.md`).
- [x] Repository implementation audited across `backend/ai/`, `backend/experiment/`, `config/`, `data/`, `frontend/`, and `tests/`.
- [x] Canonical experiment identified and formalized in [`config/experiment.json`](file:///E:/Technical_Projects/2026-SIH/SIH26174/config/experiment.json) with stable step IDs `S1`..`S5`.
- [x] Preconditions, completion conditions, failure conditions, objects, actions, timeouts, and recovery instructions fully defined.
- [x] Public AI result contract frozen per `docs/architecture.md`.
- [x] Status enum and schema mismatches thoroughly documented.
- [x] Compatibility strategy logged in [`docs/decision.md`](file:///E:/Technical_Projects/2026-SIH/SIH26174/docs/decision.md).
- [x] Step 1 contract and config tests created and validated.
- [x] Hard boundaries respected (no model training, no HMR implementation, no unauthorized refactoring).
