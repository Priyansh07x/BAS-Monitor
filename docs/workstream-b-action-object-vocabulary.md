# Workstream B Step 2 — Procedure Action/Object Vocabulary

**Date:** 2026-09-23  
**Gate:** B2 — Procedure Action/Object Vocabulary Definition  
**Author:** Workstream B (AI / Procedure Intelligence)

---

## 1. Scope

This document establishes the canonical **Procedure Action Vocabulary** and **Object Vocabulary** for the BAS Monitor demonstration experiment (`EXP-001` — *Microgravity Sample Transfer and Containment Procedure*).

> [!IMPORTANT]
> **Clear Conceptual Separation:**
> This document strictly defines the **procedure-level vocabulary** required by the experiment protocol and sequence validation engine. It does **NOT** represent the current Part-1 machine learning classifier classes (`catch` / `not-catch`). The procedure layer and ML model layers are distinct architectural entities. Part 2 must not force procedure actions into the legacy two-class model or fabricate mappings.

---

## 2. Canonical Objects

The experiment defines four physical/logical entities within the workspace:

| Object ID | Role in Experiment | Actions Using It | Required Evidence | Ambiguity & Notes | Object Category |
|---|---|---|---|---|---|
| `RED_SAMPLE` | Primary biological/chemical sample specimen | `PICK_RED`, `PLACE_RED` | 2D/3D bounding box detection, distinct visual/color signature, proximity to hand/rack | Visual similarity to blue sample if lighting shifts; occlusion during grasp | Manipulated Object |
| `BLUE_SAMPLE` | Secondary biological/chemical sample specimen | `PICK_BLUE`, `PLACE_BLUE` | 2D/3D bounding box detection, distinct visual/color signature, proximity to hand/rack | Visual similarity to red sample if lighting shifts; occlusion during grasp | Manipulated Object |
| `SAMPLE_CONTAINER` | Microgravity receptacle vessel | `PLACE_RED`, `PLACE_BLUE` | Receptacle bounding box, open chamber aperture location, spatial anchoring to rack | Hand occluding container opening; ambiguous sample entry vs hover | Destination / Container |
| `CONTAINER_LID` | Mechanical sealing mechanism | `CLOSE_LID` | Lid boundary detection, hinge/latch engagement state, spatial alignment with container | Distinguishing partial closure from fully locked state | Control / Sealing Mechanism |

### Detailed Object Specifications

#### 1. `RED_SAMPLE`
- **Object ID:** `RED_SAMPLE`
- **Role in Experiment:** Test specimen 1 to be transferred from the payload holding tray into the sample container.
- **Actions Using It:** `PICK_RED` (acquisition), `PLACE_RED` (deposition).
- **Required Evidence:** Bounding box detection on sample rack, color segmentation, hand contact / IoU overlap.
- **Potential Ambiguity:** Partial occlusion by astronaut fingers when grasped; color distortion under monochrome or low-light camera settings.
- **Object Category:** Manipulated Object.

#### 2. `BLUE_SAMPLE`
- **Object ID:** `BLUE_SAMPLE`
- **Role in Experiment:** Test specimen 2 to be transferred from the payload holding tray into the sample container.
- **Actions Using It:** `PICK_BLUE` (acquisition), `PLACE_BLUE` (deposition).
- **Required Evidence:** Bounding box detection on sample rack, color segmentation, hand contact / IoU overlap.
- **Potential Ambiguity:** Occlusion during grip; spatial proximity to `RED_SAMPLE` if adjacent on holding tray.
- **Object Category:** Manipulated Object.

#### 3. `SAMPLE_CONTAINER`
- **Object ID:** `SAMPLE_CONTAINER`
- **Role in Experiment:** Main reaction / containment vessel receiving both test samples.
- **Actions Using It:** `PLACE_RED` (target receptacle), `PLACE_BLUE` (target receptacle), `CLOSE_LID` (base fixture).
- **Required Evidence:** Stable bounding box at workstation payload bay; aperture clear/entry region.
- **Potential Ambiguity:** Bounding box overlap between astronaut hand hovering over opening vs true insertion.
- **Object Category:** Destination / Container.

#### 4. `CONTAINER_LID`
- **Object ID:** `CONTAINER_LID`
- **Role in Experiment:** Protective sealing lid to isolate the samples post-deposition.
- **Actions Using It:** `CLOSE_LID` (manipulation and latching).
- **Required Evidence:** Lid keypoints / orientation bounding box, contact with hand, mating contact with `SAMPLE_CONTAINER`.
- **Potential Ambiguity:** Differentiating between operator touching lid vs exerting closing force; verifying final seal engagement.
- **Object Category:** Control / Sealing Mechanism.

---

## 3. Canonical Procedure Actions

The protocol requires five discrete, ordered operational steps (`S1` through `S5`):

| Action ID | Target Object | Start Condition | Completion Condition | Allowed Next Step | Timeout |
|:---:|:---:|:---|:---|:---:|:---:|
| `PICK_RED` | `RED_SAMPLE` | Hand approaches red sample on rack | Hand firmly grasps and lifts red sample clear of rack | `S2` (`PLACE_RED`) | 30s |
| `PLACE_RED` | `RED_SAMPLE` | Hand carries red sample toward container | Red sample released inside container, hand retracts | `S3` (`PICK_BLUE`) | 30s |
| `PICK_BLUE` | `BLUE_SAMPLE` | Hand approaches blue sample on rack | Hand firmly grasps and lifts blue sample clear of rack | `S4` (`PLACE_BLUE`) | 30s |
| `PLACE_BLUE` | `BLUE_SAMPLE` | Hand carries blue sample toward container | Blue sample released inside container, hand retracts | `S5` (`CLOSE_LID`) | 30s |
| `CLOSE_LID` | `CONTAINER_LID` | Hand contacts open container lid | Lid rotated into closed position and latch engaged | `COMPLETED` | 25s |

---

### Detailed Action Specifications

#### 1. `PICK_RED`
- **Action ID:** `PICK_RED`
- **Target Object:** `RED_SAMPLE`
- **Procedure Meaning:** Astronaut locates, grasps, and lifts the red sample from its rest slot on the payload holding tray.
- **Start Condition:** Hand enters proximity zone of `RED_SAMPLE` on the holding tray (Euclidean distance $\le 0.15$).
- **Completion Condition:** Hand bounding box overlaps `RED_SAMPLE` ($\text{IoU} \ge 0.05$ or state `HOLDING`) and specimen is displaced from rest coordinates.
- **Expected Interaction Evidence:** Hand state transitions `IDLE` $\rightarrow$ `APPROACHING` $\rightarrow$ `HOLDING`.
- **Expected Temporal Evidence:** Sustained hand-object contact over a temporal window ($\ge 5$ consecutive frames).
- **Minimum Duration:** *UNSPECIFIED — requires dataset/experiment video calibration (tentatively 0.5s).*
- **Maximum Duration / Timeout:** 30.0 seconds.
- **Confidence Threshold:** 0.70.
- **Allowed Next Procedure Step:** `S2` (`PLACE_RED`).
- **Recovery Instruction:** Return hand to starting position and re-acquire the red sample.
- **Failure / Ambiguity Cases:** Grasping the blue sample instead (out-of-order `PICK_BLUE`); hand hovering without making contact; knocking sample loose.

#### 2. `PLACE_RED`
- **Action ID:** `PLACE_RED`
- **Target Object:** `RED_SAMPLE` (into `SAMPLE_CONTAINER`)
- **Procedure Meaning:** Astronaut transfers the held red sample and deposits it securely inside the sample container aperture.
- **Start Condition:** Hand holding `RED_SAMPLE` moves toward `SAMPLE_CONTAINER` entry aperture.
- **Completion Condition:** `RED_SAMPLE` centroid enters `SAMPLE_CONTAINER` bounds, hand releases grip ($\text{IoU} \rightarrow 0$), and hand retracts.
- **Expected Interaction Evidence:** Hand + `RED_SAMPLE` cluster reaches `SAMPLE_CONTAINER` boundary $\rightarrow$ `RELEASING` $\rightarrow$ `IDLE`.
- **Expected Temporal Evidence:** Trajectory from holding rack to container opening followed by release event.
- **Minimum Duration:** *UNSPECIFIED — requires dataset/experiment video calibration (tentatively 0.5s).*
- **Maximum Duration / Timeout:** 30.0 seconds.
- **Confidence Threshold:** 0.70.
- **Allowed Next Procedure Step:** `S3` (`PICK_BLUE`).
- **Recovery Instruction:** Retrieve red sample if dislodged and place firmly inside the container.
- **Failure / Ambiguity Cases:** Dropping sample outside container aperture; placing blue sample prematurely; releasing sample before container entry.

#### 3. `PICK_BLUE`
- **Action ID:** `PICK_BLUE`
- **Target Object:** `BLUE_SAMPLE`
- **Procedure Meaning:** Astronaut locates, grasps, and lifts the blue sample from its rest slot on the payload holding tray.
- **Start Condition:** Hand enters proximity zone of `BLUE_SAMPLE` on the holding tray after `S2` completion.
- **Completion Condition:** Hand bounding box overlaps `BLUE_SAMPLE` ($\text{IoU} \ge 0.05$ or state `HOLDING`) and specimen is displaced from rest coordinates.
- **Expected Interaction Evidence:** Hand state transitions `IDLE` $\rightarrow$ `APPROACHING` $\rightarrow$ `HOLDING`.
- **Expected Temporal Evidence:** Sustained contact over sliding window on `BLUE_SAMPLE` coordinates.
- **Minimum Duration:** *UNSPECIFIED — requires dataset/experiment video calibration (tentatively 0.5s).*
- **Maximum Duration / Timeout:** 30.0 seconds.
- **Confidence Threshold:** 0.70.
- **Allowed Next Procedure Step:** `S4` (`PLACE_BLUE`).
- **Recovery Instruction:** Return hand to starting position and re-acquire the blue sample.
- **Failure / Ambiguity Cases:** Re-picking red sample; attempting to close lid prematurely without blue sample (`S5` before `S3`).

#### 4. `PLACE_BLUE`
- **Action ID:** `PLACE_BLUE`
- **Target Object:** `BLUE_SAMPLE` (into `SAMPLE_CONTAINER`)
- **Procedure Meaning:** Astronaut transfers the held blue sample and deposits it securely into the container.
- **Start Condition:** Hand holding `BLUE_SAMPLE` moves toward `SAMPLE_CONTAINER` aperture.
- **Completion Condition:** `BLUE_SAMPLE` centroid enters container bounds, grip released, hand retracts.
- **Expected Interaction Evidence:** Hand + `BLUE_SAMPLE` enters container bounding box $\rightarrow$ `RELEASING` $\rightarrow$ `IDLE`.
- **Expected Temporal Evidence:** Controlled trajectory and release inside container chamber.
- **Minimum Duration:** *UNSPECIFIED — requires dataset/experiment video calibration (tentatively 0.5s).*
- **Maximum Duration / Timeout:** 30.0 seconds.
- **Confidence Threshold:** 0.70.
- **Allowed Next Procedure Step:** `S5` (`CLOSE_LID`).
- **Recovery Instruction:** Retrieve blue sample if dislodged and place firmly inside the container.
- **Failure / Ambiguity Cases:** Blue sample bouncing or floating out under microgravity; closing lid while sample is still outside.

#### 5. `CLOSE_LID`
- **Action ID:** `CLOSE_LID`
- **Target Object:** `CONTAINER_LID`
- **Procedure Meaning:** Astronaut grasps the open container lid, aligns it over the container aperture, and seals/latches it firmly.
- **Start Condition:** Hand enters proximity zone of `CONTAINER_LID` and makes physical contact.
- **Completion Condition:** Lid angle reaches $0^\circ$ (flush with `SAMPLE_CONTAINER`), latch pressure applied, hand retracts to rest.
- **Expected Interaction Evidence:** Hand `HOLDING` / `OPERATING` `CONTAINER_LID` until lid geometry intersects container rim.
- **Expected Temporal Evidence:** Downward/rotational trajectory of lid keypoints ending in stationary sealed alignment.
- **Minimum Duration:** *UNSPECIFIED — requires dataset/experiment video calibration (tentatively 0.5s).*
- **Maximum Duration / Timeout:** 25.0 seconds.
- **Confidence Threshold:** 0.70.
- **Allowed Next Procedure Step:** `COMPLETED` (Procedure termination).
- **Recovery Instruction:** Re-align container lid and press firmly until locked.
- **Failure / Ambiguity Cases:** Lid left ajar; closing lid before all samples are inserted; accidental reopening.

---

## 4. Current Part-1 Model Classes

The machine learning classifier exported from early Part-1 Google Colab experiments contains only two classes:
1. `catch`
2. `not-catch`

### Root Cause of the Integration Gap:
- The 2-class `catch` / `not-catch` model was trained as a generic binary interaction baseline.
- It has **zero knowledge** of:
  - Object identity (`RED_SAMPLE` vs `BLUE_SAMPLE` vs `SAMPLE_CONTAINER` vs `CONTAINER_LID`)
  - Directional semantics (`PICK` vs `PLACE`)
  - Multi-stage sequence states (`S1` through `S5`)
- Therefore, the current 2-class model **cannot directly output five-class procedure predictions**.

---

## 5. Procedure Action ↔ Current Model Relationship

> [!WARNING]
> **NO DIRECT ONE-TO-ONE MAPPING CURRENTLY EXISTS.**

- `catch` $\neq$ `PICK_RED`
- `not-catch` $\neq$ `IDLE` or `CLOSE_LID`

Part 2 **does not** create synthetic mappings between `catch`/`not-catch` and procedure actions. The Part-1 model is actively being modified/retrained on Colab. Until a revised model with multi-class action and object recognition is exported, the two domains remain strictly decoupled.

### What Part 2 Can Build & Verify Independently:
Even without an active 5-class neural network, Part 2 provides complete deterministic infrastructure:
1. Canonical experiment protocol loading ([`ProcedureManager`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/experiment/procedure_manager.py))
2. Finite State Machine sequence verification ([`SequenceValidatorFSM`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/experiment/sequence_validator.py))
3. Skipped step and out-of-order detection
4. Geometric hand-object interaction dynamics ([`InteractionEngine`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/experiment/interaction_logic.py))
5. Voice alerts, logging, and GUI telemetry dispatch
6. Architecture contract compliance tests ([`docs/architecture.md`](file:///E:/Technical_Projects/2026-SIH/SIH26174/docs/architecture.md))

---

## 6. Future AI → Procedure Interface

When the revised Part-1 model or intermediate feature extractor is integrated, the AI layer must provide the following standard record to the procedure manager / sequence validator:

### Raw Inference Output (Internal Workstream B):
```python
{
    "action": "PICK_RED",          # Classified action string
    "object": "RED_SAMPLE",        # Detected target object label
    "confidence": 0.94,            # Continuous score [0.0 - 1.0]
    "timestamp": "2026-09-23T20:00:05.123", # ISO timestamp
    "evidence": {                  # Rich perception evidence (optional/internal)
        "hand_state": "HOLDING",
        "iou": 0.32,
        "keypoints_count": 33
    }
}
```

### Public Seam Output (Facing Workstream A / GUI):
Must conform to the frozen [`docs/architecture.md`](file:///E:/Technical_Projects/2026-SIH/SIH26174/docs/architecture.md) contract:
```json
{
  "timestamp": "2026-09-23T20:00:05.123",
  "action": "PICK_RED",
  "object": "RED_SAMPLE",
  "confidence": 0.94,
  "expected_step": "S1",
  "detected_step": "S1",
  "status": "VALID",
  "next_step": "S2"
}
```

---

## 7. FSM Compatibility Analysis

| Backend Component | Canonical Vocabulary Support | Current Role / Behavior |
|---|:---:|---|
| [`ProcedureManager`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/experiment/procedure_manager.py) | **100% Compatible** | Loads `config/experiment.json`, indexes `S1`–`S5` by `step_id`, `step_number`, and `expected_action`. |
| [`SequenceValidatorFSM`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/experiment/sequence_validator.py) | **100% Compatible** | Evaluates incoming action string against `expected_step.expected_action`; correctly handles `VALID`, `SKIPPED`, `OUT_OF_ORDER`, and `UNRECOGNIZED`. |
| [`InteractionEngine`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/experiment/interaction_logic.py) | **100% Compatible** | Computes 2D IoU and euclidean proximity between hands and objects (`RED_SAMPLE`, `BLUE_SAMPLE`, `CONTAINER_LID`, `SAMPLE_CONTAINER`). |
| [`ActionClassifier`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/action_classifier.py) | **Decoupled** | Contains `DEFAULT_ACTIONS` array including all 5 procedure actions; falls back to interaction heuristic during development. |

---

## 8. Ambiguities and Missing Information

The following items are not specified in the current repository and represent open parameters to be finalized during dataset collection (Phase B3):
1. **Color / Visual Thresholds:** Exact RGB/HSV range or bounding dimensions for `RED_SAMPLE` vs `BLUE_SAMPLE`.
2. **Physical Workspace Geometry:** Relative 3D positions of the holding tray, container, and camera viewpoints.
3. **Minimum Action Durations:** The exact minimum frame count or duration in seconds for each action before temporal confirmation fires (currently defaulting to heuristic sliding window of 30 frames).
4. **Microgravity Dynamics:** Realistic floating/rotation kinematics when samples are released in zero-g.

---

## 9. Gate B2 Acceptance Checklist

- [x] Canonical procedure actions defined (`PICK_RED`, `PLACE_RED`, `PICK_BLUE`, `PLACE_BLUE`, `CLOSE_LID`).
- [x] Canonical objects defined (`RED_SAMPLE`, `BLUE_SAMPLE`, `SAMPLE_CONTAINER`, `CONTAINER_LID`).
- [x] Clear distinction between Procedure Actions, Current Model Classes (`catch`/`not-catch`), and Future AI interfaces established.
- [x] Confirmed NO synthetic or fake mappings exist between `catch`/`not-catch` and procedure actions.
- [x] Start conditions, completion conditions, temporal evidence, timeouts, confidence thresholds, and recovery instructions specified for all 5 actions.
- [x] FSM compatibility verified across [`procedure_manager.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/experiment/procedure_manager.py), [`sequence_validator.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/experiment/sequence_validator.py), and [`interaction_logic.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/experiment/interaction_logic.py).
- [x] Missing parameters explicitly cataloged.
- [x] Test suite [`tests/test_action_object_vocabulary.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_action_object_vocabulary.py) created and passing.
