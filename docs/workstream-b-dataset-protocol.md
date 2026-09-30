# Workstream B Step 3 — Dataset Planning / Collection Protocol

**Date:** 2026-09-23  
**Gate:** B3 — Dataset Planning / Collection Protocol  
**Author:** Workstream B (AI / Procedure Intelligence)

---

## 1. Scope

This document defines the **canonical dataset specification** for the BAS Monitor Workstream B AI action-recognition model. It covers collection protocol, taxonomy, split strategy, label schema, directory structure, naming conventions, quality requirements, and Part-1 handoff details for the *Microgravity Sample Transfer and Containment Procedure* (`EXP-001`).

### What This Document Is

- A precise protocol for future human data collection.
- A handoff specification to the Part-1 (Colab) training team.
- A machine-readable companion schema (`config/dataset_spec.json`).

### What This Document Is Not

- An actual dataset (no videos have been collected under this protocol yet).
- A model architecture specification.
- A training recipe.

> [!IMPORTANT]
> **No experiment-specific dataset has been collected under this protocol yet.**
> The existing `training/data/raw/hmdb51_sta/` directory contains the **HMDB51 dataset** (51 generic action classes). HMDB51 is the **current Part-1 baseline dataset source** — it is actively used for training the baseline `catch`/`not-catch` model and may support incremental class expansion in later SIH stages. However, HMDB51 generic labels (`catch`, `pick`, `pour`, etc.) are **not equivalent to and do not directly represent** the EXP-001 procedure actions (`PICK_RED`, `PLACE_RED`, `PICK_BLUE`, `PLACE_BLUE`, `CLOSE_LID`). An experiment-specific labeled dataset is a **future collection requirement** for direct five-action EXP-001 recognition.

---

### Existing Training Artifacts (Audit)

| Artifact | Location | Content | Role |
|---|---|---|---|
| **HMDB51 dataset** | `training/data/raw/hmdb51_sta/` | 51 generic motion classes (~200 clips each) including `catch`, `pick`, `pour`, `throw` | **Current Part-1 baseline dataset source.** Used for binary `catch`/`not-catch` training and future incremental class expansion. Generic labels do NOT map directly to EXP-001 procedure actions. `catch` ≠ `PICK_RED`. |
| **Exported Part-1 model** | Not present in repository (`models/` does not exist) | Binary classifier: `catch` / `not-catch` | **Not integrated.** Model under active revision in Colab. |
| **Sample run pickle** | `data/sample_run.pkl` | 4 placeholder AI result dicts simulating `EXP-001` | Simulation only; not real recognition. |
| **Test recordings** | `data/videos/REC_*.mp4` | GUI test recordings from Workstream A | No action labels; not usable as training data. |

---

## 2. Dataset Objective

The dataset must provide sufficient evidence to train and evaluate an AI system capable of:

1. **Action classification** — Distinguishing which of the 5 procedure actions is occurring in a video temporal window.
2. **Object identification** — Associating the detected action with the correct target object (`RED_SAMPLE`, `BLUE_SAMPLE`, `SAMPLE_CONTAINER`, `CONTAINER_LID`).
3. **Temporal progression** — Detecting action onset, sustained execution, and completion across multiple frames.
4. **Procedural context** — Providing sufficient signal so the FSM validator can correctly determine whether the detected action corresponds to the expected step.

> [!NOTE]
> **Architectural responsibility boundary:**
> The AI model is responsible for *detecting what action is occurring*.
> The deterministic FSM (`SequenceValidatorFSM`) is responsible for *evaluating whether that action is correct at the current procedure step*.
> The dataset serves the AI detection layer; it should not attempt to replicate the FSM's sequence logic.

---

## 3. Canonical Actions

The dataset must provide coverage for all 5 procedure actions defined in `config/experiment.json`:

| Action ID | Step | Target Object | Action Meaning |
|:---:|:---:|:---:|---|
| `PICK_RED` | S1 | `RED_SAMPLE` | Grasp and lift the red specimen from the holding tray |
| `PLACE_RED` | S2 | `RED_SAMPLE` | Transfer and deposit the red specimen into the container |
| `PICK_BLUE` | S3 | `BLUE_SAMPLE` | Grasp and lift the blue specimen from the holding tray |
| `PLACE_BLUE` | S4 | `BLUE_SAMPLE` | Transfer and deposit the blue specimen into the container |
| `CLOSE_LID` | S5 | `CONTAINER_LID` | Align, rotate, and latch the container lid |

**Scope note for this protocol:** This collection protocol targets the 5 EXP-001 procedure actions listed above. HMDB51 generic action classes (e.g., `catch`, `pick`) are **not equivalent to** and **must not be substituted for** these procedure-specific labels. HMDB51 remains the current Part-1 baseline dataset (useful for general pretraining and incremental class expansion), but the five procedure actions require their own experiment-specific labeled data. Every procedure-class label in this dataset must correspond exactly to an EXP-001 canonical action or an explicitly defined negative/failure category.

---

## 4. Canonical Objects

All four canonical objects must be detectable within clips:

| Object ID | Type | Appearance Note |
|:---:|:---:|---|
| `RED_SAMPLE` | Manipulated specimen | UNSPECIFIED — exact dimensions and color values require experimental calibration |
| `BLUE_SAMPLE` | Manipulated specimen | UNSPECIFIED — must be distinguishable from `RED_SAMPLE` under planned lighting |
| `SAMPLE_CONTAINER` | Destination vessel | UNSPECIFIED — fixed position on payload rack; exact geometry TBD |
| `CONTAINER_LID` | Sealing mechanism | UNSPECIFIED — attachment/hinge type and closure geometry TBD |

Object visual properties are not fabricated here. Exact RGB ranges, dimensions, and spatial positions must be determined during experiment setup calibration before collection begins.

---

## 5. Dataset Taxonomy

### 5A. Positive Action Examples (Primary)

Standard, complete executions of each procedure action:

| Category | Action | Object | Expected Clip Content |
|---|---|---|---|
| `POSITIVE_COMPLETE` | `PICK_RED` | `RED_SAMPLE` | Approach → contact → lift → clear of rack |
| `POSITIVE_COMPLETE` | `PLACE_RED` | `RED_SAMPLE` | Carry → align over container → release → retract |
| `POSITIVE_COMPLETE` | `PICK_BLUE` | `BLUE_SAMPLE` | Approach → contact → lift → clear of rack |
| `POSITIVE_COMPLETE` | `PLACE_BLUE` | `BLUE_SAMPLE` | Carry → align over container → release → retract |
| `POSITIVE_COMPLETE` | `CLOSE_LID` | `CONTAINER_LID` | Contact → rotate → press → latch → retract |

### 5B. Incomplete Action Examples

Partial executions that should be classified as `INCOMPLETE` and do not trigger FSM validation:

| Category | Example |
|---|---|
| `INCOMPLETE_APPROACH` | Hand approaches `RED_SAMPLE` / `BLUE_SAMPLE` without making contact |
| `INCOMPLETE_PICKUP` | Sample grasped but not fully lifted from rack |
| `INCOMPLETE_PLACE` | Sample carried to container but not released; hand withdraws with sample |
| `INCOMPLETE_LID` | Lid contacted but not fully closed or latched |
| `INCOMPLETE_ABORT` | Action started then abandoned mid-motion |

### 5C. Procedural Failure Examples

These represent actual EXP-001 procedure violations. They are **not** arbitrary negative examples:

| Category | Failure Type | Description |
|---|---|---|
| `FAILURE_WRONG_ACTION` | Wrong action at current step | e.g., `CLOSE_LID` performed when `PICK_RED` expected (S5 before S1) |
| `FAILURE_SKIPPED_STEP` | Step skipped | e.g., `PICK_BLUE` performed without completing `PLACE_RED` |
| `FAILURE_OUT_OF_ORDER` | Previous step repeated | e.g., `PICK_RED` performed again after S2 |
| `FAILURE_REPEATED_STEP` | Same step performed twice consecutively | e.g., `PICK_RED` → `PICK_RED` |
| `FAILURE_PREMATURE` | Action performed before precondition met | e.g., `CLOSE_LID` before samples are placed |
| `FAILURE_TIMEOUT` | Step initiated but not completed within timeout | e.g., sample picked but not placed within 30s |
| `FAILURE_WRONG_OBJECT` | Correct action form, wrong object | e.g., `PICK`-like motion toward `CONTAINER_LID` when `RED_SAMPLE` is expected |

> [!WARNING]
> These failure examples represent **real procedure errors** and must be staged authentically during collection. They should not be created by distorting or corrupting normal clips post-hoc.

### 5D. Visual and Environmental Variation

Each of the following variation dimensions must be covered across the collection matrix:

| Dimension | Required Values |
|---|---|
| **Subject** | Multiple different people |
| **Hand** | Left hand only; Right hand only |
| **Action Speed** | Slow / Normal / Fast |
| **Viewpoint** | Front-facing / Side angle / Overhead / Oblique |
| **Camera Distance** | Close (< 1m) / Medium (1–2m) / Far (> 2m) — PROPOSED TARGETS, require calibration |
| **Lighting** | Standard / Bright / Dim / Shadow |
| **Occlusion** | No occlusion / Partial hand occlusion / Full object occlusion |
| **Object Position** | Fixed / Varied within rack bounds |
| **Pause** | Continuous execution / Mid-action pause / Hesitation |
| **Orientation** | 0° / 45° / 90° / 135° / 180° / 225° / 270° (see §9) |

---

## 6. Collection Matrix

The matrix below defines all collection dimensions, their required values, and collection status. **No data has been collected yet** under this protocol.

| Dimension | Values / Categories | Purpose | Baseline Target | Stretch Target | Status |
|---|---|---|:---:|:---:|:---:|
| **Subject count** | Unique individual operators | Generalization across operators | 5 subjects | 10 subjects | `NOT COLLECTED` |
| **Sessions per subject** | Recording sessions per person | Within-subject variability | 2 sessions | 4 sessions | `NOT COLLECTED` |
| **Takes per action per session** | Repeated trials of same action | Action variability within subject | 3 takes | 5 takes | `NOT COLLECTED` |
| **Positive actions** | 5 × complete executions | Primary classifier training data | 150 clips | 300 clips | `NOT COLLECTED` |
| **Incomplete actions** | 5 categories | Negative/boundary training | 50 clips | 100 clips | `NOT COLLECTED` |
| **Procedural failures** | 7 failure types | Failure detection and FSM triggering | 70 clips | 140 clips | `NOT COLLECTED` |
| **Hand** | Left / Right | Handedness generalization | Both covered | Both balanced | `NOT COLLECTED` |
| **Action speed** | Slow / Normal / Fast | Speed robustness | All 3 covered | Balanced | `NOT COLLECTED` |
| **Viewpoint** | Front / Side / Overhead / Oblique | View invariance | 2 viewpoints | 4 viewpoints | `NOT COLLECTED` |
| **Lighting** | Standard / Bright / Dim / Shadow | Lighting robustness | 2 conditions | 4 conditions | `NOT COLLECTED` |
| **Occlusion** | None / Partial / Full | Occlusion robustness | None + Partial | All 3 | `NOT COLLECTED` |
| **Orientation** | 0°, 45°, 90°, 135°, 180°, 225°, 270° | Microgravity orientation robustness | 0° + 90° | All 7 | `NOT COLLECTED` |

> [!NOTE]
> **All quantities above are PROPOSED TARGETS pending experimental validation.**
> The baseline targets are minimum recommended coverage for an evaluable model. The stretch targets are aspirational goals. Neither set has been experimentally validated as sufficient.

---

## 7. Sample / Clip Definition

Each dataset sample is one **clip** — a temporally bounded video segment corresponding to one action instance.

### Clip Specification

| Field | Description |
|---|---|
| `sample_id` | Unique deterministic identifier (see §13 naming convention) |
| `source_video` | Relative path to the original raw recording file |
| `clip_start` | Start timestamp in source video (seconds, float) |
| `clip_end` | End timestamp in source video (seconds, float); `null` for incomplete actions |
| `action_label` | Canonical action string (`PICK_RED`, `PLACE_RED`, `PICK_BLUE`, `PLACE_BLUE`, `CLOSE_LID`) |
| `object_label` | Canonical object string (`RED_SAMPLE`, `BLUE_SAMPLE`, `SAMPLE_CONTAINER`, `CONTAINER_LID`) |
| `procedure_step` | Step ID (`S1`–`S5`) |
| `completion_state` | `COMPLETE` / `INCOMPLETE` / `FAILURE` |
| `failure_type` | From §5C taxonomy, or `null` for positive/complete examples |
| `subject_id` | Anonymized subject identifier |
| `session_id` | Session identifier within a subject |
| `take_id` | Take number within a session |
| `hand` | `LEFT` / `RIGHT` / `BOTH` |
| `speed_category` | `SLOW` / `NORMAL` / `FAST` |
| `viewpoint` | `FRONT` / `SIDE` / `OVERHEAD` / `OBLIQUE` |
| `orientation_deg` | Camera/scene rotation in degrees (0, 45, 90, 135, 180, 225, 270) |
| `lighting` | `STANDARD` / `BRIGHT` / `DIM` / `SHADOW` |
| `occlusion` | `NONE` / `PARTIAL` / `FULL` |
| `split` | `TRAIN` / `VALIDATION` / `TEST` |
| `orientation_category` | `NORMAL` / `MODERATE_ROTATION` / `EXTREME_ROTATION` |
| `procedure_correctness` | `CORRECT` / `INCORRECT` |
| `notes` | Free text for unusual conditions |

### Temporal Model Requirement

Clips must have sufficient temporal duration for the action to be recognizable. A minimum of **8 frames** must be present between action onset and action completion. Single-frame classification is explicitly not sufficient for actions like `PICK_RED`, `PLACE_RED`, `PICK_BLUE`, `PLACE_BLUE`, or `CLOSE_LID`.

**Proposed minimum clip length:** 0.5 seconds at 15+ FPS (approximately 8 frames minimum).  
**Proposed maximum clip length:** 35 seconds (accounting for 30s timeout + buffer).  
Both targets are PROPOSED — requires validation during collection.

---

## 8. Split Strategy and Leakage Prevention

> [!CAUTION]
> **No random frame-level or clip-level splitting.** Splitting at frame or clip level with any shared subject across splits introduces label leakage and inflates test metrics.

### Primary Split Rule: Subject-Level Isolation

All clips from a given `subject_id` are assigned to exactly one split:

```
subject_ids → randomly assign subjects to splits (not clips):
  TRAIN:       70% of subjects
  VALIDATION:  15% of subjects
  TEST:        15% of subjects
```

This guarantees zero same-person clips across splits.

### Secondary Rule: Session Integrity

Within a subject's split assignment, all sessions and takes from that subject are kept in the same split. No cherry-picking individual sessions across boundaries.

### Split Percentages (Proposed Targets)

| Split | Subjects | Approximate Clips | Purpose |
|:---:|:---:|:---:|---|
| `TRAIN` | 70% | ~70% of total | Model training |
| `VALIDATION` | 15% | ~15% of total | Hyperparameter tuning, early stopping |
| `TEST` | 15% | ~15% of total | Final held-out evaluation; must not be used during training decisions |

### TEST Set Lockdown

The TEST set must be determined before any training begins and must not be inspected for performance until evaluation time. Its composition must be logged in the dataset manifest at collection time.

### Evaluation Subsets (within TEST)

The TEST set must be tagged to support subset evaluation:

| Evaluation Subset | Contents |
|---|---|
| `EVAL_NORMAL` | Standard orientation (0°), normal lighting, no occlusion |
| `EVAL_ORIENTATION` | Non-zero orientation angles (45°–270°) |
| `EVAL_OCCLUSION` | Clips with partial or full occlusion |
| `EVAL_LIGHTING` | Bright / Dim / Shadow lighting conditions |
| `EVAL_VIEWPOINT` | Non-front viewpoints (Side / Overhead / Oblique) |
| `EVAL_FAILURE` | Procedural failure examples (`FAILURE_*` categories) |
| `EVAL_INCOMPLETE` | Incomplete action examples |

These subsets must be tagged in annotations, **not** physically separated, to prevent leakage from subset-specific over-selection.

---

## 9. Orientation Metadata

Gate B4 will implement the full orientation robustness training strategy. This gate only establishes the required orientation metadata fields so that collection can record this information from the start.

### Orientation Categories

| Category | Degrees | Notes |
|:---:|:---:|---|
| `NORMAL` | 0° | Standard upright camera |
| `MODERATE_ROTATION` | 45°, 135°, 225° | Diagonal orientations |
| `EXTREME_ROTATION` | 90°, 180°, 270° | Sideways / inverted / opposite |

### Collection Protocol Note

Videos should be collected with camera orientation metadata recorded at time of capture. **Do not synthetically rotate collected videos post-hoc in this gate.** Rotation augmentation strategy belongs to B4.

**Orientation coverage in B3 collection protocol:**
- All clips must record their `orientation_deg` and `orientation_category` in the annotation manifest.
- At minimum: 0° (normal) orientation must be fully covered.
- 90° and 180° coverage: PROPOSED TARGET for later collection passes.

---

## 10. Negative and Failure Matrix

### AI Recognition Failures (model difficulty examples)

These are clips where the action is genuine but the AI may struggle:

| Failure Mode | Category Tag | Description |
|---|:---:|---|
| Wrong action | `AI_WRONG_ACTION` | Model misclassifies the action |
| Missed action | `AI_MISSED_ACTION` | Action occurs but model outputs `IDLE` or wrong class |
| Low confidence ambiguous | `AI_LOW_CONFIDENCE` | Action is visually ambiguous (two similar actions in same clip) |
| Occlusion | `AI_OCCLUSION` | Object or hand occluded during critical phase |
| Poor lighting | `AI_LIGHTING` | Low light degrades visual signal |
| Motion blur | `AI_BLUR` | Fast motion produces blurred frames |
| Fast motion | `AI_FAST_MOTION` | Action performed much faster than typical |
| Multiple objects visible | `AI_MULTI_OBJECT` | Both `RED_SAMPLE` and `BLUE_SAMPLE` visible simultaneously |

### Procedure Violations (FSM-level failures)

These represent procedurally incorrect executions — they are **real EXP-001 failures**:

| Failure Mode | Category Tag | Description |
|---|:---:|---|
| Skipped step | `PROC_SKIPPED_STEP` | A step is entirely omitted (e.g., S2 skipped, S3 performed) |
| Out-of-order step | `PROC_OUT_OF_ORDER` | Previous step attempted after current step should be active |
| Repeated step | `PROC_REPEATED_STEP` | Same step performed twice consecutively |
| Premature action | `PROC_PREMATURE` | Step performed before its preconditions are met |
| Incomplete execution | `PROC_INCOMPLETE` | Step initiated but not completed (timeout case) |
| Wrong object | `PROC_WRONG_OBJECT` | Action directed at incorrect object |

> [!NOTE]
> **Architectural responsibility:**
> AI recognition failures test the **perception model's** accuracy.
> Procedure violations test the **FSM's** anomaly detection using real perception outputs.
> Both categories must exist in the dataset, but they serve different evaluation purposes.

---

## 11. Label / Metadata Schema

The machine-readable schema companion is defined in `config/dataset_spec.json`. Below is the normative field list for each annotation record:

### Required Annotation Fields

| Field | Type | Values / Format | Required |
|---|:---:|---|:---:|
| `sample_id` | string | Deterministic (see §13) | Yes |
| `subject_id` | string | Anonymized (e.g., `SUB_001`) | Yes |
| `session_id` | string | e.g., `SES_001` | Yes |
| `take_id` | integer | 1-indexed | Yes |
| `source_video` | string | Relative path from `data/dataset/raw/` | Yes |
| `clip_start` | float | Seconds from video start | Yes |
| `clip_end` | float | Seconds from video start, or `null` | Yes |
| `procedure_id` | string | `EXP-001` | Yes |
| `procedure_version` | string | `1.0.0` | Yes |
| `step_id` | string | `S1`–`S5`, or `null` for non-step failures | Yes |
| `action` | string | Canonical action or `NONE` | Yes |
| `object` | string | Canonical object or `NONE` | Yes |
| `hand` | string | `LEFT` / `RIGHT` / `BOTH` | Yes |
| `speed_category` | string | `SLOW` / `NORMAL` / `FAST` | Yes |
| `viewpoint` | string | `FRONT` / `SIDE` / `OVERHEAD` / `OBLIQUE` | Yes |
| `orientation_deg` | integer | 0, 45, 90, 135, 180, 225, 270 | Yes |
| `orientation_category` | string | `NORMAL` / `MODERATE_ROTATION` / `EXTREME_ROTATION` | Yes |
| `lighting` | string | `STANDARD` / `BRIGHT` / `DIM` / `SHADOW` | Yes |
| `occlusion` | string | `NONE` / `PARTIAL` / `FULL` | Yes |
| `completion_state` | string | `COMPLETE` / `INCOMPLETE` / `FAILURE` | Yes |
| `procedure_correctness` | string | `CORRECT` / `INCORRECT` | Yes |
| `failure_type` | string | Category tag from §10 or `null` | Conditional |
| `split` | string | `TRAIN` / `VALIDATION` / `TEST` | Yes |
| `notes` | string | Free text | No |

---

## 12. Dataset Directory Structure

No dataset files are created in this gate. The following is the **planned directory layout** for future collection:

```
data/
└── dataset/
    ├── raw/                        ← Original unprocessed video recordings
    │   ├── SUB_001_SES_001.mp4
    │   ├── SUB_001_SES_002.mp4
    │   └── ...
    ├── clips/                      ← Trimmed action clips (extracted from raw)
    │   ├── TRAIN/
    │   │   ├── PICK_RED/
    │   │   ├── PLACE_RED/
    │   │   ├── PICK_BLUE/
    │   │   ├── PLACE_BLUE/
    │   │   └── CLOSE_LID/
    │   ├── VALIDATION/
    │   │   └── (same structure)
    │   └── TEST/
    │       └── (same structure)
    ├── annotations/                ← One JSON manifest per clip
    │   └── EXP-001_v1.0.0_annotations.json
    ├── manifests/                  ← Split manifests (lists of sample_ids per split)
    │   ├── train_manifest.json
    │   ├── validation_manifest.json
    │   └── test_manifest.json
    └── README.md                   ← Dataset collection notes and provenance
```

The `training/data/raw/hmdb51_sta/` HMDB51 data must **not** be co-mingled with the EXP-001 experiment-specific dataset. They serve different purposes (generic pretraining vs procedure-specific evaluation).

---

## 13. Naming Convention

### Raw Recording Files

Format: `{SUBJECT_ID}_{SESSION_ID}.mp4`

Example: `SUB_001_SES_001.mp4`

### Clip Files

Format: `{SUBJECT_ID}_{SESSION_ID}_T{TAKE_ID:02d}_{ACTION}_{OBJECT}_{COMPLETION_STATE}.mp4`

Examples:
- `SUB_001_SES_001_T01_PICK_RED_RED_SAMPLE_COMPLETE.mp4`
- `SUB_002_SES_003_T02_CLOSE_LID_CONTAINER_LID_INCOMPLETE.mp4`
- `SUB_003_SES_001_T01_PICK_BLUE_BLUE_SAMPLE_FAILURE_SKIPPED_STEP.mp4`

### Annotation Sample ID

Format: `EXP001_{SUBJECT_ID}_{SESSION_ID}_T{TAKE_ID:02d}_{ACTION}_{OBJECT}`

Example: `EXP001_SUB_001_SES_001_T01_PICK_RED_RED_SAMPLE`

### Rules

- No spaces in filenames; use underscores.
- Subject IDs must be anonymized (no real names).
- Take IDs are zero-padded two-digit integers.
- Action and object labels use canonical vocabulary exactly.

---

## 14. Data Quality Checklist

For every collected clip, verify before adding to the dataset:

- [ ] Video file is readable and not corrupt.
- [ ] Clip has valid `clip_start` and `clip_end` timestamps.
- [ ] Minimum 8 frames between start and end (or minimum duration met).
- [ ] `action` is from the canonical vocabulary.
- [ ] `object` is from the canonical vocabulary.
- [ ] `subject_id` is correctly assigned.
- [ ] `session_id` is correctly assigned.
- [ ] `split` assignment is consistent with subject-level split rule (same subject → same split).
- [ ] `completion_state` correctly reflects what occurred.
- [ ] `failure_type` is populated if `completion_state` is `FAILURE`.
- [ ] No duplicate `sample_id` in annotation manifest.
- [ ] No cross-split subject leakage (same subject does not appear in two different splits).
- [ ] Annotation is consistent with video content (spot-check).
- [ ] Object is visible in the clip (adequate frame content).
- [ ] Orientation metadata correctly recorded.
- [ ] Lighting condition metadata correctly recorded.

**These checks have not been executed — no actual dataset exists yet.**

---

## 15. Part-1 Handoff Specification

This section answers: *"What exactly should the revised Part-1 model-training pipeline receive from Part 2's dataset specification?"*

### Canonical Labels

```
Actions (5):
  PICK_RED
  PLACE_RED
  PICK_BLUE
  PLACE_BLUE
  CLOSE_LID

Objects (4):
  RED_SAMPLE
  BLUE_SAMPLE
  SAMPLE_CONTAINER
  CONTAINER_LID
```

### Metadata Fields Required

All fields from §11 must be populated per sample. Specifically, training must respect:
- `subject_id` for split assignment (not random clip-level splits)
- `completion_state` for filtering positive vs negative examples
- `failure_type` for dedicated failure evaluation subsets
- `orientation_deg` and `orientation_category` for robustness evaluation

### Split Rules

Enforce subject-level isolation: no subject appears in more than one of TRAIN / VALIDATION / TEST.

### Failure Categories

Part 1 must receive both positive examples (`POSITIVE_COMPLETE`) and negative examples (`INCOMPLETE_*`, `FAILURE_*`) as labeled in §5. The training pipeline may choose how to use negatives (e.g., as a background/null class, or a separate "violation" class).

### Orientation Metadata

All clips must carry `orientation_deg`. Part 1's training augmentation strategy for microgravity orientation robustness will be defined in Gate B4.

### Object/Action Relationships

As defined in §3 and §4 — each action is paired with exactly one target object. The object-action relationship is fixed for the procedure:

| Action | Object |
|---|---|
| `PICK_RED` | `RED_SAMPLE` |
| `PLACE_RED` | `RED_SAMPLE` |
| `PICK_BLUE` | `BLUE_SAMPLE` |
| `PLACE_BLUE` | `BLUE_SAMPLE` |
| `CLOSE_LID` | `CONTAINER_LID` |

### Temporal Boundaries

Clip boundaries must be provided as timestamps, not frame numbers, to remain resolution-independent.

### Architecture Freedom

**Part 1 is free to choose the final neural architecture** (1D-TCN, R(2+1)D, SlowFast, or other). This specification defines the data and label contract, not the model internals. The model must produce:

```python
{
    "action": "PICK_RED",    # one of the 5 canonical action strings
    "object": "RED_SAMPLE",  # one of the 4 canonical object strings
    "confidence": 0.94,      # float [0.0, 1.0]
    "timestamp": "..."       # ISO 8601
}
```

---

## 16. Current Dataset and Model Situation

### A. Current Part-1 Baseline Dataset — HMDB51

- **Location:** `training/data/raw/hmdb51_sta/`
- **Role:** Current Part-1 dataset source. Used for training the baseline binary classifier and for general human action recognition pretraining.
- **Current Model Classes:** `catch`, `not-catch` (binary).
- **Incremental Expansion:** As the SIH project progresses through later development stages, additional useful HMDB51 action classes may be incorporated to improve model breadth.
- **Existing HMDB51 classes relevant to manipulation:** `catch`, `pick`, `pour`, `throw` — these have surface similarity to EXP-001 motions but are **not equivalent to** and **must not be mapped as** the procedure-specific labels.

### B. Current Part-1 Exported Model Status

- **Classes:** `catch`, `not-catch`
- **Status:** Exported from Colab training on HMDB51; **NOT integrated into Part 2**.
- **Part 2 dependency:** None — Part 2 continues model-independently.

### C. Why HMDB51 Alone Cannot Serve Direct EXP-001 Procedure Recognition

HMDB51 is the correct foundation for the Part-1 pretraining phase. However, its generic labels cannot directly represent EXP-001 procedure semantics:

1. No HMDB51 class encodes object color identity (`RED_SAMPLE` vs `BLUE_SAMPLE`).
2. No HMDB51 class encodes container-relative placement actions (`PLACE_RED`, `PLACE_BLUE`).
3. No HMDB51 class represents `CLOSE_LID` as an explicit experiment-specific sealing action.
4. `catch` ≠ `PICK_RED`; `pick` ≠ `PICK_BLUE` — the motions may be similar but the procedural semantics are distinct.
5. HMDB51 was not recorded in the EXP-001 workspace environment.

**This does not make HMDB51 irrelevant.** It remains the active Part-1 baseline. It is simply not the final experiment-specific dataset for five-action EXP-001 procedure recognition.

**Do not delete, modify, or retrain the current model in this gate.**

### D. Future Dataset Pathway

The staged dataset expansion for the SIH project is:

| Stage | Dataset | Purpose |
|---|---|---|
| **Current** | HMDB51 | General pretraining baseline; `catch`/`not-catch` binary model |
| **Near-term** | Additional HMDB51 classes (if applicable) | Incremental class expansion during SIH development |
| **Future** | EXP-001-specific labeled clips (this protocol) | Direct 5-action procedure recognition |
| **Future** | Other relevant external or custom datasets | Microgravity robustness, object-specific recognition |

The future revised Part-1 model will use this dataset specification as its label contract. The revised model may be built on top of the existing HMDB51-pretrained backbone (transfer learning) or from scratch — that decision belongs to Part 1.

---

## 17. Open Parameters

The following parameters are currently unspecified and must be determined before or during dataset collection:

| Parameter | Status | Resolution Path |
|---|:---:|---|
| Exact object dimensions (`RED_SAMPLE`, `BLUE_SAMPLE`, `SAMPLE_CONTAINER`, `CONTAINER_LID`) | `UNSPECIFIED` | Physical experiment setup required |
| Object color/spectral properties (RGB/HSV ranges) | `UNSPECIFIED` | Calibration under planned lighting |
| Physical workspace geometry (rack positions, container location) | `UNSPECIFIED` | Lab setup documentation required |
| Camera mounting position and angle | `UNSPECIFIED — PROPOSED TARGET` | Standard overhead or 45° front mount |
| Camera field of view and resolution | `UNSPECIFIED — PROPOSED TARGET` | Minimum 720p @ 15 FPS |
| Minimum action duration threshold (seconds) | `UNSPECIFIED — PROPOSED TARGET` | 0.5s proposed; requires validation |
| Microgravity object motion dynamics | `UNSPECIFIED` | Requires ISS/parabolic flight or simulation |
| Exact total clip counts per action | `PROPOSED TARGET` | See §6 collection matrix |
| Lighting lux ranges | `UNSPECIFIED — PROPOSED TARGET` | To be defined per condition |

---

## 18. Acceptance Criteria

- [x] Dataset protocol document created (`docs/workstream-b-dataset-protocol.md`).
- [x] Machine-readable schema created (`config/dataset_spec.json`).
- [x] All 5 canonical procedure actions covered in taxonomy.
- [x] All 4 canonical objects covered.
- [x] Steps S1–S5 represented.
- [x] Positive, incomplete, and failure example categories defined.
- [x] Collection matrix with baseline and stretch targets provided.
- [x] Subject-level split strategy with leakage prevention defined.
- [x] TEST set evaluation subsets defined.
- [x] Orientation metadata protocol defined (B4 deferred).
- [x] Failure matrix distinguishing AI recognition failures vs procedure violations.
- [x] Naming convention established.
- [x] Data quality checklist created.
- [x] Part-1 handoff specification written.
- [x] Current `catch`/`not-catch` model documented as NOT integrated into Part 2 and insufficient alone for direct EXP-001 procedure recognition.
- [x] HMDB51 documented as current Part-1 baseline dataset source; its generic labels documented as distinct from EXP-001 procedure-specific labels.
- [x] All open physical parameters listed as `UNSPECIFIED`.
- [x] No actual dataset was fabricated or collected in this gate.
- [x] Tests pass for dataset schema validation.
