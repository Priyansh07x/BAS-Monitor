# Workstream B — Gate B16.1 AI Failure Injection & Perception Robustness Suite Report

> **ISRO SIH26174 BAS Experiment Monitor — Workstream B (AI & Procedure Intelligence)**  
> **Sub-Gate:** B16.1 — AI Failure Injection & Perception Robustness Suite  
> **Author:** Antigravity (AI Assistant)  
> **Date:** 2026-09-30  
> **Status:** **PASSED** (17/17 Focused Tests PASS; 934/934 Full Repository Regression PASS across 42 Test Suites)

---

## 1. Executive Summary

Sub-Gate **B16.1: AI Failure Injection & Perception Robustness Suite** implements a dedicated, non-duplicative integration test suite in [`tests/test_b16_1_ai_failure_injection.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_b16_1_ai_failure_injection.py) covering all eight (8) AI and perception failure modes (**F01–F08**) identified in the approved Gate B16.0 audit:

- **F01 (Wrong Action)**: Semantic action-object mismatch & transient noise burst
- **F02 (Missed Action)**: Intermittent frame drop & complete IDLE leak
- **F03 (Low Confidence)**: Sub-marginal ($< 0.50$) isolation & marginal ($[0.50, 0.70)$) evidence accumulation
- **F04 (Occlusion)**: Missing object bounding box, hand occlusion / reach violation, and synthetic cutout patch
- **F05 (Poor Lighting)**: Extreme underexposure (pure black $0$), overexposure (pure white $255$), and synthetic brightness disturbances
- **F06 (Blur)**: Gaussian defocus blur ($k=7$) keypoint degradation stability
- **F07 (Fast Motion)**: Rapid alternating candidate flicker & single-slot latest-frame replacement dynamics
- **F08 (Multiple Objects)**: Cluttered workspace distractor objects on holding tray & 3D bounding volume reach selection

The suite formally validates that perception-level noise, visual disturbances, and multimodal contradictions are strictly contained by **B11** (`MultimodalConsistencyEvaluator` & `UncertaintyHandler`) and **B10** (`TemporalConfirmationEngine`), completely isolating the authoritative **FSM** (`SequenceValidatorFSM`) from spurious step transitions and preventing false alarm recovery events in `RecoveryManager` (**B12**).

---

## 2. Scope & Implementation Summary

Sub-Gate B16.1 adheres strictly to the non-regression and boundary invariants established across Workstream B:

1. **No Duplicate Architectures**: Reuses existing core modules ([`inference_pipeline.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/inference_pipeline.py), [`multimodal_consistency.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/multimodal_consistency.py), [`uncertainty_handler.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/uncertainty_handler.py), [`temporal_filter.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/temporal_filter.py), [`inference_worker.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/inference_worker.py), [`augmentation.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/augmentation.py), [`spatial_geometry.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/hmr/spatial_geometry.py)) without creating redundant failure handlers.
2. **Authority Isolation**: Verifies that B16 is purely a test/injection authority without introducing new production layers or background threads.
3. **Contract Strictness**: Validates that all public outputs returned by `InferencePipeline.process_frame_public()` strictly conform to the frozen 8-field public AI contract ([`docs/architecture.md`](file:///E:/Technical_Projects/2026-SIH/SIH26174/docs/architecture.md) §2).
4. **Epistemic Honesty**: Testing uses deterministic synthetic/mock injection. No claims of real physical camera robustness or trained neural network weights are made.

---

## 3. Exact Files and Tests Added or Modified

| File | Action | Purpose / Content |
|---|:---:|---|
| [`tests/test_b16_1_ai_failure_injection.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_b16_1_ai_failure_injection.py) | **CREATED** | 17 dedicated integration tests covering F01–F08 and cross-authority zero-recovery noise stress. |
| [`tests/test_workstream_b_progress.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_workstream_b_progress.py) | **MODIFIED** | Updated baseline progress assertion to reflect completed Phase B14 and B14.3. |
| [`docs/workstream-b-progress.md`](file:///E:/Technical_Projects/2026-SIH/SIH26174/docs/workstream-b-progress.md) | **MODIFIED** | Updated current gate to B16.1 PASSED, recorded B15 and B16.1 in completed gates and changelog, set B16.2 as next gate. |
| [`docs/decision.md`](file:///E:/Technical_Projects/2026-SIH/SIH26174/docs/decision.md) | **MODIFIED** | Added `[DECISION-060]` documenting Gate B16.1 decisions, invariants, and pass status. |
| [`docs/Workstream B — Gate B16.1 AI Failure Injection & Perception Robustness Suite Report.md`](file:///E:/Technical_Projects/2026-SIH/SIH26174/docs/Workstream%20B%20%E2%80%94%20Gate%20B16.1%20AI%20Failure%20Injection%20%26%20Perception%20Robustness%20Suite%20Report.md) | **CREATED** | This formal completion report. |

---

## 4. F01–F08 Perception Failure Modes Coverage Matrix

| Code | Failure Mode | Test Method(s) | Containment / Defense Authority | Verified Behavior | Status |
|:---:|---|---|:---:|---|:---:|
| **F01** | Wrong Action | `test_f01_semantic_action_object_mismatch_containment`<br>`test_f01_transient_wrong_action_burst_filtered_by_b10` | B11.2 (Multimodal Consistency)<br>B10 (Temporal Filter) | Semantic mismatch (`PICK_RED` + `BLUE_SAMPLE`) flagged as `CROSS_MODAL_CONFLICT` and rejected (`is_reliable=False`). 2-frame spike of `PLACE_RED` rejected by 3/5 M-of-N window. FSM unchanged at S1; 0 recovery alerts. | **PASS** |
| **F02** | Missed Action | `test_f02_intermittent_frame_drop_and_recovery_to_nominal`<br>`test_f02_complete_missed_action_idle_leak` | B10 (Temporal Filter)<br>FSM (Sequence Validator) | Intermittent frame loss (`[PICK_RED, IDLE, PICK_RED, IDLE, PICK_RED]`) confirms S1 upon accumulating 3 valid frames in rolling window of 5. Complete IDLE leak maintains nominal S1 expectation without raising false alarms. | **PASS** |
| **F03** | Low Confidence | `test_f03_sub_marginal_confidence_isolation`<br>`test_f03_marginal_confidence_evidence_accumulation_and_resolution` | B11.1 (Uncertainty Handler) | Sub-marginal confidence ($< 0.50$) marked `UNCERTAIN_LOW_CONFIDENCE` (`is_reliable=False`) and isolated. Marginal confidence ($0.60$) accumulates across 3 frames (`UNCERTAIN_EVIDENCE_ACCUMULATING` $\to$ `RESOLVED`), feeding B10 only when resolved. | **PASS** |
| **F04** | Occlusion | `test_f04_missing_target_object_occlusion_containment`<br>`test_f04_hand_occlusion_spatial_contradiction_containment`<br>`test_f04_synthetic_cutout_occlusion_augmentation_stability` | B11.2 (Multimodal Consistency)<br>AugmentationEngine | Target object occlusion flagged as `UNCERTAIN_MISSING_OBJECT`. Hand occlusion / reach violation flagged as `UNCERTAIN_SPATIAL_CONTRADICTION`. Synthetic 20% cutout patch processed deterministically. FSM isolated. | **PASS** |
| **F05** | Poor Lighting | `test_f05_extreme_underexposure_and_overexposure_containment`<br>`test_f05_synthetic_brightness_disturbance_stability` | InferencePipeline<br>AugmentationEngine | Pure black ($0$) and pure white ($255$) frames produce graceful fallback to `action="IDLE", status="VALID"`. Dim/bright augmented frames maintain schema contracts with zero exceptions. | **PASS** |
| **F06** | Blur | `test_f06_synthetic_gaussian_blur_and_keypoint_degradation` | FrameProcessor<br>AugmentationEngine | Gaussian defocus blur ($k=7$) preserves 111-D feature vector extraction determinism and pipeline throughput without crashing or mutating FSM state. | **PASS** |
| **F07** | Fast Motion | `test_f07_rapid_action_flicker_containment`<br>`test_f07_fast_motion_single_slot_buffer_drop_dynamics` | B10 (Temporal Filter)<br>LatestFrameBuffer | Alternating candidate flicker (`[PICK_RED, PICK_BLUE, ...]`) prevents either candidate from reaching majority threshold ($M=3$). Single-slot `LatestFrameBuffer` replaces unconsumed frames in $O(1)$ (9/10 dropped) with zero queue latency. | **PASS** |
| **F08** | Multiple Objects | `test_f08_cluttered_workspace_distractor_object_handling`<br>`test_f08_3d_spatial_geometry_multiple_volume_selection` | InteractionEngine<br>B14.2 Spatial Geometry | All 4 canonical objects present on tray: hand closest to `RED_SAMPLE` evaluates cleanly without distractor interference and validates S1. 3D spatial geometry selects correct EXP-001 bounding volume. | **PASS** |

---

## 5. Detailed Expected vs. Observed Behaviors

### F01: Wrong Action
- **Scenario 1 (Semantic Mismatch)**: Classifier claims `PICK_RED` while detector observes `BLUE_SAMPLE`.
  - *Expected*: B11 evaluates `CATEGORY_CROSS_MODAL_CONFLICT` (`is_reliable=False`), uncertainty handler logs conflict, B10 temporal filter is bypassed, FSM remains expecting S1, RecoveryManager is IDLE.
  - *Observed*: `consistency["category"] == "CROSS_MODAL_CONFLICT"`, `is_reliable == False`, FSM `current_step_index == 0`, `last_recovery_event is None`, `rec_mgr.state == "IDLE"`. **MATCH**.
- **Scenario 2 (Transient Burst)**: 2-frame spurious spike of `PLACE_RED` during step S1.
  - *Expected*: B10 requires 3/5 votes to confirm. 2-frame spike is rejected without confirming. FSM remains at S1.
  - *Observed*: `fsm.current_step_index == 0`, `fsm.get_current_expected_step().step_id == "S1"`, zero anomalies recorded. **MATCH**.

### F02: Missed Action
- **Scenario 1 (Intermittent Frame Loss)**: 5-frame sequence: `[PICK_RED, IDLE, PICK_RED, IDLE, PICK_RED]`.
  - *Expected*: Intermittent IDLE frames clear cooldown but preserve rolling window accumulation; 3 valid PICK_RED frames in the 5-frame window trigger confirmation and advance FSM to S2.
  - *Observed*: On frame 4, B10 confirms `vote_count=3`, FSM transitions `current_step_index == 1`, public status is `"VALID"`, `next_step == "S2"`. **MATCH**.
- **Scenario 2 (Complete Missed Action / IDLE Leak)**: 10 consecutive frames of total detection failure (`IDLE`).
  - *Expected*: Pipeline returns `action="IDLE", status="VALID", expected_step="S1"`. FSM remains locked at S1 without raising false errors.
  - *Observed*: 10 frames evaluated, `fsm.current_step_index == 0`, `len(fsm.anomalies) == 0`, `last_recovery_event is None`. **MATCH**.

### F03: Low Confidence
- **Scenario 1 (Sub-Marginal Confidence)**: Perception outputs `conf = 0.35 < 0.50`.
  - *Expected*: `UncertaintyHandler` marks `UNCERTAIN_LOW_CONFIDENCE` (`is_reliable=False`), B10 is bypassed, FSM is not invoked.
  - *Observed*: `last_uncertainty_result["state"] == "UNCERTAIN_LOW_CONFIDENCE"`, `is_reliable == False`, `fsm.current_step_index == 0`. **MATCH**.
- **Scenario 2 (Marginal Confidence Accumulation)**: Perception outputs `conf = 0.60` (in $[0.50, 0.70)$).
  - *Expected*: Frames 0 & 1 accumulate evidence (`UNCERTAIN_EVIDENCE_ACCUMULATING`, `is_reliable=False`). Frame 2 resolves (`RESOLVED`, `is_reliable=True`). Subsequent frames allow B10 confirmation and step transition.
  - *Observed*: Frames 0 & 1 marked `UNCERTAIN_EVIDENCE_ACCUMULATING`; frame 2 marked `RESOLVED`; subsequent nominal frames confirm S1 in FSM. **MATCH**.

### F04: Occlusion
- **Scenario 1 (Target Object Occluded)**: Object detector returns `[]`.
  - *Expected*: B11 flags `UNCERTAIN_MISSING_OBJECT` (`is_reliable=False`), FSM is not invoked, recovery is not triggered.
  - *Observed*: `last_consistency_result["category"] == "UNCERTAIN_MISSING_OBJECT"`, `is_reliable == False`, FSM unchanged. **MATCH**.
- **Scenario 2 (Hand Occlusion / Spatial Veto)**: Hands far away ($> 1.50$ reach scale) while manipulation action is reported.
  - *Expected*: `InteractionEngine` reports `IDLE`, B11 flags `UNCERTAIN_SPATIAL_CONTRADICTION` (`is_reliable=False`), FSM is isolated.
  - *Observed*: `last_consistency_result["category"] == "UNCERTAIN_SPATIAL_CONTRADICTION"`, `is_reliable == False`, FSM unchanged. **MATCH**.

### F05: Poor Lighting
- **Scenario 1 (Black/White Frames)**: Ingestion of pure black ($0$) and pure white ($255$) frames.
  - *Expected*: Zero exceptions, safe fallback to `action="IDLE", status="VALID"`, public schema strictly preserved.
  - *Observed*: Zero exceptions; `status == "VALID"`, `action == "IDLE"`, schema valid. **MATCH**.

### F06: Blur
- **Scenario 1 (Gaussian Defocus Blur)**: Ingestion of $k=7$ blurred frames.
  - *Expected*: 111-D feature vector extraction is deterministic; pipeline throughput remains stable; FSM is not corrupted.
  - *Observed*: Pipeline processes cleanly; contract valid; FSM unchanged. **MATCH**.

### F07: Fast Motion
- **Scenario 1 (Candidate Flicker)**: Rapid alternating actions `[PICK_RED, PICK_BLUE, PICK_RED, PICK_BLUE]`.
  - *Expected*: Alternating sequence prevents either candidate from reaching majority threshold ($M=3$). FSM stays at S1.
  - *Observed*: `fsm.current_step_index == 0`, `len(fsm.anomalies) == 0`. **MATCH**.
- **Scenario 2 (Single-Slot Buffer Replacement)**: Producer submits 10 frames in rapid succession.
  - *Expected*: `LatestFrameBuffer` replaces unconsumed frames atomically in $O(1)$; `submission_count == 10`, `replacement_count == 9`; latest frame is popped.
  - *Observed*: `submission_count == 10`, `replacement_count == 9`, popped frame timestamp is `9.0`. **MATCH**.

### F08: Multiple Objects
- **Scenario 1 (Cluttered Workspace)**: All 4 canonical objects in FOV. Hand closest to `RED_SAMPLE`.
  - *Expected*: `InteractionEngine` identifies target as `RED_SAMPLE`; distractor objects are ignored; S1 confirms.
  - *Observed*: S1 confirmed; `res["object"] == "RED_SAMPLE"`, `res["next_step"] == "S2"`. **MATCH**.
- **Scenario 2 (3D Volume Selection)**: Wrist positioned at $(-0.14, 0.01, 0.00)$.
  - *Expected*: 3D kinematics matches `RED_SAMPLE` volume (center $-0.15$) and excludes `BLUE_SAMPLE` volume (center $+0.275$).
  - *Observed*: `relations["RED_SAMPLE"].is_within_reach == True`, `relations["BLUE_SAMPLE"].is_within_reach == False`. **MATCH**.

---

## 6. Authority Boundary & Invariant Verification

Sub-Gate B16.1 explicitly verified the cross-authority and architectural invariants across 100 continuous cycles of varied visual noise (occlusion, brightness disturbances, blurs, cross-modal conflicts, sub-marginal confidences, and idle frames):

```
Perception Layer (ObjectDetector / Pose / Hand / Augmentation)
       ↓
Multimodal Consistency Evaluator (B11.2)  ──[If Unreliable]──→ Isolates B10 & FSM
       ↓ [If Reliable]
Uncertainty Handler (B11.1)               ──[If Marginal]───→ Bounded Accumulator (Isolates B10 & FSM)
       ↓ [If Confident / Resolved]
Temporal Confirmation Engine (B10)        ──[If < M Votes]──→ Accumulates Window (Isolates FSM)
       ↓ [If M-of-N Confirmed]
Sequence Validator FSM (B9)               ──[Sole Procedural Authority]
       ↓ [If FSM Sequence Violation]
Recovery Manager (B12)                    ──[Authoritative Procedural Recovery Only]
```

### Critical Invariant Check
- **Noise Test**: 100 consecutive frames of random visual noise and perception conflicts processed by live `InferencePipeline`.
- **Observed FSM State**: `fsm.current_step_index == 0` (zero false transitions).
- **Observed Recovery State**: `pipeline.last_recovery_event is None`, `rec_mgr.state == "IDLE"`, `len(rec_mgr.recovery_history) == 0` (**zero false alarms**).

---

## 7. Passive Telemetry & Diagnostic Verification

The passive telemetry aggregator ([`TelemetryDiagnosticAggregator`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/telemetry_aggregator.py) from Gate B13) recorded all failure injection cycles without mutating pipeline state or violating thread safety:

- Multimodal consistency categories recorded: `CATEGORY_CROSS_MODAL_CONFLICT`, `CATEGORY_UNCERTAIN_MISSING_OBJECT`, `CATEGORY_UNCERTAIN_SPATIAL_CONTRADICTION`, `CATEGORY_UNCERTAIN_LOW_OBJECT_CONFIDENCE`, `CATEGORY_IDLE`.
- Uncertainty states tracked: `UNCERTAIN_LOW_CONFIDENCE`, `UNCERTAIN_EVIDENCE_ACCUMULATING`, `RESOLVED`, `CONFIDENT`.
- Single-slot buffer metrics verified: Frame drop count increments on producer speedups; frame drop rate accurately calculated.

---

## 8. Public Contract & Public Schema Compliance

Across all 17 integration tests and 100 noise stress cycles, every dictionary returned by `InferencePipeline.process_frame_public()` was validated against `AIResultAdapter.validate_public_contract()`:

- **Exact 8 Fields Present**: `timestamp`, `action`, `object`, `confidence`, `expected_step`, `detected_step`, `status`, `next_step`.
- **Status Enum**: Strictly restricted to `{"VALID", "SKIPPED", "OUT_OF_SEQUENCE"}`.
- **Action & Object Vocabulary**: Strictly canonical EXP-001 vocabulary (`PICK_RED`, `PLACE_RED`, `PICK_BLUE`, `PLACE_BLUE`, `CLOSE_LID`, `IDLE`; `RED_SAMPLE`, `BLUE_SAMPLE`, `SAMPLE_CONTAINER`, `CONTAINER_LID`, `NONE`).
- **Pass Rate**: 100% contract compliance across all failure conditions.

---

## 9. Test Execution Summary

### 1. Focused B16.1 Test Suite
```
pytest tests/test_b16_1_ai_failure_injection.py -v
============================= 17 passed in 1.80s ==============================
```

### 2. Relevant B9–B14 Subsystem Regressions
- `tests/test_b9_*.py` (AI ↔ FSM Integration): **40/40 PASS**
- `tests/test_b10_*.py` (Temporal Confirmation): **70/70 PASS**
- `tests/test_b11_*.py` (Uncertainty & Consistency): **74/74 PASS**
- `tests/test_b12_*.py` (Procedural Recovery): **38/38 PASS**
- `tests/test_b13_*.py` (Telemetry & Diagnostics): **53/53 PASS**
- `tests/test_b14_*.py` (3D HMR & Spatial Kinematics): **87/87 PASS**

### 3. Full Repository Regression
```
pytest --ignore=tests/test_video_recorder.py
======================= 934 passed in 247.26s (0:04:07) =======================
```
*Note: `test_video_recorder.py` is ignored because it requires the external system `cv2` binary on the host.*

---

## 10. Epistemic Limitations & Deferred Real-World Checks

In accordance with Workstream B epistemic governance rules:
1. **No Physical Neural Robustness Claim**: Tests utilize deterministic mocks, synthetic augmentations, and mathematical heuristics. No claims regarding physical neural network robustness under uncollected microgravity optical conditions are made.
2. **No Physical Hardware Dependency**: Tests do not require real camera hardware, Hailo-8L NPU accelerators, or commercial SMPL body mesh assets.
3. **No EXP-001 Procedure Dataset Claims**: Physical dataset collection remains deferred per the Gate B3 protocol.

---

## 11. Formal Gate Decision & Recommendation

### Gate Decision
**Gate B16.1 (AI Failure Injection & Perception Robustness Suite) is formally PASSED.**

### Recommendation for Next Sub-Gate
Proceed to **Gate B16.2: Procedure Error Injection Suite**.
- **Scope of B16.2**: Implement a dedicated test suite verifying the six (6) procedure failure modes:
  - P01: Skipped step (e.g. S1 $\to$ S3 jump)
  - P02: Out-of-order step (e.g. S1 $\to$ S4)
  - P03: Repeated step (e.g. executing S1 again while expecting S2)
  - P04: Premature action (e.g. closing lid S5 before sample transfer)
  - P05: Incomplete action / intermediate abort
  - P06: Timeout violation ($> 30\text{s}$ inactivity on active step)
- Validate end-to-end event generation in `RecoveryManager` (**B12**), out-of-band Qt signals, debounced speech guidance, and recovery event telemetry.
