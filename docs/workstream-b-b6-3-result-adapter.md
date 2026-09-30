# Workstream B — Sub-Gate B6.3: Public AI Result Adapter & Contract Compliance Specification

## 1. Executive Summary

This document specifies the implementation and validation of **Sub-Gate B6.3: Public AI Result Adapter & Contract Compliance** for the ISRO SIH26174 BAS Experiment Monitor.

The primary objective of Gate B6.3 is to establish an explicit, robust transformation barrier between the internal multimodal perception graph (rectification, YOLO bounding boxes, MediaPipe 3D body pose mesh, 21-keypoint hand skeletons, 111-D flattened vectors, and 1D-TCN action logits) and the downstream public API consumers (e.g., FSM, ProcedureManager, UI HUD, REST endpoints).

The public contract is strictly frozen in [`architecture.md`](../architecture.md) §2 to an **exact 8-field schema**. Under no circumstances may internal perception keys leak, nor may required public keys be omitted or renamed.

---

## 2. Frozen 8-Field Public Schema Specification

The public contract dictionary returned by `AIResultAdapter.adapt()` or `InferencePipeline.process_frame_public()` adheres strictly to the following 8 keys:

```json
{
  "timestamp": "2026-09-24T18:00:00.000000",
  "action": "PICK_RED",
  "object": "RED_SAMPLE",
  "confidence": 0.95,
  "expected_step": "S1",
  "detected_step": "S1",
  "status": "VALID",
  "next_step": "S2"
}
```

### 2.1 Field Definitions & Value Constraints

| Field | Type | Allowed Values / Formats | Validation & Normalization Policy |
| :--- | :--- | :--- | :--- |
| `timestamp` | `str` | ISO-8601 string (`YYYY-MM-DDTHH:MM:SS.ffffff`) | Preserves ISO strings; converts numeric Unix epoch timestamps (seconds) to ISO-8601; defaults to `datetime.now().isoformat()` if null. |
| `action` | `str` | `PICK_RED`, `PLACE_RED`, `PICK_BLUE`, `PLACE_BLUE`, `CLOSE_LID`, `IDLE` | Validated against canonical EXP-001 vocabulary; legacy strings (`PIPETTE_TRANSFER`, `INSERT_ANALYZER`, etc.) are rejected and remapped to `IDLE`. |
| `object` | `str` | `RED_SAMPLE`, `BLUE_SAMPLE`, `SAMPLE_CONTAINER`, `CONTAINER_LID`, `NONE` | Validated against canonical EXP-001 vocabulary; legacy strings (`SAMPLE_VIAL`, `PIPETTE`, etc.) remapped to canonical default based on action or `NONE`. |
| `confidence` | `float` | `[0.0, 1.0]` | Float parsed and clamped strictly between 0.0 and 1.0; non-numeric values default to 0.0. Rounded to 2 decimal places. |
| `expected_step` | `str` \| `None` | `S1`, `S2`, `S3`, `S4`, `S5`, or `None` | Normalized from strings/integers (`"1"`, `1`, `"s1"` $\to$ `"S1"`). Defaults to current detected step or `None`. |
| `detected_step` | `str` \| `None` | `S1`, `S2`, `S3`, `S4`, `S5`, or `None` | Inferred directly from canonical action (`PICK_RED` $\to$ `S1`, etc.) or normalized from raw input. |
| `status` | `str` | `VALID`, `SKIPPED`, `OUT_OF_SEQUENCE` | Strictly mapped from internal FSM/perception statuses to the 3-state public status vocabulary. |
| `next_step` | `str` \| `None` | `S1`, `S2`, `S3`, `S4`, `S5`, or `None` | Inferred from sequence progression (`S1` $\to$ `S2`, `S2` $\to$ `S3`, `S3` $\to$ `S4`, `S4` $\to$ `S5`, `S5` $\to$ `None`). |

---

## 3. Status Vocabulary Mapping

Internal subsystems use diverse diagnostic statuses (`UNCERTAIN`, `OUT_OF_ORDER`, `LOW_CONFIDENCE`, `UNRECOGNIZED`, `ANOMALY`, `IGNORED`). The public adapter maps these exhaustively into the frozen 3-state vocabulary:

| Internal Pipeline / FSM Status | Public Contract `status` | Rationale |
| :--- | :--- | :--- |
| `VALID`, `IDLE`, `RUNNING`, `COMPLETED` | `VALID` | Action aligns with procedure expectations or nominal steady-state. |
| `SKIPPED` | `SKIPPED` | User or operator skipped a mandatory protocol step. |
| `OUT_OF_SEQUENCE`, `OUT_OF_ORDER` | `OUT_OF_SEQUENCE` | Action performed out of expected procedural order. |
| `UNCERTAIN`, `LOW_CONFIDENCE` | `OUT_OF_SEQUENCE` | Action below confidence threshold; requires supervisor verification. |
| `UNRECOGNIZED`, `ANOMALY`, `IGNORED` | `OUT_OF_SEQUENCE` | Out-of-vocabulary or anomalous action detected. |
| `*` (Any unmapped / unknown status) | `OUT_OF_SEQUENCE` | Fail-safe default prevents unvalidated strings from propagating. |

---

## 4. Pipeline Integration Architecture

The adapter is integrated into the inference pipeline via two complementary paths:

```
                          ┌─────────────────────────────┐
                          │   Raw Input Video Frame     │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │     InferencePipeline       │
                          │   (Rectification + YOLO +   │
                          │  MediaPipe + 1D-TCN + HUD)  │
                          └──────────────┬──────────────┘
                                         │ Internal Multimodal Payload
                                         │ (Rect, Mesh, Hands, IoU, Logits)
                                         ▼
                          ┌─────────────────────────────┐
                          │       AIResultAdapter       │
                          │    - Normalization          │
                          │    - Vocabulary Validation  │
                          │    - Status Mapping         │
                          │    - Schema Enforcement     │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                         ┌───────────────────────────────┐
                         │ Frozen 8-Field Public Payload │
                         │ (timestamp, action, object,   │
                         │  confidence, expected_step,   │
                         │  detected_step, status, next) │
                         └───────────────────────────────┘
```

Callers can invoke `pipeline.process_frame_public(frame)` directly to obtain the clean 8-field public dictionary or call `AIResultAdapter.adapt(internal_dict)` on raw outputs.

---

## 5. Epistemic Boundaries & Real-Inference Constraints

1. **Deterministic Adapter Overhead:** The adapter transformation is strictly deterministic and lightweight ($<0.05$ ms latency overhead), adding zero neural compute.
2. **Heuristic vs Neural Operation:** In environments where Hailo-8 or TFLite compiled models are absent, the upstream `InferencePipeline` operates on heuristic mock engines. The adapter ensures that regardless of the perception backend, downstream consumers always receive a compliant 8-field payload.
3. **Immutability of Public Contract:** Changes to `architecture.md` §2 are frozen; all future additions must either be internal pipeline fields or routed through formal schema revision gates.
