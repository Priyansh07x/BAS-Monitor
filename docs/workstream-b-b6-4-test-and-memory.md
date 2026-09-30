# Workstream B — Sub-Gate B6.4: Inference Pipeline Test Suite & Living Memory Update

## 1. Executive Summary

This document specifies the validation and completion of **Sub-Gate B6.4: Inference Pipeline Test Suite & Living Memory Update** for the ISRO SIH26174 BAS Experiment Monitor.

Gate B6.4 completes the end-to-end integration and verification of the Gate B6 (Real Inference Pipeline) sub-gates:
- **B6.1:** Vocabulary alignment (`EXP-001`) & Step 0 camera rectification hook
- **B6.2:** Multi-detector perception coordination (`ObjectDetector` + `PoseDetector` + `HandDetector` + `InteractionEngine` + deterministic 5-action heuristic fallback)
- **B6.3:** Public AI Result Adapter enforcing the frozen 8-field public AI schema (`docs/architecture.md` §2)
- **B6.4:** Comprehensive inference pipeline test suite ([`tests/test_inference_pipeline.py`](../tests/test_inference_pipeline.py)) and living memory update

---

## 2. Complete B6 Runtime Architecture & Data Flow

The B6 inference sequence runs synchronously per incoming frame:

```
                          ┌─────────────────────────────┐
                          │   Raw Input Video Frame     │
                          │     (H x W x 3 uint8)       │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Step 0: Camera Rectification Hook     │
                     │  - Default identity bypass (disabled) │
                     │  - Static physical mounting transform │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Step 1-3: Multi-Detector Perception   │
                     │  - ObjectDetector (EXP-001 Classes)   │
                     │  - PoseDetector (33 3D Joint Mesh)    │
                     │  - HandDetector (21 Hand Landmarks)   │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Step 4: Spatial Interaction Engine    │
                     │  - Hand-Object Euclidean reach radii  │
                     │  - Secondary container spatial metrics│
                     │    (IoU, distance, scale proximity)   │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Step 5: Feature Vector Extraction     │
                     │  - 99 3D Body Landmarks (33 x 3)      │
                     │  - 12 Object Centroids (4 x 3)        │
                     │  = Exactly 111-D feature vector       │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Step 6: 30-Frame Temporal Window      │
                     │  - Stacks into (1, 30, 111) tensor    │
                     │  - 5-frame warm-up behavior           │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Step 7: Action Recognition            │
                     │  - TFLite 1D-TCN (if checkpoint exists│
                     │  - Deterministic 5-Action Fallback    │
                     │    (PICK/PLACE RED/BLUE, CLOSE_LID)   │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Step 8: Public AI Result Adapter      │
                     │  - Normalizes timestamps & confidence │
                     │  - Validates canonical vocabularies   │
                     │  - Maps internal status to 3-state    │
                     │  - Strips internal perception payload │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Frozen 8-Field Public Contract        │
                     │ {"timestamp", "action", "object",     │
                     │  "confidence", "expected_step",       │
                     │  "detected_step", "status", "next"}   │
                     └───────────────────────────────────────┘
```

---

## 3. Contracts & Invariants Validated

### 3.1 111-D Spatial Feature Contract
- **Shape:** Exactly `(111,)` float32 array.
- **Indices 0–98:** 33 body pose landmarks $\times$ 3 coordinates $(x, y, z) = 99$ features.
- **Indices 99–110:** Up to 4 primary detected objects $\times$ 3 parameters $(c_x, c_y, \text{confidence}) = 12$ features (zero-padded if $<4$ objects).
- **Temporal Stacking:** Sliding window stacks 30 vectors into `(1, 30, 111)` input tensor.

### 3.2 Frozen 8-Field Public AI Schema
- Conforms strictly to [`docs/architecture.md`](../architecture.md) §2:
  `{"timestamp", "action", "object", "confidence", "expected_step", "detected_step", "status", "next_step"}`.
- **Zero Diagnostics Leakage:** Internal perception fields (`objects`, `pose`, `hands`, `interaction`, `rectification_applied`, `frame_index`, `metadata`, `annotated_frame`) are strictly isolated and never leak into the public result dictionary.

### 3.3 Public Status Mapping Table

| Internal Pipeline / FSM State | Public Contract `status` | Epistemic Rationale |
| :--- | :--- | :--- |
| `VALID`, `IDLE`, `RUNNING`, `COMPLETED` | `VALID` | Protocol execution is nominal or within expected rest state. |
| `SKIPPED` | `SKIPPED` | Step skipped by operator. |
| `OUT_OF_SEQUENCE`, `OUT_OF_ORDER` | `OUT_OF_SEQUENCE` | Step attempted out of protocol order. |
| `UNCERTAIN`, `LOW_CONFIDENCE` | `OUT_OF_SEQUENCE` | Visual confidence below threshold; supervisor review required. |
| `UNRECOGNIZED`, `ANOMALY`, `IGNORED` | `OUT_OF_SEQUENCE` | Out-of-vocabulary or anomalous action detected. |
| `<unknown / unmapped>` | `OUT_OF_SEQUENCE` | Safe fallback ensures unvalidated strings do not propagate. |

---

## 4. Epistemic Boundaries & Limitations

1. **Neural Model Artifacts Status:**
   - **TFLite (`data/temporal_action.tflite`):** NOT COMPILED / NOT PRESENT.
   - **YOLO Detection Weights:** NOT PRESENT.
   - **Hailo-8L HEF Binary (`data/vision_pipeline.hef`):** NOT PRESENT.
   - The absence of these model weights is safely handled via edge CPU fallbacks without raising unhandled exceptions or crashing the runtime.
2. **Heuristic Development Fallback:**
   - The 5-action classification heuristic in `ActionClassifier.classify()` is strictly a development and integration scaffold. It uses spatial proximity and container geometry to simulate action triggers. It is **NOT** a trained neural network.
3. **Status Enum Translation:**
   - The internal FSM in Workstream B maintains richer states (`UNCERTAIN`, `LOW_CONFIDENCE`, `OUT_OF_ORDER`). The public adapter maps these to `OUT_OF_SEQUENCE` to satisfy the frozen `architecture.md` 3-state contract without altering FSM internals.

---

## 5. Verification & Test Suite Summary

- **Focused Test Suite:** [`tests/test_inference_pipeline.py`](../tests/test_inference_pipeline.py) — 65/65 tests PASSED in 1.12s.
- **Contract Test Suite:** [`tests/test_b6_3_result_adapter.py`](../tests/test_b6_3_result_adapter.py) — 60/60 tests PASSED in 0.86s.
- **Coordination Test Suite:** [`tests/test_b6_2_multi_detector_coordination.py`](../tests/test_b6_2_multi_detector_coordination.py) — 18/18 tests PASSED in 0.35s.
- **Vocabulary Test Suite:** [`tests/test_b6_1_vocabulary_rectification.py`](../tests/test_b6_1_vocabulary_rectification.py) — 15/15 tests PASSED in 0.31s.
- **Full Workstream B Regression:** 377/377 tests PASSED in 64.16s across 18 test modules.
