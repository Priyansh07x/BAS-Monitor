# Workstream B — Gate B7.3: Threading Verification & Living Memory Update

**ISRO SIH26174 BAS Experiment Monitor — Workstream B**  
**Gate:** B7.3  
**Status:** PASSED  
**Date:** 2026-09-24  

---

## 1. Executive Summary

Gate B7.3 serves as the final verification and living memory consolidation gate for the non-blocking AI execution architecture (Sub-gates B7.0–B7.3). 

Comprehensive verification tests in [`tests/test_b7_3_threading_verification.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_b7_3_threading_verification.py) validate all 40 mandatory criteria across thread topology, frame buffer replacement dynamics, producer non-blocking guarantees, Qt signal bridging, error containment, and memory/queue constraints.

---

## 2. Verified Thread Topology

```
┌────────────────────────────────────────────────────────┐
│ Camera Ingestion / Video Loader (Producer Boundary)    │
│  - Pull-driven via Frontend Polling (Bridge.getCameraFrame)
│  - Acquires frame synchronously (Camera.read)          │
│  - Non-blocking O(1) submission to LatestFrameBuffer   │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼ (at most 1 pending frame in buffer)
┌────────────────────────────────────────────────────────┐
│ LatestFrameBuffer (backend/ai/inference_worker.py)     │
│  - Single-slot thread-safe replacement buffer          │
│  - Stale unread frame replaced on newer frame arrival  │
│  - Wakes worker thread immediately via Condition       │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼ (consumed sequentially)
┌────────────────────────────────────────────────────────┐
│ Dedicated Worker Thread ("AIInferenceWorker")          │
│  - Confines InferencePipeline execution to this thread │
│  - Never blocks camera preview or disk recording       │
│  - Catches exceptions and emits safe fallback results  │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼ (generic callback)
┌────────────────────────────────────────────────────────┐
│ Qt Bridge Boundary (backend/bridge.py)                 │
│  - Converts callback into Qt Queued Signals:           │
│      * aiResultReady(dict)                             │
│      * aiResultJsonReady(str)                          │
│  - Dispatches safely to UI, FSM, and loggers           │
└────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Verification Results

### 3.1. Worker Lifecycle & Idempotency
- **Single Instance:** `AppState` creates and owns exactly one canonical `InferenceWorker` instance; `Bridge` references it directly without creating duplicate workers.
- **Start / Stop / Restart:** Repeated `start()`, `stop()`, and `shutdown()` calls are completely idempotent and thread-safe.
- **Thread Naming:** Background thread is named `"AIInferenceWorker"`.
- **Zero Thread Leaks:** Clean shutdown terminates worker within $< 0.5$ s; no lingering threads remain after multiple start/stop cycles.

### 3.2. Latest-Frame Single-Slot Replacement Dynamics
- **Bounded Buffer:** `LatestFrameBuffer` strictly contains 0 or 1 item.
- **Non-Blocking Ingestion:** Producer `submit_frame()` completes in $< 0.05$ ms, unhindered by long inference times.
- **Stale Frame Dropping:** Under rapid frame flooding (100+ frames), intermediate unconsumed frames are replaced immediately, and the freshest frame is guaranteed to be processed.

### 3.3. Thread Confinement & Concurrency Protection
- **Worker Confinement:** `InferencePipeline.process_frame_public()` executes exclusively on `AIInferenceWorker`, never on the main Qt GUI thread or camera thread.
- **Serialization:** Zero concurrent invocations of the underlying multi-detector graph or temporal buffer stack occur.
- **Frame Reference Safety:** Verified that upstream producers (`Camera`, `VideoRecorder`, `IPStreamer`) and downstream perception pipelines do not mutate frame buffers in-place.

### 3.4. Qt Signal Boundary & Contract Compliance
- **Zero Qt in Worker:** `backend/ai/inference_worker.py` contains 0 Qt/PySide imports.
- **Queued Signal Dispatch:** `Bridge._on_ai_result_from_worker` converts callback events to `aiResultReady` and `aiResultJsonReady`.
- **Frozen Contract:** Public result strictly complies with the 8-field schema (`timestamp`, `action`, `object`, `confidence`, `expected_step`, `detected_step`, `status`, `next_step`).
- **Zero Diagnostic Leakage:** Internal bounding boxes, keypoints, logits, and feature arrays are completely shielded from public listeners.

### 3.5. Error Containment & Missing Weights Fallback
- **Exception Isolation:** Inference crashes (e.g. simulated exceptions or faulty inputs) are caught and logged; worker thread continues processing subsequent frames.
- **Callback Crash Isolation:** Exceptions raised within UI/consumer callbacks do not kill the worker thread.
- **Missing Weights Resilience:** Runs in edge CPU heuristic mode without physical HEF/ONNX weights, outputting valid `EXP-001` contract dictionaries.

---

## 4. Test Suite Summary

- **Focused Test Suite:** [`tests/test_b7_3_threading_verification.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_b7_3_threading_verification.py) — 40/40 PASSED in 2.11 s.
- **Integration Test Suite:** [`tests/test_inference_bridge_integration.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_inference_bridge_integration.py) — 22/22 PASSED in 0.99 s.
- **Worker Test Suite:** [`tests/test_inference_worker.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_inference_worker.py) — 15/15 PASSED in 1.34 s.
- **Workstream B Full Regression:** 479/479 PASSED in 66.37 s (`pytest --ignore=tests/test_video_recorder.py`).
- **Recorder Regression:** `tests/test_video_recorder.py` fails during collection due to pre-existing missing `cv2` binary in Python 3.14 test environment (`ENVIRONMENT-BLOCKED`).

---

## 5. Explicit Boundary for Next Gate (B8)

Completion of Gate B7 establishes a proven non-blocking execution architecture. It does **not** imply:
1. 30 FPS camera preview is benchmarked under production load.
2. AI perception runs at an optimized 10–15 FPS target.
3. Dynamic frame-rate matching, FPS decimation, or throttling is active.
4. Physical neural model weights are loaded.

These optimizations are explicitly deferred to **Gate B8 (Frame-Rate Strategy)**.
