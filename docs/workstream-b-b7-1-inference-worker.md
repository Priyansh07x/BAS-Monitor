# Workstream B — Sub-Gate B7.1: Inference Worker & Latest-Frame Buffer

## 1. Executive Summary

This document records the architectural specification and implementation of **Sub-Gate B7.1: Inference Worker & Latest-Frame Buffer** for the ISRO SIH26174 BAS Experiment Monitor.

Gate B7 decouples real-time video capture and GUI rendering from the multimodal AI perception graph. In B7.1, we implemented the foundational pure-Python worker and single-slot replacement buffer:
- **`backend/ai/inference_worker.py`**: Contains `LatestFrameBuffer` (thread-safe single-slot replacement buffer) and `InferenceWorker` (dedicated background thread lifecycle manager).
- **Decoupled Concurrency**: Upstream producers (e.g. camera capture at 30 FPS) submit frames asynchronously without blocking; the worker thread consumes the freshest pending frame (10–15 FPS inference), atomically discarding stale unconsumed intermediate frames.
- **Pure Python Design**: Zero Qt/PySide6 framework imports in the worker module, preserving portable testability and clean architectural decoupling.
- **Thread Confinement**: The full perception pipeline (`InferencePipeline`, detectors, interaction engine, action classifier deque) is owned and executed exclusively on the background worker thread.

---

## 2. Architecture & Data Flow

```
   [Camera Thread / Video Loader (30 FPS)]
                     │
                     │ submit_frame(frame, timestamp, ...) (Non-blocking)
                     ▼
       ┌───────────────────────────────┐
       │      LatestFrameBuffer        │
       │   - Single-Slot Storage       │
       │   - Atomic Overwrite          │
       │   - Condition Wakeup          │
       └──────────────┬────────────────┘
                      │
                      │ get(timeout=0.1)
                      ▼
       ┌───────────────────────────────┐
       │   InferenceWorker Thread      │
       │   - process_frame_public()    │
       │   - Full 7-Step Perception    │
       │   - 111-D Spatial Features    │
       │   - Heuristic / Fallback ML   │
       └──────────────┬────────────────┘
                      │
                      │ Dispatches 8-Field Public Contract
                      ▼
       ┌───────────────────────────────┐
       │  Result Callback / Queue      │
       │  - {timestamp, action,        │
       │     object, confidence,       │
       │     expected_step,            │
       │     detected_step, status,    │
       │     next_step}                │
       └───────────────────────────────┘
```

---

## 3. Buffer Semantics & Concurrency Rules

### 3.1 Latest-Frame Replacement Buffer
1. **Single-Slot Capacity**: Stores at most one pending frame `(frame, timestamp, metadata, expected_step)`.
2. **Atomic Replacement**: When the producer submits a new frame while a previous frame is pending unconsumed, the older frame is immediately and atomically overwritten.
3. **Non-Blocking Producer**: `submit_frame()` returns immediately (`O(1)`), preventing camera pipeline latency or frame drops.
4. **Immediate Consumer Wakeup**: `threading.Condition.notify()` wakes the sleeping inference thread as soon as a frame arrives.

### 3.2 Worker Lifecycle & Control
- `start()`: Idempotently spawns and starts the daemon background thread `AIInferenceWorker`.
- `stop(timeout)`: Signals termination, wakes waiting condition variables, and joins the thread cleanly within `timeout`.
- `pause()`: Suspends inference loop while allowing incoming frames to stage in the single-slot buffer.
- `resume()`: Resumes inference loop, immediately picking up the freshest staged frame.
- `reset()`: Flushes pending frames, drains internal result queues, and resets pipeline temporal buffers (`_vector_buffer`, `_frame_index`, interaction tracker).
- `shutdown(timeout)`: Stops worker, drains buffers, and releases underlying hardware/detector resources (`pipeline.release()`).

---

## 4. Error Resilience & Contract Conformance

1. **Exception Isolation**: Runtime exceptions during pipeline execution are caught, logged, and increment `error_count`. They never terminate the worker thread.
2. **Fallback Delivery**: If an unexpected exception occurs during frame evaluation, a safe default public contract is delivered:
   ```json
   {
       "timestamp": "<ISO-8601>",
       "action": "IDLE",
       "object": "NONE",
       "confidence": 0.0,
       "expected_step": "<expected_step>",
       "detected_step": null,
       "status": "OUT_OF_SEQUENCE",
       "next_step": null
   }
   ```
3. **Missing Neural Checkpoints**: In the absence of trained neural weights (`data/action_model.pt`), the worker seamlessly runs the deterministic heuristic fallback pipeline.

---

## 5. Verification Matrix (Gate B7.1)

| Requirement | Description | Test Case | Status |
| :--- | :--- | :--- | :--- |
| **REQ-B7.1-01** | Single-slot buffer lifecycle & pending state | `test_buffer_single_slot_lifecycle` | PASSED |
| **REQ-B7.1-02** | Atomic frame replacement under producer speedup | `test_buffer_replacement_semantics` | PASSED |
| **REQ-B7.1-03** | Invalid / empty frame rejection | `test_buffer_rejects_empty_or_none` | PASSED |
| **REQ-B7.1-04** | Buffer clear and condition broadcast | `test_buffer_clear_and_notify_all` | PASSED |
| **REQ-B7.1-05** | Condition wake upon submission | `test_buffer_condition_wake_immediate` | PASSED |
| **REQ-B7.1-06** | Worker start / stop idempotency & clean join | `test_worker_start_stop_idempotent` | PASSED |
| **REQ-B7.1-07** | Worker pause and resume semantics | `test_worker_pause_resume` | PASSED |
| **REQ-B7.1-08** | Worker reset flushes buffer & pipeline state | `test_worker_reset_clears_buffer_and_pipeline` | PASSED |
| **REQ-B7.1-09** | Worker shutdown idempotency & release | `test_worker_shutdown_idempotent` | PASSED |
| **REQ-B7.1-10** | Non-running submission rejection | `test_worker_submit_when_not_running` | PASSED |
| **REQ-B7.1-11** | Worker processes freshest frame under load | `test_worker_latest_frame_replacement_under_load` | PASSED |
| **REQ-B7.1-12** | Frozen 8-field public AI contract compliance | `test_worker_result_contract_and_callback` | PASSED |
| **REQ-B7.1-13** | Dedicated background thread execution | `test_worker_thread_isolation` | PASSED |
| **REQ-B7.1-14** | Exception containment and fault recovery | `test_worker_resilience_to_pipeline_exceptions` | PASSED |
| **REQ-B7.1-15** | Pure Python zero Qt dependency verification | `test_worker_no_qt_imports` | PASSED |

---

## 6. Living Memory & Roadmap Integration

- **Sub-Gate B7.1 Status**: COMPLETED (15/15 tests passing in `tests/test_inference_worker.py`).
- **Next Sub-Gate**: **B7.2 — Bridge & Signal Integration** (Connecting `InferenceWorker` to the Workstream A GUI/FSM Qt bridge via thread-safe Qt signals and callbacks).
