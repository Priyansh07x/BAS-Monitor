# Workstream B — Sub-Gate B7.2: Bridge & Signal Integration

## 1. Executive Summary

This document specifies the architecture, implementation, and verification of **Sub-Gate B7.2: Bridge & Signal Integration** for the ISRO SIH26174 BAS Experiment Monitor.

Gate B7.2 connects the pure-Python background [`InferenceWorker`](../backend/ai/inference_worker.py) (from Gate B7.1) to the Workstream A runtime ([`Bridge`](../backend/bridge.py) and [`AppState`](../backend/app_state.py)) via thread-safe Qt queued signals (`aiResultReady` and `aiResultJsonReady`).

Key Guarantees Established:
1. **Single Runtime Owner:** `InferenceWorker` is instantiated and owned exclusively by `AppState`.
2. **Non-Blocking Frame Submission:** Camera frame acquisition in `Bridge.getCameraFrame()` submits frames to `InferenceWorker.submit_frame()` in $O(1)$ without waiting for AI perception to execute.
3. **Decoupled Concurrency:** Frame ingestion (30 FPS) and neural/heuristic inference (10–15 FPS) operate on distinct threads without lock contention or FIFO latency buildup.
4. **Thread-Safe Signal Boundary:** Worker result callback dispatches the frozen 8-field public AI contract dictionary across the thread boundary into Qt queued signals (`aiResultReady = Signal(dict)`, `aiResultJsonReady = Signal(str)`).
5. **Zero Qt in Worker:** `InferenceWorker` remains 100% pure Python with zero Qt/PySide imports.

---

## 2. Target Runtime Architecture & Data Flow

```
Camera Frame Acquisition (30 FPS)
            │
            ▼
   Bridge.getCameraFrame()
   ├── VideoRecorder.enqueue_frame() (non-blocking)
   ├── IPStreamer.update_frame()    (non-blocking)
   └── InferenceWorker.submit_frame(frame, timestamp, metadata, expected_step)
            │
            ▼
┌───────────────────────────────────────┐
│ LatestFrameBuffer (Single-Slot)       │
│ - Overwrites unconsumed stale frames  │
│ - Wakes AIInferenceWorker thread      │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ AIInferenceWorker Background Thread   │
│ - InferencePipeline.process_frame_public()
│ - 111-D Spatial Features & Buffering  │
│ - Deterministic Heuristic Fallback    │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ Result Callback Boundary              │
│ - Invokes Bridge._on_ai_result_from_worker()
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ Qt Queued Signal Boundary             │
│ - Bridge.aiResultReady.emit(dict)     │
│ - Bridge.aiResultJsonReady.emit(str)  │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ Application Consumers                 │
│ - QWebChannel Frontend Bridge         │
│ - Future Gate B9 FSM SequenceValidator│
│ - Future Gate B12 Voice / TTS Alerts  │
└───────────────────────────────────────┘
```

---

## 3. Component Ownership & Lifecycle

### 3.1 Runtime Ownership
- `AppState` owns the canonical `InferenceWorker` instance (`self.inference_worker = inference_worker or InferenceWorker()`).
- `Bridge` references `self.inference_worker = self.state.inference_worker` and registers its internal callback `self.inference_worker.result_callback = self._on_ai_result_from_worker`.

### 3.2 Bridge Control API
The following Bridge slots expose clean, non-blocking lifecycle management:
- `startAI()` / `startInference()`: Starts the background worker thread.
- `stopAI()` / `stopInference()`: Stops the worker thread cleanly.
- `pauseAI()` / `pauseInference()`: Suspends frame evaluation while retaining staged buffer frames.
- `resumeAI()` / `resumeInference()`: Resumes frame evaluation on the newest staged frame.
- `resetAI()` / `resetInference()`: Flushes buffer and resets pipeline temporal history.
- `getLatestAIResult()`: Returns the most recent public AI contract JSON string or `"null"`.
- `isAIRunning()` / `isAIPaused()`: Inspects current worker lifecycle states.

### 3.3 Application Shutdown Integration
During application exit (`app.aboutToQuit`), `Bridge.shutdown()` is called:
- Gracefully signals `InferenceWorker.shutdown(timeout=2.0)`.
- Joins the worker thread within timeout, guaranteeing no orphan background threads remain active.

---

## 4. Frame Ownership & Mutation Analysis

| Component | Access Mode | Mutation Risk | Protection Strategy |
| :--- | :--- | :--- | :--- |
| **`VideoRecorder`** | Background queue | Read-only | Enqueues frame reference; `cv2.VideoWriter.write()` reads pixel buffer without mutating in-place. |
| **`IPStreamer`** | Background server | Read-only | Calls `cv2.resize()`, which creates a newly allocated resized array in memory. |
| **`InferencePipeline`** | Worker thread | Read-only | Rectification and detector letterboxing allocate new transformed ndarrays; HUD annotations explicitly allocate a new canvas via `frame.copy()`. |
| **`Bridge`** | Main/GUI thread | Read-only | Passes frame reference to worker and converts to base64 JPEG via `cv2.imencode()`. |

**Conclusion:** Zero in-place mutation occurs across all concurrent consumers. Passing the NumPy array reference is safe, fast, and eliminates unnecessary memory copying.

---

## 5. Qt Signal Boundary & Public Contract Payload

The signal payload emitted on `aiResultReady` strictly matches the frozen 8-field public AI contract defined in `docs/architecture.md` §2:

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

Zero internal perception diagnostics (bounding boxes, 3D keypoints, hand landmark arrays, interaction distance floats, raw logits) are leaked into the signal payload.

---

## 6. Error Resilience & Missing Artifact Handling

1. **Exception Isolation:** Unhandled exceptions in the perception pipeline are caught in `InferenceWorker._worker_loop`, logged to `SystemLogger`, and trigger safe default fallback payload delivery (`action="IDLE"`, `confidence=0.0`, `status="OUT_OF_SEQUENCE"`). Bridge and the GUI thread remain completely unaffected.
2. **Missing Checkpoints:** In the absence of trained neural weights (`data/action_model.pt`, `data/vision_pipeline.hef`), `InferencePipeline` operates on deterministic 5-action heuristic fallbacks.

---

## 7. Deferred Items & Next Gates

- **Gate B7.3:** Comprehensive non-blocking execution threading verification and living memory update.
- **Gate B8:** Frame-Rate Strategy & FPS Optimization (tuning camera 30 FPS vs AI 10–15 FPS sampling).
- **Gate B9:** FSM Integration (connecting `aiResultReady` signal to `SequenceValidatorFSM`).
- **Gate B12:** Voice Alert Service integration on procedure missteps.
