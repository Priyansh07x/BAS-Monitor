# Workstream B — Gate B8.1: Frame-Rate Telemetry & Timing Instrumentation

**ISRO SIH26174 BAS Experiment Monitor — Workstream B**  
**Gate:** B8.1 — Frame-Rate Telemetry & Timing Instrumentation  
**Status:** PASSED  
**Date:** 2026-09-24  

---

## 1. Executive Summary

Gate B8.1 establishes deterministic monotonic timing and frame-rate telemetry across the video acquisition boundary ([`backend/bridge.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/bridge.py)), single-slot buffer ([`LatestFrameBuffer`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/inference_worker.py)), and edge perception worker ([`InferenceWorker`](file:///E:/Technical_Projects/2026-SIH/SIH26174/backend/ai/inference_worker.py)).

This gate provides honest, sub-millisecond precision instrumentation of the runtime before rate-matching strategies or throttling (Gate B8.2/B8.3) are introduced.

---

## 2. Telemetry & Timing Architecture

```
Stage A: Camera Acquisition Boundary (Bridge.getCameraFrame)
  - Record monotonic timestamp: t_capture = time.monotonic()
  - Increment camera acquisition counter (_camera_frame_count)
  - Append to sliding-window timestamp deque (_camera_ingest_times)
       ↓
Stage B: Single-Slot Buffer Ingestion (LatestFrameBuffer.put)
  - Attach capture_monotonic to frame metadata
  - Increment submission counter (_submission_count)
  - If pending frame exists: increment replacement counter (_replacement_count)
       ↓
Stage C: Background Worker Pickup (InferenceWorker._worker_loop)
  - Record pickup monotonic timestamp: t_pickup = time.monotonic()
  - Calculate frame age at worker pickup: frame_age_ms = (t_pickup - capture_monotonic) * 1000
       ↓
Stage D: Inference Start (InferencePipeline.process_frame_public)
  - Record inference start timestamp: t_infer_start = time.monotonic()
       ↓
Stage E: Inference Completion
  - Record inference end timestamp: t_infer_end = time.monotonic()
  - Calculate inference duration: infer_dur_ms = (t_infer_end - t_infer_start) * 1000
  - Increment processed counter (_processed_count)
  - Append to rolling processing times deque (_processing_times)
       ↓
Stage F: Public Result Callback & Dispatch (Bridge._on_ai_result_from_worker)
  - Record dispatch duration: dispatch_ms = (t_dispatch_end - t_dispatch_start) * 1000
  - Calculate end-to-end latency: end_to_end_ms = (t_dispatch_end - capture_monotonic) * 1000
  - Increment result emission counter (_emitted_count)
  - Dispatches frozen 8-field public contract to Qt queued signals
```

---

## 3. Telemetry Metrics Specification

| Metric | Source Stage | Units / Type | Description |
|---|---|---|---|
| `frames_submitted` | Stage B | Integer | Cumulative count of frames passed to `submit_frame()` / `LatestFrameBuffer.put()`. |
| `frames_replaced` | Stage B | Integer | Cumulative count of unconsumed pending frames overwritten by newer frames. |
| `frames_processed` | Stage E | Integer | Cumulative count of frames successfully analyzed by `InferencePipeline`. |
| `error_count` | Stage E | Integer | Cumulative count of caught exceptions inside `_worker_loop()`. |
| `results_emitted` | Stage F | Integer | Cumulative count of public AI results dispatched via callback. |
| `camera_ingest_fps` | Stage A / B | FPS (float) | Rolling ingestion frequency calculated over a 2.0-second window. |
| `ai_processing_fps` | Stage E | FPS (float) | Rolling processing frequency calculated over a 2.0-second window. |
| `frame_drop_rate` | Stage B | Float ($[0, 1]$) | Drop ratio $\frac{\text{frames\_replaced}}{\text{frames\_submitted}}$. |
| `mean_inference_ms` | Stage D $\to$ E | Milliseconds | Arithmetic mean of inference duration across rolling window. |
| `p95_inference_ms` | Stage D $\to$ E | Milliseconds | 95th percentile inference duration across rolling window. |
| `mean_frame_age_ms` | Stage A $\to$ C | Milliseconds | Mean latency between camera capture and worker dequeuing. |
| `p95_frame_age_ms` | Stage A $\to$ C | Milliseconds | 95th percentile frame age at worker pickup. |
| `mean_dispatch_latency_ms`| Stage F | Milliseconds | Mean callback execution latency. |
| `mean_end_to_end_ms` | Stage A $\to$ F | Milliseconds | Total latency from camera acquisition to callback completion. |
| `target_fps` | Reference | Dict | System targets: Camera $30.0$, Display $30.0$, AI $[10.0, 15.0]$. |

---

## 4. Architectural Separation & Frozen Contract Preservation

- **Monotonic Timing Source:** All internal elapsed durations and rates strictly use `time.monotonic()`. `datetime.now().isoformat()` is preserved only for wall-clock date formatting in the public result dictionary.
- **Zero Public Schema Contamination:** Public AI results returned by `AIResultAdapter` and emitted over Qt signals (`aiResultReady`, `aiResultJsonReady`) strictly contain the frozen 8 fields:
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
- **Dedicated Telemetry APIs:** Telemetry is queried via dedicated inspector methods and QWebChannel slots:
  - `InferenceWorker.get_telemetry() -> Dict[str, Any]`
  - `Bridge.getAITelemetry() -> str` (JSON)
  - `Bridge.getCameraTelemetry() -> str` (JSON)
  - `Bridge.get_camera_telemetry() -> Dict[str, Any]`

---

## 5. Architectural Clarification & Camera Execution Correction

The B8.0 audit identified an inaccurate description in earlier documentation diagrams implying that camera capture runs on an independent background producer thread.

### Corrected Runtime Reality:
1. Camera capture is **synchronous and pull-driven**.
2. [`frontend/assets/js/app.js`](file:///E:/Technical_Projects/2026-SIH/SIH26174/frontend/assets/js/app.js) executes `setInterval(..., 100)` (10 Hz polling rate).
3. Each tick invokes `Bridge.getCameraFrame()`, which calls `Camera.read()` synchronously on the main thread.
4. `Bridge.getCameraFrame()` submits the frame non-blocking into `LatestFrameBuffer` for consumption by the dedicated background thread `"AIInferenceWorker"`.

---

## 6. Target Rates vs Actual Measured Reality

| Dimension | Target Goal (Reference) | Actual Measured Reality (Gate B8.1) |
|---|:---:|---|
| **Camera Ingestion** | $30\text{ FPS}$ | Poll-driven at $10\text{ FPS}$ ($100\text{ ms}$ interval from frontend `app.js`). |
| **Browser Display** | $30\text{ FPS}$ | Capped at $\approx 10\text{ FPS}$ due to frontend `setInterval(100ms)`. |
| **AI Perception** | $10\text{--}15\text{ FPS}$ | Ingests at $10\text{ FPS}$; processes synchronously on CPU heuristic fallback in $< 1\text{ ms}$; drops $0$ frames when $T_{\text{inf}} < 100\text{ ms}$. |

---

## 7. Verification Summary

Comprehensive unit and integration tests in [`tests/test_b8_1_frame_rate_telemetry.py`](file:///E:/Technical_Projects/2026-SIH/SIH26174/tests/test_b8_1_frame_rate_telemetry.py) validate all 18 criteria:
- **12/12 PASSED** in 1.53 s.
- Monotonic timestamping verified across stages A–F.
- Single-slot buffer replacement counting verified under frame bursts.
- Rolling FPS verified across 2.0s time windows.
- Zero/insufficient sample edge cases handled without exceptions.
- Zero leakage into frozen 8-field public AI contract.
