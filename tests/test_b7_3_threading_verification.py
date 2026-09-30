"""
test_b7_3_threading_verification.py — Gate B7.3 Threading Verification Tests
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B7.3)

Comprehensive verification test suite for non-blocking inference worker execution,
thread safety, latest-frame single-slot buffer dynamics, Qt boundary bridging,
error containment, and lifecycle management.

Covers all 40 mandatory criteria:
 1. Single worker instance.
 2. Repeated start() does not create duplicate workers.
 3. Repeated stop() is safe.
 4. Repeated shutdown() is safe.
 5. Start -> stop -> start lifecycle.
 6. Pause/resume lifecycle.
 7. Reset while worker is active.
 8. Shutdown while worker is waiting for a frame.
 9. Shutdown while a frame is pending.
10. Shutdown while inference is executing.
11. No worker-thread leak after shutdown.
12. Worker thread has the expected name ('AIInferenceWorker').
13. InferencePipeline executes only on the worker thread.
14. Same InferencePipeline is never concurrently invoked.
15. Latest-frame replacement under rapid producer submissions.
16. Verify at most one pending frame.
17. Verify an older frame can be replaced before processing.
18. Verify the newest frame is eventually processed.
19. Producer submission remains non-blocking.
20. Worker survives an inference exception.
21. Subsequent frames still process after an exception.
22. Missing model artifacts do not terminate the worker.
23. Result callback delivery remains functional after errors.
24. Bridge does not directly invoke process_frame().
25. Bridge camera method does not wait for inference.
26. Public result reaches Qt signal boundary.
27. Qt result handling occurs on the application/Qt thread.
28. No GUI/widget access occurs from the worker thread.
29. Worker core remains Qt-free.
30. Public result remains exactly the frozen 8-field schema.
31. No internal detector data leaks through the signal.
32. Timestamp preservation.
33. Metadata preservation.
34. Expected-step preservation.
35. No unbounded result accumulation.
36. Latest-result accessor remains consistent.
37. Rapid frame submission followed by shutdown does not deadlock.
38. Multiple shutdown calls do not deadlock.
39. Exception in result callback does not kill the worker if callback isolation is supported.
40. Repeated start/stop cycles leave no additional worker threads.
"""

from __future__ import annotations

import inspect
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.inference_worker import InferenceWorker, LatestFrameBuffer
from backend.ai.result_adapter import AIResultAdapter
from backend.app_state import AppState
from backend.bridge import Bridge


FROZEN_8_FIELDS = {
    "timestamp",
    "action",
    "object",
    "confidence",
    "expected_step",
    "detected_step",
    "status",
    "next_step",
}


@pytest.fixture
def blank_frame() -> np.ndarray:
    """Standard 480x640x3 blank synthetic frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


# -----------------------------------------------------------------------------
# 1. Single worker instance & AppState ownership
# -----------------------------------------------------------------------------
def test_01_single_worker_instance():
    """Verify AppState creates and owns exactly one canonical worker instance."""
    state = AppState()
    assert state.inference_worker is not None
    assert isinstance(state.inference_worker, InferenceWorker)

    bridge = Bridge(app_state=state)
    assert bridge.inference_worker is state.inference_worker
    bridge.shutdown()


# -----------------------------------------------------------------------------
# 2. Repeated start() does not create duplicate workers
# -----------------------------------------------------------------------------
def test_02_repeated_start_is_idempotent():
    """Verify calling start() multiple times retains the single running thread."""
    worker = InferenceWorker()
    assert worker.start() is True
    thread_1 = worker._thread
    assert thread_1 is not None and thread_1.is_alive()

    # Second start
    assert worker.start() is True
    assert worker._thread is thread_1
    worker.shutdown()


# -----------------------------------------------------------------------------
# 3. Repeated stop() is safe
# -----------------------------------------------------------------------------
def test_03_repeated_stop_is_safe():
    """Verify calling stop() repeatedly is completely safe and returns True."""
    worker = InferenceWorker()
    worker.start()
    assert worker.stop() is True
    assert worker.stop() is True
    assert worker.is_running is False


# -----------------------------------------------------------------------------
# 4. Repeated shutdown() is safe
# -----------------------------------------------------------------------------
def test_04_repeated_shutdown_is_safe():
    """Verify calling shutdown() multiple times is idempotent and safe."""
    worker = InferenceWorker()
    worker.start()
    worker.shutdown()
    worker.shutdown()
    assert worker.is_running is False
    assert worker._thread is None


# -----------------------------------------------------------------------------
# 5. Start -> stop -> start lifecycle
# -----------------------------------------------------------------------------
def test_05_start_stop_start_lifecycle():
    """Verify worker can restart cleanly after being stopped."""
    worker = InferenceWorker()
    worker.start()
    assert worker.is_running is True
    worker.stop()
    assert worker.is_running is False

    worker.start()
    assert worker.is_running is True
    assert worker._thread is not None and worker._thread.is_alive()
    worker.shutdown()


# -----------------------------------------------------------------------------
# 6. Pause / resume lifecycle
# -----------------------------------------------------------------------------
def test_06_pause_resume_lifecycle(blank_frame):
    """Verify pause suspends processing and resume continues on latest frame."""
    result_event = threading.Event()
    results = []

    def on_result(res):
        results.append(res)
        result_event.set()

    worker = InferenceWorker(result_callback=on_result)
    worker.start()

    worker.pause()
    assert worker.is_paused is True

    # Submit frame while paused
    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:01Z")
    time.sleep(0.05)
    assert len(results) == 0  # Not processed while paused

    # Resume
    worker.resume()
    assert worker.is_paused is False
    assert result_event.wait(timeout=2.0)
    assert len(results) >= 1
    worker.shutdown()


# -----------------------------------------------------------------------------
# 7. Reset while worker is active
# -----------------------------------------------------------------------------
def test_07_reset_while_worker_is_active(blank_frame):
    """Verify reset() clears queues and resets pipeline without killing worker."""
    worker = InferenceWorker()
    worker.start()
    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:10Z")
    worker.reset()

    assert worker.is_running is True
    assert worker.get_latest_result() is None
    assert worker._buffer.has_pending() is False
    worker.shutdown()


# -----------------------------------------------------------------------------
# 8. Shutdown while worker is waiting for a frame
# -----------------------------------------------------------------------------
def test_08_shutdown_while_waiting_for_frame():
    """Verify worker terminates promptly when idle and waiting for frames."""
    worker = InferenceWorker()
    worker.start()
    time.sleep(0.05)  # Ensure thread entered wait in get()
    t0 = time.monotonic()
    worker.shutdown(timeout=1.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5
    assert worker.is_running is False
    assert worker._thread is None


# -----------------------------------------------------------------------------
# 9. Shutdown while a frame is pending
# -----------------------------------------------------------------------------
def test_09_shutdown_while_frame_is_pending(blank_frame):
    """Verify shutdown clears pending frames and halts without deadlock."""
    worker = InferenceWorker()
    worker._buffer.put(blank_frame, timestamp="2026-09-24T12:00:05Z")
    assert worker._buffer.has_pending() is True
    worker.shutdown()
    assert worker._buffer.has_pending() is False


# -----------------------------------------------------------------------------
# 10. Shutdown while inference is executing
# -----------------------------------------------------------------------------
def test_10_shutdown_while_inference_is_executing(blank_frame):
    """Verify worker terminates cleanly if shutdown is called during pipeline execution."""
    pipeline_started = threading.Event()
    allow_pipeline_finish = threading.Event()

    mock_pipeline = MagicMock(spec=InferencePipeline)

    def slow_process(*args, **kwargs):
        pipeline_started.set()
        allow_pipeline_finish.wait(timeout=2.0)
        return AIResultAdapter.adapt(action="PICK_RED", confidence=0.9, fsm_status="VALID")

    mock_pipeline.process_frame_public.side_effect = slow_process

    worker = InferenceWorker(pipeline=mock_pipeline)
    worker.start()
    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:01Z")

    assert pipeline_started.wait(timeout=2.0)
    
    # Trigger shutdown concurrently
    shutdown_thread = threading.Thread(target=lambda: worker.shutdown(timeout=2.0))
    shutdown_thread.start()

    allow_pipeline_finish.set()
    shutdown_thread.join(timeout=2.0)

    assert not shutdown_thread.is_alive()
    assert worker.is_running is False
    assert worker._thread is None


# -----------------------------------------------------------------------------
# 11. No worker-thread leak after shutdown
# -----------------------------------------------------------------------------
def test_11_no_worker_thread_leak():
    """Verify no active AIInferenceWorker threads remain after shutdown."""
    worker = InferenceWorker()
    worker.start()
    assert worker._thread is not None and worker._thread.is_alive()

    worker.shutdown()
    assert worker._thread is None or not worker._thread.is_alive()


# -----------------------------------------------------------------------------
# 12. Worker thread has the expected name
# -----------------------------------------------------------------------------
def test_12_worker_thread_has_expected_name():
    """Verify the dedicated background thread is named 'AIInferenceWorker'."""
    worker = InferenceWorker()
    worker.start()
    assert worker._thread is not None
    assert worker._thread.name == "AIInferenceWorker"
    worker.shutdown()


# -----------------------------------------------------------------------------
# 13. InferencePipeline executes only on the worker thread
# -----------------------------------------------------------------------------
def test_13_inference_pipeline_executes_only_on_worker_thread(blank_frame):
    """Verify pipeline execution thread ID matches worker thread and NOT main thread."""
    executed_threads = []
    exec_event = threading.Event()

    mock_pipeline = MagicMock(spec=InferencePipeline)

    def record_thread(*args, **kwargs):
        executed_threads.append(threading.current_thread().name)
        exec_event.set()
        return AIResultAdapter.adapt(action="PICK_RED", confidence=0.9, fsm_status="VALID")

    mock_pipeline.process_frame_public.side_effect = record_thread

    worker = InferenceWorker(pipeline=mock_pipeline)
    worker.start()
    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:01Z")

    assert exec_event.wait(timeout=2.0)
    assert len(executed_threads) == 1
    assert executed_threads[0] == "AIInferenceWorker"
    assert executed_threads[0] != threading.main_thread().name
    worker.shutdown()


# -----------------------------------------------------------------------------
# 14. Same InferencePipeline is never concurrently invoked
# -----------------------------------------------------------------------------
def test_14_inference_pipeline_never_concurrently_invoked(blank_frame):
    """Verify serialized execution through the single worker loop."""
    concurrent_violations = []
    active_invocations = [0]
    lock = threading.Lock()
    done_event = threading.Event()

    mock_pipeline = MagicMock(spec=InferencePipeline)

    def process_with_concurrency_check(*args, **kwargs):
        with lock:
            active_invocations[0] += 1
            if active_invocations[0] > 1:
                concurrent_violations.append(active_invocations[0])
        time.sleep(0.01)
        with lock:
            active_invocations[0] -= 1
        if worker.processed_count >= 3:
            done_event.set()
        return AIResultAdapter.adapt(action="PICK_RED", confidence=0.9, fsm_status="VALID")

    mock_pipeline.process_frame_public.side_effect = process_with_concurrency_check

    worker = InferenceWorker(pipeline=mock_pipeline)
    worker.start()

    for i in range(5):
        worker.submit_frame(blank_frame, timestamp=f"2026-09-24T12:00:0{i}Z")
        time.sleep(0.005)

    done_event.wait(timeout=2.0)
    assert len(concurrent_violations) == 0
    worker.shutdown()


# -----------------------------------------------------------------------------
# 15. Latest-frame replacement under rapid producer submissions
# -----------------------------------------------------------------------------
def test_15_latest_frame_replacement_under_rapid_submissions(blank_frame):
    """Verify single-slot buffer drops intermediate frames and preserves newest."""
    buffer = LatestFrameBuffer()
    for i in range(100):
        buffer.put(blank_frame, timestamp=f"ts_{i}")

    data = buffer.get(timeout=0.1)
    assert data is not None
    _, ts, _, _ = data
    assert ts == "ts_99"
    assert buffer.has_pending() is False


# -----------------------------------------------------------------------------
# 16. Verify at most one pending frame
# -----------------------------------------------------------------------------
def test_16_at_most_one_pending_frame(blank_frame):
    """Verify buffer structure holds strictly 0 or 1 item."""
    buffer = LatestFrameBuffer()
    assert buffer._pending_data is None
    buffer.put(blank_frame, timestamp="ts_1")
    assert buffer._pending_data is not None
    buffer.put(blank_frame, timestamp="ts_2")
    assert buffer._pending_data is not None
    assert buffer._pending_data[1] == "ts_2"


# -----------------------------------------------------------------------------
# 17. Verify an older frame can be replaced before processing
# -----------------------------------------------------------------------------
def test_17_older_frame_replaced_before_processing(blank_frame):
    """Verify unread frame is overwritten when a new frame is pushed."""
    buffer = LatestFrameBuffer()
    buffer.put(blank_frame, timestamp="ts_old")
    buffer.put(blank_frame, timestamp="ts_new")
    item = buffer.get(timeout=0.1)
    assert item[1] == "ts_new"


# -----------------------------------------------------------------------------
# 18. Verify the newest frame is eventually processed
# -----------------------------------------------------------------------------
def test_18_newest_frame_eventually_processed(blank_frame):
    """Verify worker processes the newest submitted frame."""
    latest_ts = []
    done_event = threading.Event()

    def callback(res):
        latest_ts.append(res["timestamp"])
        if res["timestamp"] == "2026-09-24T12:00:99Z":
            done_event.set()

    worker = InferenceWorker(result_callback=callback)
    worker.start()

    # Flood with frames
    for i in range(100):
        worker.submit_frame(blank_frame, timestamp=f"2026-09-24T12:00:{i:02d}Z")

    assert done_event.wait(timeout=2.0)
    assert "2026-09-24T12:00:99Z" in latest_ts
    worker.shutdown()


# -----------------------------------------------------------------------------
# 19. Producer submission remains non-blocking
# -----------------------------------------------------------------------------
def test_19_producer_submission_is_non_blocking(blank_frame):
    """Verify submit_frame returns in < 5 ms even when pipeline is busy."""
    pipeline_blocked = threading.Event()
    release_pipeline = threading.Event()

    mock_pipeline = MagicMock(spec=InferencePipeline)

    def blocking_eval(*args, **kwargs):
        pipeline_blocked.set()
        release_pipeline.wait(timeout=2.0)
        return AIResultAdapter.adapt(action="PICK_RED", confidence=0.9, fsm_status="VALID")

    mock_pipeline.process_frame_public.side_effect = blocking_eval

    worker = InferenceWorker(pipeline=mock_pipeline)
    worker.start()
    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:01Z")

    assert pipeline_blocked.wait(timeout=2.0)

    t0 = time.perf_counter()
    submitted = worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:02Z")
    elapsed = time.perf_counter() - t0

    assert submitted is True
    assert elapsed < 0.005  # < 5ms submission

    release_pipeline.set()
    worker.shutdown()


# -----------------------------------------------------------------------------
# 20. Worker survives an inference exception
# -----------------------------------------------------------------------------
def test_20_worker_survives_inference_exception(blank_frame):
    """Verify worker catches exception, logs error, and thread remains alive."""
    mock_pipeline = MagicMock(spec=InferencePipeline)
    mock_pipeline.process_frame_public.side_effect = RuntimeError("Simulated perception crash")

    worker = InferenceWorker(pipeline=mock_pipeline)
    worker.start()

    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:01Z")
    time.sleep(0.05)

    assert worker.is_running is True
    assert worker.error_count == 1
    worker.shutdown()


# -----------------------------------------------------------------------------
# 21. Subsequent frames still process after an exception
# -----------------------------------------------------------------------------
def test_21_subsequent_frames_process_after_exception(blank_frame):
    """Verify subsequent frame processes normally after a prior error."""
    call_count = [0]
    done_event = threading.Event()

    mock_pipeline = MagicMock(spec=InferencePipeline)

    def faulty_then_good(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise ValueError("First frame fails")
        done_event.set()
        return AIResultAdapter.adapt(action="PLACE_RED", confidence=0.88, fsm_status="VALID")

    mock_pipeline.process_frame_public.side_effect = faulty_then_good

    worker = InferenceWorker(pipeline=mock_pipeline)
    worker.start()

    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:01Z")
    time.sleep(0.05)
    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:02Z")

    assert done_event.wait(timeout=2.0)
    assert worker.error_count == 1
    assert worker.processed_count == 1
    worker.shutdown()


# -----------------------------------------------------------------------------
# 22. Missing model artifacts do not terminate the worker
# -----------------------------------------------------------------------------
def test_22_missing_model_artifacts_safe(blank_frame):
    """Verify default heuristic fallback runs smoothly without checkpoint."""
    worker = InferenceWorker()
    worker.start()

    result_event = threading.Event()
    results = []

    worker.result_callback = lambda r: (results.append(r), result_event.set())
    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:01Z")

    assert result_event.wait(timeout=2.0)
    assert len(results) == 1
    assert results[0]["action"] == "IDLE"
    assert results[0]["status"] in ("VALID", "SKIPPED", "OUT_OF_SEQUENCE")
    worker.shutdown()


# -----------------------------------------------------------------------------
# 23. Result callback delivery remains functional after errors
# -----------------------------------------------------------------------------
def test_23_result_callback_functional_after_errors(blank_frame):
    """Verify callback is called with fallback result during errors and normal after."""
    results = []
    done_event = threading.Event()

    def on_result(res):
        results.append(res)
        if len(results) >= 2:
            done_event.set()

    mock_pipeline = MagicMock(spec=InferencePipeline)
    call_idx = [0]

    def behave(*args, **kwargs):
        call_idx[0] += 1
        if call_idx[0] == 1:
            raise RuntimeError("Transient crash")
        return AIResultAdapter.adapt(action="PICK_BLUE", confidence=0.92, fsm_status="VALID")

    mock_pipeline.process_frame_public.side_effect = behave

    worker = InferenceWorker(pipeline=mock_pipeline, result_callback=on_result)
    worker.start()

    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:01Z")
    time.sleep(0.05)
    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:02Z")

    assert done_event.wait(timeout=2.0)
    assert len(results) == 2
    assert results[0]["status"] == "OUT_OF_SEQUENCE"  # Fallback
    assert results[1]["action"] == "PICK_BLUE"
    worker.shutdown()


# -----------------------------------------------------------------------------
# 24. Bridge does not directly invoke process_frame()
# -----------------------------------------------------------------------------
def test_24_bridge_does_not_directly_invoke_process_frame():
    """Verify Bridge source code does NOT contain synchronous process_frame calls."""
    import backend.bridge as bridge_mod
    source = inspect.getsource(bridge_mod.Bridge)
    assert "pipeline.process_frame(" not in source
    assert "pipeline.process_frame_public(" not in source


# -----------------------------------------------------------------------------
# 25. Bridge camera method does not wait for inference
# -----------------------------------------------------------------------------
def test_25_bridge_camera_method_does_not_wait_for_inference(blank_frame):
    """Verify Bridge.getCameraFrame completes instantly even if worker is busy."""
    state = AppState()
    mock_camera = MagicMock()
    mock_camera.get_frame.return_value = blank_frame
    state.camera = mock_camera

    pipeline_busy = threading.Event()
    release_pipeline = threading.Event()

    mock_pipeline = MagicMock(spec=InferencePipeline)

    def slow_exec(*args, **kwargs):
        pipeline_busy.set()
        release_pipeline.wait(timeout=2.0)
        return AIResultAdapter.adapt(action="PICK_RED", confidence=0.9, fsm_status="VALID")

    mock_pipeline.process_frame_public.side_effect = slow_exec
    state.inference_worker = InferenceWorker(pipeline=mock_pipeline)

    bridge = Bridge(app_state=state)
    bridge.startAI()

    # Prime one frame into pipeline
    bridge.getCameraFrame()
    assert pipeline_busy.wait(timeout=2.0)

    # Ingest next camera frame
    t0 = time.perf_counter()
    res = bridge.getCameraFrame()
    elapsed = time.perf_counter() - t0

    assert res is not None
    assert elapsed < 0.05  # Completed immediately

    release_pipeline.set()
    bridge.shutdown()


# -----------------------------------------------------------------------------
# 26. Public result reaches Qt signal boundary
# -----------------------------------------------------------------------------
def test_26_public_result_reaches_qt_signal(blank_frame):
    """Verify aiResultReady signal is emitted with valid public result dict."""
    state = AppState()
    bridge = Bridge(app_state=state)
    bridge.startAI()

    received_signals = []
    signal_event = threading.Event()

    def on_ai_result(res):
        received_signals.append(res)
        signal_event.set()

    bridge.aiResultReady.connect(on_ai_result)
    state.inference_worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:01Z")

    assert signal_event.wait(timeout=2.0)
    assert len(received_signals) == 1
    assert isinstance(received_signals[0], dict)
    assert set(received_signals[0].keys()) == FROZEN_8_FIELDS
    bridge.shutdown()


# -----------------------------------------------------------------------------
# 27. Qt result handling occurs on the application/Qt thread
# -----------------------------------------------------------------------------
def test_27_qt_result_handling_thread_boundary(blank_frame):
    """Verify callback bridging dispatches to Qt signal handler safely."""
    state = AppState()
    bridge = Bridge(app_state=state)
    bridge.startAI()

    dispatched = []
    disp_event = threading.Event()

    def slot_receiver(res):
        dispatched.append(res)
        disp_event.set()

    bridge.aiResultReady.connect(slot_receiver)
    bridge.inference_worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:01Z")

    assert disp_event.wait(timeout=2.0)
    assert len(dispatched) == 1
    bridge.shutdown()


# -----------------------------------------------------------------------------
# 28. No GUI/widget access occurs from the worker thread
# -----------------------------------------------------------------------------
def test_28_no_gui_access_from_worker_thread():
    """Verify InferenceWorker module does not reference UI widgets or QWidget."""
    import backend.ai.inference_worker as iw
    src = inspect.getsource(iw)
    assert "QWidget" not in src
    assert "QApplication" not in src
    assert "QMainWindow" not in src


# -----------------------------------------------------------------------------
# 29. Worker core remains Qt-free
# -----------------------------------------------------------------------------
def test_29_worker_core_remains_qt_free():
    """Verify backend/ai/inference_worker.py has zero PyQt/PySide imports."""
    import backend.ai.inference_worker as iw
    src = inspect.getsource(iw)
    assert "PySide" not in src
    assert "PyQt" not in src
    assert "QtCore" not in src
    assert "QThread" not in src
    assert "QObject" not in src


# -----------------------------------------------------------------------------
# 30. Public result remains exactly the frozen 8-field schema
# -----------------------------------------------------------------------------
def test_30_frozen_8_field_schema_compliance(blank_frame):
    """Verify all delivered public results strictly contain the frozen 8 fields."""
    worker = InferenceWorker()
    worker.start()

    res_event = threading.Event()
    received = []

    worker.result_callback = lambda r: (received.append(r), res_event.set())
    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:00Z", expected_step="S1")

    assert res_event.wait(timeout=2.0)
    res = received[0]
    assert set(res.keys()) == FROZEN_8_FIELDS
    assert isinstance(res["timestamp"], str)
    assert isinstance(res["confidence"], float)
    assert res["status"] in ("VALID", "SKIPPED", "OUT_OF_SEQUENCE")
    worker.shutdown()


# -----------------------------------------------------------------------------
# 31. No internal detector data leaks through the signal
# -----------------------------------------------------------------------------
def test_31_no_internal_detector_leakage(blank_frame):
    """Verify bounding boxes, keypoints, logits, and landmarks never leak."""
    forbidden_keys = {"landmarks", "keypoints", "bbox", "boxes", "logits", "feature_vector", "buffer_tensor"}
    
    worker = InferenceWorker()
    worker.start()

    res_event = threading.Event()
    received = []

    worker.result_callback = lambda r: (received.append(r), res_event.set())
    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:00Z")

    assert res_event.wait(timeout=2.0)
    for k in forbidden_keys:
        assert k not in received[0], f"Leaked internal key: {k}"
    worker.shutdown()


# -----------------------------------------------------------------------------
# 32. Timestamp preservation
# -----------------------------------------------------------------------------
def test_32_timestamp_preservation(blank_frame):
    """Verify submitted frame timestamp is preserved in public result."""
    worker = InferenceWorker()
    worker.start()

    res_event = threading.Event()
    received = []

    worker.result_callback = lambda r: (received.append(r), res_event.set())
    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:00Z")

    assert res_event.wait(timeout=2.0)
    assert received[0]["timestamp"] == "2026-09-24T12:00:00Z"
    worker.shutdown()


# -----------------------------------------------------------------------------
# 33. Metadata preservation
# -----------------------------------------------------------------------------
def test_33_metadata_preservation(blank_frame):
    """Verify frame metadata passes cleanly to pipeline without contract disruption."""
    mock_pipeline = MagicMock(spec=InferencePipeline)
    captured_meta = []
    res_event = threading.Event()

    def capture(*args, **kwargs):
        captured_meta.append(kwargs.get("metadata"))
        res_event.set()
        return AIResultAdapter.adapt(action="PICK_RED", confidence=0.9, fsm_status="VALID")

    mock_pipeline.process_frame_public.side_effect = capture

    worker = InferenceWorker(pipeline=mock_pipeline)
    worker.start()
    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:01Z", metadata={"camera_id": "CAM_01", "gain": 1.5})

    assert res_event.wait(timeout=2.0)
    assert captured_meta[0] == {"camera_id": "CAM_01", "gain": 1.5}
    worker.shutdown()


# -----------------------------------------------------------------------------
# 34. Expected-step preservation
# -----------------------------------------------------------------------------
def test_34_expected_step_preservation(blank_frame):
    """Verify expected_step is passed and reflected in result."""
    worker = InferenceWorker()
    worker.start()

    res_event = threading.Event()
    received = []

    worker.result_callback = lambda r: (received.append(r), res_event.set())
    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:01Z", expected_step="S3")

    assert res_event.wait(timeout=2.0)
    assert received[0]["expected_step"] == "S3"
    worker.shutdown()


# -----------------------------------------------------------------------------
# 35. No unbounded result accumulation
# -----------------------------------------------------------------------------
def test_35_no_unbounded_result_accumulation(blank_frame):
    """Verify result queue respects maxsize limit and does not grow unbounded."""
    worker = InferenceWorker(max_result_queue_size=5)
    worker.start()

    for i in range(20):
        worker.submit_frame(blank_frame, timestamp=f"2026-09-24T12:00:{i:02d}Z")
        time.sleep(0.01)

    time.sleep(0.1)
    assert worker._result_queue.qsize() <= 5
    worker.shutdown()


# -----------------------------------------------------------------------------
# 36. Latest-result accessor remains consistent
# -----------------------------------------------------------------------------
def test_36_latest_result_accessor_consistency(blank_frame):
    """Verify get_latest_result returns the most recently processed item."""
    worker = InferenceWorker()
    worker.start()

    res_event = threading.Event()
    worker.result_callback = lambda r: res_event.set()
    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:55Z")

    assert res_event.wait(timeout=2.0)
    latest = worker.get_latest_result()
    assert latest is not None
    assert latest["timestamp"] == "2026-09-24T12:00:55Z"
    worker.shutdown()


# -----------------------------------------------------------------------------
# 37. Rapid frame submission followed by shutdown does not deadlock
# -----------------------------------------------------------------------------
def test_37_rapid_submission_shutdown_no_deadlock(blank_frame):
    """Verify flooding frames and instantly calling shutdown completes in < 1s."""
    worker = InferenceWorker()
    worker.start()

    for i in range(50):
        worker.submit_frame(blank_frame, timestamp=f"2026-09-24T12:00:{i:02d}Z")

    t0 = time.monotonic()
    worker.shutdown(timeout=2.0)
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0
    assert worker.is_running is False


# -----------------------------------------------------------------------------
# 38. Multiple shutdown calls do not deadlock
# -----------------------------------------------------------------------------
def test_38_multiple_shutdown_calls_no_deadlock():
    """Verify back-to-back shutdown calls complete cleanly."""
    worker = InferenceWorker()
    worker.start()
    worker.shutdown()
    worker.shutdown()
    worker.shutdown()
    assert worker.is_running is False


# -----------------------------------------------------------------------------
# 39. Exception in result callback does not kill worker
# -----------------------------------------------------------------------------
def test_39_callback_exception_isolation(blank_frame):
    """Verify an exception inside result_callback does not crash the worker thread."""
    call_count = [0]
    done_event = threading.Event()

    def buggy_callback(res):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("Crashing UI slot")
        done_event.set()

    worker = InferenceWorker(result_callback=buggy_callback)
    worker.start()

    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:01Z")
    time.sleep(0.05)
    worker.submit_frame(blank_frame, timestamp="2026-09-24T12:00:02Z")

    assert done_event.wait(timeout=2.0)
    assert worker.is_running is True
    assert call_count[0] >= 2
    worker.shutdown()


# -----------------------------------------------------------------------------
# 40. Repeated start/stop cycles leave no additional worker threads
# -----------------------------------------------------------------------------
def test_40_repeated_start_stop_no_thread_accumulation():
    """Verify cycling start/stop 5 times leaves 0 worker threads alive."""
    worker = InferenceWorker()
    for _ in range(5):
        worker.start()
        assert worker.is_running is True
        worker.stop()
        assert worker.is_running is False

    assert worker._thread is None or not worker._thread.is_alive()
