"""
test_inference_bridge_integration.py — Unit Tests for Workstream B Gate B7.2: Bridge & Signal Integration
ISRO SIH26174 BAS Experiment Monitor
"""

import ast
import json
import threading
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pytest

try:
    from PySide6.QtCore import QCoreApplication, QObject, Slot, Qt
except ImportError:
    QCoreApplication = None

from backend.app_state import AppState
from backend.bridge import Bridge
from backend.ai.inference_worker import InferenceWorker
from backend.ai.inference_pipeline import InferencePipeline


@pytest.fixture(scope="module")
def qapp():
    """Ensure a QCoreApplication instance exists for Qt signals if PySide6 is installed."""
    if QCoreApplication is not None:
        app = QCoreApplication.instance()
        if app is None:
            app = QCoreApplication([])
        return app
    return None


# -----------------------------------------------------------------------------
# 1. Runtime Ownership & Lifecycle (Requirements 1, 2, 3, 17, 18, 19, 20, 24)
# -----------------------------------------------------------------------------

def test_01_worker_creation_by_correct_runtime_owner(qapp):
    """1. Worker creation by the correct runtime owner (AppState)."""
    state = AppState()
    assert hasattr(state, "inference_worker")
    assert isinstance(state.inference_worker, InferenceWorker)

    bridge = Bridge(state)
    assert bridge.inference_worker is state.inference_worker
    bridge.shutdown()


def test_02_worker_starts_exactly_once(qapp):
    """2. Worker starts exactly once; repeated starts are idempotent."""
    state = AppState()
    bridge = Bridge(state)

    assert not bridge.isAIRunning()
    assert bridge.startAI() is True
    assert bridge.isAIRunning()
    initial_thread = bridge.inference_worker._thread

    # Repeated start call should be idempotent and not create a new thread
    assert bridge.startAI() is True
    assert bridge.inference_worker._thread is initial_thread

    bridge.shutdown()


def test_03_worker_shuts_down_cleanly(qapp):
    """3. Worker shuts down cleanly without errors."""
    state = AppState()
    bridge = Bridge(state)
    bridge.startAI()
    assert bridge.isAIRunning()

    bridge.stopAI()
    assert not bridge.isAIRunning()
    assert not bridge.inference_worker.is_running
    bridge.shutdown()


def test_17_pause_resume_integration(qapp):
    """17. Pause/resume integration via Bridge."""
    state = AppState()
    bridge = Bridge(state)
    bridge.startAI()

    assert not bridge.isAIPaused()
    bridge.pauseAI()
    assert bridge.isAIPaused()
    assert bridge.inference_worker.is_paused

    bridge.resumeAI()
    assert not bridge.isAIPaused()
    assert not bridge.inference_worker.is_paused

    bridge.shutdown()


def test_18_reset_integration(qapp):
    """18. Reset integration flushes state and clears result."""
    state = AppState()
    bridge = Bridge(state)
    bridge.startAI()

    bridge.resetAI()
    assert bridge.getLatestAIResult() == "null"
    assert not bridge.inference_worker._buffer.has_pending()

    bridge.shutdown()


def test_19_shutdown_integration(qapp):
    """19. Shutdown integration releases resources cleanly."""
    state = AppState()
    bridge = Bridge(state)
    bridge.startAI()

    bridge.shutdown()
    assert not bridge.isAIRunning()
    assert bridge.inference_worker._thread is None


def test_20_no_duplicate_worker_creation(qapp):
    """20. Multiple bridge references to the same AppState share the same worker."""
    state = AppState()
    worker_1 = state.inference_worker
    bridge_1 = Bridge(state)
    bridge_2 = Bridge(state)

    assert bridge_1.inference_worker is worker_1
    assert bridge_2.inference_worker is worker_1
    bridge_1.shutdown()


def test_24_application_shutdown_does_not_leave_worker_thread_alive(qapp):
    """24. Application shutdown does not leave AI worker thread alive."""
    state = AppState()
    bridge = Bridge(state)
    bridge.startAI()
    worker_thread = bridge.inference_worker._thread
    assert worker_thread is not None
    assert worker_thread.is_alive()

    # Simulate app exit
    bridge.shutdown()
    assert not worker_thread.is_alive()


# -----------------------------------------------------------------------------
# 2. Camera Frame Acquisition & Buffer Guarantees (Requirements 4, 5, 6, 7, 8, 10, 23)
# -----------------------------------------------------------------------------

def test_04_camera_frame_submission_does_not_synchronously_execute_inference(qapp):
    """4. Camera frame submission does not synchronously execute inference."""
    state = AppState()
    bridge = Bridge(state)
    bridge.startAI()

    dummy_frame = np.full((120, 120, 3), 64, dtype=np.uint8)
    state.camera.read = lambda: dummy_frame.copy()

    # Time getCameraFrame call: must be sub-millisecond
    t0 = time.perf_counter()
    _ = bridge.getCameraFrame()
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.05, f"Camera submission blocked caller: {elapsed:.4f}s"
    bridge.shutdown()


def test_05_bridge_remains_responsive_while_worker_runs(qapp):
    """5. Bridge remains responsive while worker runs."""
    state = AppState()
    bridge = Bridge(state)
    bridge.startAI()

    # Check rapid state queries while inference worker is running
    for _ in range(10):
        t0 = time.perf_counter()
        _ = bridge.getState()
        _ = bridge.isAIRunning()
        assert (time.perf_counter() - t0) < 0.01

    bridge.shutdown()


def test_06_newer_frame_replaces_older_pending_frame(qapp):
    """6. Newer frame replaces older pending frame in the single-slot buffer."""
    state = AppState()
    bridge = Bridge(state)
    bridge.startAI()
    bridge.pauseAI()  # Pause worker so buffer queues frames

    f1 = np.full((32, 32, 3), 1, dtype=np.uint8)
    f2 = np.full((32, 32, 3), 2, dtype=np.uint8)
    f3 = np.full((32, 32, 3), 3, dtype=np.uint8)

    state.camera.read = lambda: f1
    bridge.getCameraFrame()
    state.camera.read = lambda: f2
    bridge.getCameraFrame()
    state.camera.read = lambda: f3
    bridge.getCameraFrame()

    latest_item = bridge.inference_worker._buffer.get(timeout=0.1)
    assert latest_item is not None
    assert np.array_equal(latest_item[0], f3)

    bridge.shutdown()


def test_07_recorder_behavior_remains_intact(qapp):
    """7. Recorder behavior remains intact when Bridge processes camera frames."""
    state = AppState()
    bridge = Bridge(state)
    bridge.startAI()

    enqueued_frames = []
    bridge._recorder._recording.set()
    bridge._recorder.enqueue_frame = lambda f: enqueued_frames.append(f)

    dummy_frame = np.zeros((40, 40, 3), dtype=np.uint8)
    state.camera.read = lambda: dummy_frame

    bridge.getCameraFrame()
    assert len(enqueued_frames) == 1

    bridge._recorder._recording.clear()
    bridge.shutdown()


def test_08_ip_streamer_behavior_remains_intact(qapp):
    """8. IP streamer behavior remains intact when Bridge processes camera frames."""
    state = AppState()
    bridge = Bridge(state)
    bridge.startAI()

    streamed_frames = []
    bridge._streamer._is_running = True
    bridge._streamer.update_frame = lambda f: streamed_frames.append(f)

    dummy_frame = np.zeros((40, 40, 3), dtype=np.uint8)
    state.camera.read = lambda: dummy_frame

    bridge.getCameraFrame()
    assert len(streamed_frames) == 1

    bridge._streamer._is_running = False
    bridge.shutdown()


def test_23_gui_thread_blocking_protection(qapp):
    """23. GUI-thread blocking protection: heavy inference on worker does not freeze frame ingestion."""
    slow_done = threading.Event()

    class DeliberatelySlowPipeline(InferencePipeline):
        def process_frame_public(self, **kwargs):
            time.sleep(0.08)  # 80ms inference
            slow_done.set()
            return super().process_frame_public(**kwargs)

    custom_pipeline = DeliberatelySlowPipeline()
    custom_worker = InferenceWorker(pipeline=custom_pipeline)
    state = AppState(inference_worker=custom_worker)
    bridge = Bridge(state)
    bridge.startAI()

    dummy_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    state.camera.read = lambda: dummy_frame

    t0 = time.perf_counter()
    # Rapidly call getCameraFrame 5 times from main thread
    for _ in range(5):
        bridge.getCameraFrame()
    elapsed_main = time.perf_counter() - t0

    # Total time for 5 frame ingestions must be well under the 80ms slow inference duration
    assert elapsed_main < 0.05, f"Main thread blocked during frame submission: {elapsed_main:.4f}s"
    assert slow_done.wait(timeout=2.0)

    bridge.shutdown()


# -----------------------------------------------------------------------------
# 3. Qt Signal Boundary & Public Contract (Requirements 9, 10, 11, 12, 13, 14)
# -----------------------------------------------------------------------------

def test_09_10_11_worker_callback_reaches_qt_signal_and_consumer(qapp):
    """9, 10, 11. Worker callback reaches Qt signal boundary and is received by consumer."""
    state = AppState()
    bridge = Bridge(state)

    received_results = []
    got_signal = threading.Event()

    def slot_ai_result(res: dict):
        received_results.append(res)
        got_signal.set()

    bridge.aiResultReady.connect(slot_ai_result)
    bridge.startAI()

    f = np.zeros((80, 80, 3), dtype=np.uint8)
    state.camera.read = lambda: f
    bridge.getCameraFrame()

    assert got_signal.wait(timeout=3.0)
    assert len(received_results) == 1
    bridge.shutdown()


def test_12_exact_8_field_public_result_schema(qapp):
    """12. Exact 8-field public result schema matching docs/architecture.md §2."""
    state = AppState()
    bridge = Bridge(state)
    results = []
    done = threading.Event()

    bridge.aiResultReady.connect(lambda r: (results.append(r), done.set()))
    bridge.startAI()

    f = np.zeros((80, 80, 3), dtype=np.uint8)
    state.camera.read = lambda: f
    bridge.getCameraFrame()

    assert done.wait(timeout=3.0)
    res = results[0]

    expected_keys = {
        "timestamp",
        "action",
        "object",
        "confidence",
        "expected_step",
        "detected_step",
        "status",
        "next_step",
    }
    assert set(res.keys()) == expected_keys
    assert isinstance(res["timestamp"], str)
    assert isinstance(res["action"], str)
    assert isinstance(res["object"], str)
    assert isinstance(res["confidence"], float)
    assert res["status"] in ("VALID", "SKIPPED", "OUT_OF_SEQUENCE")

    bridge.shutdown()


def test_13_no_internal_diagnostic_leakage(qapp):
    """13. No internal diagnostic leakage in public signal payload."""
    state = AppState()
    bridge = Bridge(state)
    results = []
    done = threading.Event()

    bridge.aiResultReady.connect(lambda r: (results.append(r), done.set()))
    bridge.startAI()

    f = np.zeros((80, 80, 3), dtype=np.uint8)
    state.camera.read = lambda: f
    bridge.getCameraFrame()

    assert done.wait(timeout=3.0)
    res = results[0]

    forbidden_keys = {
        "detections",
        "bounding_boxes",
        "pose_landmarks",
        "keypoints",
        "hand_landmarks",
        "hands",
        "interaction",
        "rectification_applied",
        "logits",
        "vector_buffer",
        "annotated_frame",
    }
    for k in forbidden_keys:
        assert k not in res, f"Forbidden diagnostic '{k}' found in public result payload"

    bridge.shutdown()


def test_14_timestamp_preservation(qapp):
    """14. Timestamp preservation in public AI contract output."""
    state = AppState()
    bridge = Bridge(state)
    results = []
    done = threading.Event()

    bridge.aiResultReady.connect(lambda r: (results.append(r), done.set()))
    bridge.startAI()

    f = np.zeros((64, 64, 3), dtype=np.uint8)
    state.camera.read = lambda: f
    bridge.getCameraFrame()

    assert done.wait(timeout=3.0)
    assert "timestamp" in results[0]
    assert len(results[0]["timestamp"]) > 0

    bridge.shutdown()


# -----------------------------------------------------------------------------
# 4. Error Handling, Decoupling & Boundary Tests (Requirements 15, 16, 21, 22)
# -----------------------------------------------------------------------------

def test_15_worker_exception_does_not_crash_bridge(qapp):
    """15. Worker exception does not crash Bridge or terminate worker thread."""
    call_count = 0
    got_fallback = threading.Event()
    results = []

    class FailingPipeline(InferencePipeline):
        def process_frame_public(self, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated failure inside pipeline")
            return super().process_frame_public(**kwargs)

    custom_pipeline = FailingPipeline()
    custom_worker = InferenceWorker(pipeline=custom_pipeline)
    state = AppState(inference_worker=custom_worker)
    bridge = Bridge(state)

    bridge.aiResultReady.connect(lambda r: (results.append(r), got_fallback.set()))
    bridge.startAI()

    f = np.zeros((50, 50, 3), dtype=np.uint8)
    state.camera.read = lambda: f
    bridge.getCameraFrame()

    assert got_fallback.wait(timeout=3.0)
    assert len(results) == 1
    assert results[0]["action"] == "IDLE"
    assert results[0]["confidence"] == 0.0
    assert results[0]["status"] == "OUT_OF_SEQUENCE"

    assert bridge.isAIRunning()
    assert custom_worker.error_count == 1

    bridge.shutdown()


def test_16_missing_model_checkpoint_deterministic_fallback(qapp):
    """16. Missing neural model checkpoint executes deterministic fallback without crashing."""
    state = AppState()
    bridge = Bridge(state)
    results = []
    done = threading.Event()

    bridge.aiResultReady.connect(lambda r: (results.append(r), done.set()))
    bridge.startAI()

    f = np.zeros((64, 64, 3), dtype=np.uint8)
    state.camera.read = lambda: f
    bridge.getCameraFrame()

    assert done.wait(timeout=3.0)
    assert len(results) == 1
    assert results[0]["action"] in {"IDLE", "PICK_RED", "PLACE_RED", "PICK_BLUE", "PLACE_BLUE", "CLOSE_LID"}

    bridge.shutdown()


def test_21_no_direct_bridge_pipeline_calls():
    """21. No direct Bridge -> InferencePipeline.process_frame() call."""
    import backend.bridge as bridge_mod
    with open(bridge_mod.__file__, "r", encoding="utf-8") as f:
        src = f.read()

    assert "process_frame(" not in src, "Bridge must not directly invoke process_frame()"
    assert "process_frame_public(" not in src, "Bridge must not directly invoke process_frame_public()"


def test_22_worker_contains_no_qt_imports():
    """22. Worker module contains no Qt/PySide imports (100% pure Python)."""
    import backend.ai.inference_worker as iw_mod
    with open(iw_mod.__file__, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "PySide" not in alias.name
                assert "PyQt" not in alias.name
                assert "Qt" not in alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert "PySide" not in node.module
                assert "PyQt" not in node.module
                assert "Qt" not in node.module
