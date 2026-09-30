"""
test_inference_worker.py — Unit Tests for Workstream B Inference Worker & Latest-Frame Buffer (B7.1)
ISRO SIH26174 BAS Experiment Monitor
"""

import ast
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pytest

from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.inference_worker import LatestFrameBuffer, InferenceWorker


# -----------------------------------------------------------------------------
# 1. LatestFrameBuffer Tests
# -----------------------------------------------------------------------------

def test_buffer_single_slot_lifecycle():
    """Buffer holds at most one frame and tracks pending state accurately."""
    buf = LatestFrameBuffer()
    assert not buf.has_pending()
    assert buf.get(timeout=0.01) is None

    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    ok = buf.put(frame1, timestamp="2026-09-24T18:00:00Z", metadata={"source": "cam"}, expected_step="S1")
    assert ok is True
    assert buf.has_pending()

    item = buf.get(timeout=0.1)
    assert item is not None
    frame_out, ts_out, meta_out, step_out = item
    assert np.array_equal(frame_out, frame1)
    assert ts_out == "2026-09-24T18:00:00Z"
    assert meta_out == {"source": "cam"}
    assert step_out == "S1"
    assert not buf.has_pending()


def test_buffer_replacement_semantics():
    """Submitting new frames replaces unconsumed pending frames atomically."""
    buf = LatestFrameBuffer()

    f1 = np.full((10, 10, 3), 1, dtype=np.uint8)
    f2 = np.full((10, 10, 3), 2, dtype=np.uint8)
    f3 = np.full((10, 10, 3), 3, dtype=np.uint8)

    buf.put(f1, timestamp="ts-1")
    buf.put(f2, timestamp="ts-2")
    buf.put(f3, timestamp="ts-3")

    # Only f3 should be retrieved
    item = buf.get(timeout=0.1)
    assert item is not None
    assert np.array_equal(item[0], f3)
    assert item[1] == "ts-3"

    # Buffer is now empty
    assert buf.get(timeout=0.01) is None


def test_buffer_rejects_empty_or_none():
    """Buffer rejects None or 0-sized arrays."""
    buf = LatestFrameBuffer()
    assert buf.put(None) is False
    assert buf.put(np.array([])) is False
    assert not buf.has_pending()


def test_buffer_clear_and_notify_all():
    """Buffer clear empties pending slot and notify_all wakes waiting get."""
    buf = LatestFrameBuffer()
    buf.put(np.zeros((10, 10, 3), dtype=np.uint8))
    assert buf.has_pending()

    buf.clear()
    assert not buf.has_pending()
    assert buf.get(timeout=0.01) is None


def test_buffer_condition_wake_immediate():
    """Consumer thread waiting in get() is woken immediately upon put()."""
    buf = LatestFrameBuffer()
    retrieved = []
    ready = threading.Event()

    def consumer():
        ready.set()
        data = buf.get(timeout=2.0)
        if data:
            retrieved.append(data)

    t = threading.Thread(target=consumer, daemon=True)
    t.start()
    assert ready.wait(timeout=1.0)

    # Brief delay to ensure thread is inside buf.get()
    time.sleep(0.05)
    f = np.ones((5, 5, 3), dtype=np.uint8)
    buf.put(f, timestamp="ts-wake")

    t.join(timeout=1.0)
    assert not t.is_alive()
    assert len(retrieved) == 1
    assert retrieved[0][1] == "ts-wake"


# -----------------------------------------------------------------------------
# 2. InferenceWorker Lifecycle Tests
# -----------------------------------------------------------------------------

def test_worker_start_stop_idempotent():
    """Worker start and stop are idempotent and manage thread cleanly."""
    worker = InferenceWorker()
    assert not worker.is_running
    assert worker._thread is None

    # Start
    assert worker.start() is True
    assert worker.is_running
    assert worker._thread is not None
    assert worker._thread.is_alive()

    # Redundant start
    assert worker.start() is True
    assert worker.is_running

    # Stop
    assert worker.stop(timeout=2.0) is True
    assert not worker.is_running
    assert worker._thread is None

    # Redundant stop
    assert worker.stop(timeout=2.0) is True
    worker.shutdown()


def test_worker_pause_resume():
    """Paused worker suspends execution and resumes when signaled."""
    processed_results = []
    resume_event = threading.Event()

    def on_result(res):
        processed_results.append(res)
        resume_event.set()

    worker = InferenceWorker(result_callback=on_result)
    worker.start()
    time.sleep(0.05)
    worker.pause()
    assert worker.is_paused

    f1 = np.zeros((100, 100, 3), dtype=np.uint8)
    worker.submit_frame(f1, timestamp="2026-09-24T18:00:00Z")

    # While paused, nothing should be processed
    time.sleep(0.1)
    assert len(processed_results) == 0

    # Resume and wait for result
    worker.resume()
    assert not worker.is_paused
    assert resume_event.wait(timeout=2.0)
    assert len(processed_results) == 1
    assert processed_results[0]["timestamp"] == "2026-09-24T18:00:00Z"

    worker.shutdown()


def test_worker_reset_clears_buffer_and_pipeline():
    """Reset flushes pending frame, drains result queue, and resets pipeline state."""
    worker = InferenceWorker()
    worker.start()
    time.sleep(0.05)
    worker.pause()  # pause so pending frame stays in buffer

    f = np.zeros((50, 50, 3), dtype=np.uint8)
    worker.submit_frame(f, timestamp="2026-09-24T18:00:00Z")
    assert worker._buffer.has_pending()

    worker.reset()
    assert not worker._buffer.has_pending()
    assert worker.get_latest_result() is None
    assert worker.pop_result(timeout=0.01) is None

    worker.shutdown()


def test_worker_shutdown_idempotent():
    """Shutdown cleans up thread and releases pipeline resources."""
    pipeline = InferencePipeline()
    worker = InferenceWorker(pipeline=pipeline)
    worker.start()
    assert worker.is_running

    worker.shutdown()
    assert not worker.is_running
    assert worker._thread is None

    # Redundant shutdown should not error
    worker.shutdown()


# -----------------------------------------------------------------------------
# 3. Producer Submission & Replacement Semantics Tests
# -----------------------------------------------------------------------------

def test_worker_submit_when_not_running():
    """Submitting frame when worker is stopped returns False."""
    worker = InferenceWorker()
    f = np.zeros((10, 10, 3), dtype=np.uint8)
    assert worker.submit_frame(f) is False


def test_worker_latest_frame_replacement_under_load():
    """When producer submits multiple frames quickly, worker processes freshest frame."""
    results = []
    done_event = threading.Event()

    def on_result(res):
        results.append(res)
        if res["timestamp"] == "ts-freshest":
            done_event.set()

    worker = InferenceWorker(result_callback=on_result)
    worker.start()
    time.sleep(0.05)
    worker.pause()  # Pause to stage frames

    f = np.zeros((64, 64, 3), dtype=np.uint8)
    worker.submit_frame(f, timestamp="ts-old-1")
    worker.submit_frame(f, timestamp="ts-old-2")
    worker.submit_frame(f, timestamp="ts-freshest")  # Replaces ts-old-1 and ts-old-2

    worker.resume()
    assert done_event.wait(timeout=2.0)

    # Verify that ts-freshest was processed and older unconsumed frames were dropped
    timestamps = [r["timestamp"] for r in results]
    assert "ts-freshest" in timestamps
    assert "ts-old-1" not in timestamps
    assert "ts-old-2" not in timestamps

    worker.shutdown()


# -----------------------------------------------------------------------------
# 4. Result Delivery & Contract Compliance Tests
# -----------------------------------------------------------------------------

def test_worker_result_contract_and_callback():
    """Worker delivers valid 8-field public AI contract matching architecture.md §2."""
    delivered = []
    got_result = threading.Event()

    def callback(res):
        delivered.append(res)
        got_result.set()

    worker = InferenceWorker(result_callback=callback)
    worker.start()

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    worker.submit_frame(frame, timestamp="2026-09-24T18:00:00Z", metadata={"cam_id": 0}, expected_step="S1")

    assert got_result.wait(timeout=2.0)
    assert len(delivered) == 1
    res = delivered[0]

    # Verify frozen 8-field public AI contract (docs/architecture.md §2)
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
    assert res["expected_step"] in (None, "S1", "S2", "S3", "S4", "S5")
    assert res["detected_step"] in (None, "S1", "S2", "S3", "S4", "S5")
    assert res["status"] in ("VALID", "SKIPPED", "OUT_OF_SEQUENCE")

    # Verify latest result accessor and queue
    assert worker.get_latest_result() == res
    popped = worker.pop_result(timeout=0.1)
    assert popped == res

    worker.shutdown()


def test_worker_thread_isolation():
    """Inference is executed strictly on the background worker thread."""
    worker_thread_ids = []
    done = threading.Event()

    class ThreadCapturePipeline(InferencePipeline):
        def process_frame_public(self, **kwargs):
            worker_thread_ids.append(threading.get_ident())
            return super().process_frame_public(**kwargs)

    pipeline = ThreadCapturePipeline()
    worker = InferenceWorker(pipeline=pipeline, result_callback=lambda r: done.set())
    worker.start()

    main_tid = threading.get_ident()
    f = np.zeros((64, 64, 3), dtype=np.uint8)
    worker.submit_frame(f, timestamp="2026-09-24T18:00:00Z")

    assert done.wait(timeout=2.0)
    assert len(worker_thread_ids) == 1
    assert worker_thread_ids[0] != main_tid
    assert worker_thread_ids[0] == worker._thread.ident

    worker.shutdown()


# -----------------------------------------------------------------------------
# 5. Error Resilience & Fallback Tests
# -----------------------------------------------------------------------------

def test_worker_resilience_to_pipeline_exceptions():
    """Exceptions during frame execution do not terminate worker thread; subsequent frames succeed."""
    call_count = 0
    second_done = threading.Event()

    class FaultyPipeline(InferencePipeline):
        def process_frame_public(self, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated pipeline internal crash")
            return super().process_frame_public(**kwargs)

    pipeline = FaultyPipeline()
    results = []

    def on_result(res):
        results.append(res)
        if call_count >= 2:
            second_done.set()

    worker = InferenceWorker(pipeline=pipeline, result_callback=on_result)
    worker.start()

    f = np.zeros((64, 64, 3), dtype=np.uint8)
    # Frame 1 fails
    worker.submit_frame(f, timestamp="2026-09-24T18:00:01Z", expected_step="S1")
    time.sleep(0.1)

    assert worker.is_running
    assert worker.error_count == 1
    # Fallback result delivered on error
    assert len(results) == 1
    assert results[0]["action"] == "IDLE"
    assert results[0]["confidence"] == 0.0

    # Frame 2 succeeds
    worker.submit_frame(f, timestamp="2026-09-24T18:00:02Z", expected_step="S1")
    assert second_done.wait(timeout=2.0)
    assert len(results) == 2
    assert worker.processed_count == 1  # 1 successful
    assert worker.error_count == 1

    worker.shutdown()


def test_worker_no_qt_imports():
    """InferenceWorker module is pure Python with zero Qt/PySide6 dependency."""
    import backend.ai.inference_worker as iw_mod
    with open(iw_mod.__file__, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module)

    for mod in imported_modules:
        assert "PySide" not in mod, f"Forbidden Qt import found: {mod}"
        assert "PyQt" not in mod, f"Forbidden Qt import found: {mod}"
        assert "QtCore" not in mod, f"Forbidden Qt import found: {mod}"
