"""
test_b8_1_frame_rate_telemetry.py — Unit & Integration Test Suite for Gate B8.1
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B8.1)

Validates:
1. Monotonic timestamp generation across stages.
2. Camera ingestion counter.
3. AI processing counter.
4. Frame replacement / drop counter.
5. Error counter.
6. Result emission counter.
7. Rolling FPS calculation with monotonic timestamps.
8. Zero / insufficient sample handling (returns 0.0 without errors).
9. Mean latency calculation.
10. p95 latency calculation.
11. Frame age calculation (worker pickup - capture monotonic).
12. Inference duration calculation (inference end - inference start).
13. Single-slot buffer replacement semantics strictly preserved.
14. Public 8-field schema strictly unchanged (zero internal telemetry leakage).
15. No datetime-based elapsed-time calculation for latency metrics.
16. Deterministic statistics with fixed sample arrays.
17. Counter and telemetry reset behavior.
18. Worker shutdown preserves telemetry consistency.
"""

import json
import time
from typing import Any, Dict
import numpy as np
import pytest

from backend.ai.inference_worker import InferenceWorker, LatestFrameBuffer
from backend.ai.inference_pipeline import InferencePipeline
from backend.app_state import AppState
from backend.bridge import Bridge


FROZEN_PUBLIC_KEYS = {
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
def dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class MockPipeline:
    """Mock pipeline with controllable delay."""
    def __init__(self, delay_sec: float = 0.0, raise_error: bool = False):
        self.delay_sec = delay_sec
        self.raise_error = raise_error
        self.call_count = 0

    def process_frame_public(
        self,
        frame: np.ndarray,
        annotate: bool = True,
        timestamp: Any = None,
        metadata: Any = None,
        expected_step: Any = None,
        fsm_status: Any = None,
        next_step: Any = None,
        rectifier: Any = None,
    ) -> Dict[str, Any]:
        self.call_count += 1
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)
        if self.raise_error:
            raise RuntimeError("Simulated perception pipeline crash")
        return {
            "timestamp": "2026-09-24T12:00:00",
            "action": "PICK_RED",
            "object": "RED_SAMPLE",
            "confidence": 0.95,
            "expected_step": expected_step or "S1",
            "detected_step": "S1",
            "status": "VALID",
            "next_step": "S2",
        }

    def reset(self):
        pass

    def release(self):
        pass


# =============================================================================
# 1. Monotonic Timestamp Generation & Timing Source
# =============================================================================

def test_monotonic_timestamp_generation(dummy_frame):
    """Verifies that frame submissions attach a monotonic timestamp when absent."""
    worker = InferenceWorker(pipeline=MockPipeline())
    worker.start()

    t_before = time.monotonic()
    worker.submit_frame(dummy_frame, timestamp="2026-09-24T12:00:00")
    t_after = time.monotonic()

    # Dequeue from worker buffer
    data = worker._buffer.get(timeout=0.2)
    assert data is not None
    _, _, metadata, _ = data
    cap_mono = worker._buffer.last_capture_monotonic
    assert cap_mono is not None
    assert isinstance(cap_mono, float)
    assert t_before <= cap_mono <= t_after

    # Test explicit capture_monotonic in metadata
    t_custom = 12345.678
    worker.submit_frame(dummy_frame, metadata={"capture_monotonic": t_custom})
    worker._buffer.get(timeout=0.2)
    assert worker._buffer.last_capture_monotonic == t_custom

    worker.shutdown()


def test_no_datetime_based_latency_calculation():
    """Verifies that timing calculation methods strictly operate on float monotonic timestamps."""
    # Rolling FPS and latency methods require numerical sequence
    timestamps = [100.0, 100.1, 100.2, 100.3]
    fps = InferenceWorker._compute_rolling_fps(timestamps, window_sec=1.0)
    assert isinstance(fps, float)
    assert abs(fps - 10.0) < 0.1

    latencies = [15.2, 18.5, 22.1]
    mean_val = InferenceWorker._compute_mean(latencies)
    assert isinstance(mean_val, float)
    assert abs(mean_val - np.mean(latencies)) < 1e-5


# =============================================================================
# 2. Camera Ingestion, Ingestion Counter, and Rolling FPS
# =============================================================================

def test_camera_ingestion_counter_and_telemetry(dummy_frame):
    """Verifies Bridge camera ingestion counting and monotonic timestamp recording."""
    app_state = AppState()
    bridge = Bridge(app_state)

    # Mock camera.read() to return dummy_frame
    app_state.camera.read = lambda: dummy_frame

    assert bridge._camera_frame_count == 0
    t1 = time.monotonic()
    bridge.getCameraFrame()
    bridge.getCameraFrame()
    bridge.getCameraFrame()

    assert bridge._camera_frame_count == 3
    assert len(bridge._camera_ingest_times) == 3
    assert bridge._camera_ingest_times[0] >= t1

    telemetry = bridge.get_camera_telemetry()
    assert telemetry["camera_frames_acquired"] == 3
    assert "camera_ingest_fps" in telemetry
    assert telemetry["target_fps"] == 30.0

    # Test QWebChannel JSON serialization slot
    json_str = bridge.getCameraTelemetry()
    parsed = json.loads(json_str)
    assert parsed["camera_frames_acquired"] == 3

    # Reset
    bridge.reset_camera_telemetry()
    assert bridge._camera_frame_count == 0
    assert len(bridge._camera_ingest_times) == 0


# =============================================================================
# 3. Buffer Replacement / Drop Counter & Semantics Preservation
# =============================================================================

def test_frame_replacement_counter(dummy_frame):
    """Verifies that LatestFrameBuffer accurately tracks submission and drop/replacement counts."""
    buffer = LatestFrameBuffer()

    assert buffer.submission_count == 0
    assert buffer.replacement_count == 0
    assert not buffer.has_pending()

    # First frame (no replacement)
    buffer.put(dummy_frame)
    assert buffer.submission_count == 1
    assert buffer.replacement_count == 0
    assert buffer.has_pending()

    # Second frame without consuming first (triggers 1 replacement)
    buffer.put(dummy_frame)
    assert buffer.submission_count == 2
    assert buffer.replacement_count == 1

    # Third frame (triggers 2nd replacement)
    buffer.put(dummy_frame)
    assert buffer.submission_count == 3
    assert buffer.replacement_count == 2

    # Pop latest
    data = buffer.get(timeout=0.1)
    assert data is not None
    assert not buffer.has_pending()

    # Fourth frame (no replacement because slot was empty)
    buffer.put(dummy_frame)
    assert buffer.submission_count == 4
    assert buffer.replacement_count == 2

    buffer.reset_counts()
    assert buffer.submission_count == 0
    assert buffer.replacement_count == 0


def test_single_slot_buffer_semantics_preserved(dummy_frame):
    """Verifies that LatestFrameBuffer strictly retains single-slot replacement semantics."""
    buffer = LatestFrameBuffer()

    meta1 = {"id": 1}
    meta2 = {"id": 2}
    meta3 = {"id": 3}

    buffer.put(dummy_frame, metadata=meta1)
    buffer.put(dummy_frame, metadata=meta2)
    buffer.put(dummy_frame, metadata=meta3)

    # Exactly 1 frame popped, which must be meta3
    popped = buffer.get(timeout=0.1)
    assert popped is not None
    _, _, meta, _ = popped
    assert meta["id"] == 3

    # Next pop must time out (slot is now empty)
    assert buffer.get(timeout=0.05) is None


# =============================================================================
# 4. Worker Processing, Errors, and Result Emission Counters
# =============================================================================

def test_worker_processing_error_and_emission_counters(dummy_frame):
    """Verifies that InferenceWorker counts processed frames, caught errors, and dispatched results."""
    dispatched_results = []

    def callback(res):
        dispatched_results.append(res)

    mock_pipe = MockPipeline()
    worker = InferenceWorker(pipeline=mock_pipe, result_callback=callback)
    worker.start()

    worker.submit_frame(dummy_frame)
    time.sleep(0.05)
    worker.submit_frame(dummy_frame)
    time.sleep(0.05)

    assert worker.frames_processed == 2
    assert worker.results_emitted == 2
    assert len(dispatched_results) == 2
    assert worker.error_count == 0

    # Inject error in pipeline
    mock_pipe.raise_error = True
    worker.submit_frame(dummy_frame)
    time.sleep(0.05)

    assert worker.error_count == 1
    # Fallback result was dispatched
    assert worker.results_emitted == 3
    assert len(dispatched_results) == 3
    assert dispatched_results[-1]["action"] == "IDLE"
    assert dispatched_results[-1]["status"] == "OUT_OF_SEQUENCE"

    worker.shutdown()


# =============================================================================
# 5. Statistics, Percentiles, and Zero/Insufficient Sample Handling
# =============================================================================

def test_zero_and_insufficient_sample_handling():
    """Verifies that rolling FPS and percentile computations handle empty / single samples safely."""
    # Zero samples
    assert InferenceWorker._compute_rolling_fps([]) == 0.0
    assert InferenceWorker._compute_mean([]) == 0.0
    assert InferenceWorker._compute_p95([]) == 0.0

    # Single sample
    assert InferenceWorker._compute_rolling_fps([10.0]) == 0.0
    assert InferenceWorker._compute_mean([42.5]) == 42.5
    assert InferenceWorker._compute_p95([42.5]) == 42.5

    # Degenerate duration
    assert InferenceWorker._compute_rolling_fps([10.0, 10.0]) == 0.0


def test_deterministic_statistics_with_fixed_sample_data():
    """Verifies exact deterministic mean and p95 calculations with known array."""
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    expected_mean = 55.0
    expected_p95 = float(np.percentile(data, 95))

    computed_mean = InferenceWorker._compute_mean(data)
    computed_p95 = InferenceWorker._compute_p95(data)

    assert abs(computed_mean - expected_mean) < 1e-6
    assert abs(computed_p95 - expected_p95) < 1e-6


def test_rolling_fps_window_filtering():
    """Verifies rolling FPS filters timestamps outside the specified window."""
    # Timestamps spanning 10 seconds, but window is 2 seconds
    timestamps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 8.5, 9.0, 9.5, 10.0]
    # Window of 2.0s around 10.0s captures [8.0, 8.5, 9.0, 9.5, 10.0] (5 samples over 2.0s -> 4 intervals / 2.0s = 2.0 FPS)
    fps = InferenceWorker._compute_rolling_fps(timestamps, window_sec=2.0)
    assert abs(fps - 2.0) < 0.01


# =============================================================================
# 6. Latency, Frame Age, and Inference Duration Metrics
# =============================================================================

def test_frame_age_and_inference_duration_metrics(dummy_frame):
    """Verifies that frame age at worker pickup and inference duration are recorded in telemetry."""
    delay = 0.01  # 10 ms simulated inference
    mock_pipe = MockPipeline(delay_sec=delay)
    worker = InferenceWorker(pipeline=mock_pipe)
    worker.start()

    t_capture = time.monotonic() - 0.02  # Frame captured 20 ms ago
    worker.submit_frame(
        dummy_frame,
        metadata={"capture_monotonic": t_capture},
    )
    time.sleep(0.05)

    telemetry = worker.get_telemetry()

    assert telemetry["frames_processed"] == 1
    assert telemetry["frames_submitted"] == 1
    assert telemetry["frames_replaced"] == 0

    # Inference duration should be >= 10 ms
    assert telemetry["mean_inference_ms"] >= 9.0
    assert telemetry["latest_inference_duration_ms"] is not None
    assert telemetry["latest_inference_duration_ms"] >= 9.0

    # Frame age should be >= 20 ms
    assert telemetry["mean_frame_age_ms"] >= 19.0
    assert telemetry["latest_frame_age_ms"] is not None
    assert telemetry["latest_frame_age_ms"] >= 19.0

    # Target metadata
    assert telemetry["target_fps"]["camera"] == 30.0
    assert telemetry["target_fps"]["display"] == 30.0
    assert telemetry["target_fps"]["ai_min"] == 10.0
    assert telemetry["target_fps"]["ai_max"] == 15.0

    worker.shutdown()


# =============================================================================
# 7. Frozen Public 8-Field Schema Unchanged
# =============================================================================

def test_public_8_field_schema_unchanged(dummy_frame):
    """Verifies that internal telemetry is completely shielded from public result payloads."""
    delivered_result = None

    def callback(res):
        nonlocal delivered_result
        delivered_result = res

    pipeline = InferencePipeline()
    worker = InferenceWorker(pipeline=pipeline, result_callback=callback)
    worker.start()

    worker.submit_frame(dummy_frame)
    time.sleep(0.05)

    assert delivered_result is not None
    assert set(delivered_result.keys()) == FROZEN_PUBLIC_KEYS
    # Ensure zero telemetry keys leak into public contract
    for telemetry_key in [
        "capture_monotonic",
        "frame_age_ms",
        "inference_duration_ms",
        "frames_processed",
        "camera_ingest_fps",
        "ai_processing_fps",
    ]:
        assert telemetry_key not in delivered_result

    worker.shutdown()


# =============================================================================
# 8. Reset and Shutdown Telemetry Consistency
# =============================================================================

def test_telemetry_reset_and_shutdown_consistency(dummy_frame):
    """Verifies that reset clears all telemetry and shutdown leaves telemetry in a consistent state."""
    worker = InferenceWorker(pipeline=MockPipeline())
    worker.start()

    for _ in range(5):
        worker.submit_frame(dummy_frame)
    time.sleep(0.05)

    t1 = worker.get_telemetry()
    assert t1["frames_submitted"] == 5

    # Reset telemetry
    worker.reset_telemetry()
    t2 = worker.get_telemetry()
    assert t2["frames_submitted"] == 0
    assert t2["frames_replaced"] == 0
    assert t2["frames_processed"] == 0
    assert t2["mean_inference_ms"] == 0.0
    assert t2["latest_inference_duration_ms"] is None

    # Shutdown
    worker.shutdown()
    t3 = worker.get_telemetry()
    assert t3["frames_submitted"] == 0
    assert not worker.is_running
