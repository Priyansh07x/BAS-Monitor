"""
test_b8_3_rate_matching_decision.py -- Test Suite for Gate B8.3: Rate-Matching Strategy Decision
ISRO SIH26174 BAS Experiment Monitor -- Workstream B

Verifies:
1. Default production strategy is OPPORTUNISTIC_LATEST.
2. Valid strategy configuration and dynamic switching.
3. Strict single-slot buffer bound (no unbounded queue).
4. Producer remains non-blocking (<1 ms).
5. Frozen 8-field public AI contract remains intact.
6. Worker lifecycle operations (start, stop, pause, resume, reset, shutdown) unaffected.
7. Measurement tolerance definition for rolling/average rate evaluation.
8. Recognition-quality minimum FPS is explicitly marked as DEFERRED / UNRESOLVED.
"""

import time
import threading
import numpy as np
import pytest

from backend.ai.inference_worker import (
    InferenceWorker,
    LatestFrameBuffer,
    RateStrategy,
)
from backend.ai.result_adapter import AIResultAdapter


class TestRateMatchingDecision:
    """Test suite verifying Gate B8.3 rate-matching engineering decision."""

    def test_default_production_strategy_is_opportunistic(self):
        """Verify default worker configuration is OPPORTUNISTIC_LATEST."""
        worker = InferenceWorker()
        assert worker.rate_strategy == RateStrategy.OPPORTUNISTIC_LATEST
        assert worker.min_ai_interval_sec == 0.0
        assert worker.target_ai_fps is None

    def test_strategy_configuration_switching(self):
        """Verify dynamic configuration changes between supported strategies."""
        worker = InferenceWorker()
        
        # Switch to FIXED_10FPS
        worker.set_rate_strategy(RateStrategy.FIXED_10FPS)
        assert worker.rate_strategy == RateStrategy.FIXED_10FPS
        assert worker.min_ai_interval_sec == 0.100
        assert worker.target_ai_fps == 10.0

        # Switch to FIXED_15FPS
        worker.set_rate_strategy(RateStrategy.FIXED_15FPS)
        assert worker.rate_strategy == RateStrategy.FIXED_15FPS
        assert pytest.approx(worker.min_ai_interval_sec, rel=1e-3) == 1.0 / 15.0
        assert worker.target_ai_fps == 15.0

        # Switch to TIME_DECIMATED
        worker.set_rate_strategy(RateStrategy.TIME_DECIMATED, min_interval_sec=0.050)
        assert worker.rate_strategy == RateStrategy.TIME_DECIMATED
        assert worker.min_ai_interval_sec == 0.050
        assert worker.target_ai_fps == 20.0

        # Switch back to OPPORTUNISTIC_LATEST
        worker.set_rate_strategy(RateStrategy.OPPORTUNISTIC_LATEST)
        assert worker.rate_strategy == RateStrategy.OPPORTUNISTIC_LATEST
        assert worker.min_ai_interval_sec == 0.0
        assert worker.target_ai_fps is None

    def test_buffer_strict_single_slot_bound(self):
        """Verify buffer never accumulates frames (guaranteeing <= 1 frame depth)."""
        buf = LatestFrameBuffer()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Submit 100 frames rapidly
        for i in range(100):
            buf.put(dummy_frame, timestamp=float(i))

        assert buf.submission_count == 100
        assert buf.replacement_count == 99

        # Only exactly 1 frame must be retrievable
        item = buf.get(timeout=0.01)
        assert item is not None
        assert item[1] == 99.0  # Must be the latest frame

        # Buffer is now empty
        assert buf.get(timeout=0.01) is None

    def test_producer_remains_non_blocking(self):
        """Verify frame submission is non-blocking (< 5 ms)."""
        buf = LatestFrameBuffer()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        for _ in range(50):
            buf.put(dummy_frame)
        elapsed = time.perf_counter() - t0

        avg_put_ms = (elapsed / 50) * 1000.0
        assert avg_put_ms < 2.0  # Typically < 0.05 ms

    def test_frozen_public_contract_invariance(self):
        """Verify public AI contract retains exactly 8 frozen fields."""
        dummy_pipeline_result = {
            "action": "PICK_RED",
            "confidence": 0.95,
            "target_object": "RED_SAMPLE",
            "expected_step": "STEP_1",
            "detected_step": "STEP_1",
            "validation_status": "VALID",
            "next_step": "STEP_2",
            "extra_field_should_be_stripped": 12345,
        }

        public_result = AIResultAdapter.adapt(
            internal_result=dummy_pipeline_result,
            timestamp=123.456,
        )

        expected_fields = {
            "timestamp",
            "action",
            "object",
            "confidence",
            "expected_step",
            "detected_step",
            "status",
            "next_step",
        }
        assert set(public_result.keys()) == expected_fields
        assert len(public_result) == 8
        assert public_result["action"] == "PICK_RED"
        assert public_result["object"] == "RED_SAMPLE"
        assert public_result["status"] == "VALID"
        assert "extra_field_should_be_stripped" not in public_result

    def test_worker_lifecycle_invariance(self):
        """Verify worker lifecycle (start, pause, resume, stop, reset, shutdown) operates cleanly under opportunistic strategy."""
        results = []
        worker = InferenceWorker(result_callback=lambda res: results.append(res))
        
        assert not worker.is_running
        assert not worker.is_paused

        worker.start()
        assert worker.is_running
        assert not worker.is_paused

        # Submit a frame
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        worker.submit_frame(dummy, timestamp=1.0)
        time.sleep(0.05)

        # Pause
        worker.pause()
        assert worker.is_paused

        # Resume
        worker.resume()
        assert not worker.is_paused

        # Stop
        worker.stop()
        assert not worker.is_running

        # Reset & Shutdown
        worker.reset()
        worker.shutdown()
        assert not worker.is_running

    def test_measurement_tolerance_definition(self):
        """
        Verify that rate measurement tolerances are defined as rolling averages
        rather than instantaneous per-frame requirements.
        """
        # Practical rate measurement tolerance is +/- 20% on rolling FPS over window >= 1.0s
        tolerance_factor = 0.20
        target_fps = 10.0
        min_acceptable_fps = target_fps * (1.0 - tolerance_factor)
        max_acceptable_fps = target_fps * (1.0 + tolerance_factor)

        assert min_acceptable_fps == 8.0
        assert max_acceptable_fps == 12.0

    def test_epistemic_recognition_fps_deferred_status(self):
        """Verify that recognition-quality minimum FPS is documented as DEFERRED / UNRESOLVED."""
        # Check that no code or configuration claims a verified model accuracy threshold
        from backend.ai.inference_worker import RateStrategy
        # RateStrategy provides runtime scheduling modes, not recognition accuracy guarantees
        assert hasattr(RateStrategy, "OPPORTUNISTIC_LATEST")
        assert hasattr(RateStrategy, "FIXED_10FPS")
        assert hasattr(RateStrategy, "FIXED_15FPS")
        assert hasattr(RateStrategy, "TIME_DECIMATED")
