"""
test_b13_1_telemetry_aggregator.py — Gate B13.1 Verification Suite
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B13.1)

Comprehensive tests for TelemetryDiagnosticAggregator:
- Aggregator initialization & empty snapshot
- Frame counter and throughput aggregation
- Dropped frames, drop rate, and buffer occupancy
- Stage latency breakdown & rolling statistical profiling
- Frame age and end-to-end latency tracking
- Multimodal consistency diagnostic aggregation (B11.2)
- Uncertainty handling diagnostic aggregation (B11.1)
- Temporal confirmation engine diagnostic aggregation (B10)
- SequenceValidatorFSM procedure validation diagnostic aggregation (B9)
- RecoveryManager procedural recovery diagnostic aggregation (B12)
- Deterministic JSON snapshot serialization
- Thread-safe concurrent worker updates & GUI reads
- Lifecycle transitions (START, PAUSE, RESUME, RESET, STOP, COMPLETED)
- Session isolation (no stale state across sessions)
- Memory boundedness & zero unbounded growth
- Strict isolation from frozen 8-field public AI contract
- Pipeline & worker integration without decision alterations
"""

import json
import threading
import time
from typing import Any, Dict
import numpy as np
import pytest

from backend.ai.telemetry_aggregator import (
    BufferTelemetrySection,
    ConsistencyDiagnosticSection,
    LifecycleTelemetrySection,
    PipelineTelemetrySection,
    ProcedureDiagnosticSection,
    RecoveryDiagnosticSection,
    StageLatenciesSection,
    TelemetryDiagnosticAggregator,
    TelemetrySnapshot,
    TemporalDiagnosticSection,
    UncertaintyDiagnosticSection,
)
from backend.ai.inference_pipeline import InferencePipeline
from backend.ai.inference_worker import InferenceWorker, RateStrategy
from backend.ai.result_adapter import AIResultAdapter
from backend.experiment.procedure_manager import ProcedureManager
from backend.experiment.recovery_manager import RecoveryEvent, RecoveryManager
from backend.experiment.sequence_validator import SequenceValidatorFSM


# =============================================================================
# 1. Initialization & Clean Empty Snapshot
# =============================================================================

def test_aggregator_initialization_defaults():
    agg = TelemetryDiagnosticAggregator()
    snap = agg.get_snapshot()

    assert isinstance(snap, TelemetrySnapshot)
    assert snap.lifecycle.session_state == "IDLE"
    assert snap.lifecycle.session_id is None
    assert snap.pipeline.frames_submitted == 0
    assert snap.pipeline.frames_processed == 0
    assert snap.pipeline.frame_drop_rate == 0.0
    assert snap.consistency.total_evaluations == 0
    assert snap.uncertainty.total_evaluations == 0
    assert snap.temporal.total_evaluations == 0
    assert snap.procedure.current_step_index == 0
    assert snap.recovery.total_recoveries == 0


def test_clean_empty_snapshot_serialization():
    agg = TelemetryDiagnosticAggregator()
    snap_dict = agg.get_snapshot_dict()
    snap_json = agg.get_snapshot_json()

    assert isinstance(snap_dict, dict)
    assert isinstance(snap_json, str)
    parsed = json.loads(snap_json)

    assert "timestamp" in parsed
    assert "lifecycle" in parsed
    assert "pipeline" in parsed
    assert "stage_latencies" in parsed
    assert "buffer" in parsed
    assert "consistency" in parsed
    assert "uncertainty" in parsed
    assert "temporal" in parsed
    assert "procedure" in parsed
    assert "recovery" in parsed


# =============================================================================
# 2. Pipeline Performance & Latency Telemetry
# =============================================================================

def test_frame_counter_and_drop_rate_aggregation():
    agg = TelemetryDiagnosticAggregator()
    agg.start_session("TEST_SESSION")

    # Record 10 frame submissions and 2 buffer replacements
    t_base = time.monotonic()
    for i in range(10):
        agg.record_frame_submission(t_base + i * 0.033)
    agg.record_buffer_status(submission_count=10, replacement_count=2, has_pending=True)

    snap = agg.get_snapshot()
    assert snap.pipeline.frames_submitted == 10
    assert snap.pipeline.frames_replaced == 2
    assert snap.pipeline.frame_drop_rate == 0.20
    assert snap.buffer.has_pending is True


def test_worker_timings_and_percentiles():
    agg = TelemetryDiagnosticAggregator(window_size=50)
    t_base = time.monotonic()

    # Record 20 processing cycles with known latencies
    latencies = [10.0 + i for i in range(20)]
    for i, lat in enumerate(latencies):
        agg.record_worker_timings(
            process_end_mono=t_base + i * 0.05,
            infer_dur_ms=lat,
            frame_age_ms=5.0 + i * 0.5,
            dispatch_ms=1.0,
            e2e_ms=lat + 5.0 + i * 0.5 + 1.0,
        )

    snap = agg.get_snapshot()
    assert snap.pipeline.frames_processed == 20
    assert snap.pipeline.latest_inference_duration_ms == 29.0
    assert snap.pipeline.mean_inference_ms == pytest.approx(float(np.mean(latencies)), abs=0.01)
    assert snap.pipeline.p95_inference_ms == pytest.approx(float(np.percentile(latencies, 95)), abs=0.01)
    assert snap.pipeline.latest_frame_age_ms == pytest.approx(5.0 + 19 * 0.5, abs=0.01)


def test_stage_latency_breakdown_aggregation():
    agg = TelemetryDiagnosticAggregator()
    agg.record_stage_latencies(
        stage_a_rectification_ms=0.5,
        stage_b_detection_ms=8.0,
        stage_c_consistency_ms=0.2,
        stage_d_temporal_ms=0.3,
        stage_e_fsm_ms=0.4,
        stage_f_adapter_ms=0.1,
        total_pipeline_ms=9.5,
    )

    snap = agg.get_snapshot()
    stg = snap.stage_latencies
    assert stg.stage_a_rectification_ms == 0.5
    assert stg.stage_b_detection_ms == 8.0
    assert stg.total_pipeline_ms == 9.5
    assert stg.latest["stage_c_consistency_ms"] == 0.2


# =============================================================================
# 3. Multimodal Consistency Diagnostics (Gate B11.2)
# =============================================================================

def test_multimodal_consistency_diagnostics_aggregation():
    agg = TelemetryDiagnosticAggregator()

    agg.record_consistency_result({
        "category": "RELIABLE_ALIGNED",
        "consistency_score": 0.95,
        "is_reliable": True,
    })
    agg.record_consistency_result({
        "category": "CROSS_MODAL_CONFLICT",
        "consistency_score": 0.20,
        "is_reliable": False,
    })
    agg.record_consistency_result({
        "category": "UNCERTAIN_MISSING_OBJECT",
        "consistency_score": 0.40,
        "is_reliable": False,
    })

    snap = agg.get_snapshot()
    c = snap.consistency
    assert c.total_evaluations == 3
    assert c.reliable_aligned_count == 1
    assert c.cross_modal_conflict_count == 1
    assert c.missing_object_count == 1
    assert c.latest_category == "UNCERTAIN_MISSING_OBJECT"
    assert c.latest_score == 0.40
    assert c.latest_is_reliable is False


# =============================================================================
# 4. Uncertainty Diagnostics (Gate B11.1)
# =============================================================================

def test_uncertainty_diagnostics_aggregation():
    agg = TelemetryDiagnosticAggregator()

    agg.record_uncertainty_result({
        "state": "CONFIDENT",
        "confidence": 0.88,
        "margin": 0.18,
        "is_reliable": True,
        "evidence_count": 0,
    })
    agg.record_uncertainty_result({
        "state": "MARGINAL",
        "confidence": 0.62,
        "margin": 0.12,
        "is_reliable": False,
        "evidence_count": 2,
    })
    agg.record_uncertainty_result({
        "state": "RESOLVED",
        "confidence": 0.72,
        "margin": 0.02,
        "is_reliable": True,
        "evidence_count": 3,
    })

    snap = agg.get_snapshot()
    u = snap.uncertainty
    assert u.total_evaluations == 3
    assert u.confident_count == 1
    assert u.marginal_count == 1
    assert u.resolved_count == 1
    assert u.latest_state == "RESOLVED"
    assert u.latest_confidence == 0.72
    assert u.active_window_size == 3
    assert u.latest_is_reliable is True


# =============================================================================
# 5. Temporal Diagnostics (Gate B10)
# =============================================================================

def test_temporal_diagnostics_aggregation():
    agg = TelemetryDiagnosticAggregator()

    agg.record_temporal_result({
        "confirmed": False,
        "candidate": "PICK_RED:RED_SAMPLE",
        "streak": 1,
        "confirmation_progress": 0.33,
        "in_cooldown": False,
    })
    agg.record_temporal_result({
        "confirmed": True,
        "candidate": "PICK_RED:RED_SAMPLE",
        "streak": 3,
        "confirmation_progress": 1.0,
        "in_cooldown": True,
    })

    snap = agg.get_snapshot()
    t = snap.temporal
    assert t.total_evaluations == 2
    assert t.confirmed_count == 1
    assert t.cooldown_events == 1
    assert t.latest_candidate == "PICK_RED:RED_SAMPLE"
    assert t.latest_streak == 3
    assert t.latest_progress_ratio == 1.0
    assert t.is_in_cooldown is True


# =============================================================================
# 6. Procedure Validation Diagnostics (Gate B9)
# =============================================================================

def test_procedure_validation_diagnostics_aggregation():
    agg = TelemetryDiagnosticAggregator()
    fsm = SequenceValidatorFSM()
    fsm.start()

    agg.record_fsm_result({"status": "VALID", "error_type": None}, validator=fsm)
    agg.record_fsm_result({"status": "SKIPPED", "error_type": "SKIPPED_STEP"}, validator=fsm)
    agg.record_fsm_result({"status": "OUT_OF_SEQUENCE", "error_type": "OUT_OF_ORDER"}, validator=fsm)

    snap = agg.get_snapshot()
    p = snap.procedure
    assert p.valid_transitions == 1
    assert p.skipped_steps == 1
    assert p.out_of_order_events == 1
    assert p.fsm_state == "RUNNING"
    assert p.current_step_id == "S1"
    assert p.current_expected_action == "PICK_RED"


# =============================================================================
# 7. Recovery Diagnostics (Gate B12)
# =============================================================================

def test_recovery_diagnostics_aggregation():
    agg = TelemetryDiagnosticAggregator()

    ev = RecoveryEvent(
        event_type="PROCEDURAL_RECOVERY_OUT_OF_ORDER",
        expected_step="STEP-01",
        detected_action="PLACE_RED",
        procedural_status="OUT_OF_ORDER",
        explanation="Action performed before step 1",
        recovery_instruction="Return sample to tray",
        timeout_s=30.0,
    )
    agg.record_recovery_event(ev)

    snap = agg.get_snapshot()
    r = snap.recovery
    assert r.total_recoveries == 1
    assert r.out_of_order_recoveries == 1
    assert r.active_state == "RECOVERY_ACTIVE"
    assert r.latest_instruction == "Return sample to tray"
    assert r.latest_timeout_s == 30.0


# =============================================================================
# 8. Lifecycle Transitions & Duration Tracking
# =============================================================================

def test_lifecycle_transitions_and_pause_resume():
    agg = TelemetryDiagnosticAggregator()
    assert agg.get_snapshot().lifecycle.session_state == "IDLE"

    agg.start_session("LIFECYCLE_TEST")
    assert agg.get_snapshot().lifecycle.session_state == "RUNNING"
    assert agg.get_snapshot().lifecycle.session_id == "LIFECYCLE_TEST"

    time.sleep(0.05)
    agg.pause_session()
    snap_paused = agg.get_snapshot()
    assert snap_paused.lifecycle.session_state == "PAUSED"
    dur_paused = snap_paused.lifecycle.duration_seconds

    time.sleep(0.05)
    agg.resume_session()
    assert agg.get_snapshot().lifecycle.session_state == "RUNNING"

    agg.stop_session()
    snap_stopped = agg.get_snapshot()
    assert snap_stopped.lifecycle.session_state == "STOPPED"
    assert snap_stopped.lifecycle.end_time is not None

    agg.complete_session()
    assert agg.get_snapshot().lifecycle.session_state == "COMPLETED"


def test_reset_flushes_all_state():
    agg = TelemetryDiagnosticAggregator()
    agg.start_session("RESET_TEST")
    agg.record_frame_submission()
    agg.record_consistency_result({"category": "RELIABLE_ALIGNED"})
    agg.record_uncertainty_result({"state": "CONFIDENT"})
    agg.record_temporal_result({"confirmed": True})

    assert agg.get_snapshot().pipeline.frames_submitted == 1
    assert agg.get_snapshot().consistency.total_evaluations == 1

    agg.reset()
    snap = agg.get_snapshot()
    assert snap.lifecycle.session_state == "IDLE"
    assert snap.lifecycle.session_id is None
    assert snap.pipeline.frames_submitted == 0
    assert snap.consistency.total_evaluations == 0
    assert snap.uncertainty.total_evaluations == 0
    assert snap.temporal.total_evaluations == 0


# =============================================================================
# 9. Bounded Rolling Windows & Memory Safety
# =============================================================================

def test_bounded_rolling_memory_safety():
    agg = TelemetryDiagnosticAggregator(window_size=20)
    t_base = time.monotonic()

    # Append 500 records into a capacity-20 window
    for i in range(500):
        agg.record_worker_timings(
            process_end_mono=t_base + i * 0.001,
            infer_dur_ms=10.0 + (i % 5),
            frame_age_ms=2.0,
            dispatch_ms=0.5,
            e2e_ms=12.5,
        )

    assert len(agg._processing_times) == 20
    assert len(agg._inference_durations_ms) == 20
    assert agg._frames_processed == 500

    snap = agg.get_snapshot()
    assert snap.pipeline.frames_processed == 500
    assert snap.pipeline.mean_inference_ms > 0.0


# =============================================================================
# 10. Thread Safety & Concurrent Worker Updates / GUI Reads
# =============================================================================

def test_thread_safe_concurrent_updates_and_reads():
    agg = TelemetryDiagnosticAggregator()
    agg.start_session("CONCURRENCY_TEST")

    stop_event = threading.Event()
    read_snapshots = []
    errors = []

    def writer_thread():
        try:
            for i in range(200):
                agg.record_frame_submission()
                agg.record_worker_timings(
                    process_end_mono=time.monotonic(),
                    infer_dur_ms=5.0,
                    frame_age_ms=1.0,
                    dispatch_ms=0.2,
                    e2e_ms=6.2,
                )
                agg.record_consistency_result({"category": "RELIABLE_ALIGNED", "consistency_score": 0.9})
                agg.record_uncertainty_result({"state": "CONFIDENT", "confidence": 0.85})
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    def reader_thread():
        try:
            while not stop_event.is_set():
                snap = agg.get_snapshot()
                snap_json = agg.get_snapshot_json()
                read_snapshots.append(len(snap_json))
                time.sleep(0.002)
        except Exception as e:
            errors.append(e)

    t_w1 = threading.Thread(target=writer_thread)
    t_w2 = threading.Thread(target=writer_thread)
    t_r = threading.Thread(target=reader_thread)

    t_r.start()
    t_w1.start()
    t_w2.start()

    t_w1.join()
    t_w2.join()
    stop_event.set()
    t_r.join()

    assert len(errors) == 0
    assert len(read_snapshots) > 0
    snap = agg.get_snapshot()
    assert snap.pipeline.frames_processed == 400
    assert snap.consistency.total_evaluations == 400


# =============================================================================
# 11. Pipeline & Worker Integration with Aggregator
# =============================================================================

def test_pipeline_passive_ingestion_and_contract_isolation():
    agg = TelemetryDiagnosticAggregator()
    pipeline = InferencePipeline(telemetry_aggregator=agg)
    fsm = SequenceValidatorFSM(procedure_manager=ProcedureManager())
    fsm.start()
    pipeline.validator = fsm

    # Feed synthetic frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    public_res = pipeline.process_frame_public(
        frame=dummy_frame,
        timestamp="2026-09-28T12:00:00",
    )

    # 1. Verify frozen 8-field public AI contract remains unmodified
    assert isinstance(public_res, dict)
    expected_fields = {
        "timestamp", "action", "object", "confidence",
        "expected_step", "detected_step", "status", "next_step"
    }
    assert set(public_res.keys()) == expected_fields
    assert "pipeline" not in public_res
    assert "telemetry" not in public_res

    # 2. Verify aggregator captured diagnostic cycle passively
    snap = agg.get_snapshot()
    assert snap.consistency.total_evaluations >= 1
    assert snap.uncertainty.total_evaluations >= 1


def test_worker_telemetry_snapshot_api():
    agg = TelemetryDiagnosticAggregator()
    worker = InferenceWorker(telemetry_aggregator=agg)

    worker.start()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    worker.submit_frame(dummy_frame)

    time.sleep(0.1)
    worker.stop()

    snap_dict = worker.get_telemetry_snapshot()
    assert snap_dict is not None
    assert isinstance(snap_dict, dict)
    assert snap_dict["pipeline"]["frames_submitted"] >= 1


# =============================================================================
# 12. Bridge Telemetry & Diagnostic Snapshot Slots
# =============================================================================

def test_bridge_telemetry_snapshot_slot():
    from backend.app_state import AppState
    from backend.bridge import Bridge

    agg = TelemetryDiagnosticAggregator()
    worker = InferenceWorker(telemetry_aggregator=agg)
    app_state = AppState()
    app_state.inference_worker = worker

    bridge = Bridge(app_state)
    snap_json = bridge.getTelemetrySnapshot()
    assert isinstance(snap_json, str)
    parsed = json.loads(snap_json)
    assert "pipeline" in parsed
    assert "consistency" in parsed

    diag_json = bridge.getDiagnosticSnapshot()
    diag_parsed = json.loads(diag_json)
    assert diag_parsed["pipeline"] == parsed["pipeline"]
    assert diag_parsed["consistency"] == parsed["consistency"]


# =============================================================================
# 13. Session Isolation & Clean Re-Start
# =============================================================================

def test_session_isolation_across_multiple_runs():
    agg = TelemetryDiagnosticAggregator()

    # Session 1
    agg.start_session("SESSION_001")
    for _ in range(5):
        agg.record_frame_submission()
        agg.record_consistency_result({"category": "RELIABLE_ALIGNED"})
    agg.stop_session()

    snap1 = agg.get_snapshot()
    assert snap1.lifecycle.session_id == "SESSION_001"
    assert snap1.pipeline.frames_submitted == 5
    assert snap1.consistency.reliable_aligned_count == 5

    # Reset and start Session 2
    agg.reset()
    agg.start_session("SESSION_002")
    agg.record_frame_submission()

    snap2 = agg.get_snapshot()
    assert snap2.lifecycle.session_id == "SESSION_002"
    assert snap2.pipeline.frames_submitted == 1
    assert snap2.consistency.reliable_aligned_count == 0


# =============================================================================
# 14. Performance Overhead Micro-Benchmark (< 0.1 ms / frame)
# =============================================================================

def test_telemetry_aggregation_microsecond_overhead():
    agg = TelemetryDiagnosticAggregator()
    agg.start_session("PERF_TEST")

    # Measure time for 1,000 full-cycle telemetry updates
    t_start = time.perf_counter()
    for i in range(1000):
        agg.record_frame_submission(time.monotonic())
        agg.record_worker_timings(
            process_end_mono=time.monotonic(),
            infer_dur_ms=5.0,
            frame_age_ms=1.2,
            dispatch_ms=0.1,
            e2e_ms=6.3,
        )
        agg.record_stage_latencies(
            stage_a_rectification_ms=0.1,
            stage_b_detection_ms=3.5,
            stage_c_consistency_ms=0.2,
            stage_d_temporal_ms=0.2,
            stage_e_fsm_ms=0.3,
            stage_f_adapter_ms=0.1,
        )
        agg.record_consistency_result({
            "category": "RELIABLE_ALIGNED",
            "consistency_score": 0.95,
            "is_reliable": True,
        })
        agg.record_uncertainty_result({
            "state": "CONFIDENT",
            "confidence": 0.90,
            "margin": 0.20,
            "is_reliable": True,
        })
        agg.record_temporal_result({
            "confirmed": True,
            "candidate": "PICK_RED:RED_SAMPLE",
            "streak": 3,
            "confirmation_progress": 1.0,
            "in_cooldown": False,
        })

    t_elapsed = time.perf_counter() - t_start
    avg_overhead_ms = (t_elapsed / 1000.0) * 1000.0

    # Verification: Aggregation overhead must remain well under 0.1 ms (100 microseconds)
    assert avg_overhead_ms < 0.10, f"Overhead {avg_overhead_ms:.4f} ms exceeds 0.1 ms threshold"

