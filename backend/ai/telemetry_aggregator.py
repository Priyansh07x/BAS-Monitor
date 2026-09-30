"""
telemetry_aggregator.py — Unified Telemetry & Diagnostic Aggregator Core
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B13.1)

Centralized, passive, thread-safe observer aggregating pipeline performance telemetry,
stage-level latency profiles, multimodal consistency categories, uncertainty evidence
states, temporal confirmation hysteresis, authoritative FSM procedure validation,
and procedural recovery events into deterministic, serializable diagnostic snapshots.

Architectural Guarantees:
1. Pure Observer: Does NOT participate in or alter any perception, temporal, uncertainty,
   FSM validation, or recovery decisions.
2. Source of Truth: Consumes metrics and diagnostic statuses exclusively from authoritative
   subsystems (InferenceWorker, LatestFrameBuffer, MultimodalConsistencyEvaluator,
   UncertaintyHandler, TemporalConfirmationEngine, SequenceValidatorFSM, RecoveryManager).
3. Contract Isolation: Telemetry snapshots remain completely out-of-band and never modify
   the frozen 8-field public AI contract (docs/architecture.md §2).
4. Memory Bounded: Uses fixed-capacity circular ring buffers (collections.deque(maxlen=N))
   for all rolling statistics, guaranteeing O(1) appends and zero memory leaks.
5. Thread Safe: Protects internal updates and snapshot generation with lightweight mutexes
   (threading.Lock), allowing safe worker thread updates and concurrent main/GUI reads.
"""

from __future__ import annotations

import collections
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


@dataclass
class PipelineTelemetrySection:
    """Pipeline performance and throughput telemetry."""
    frames_submitted: int = 0
    frames_replaced: int = 0
    frames_processed: int = 0
    error_count: int = 0
    results_emitted: int = 0
    camera_ingest_fps: float = 0.0
    ai_processing_fps: float = 0.0
    frame_drop_rate: float = 0.0
    buffer_occupancy: int = 0
    mean_inference_ms: float = 0.0
    p50_inference_ms: float = 0.0
    p95_inference_ms: float = 0.0
    p99_inference_ms: float = 0.0
    mean_frame_age_ms: float = 0.0
    p95_frame_age_ms: float = 0.0
    mean_dispatch_latency_ms: float = 0.0
    p95_dispatch_latency_ms: float = 0.0
    mean_end_to_end_ms: float = 0.0
    p50_end_to_end_ms: float = 0.0
    p95_end_to_end_ms: float = 0.0
    p99_end_to_end_ms: float = 0.0
    latest_inference_duration_ms: Optional[float] = None
    latest_frame_age_ms: Optional[float] = None
    latest_end_to_end_ms: Optional[float] = None
    rate_strategy: str = "OPPORTUNISTIC_LATEST"
    min_ai_interval_sec: float = 0.0
    target_ai_fps: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StageLatenciesSection:
    """Stage-by-stage latency profile in milliseconds."""
    stage_a_rectification_ms: float = 0.0
    stage_b_detection_ms: float = 0.0
    stage_c_consistency_ms: float = 0.0
    stage_d_temporal_ms: float = 0.0
    stage_e_fsm_ms: float = 0.0
    stage_f_adapter_ms: float = 0.0
    total_pipeline_ms: float = 0.0
    mean: Dict[str, float] = field(default_factory=dict)
    latest: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BufferTelemetrySection:
    """LatestFrameBuffer utilization and drop metrics."""
    submission_count: int = 0
    replacement_count: int = 0
    has_pending: bool = False
    drop_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConsistencyDiagnosticSection:
    """Multimodal perception consistency diagnostics (Gate B11.2)."""
    total_evaluations: int = 0
    reliable_aligned_count: int = 0
    cross_modal_conflict_count: int = 0
    missing_object_count: int = 0
    low_object_confidence_count: int = 0
    static_object_count: int = 0
    object_flicker_count: int = 0
    action_flicker_count: int = 0
    spatial_contradiction_count: int = 0
    idle_count: int = 0
    latest_category: Optional[str] = None
    latest_score: Optional[float] = None
    latest_is_reliable: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UncertaintyDiagnosticSection:
    """Perception uncertainty & evidence accumulation diagnostics (Gate B11.1)."""
    total_evaluations: int = 0
    confident_count: int = 0
    marginal_count: int = 0
    low_confidence_count: int = 0
    conflict_count: int = 0
    resolved_count: int = 0
    confirmed_anomaly_count: int = 0
    idle_count: int = 0
    latest_state: Optional[str] = None
    latest_confidence: Optional[float] = None
    latest_margin: Optional[float] = None
    latest_is_reliable: Optional[bool] = None
    active_window_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalDiagnosticSection:
    """M-of-N temporal confirmation & hysteresis diagnostics (Gate B10)."""
    total_evaluations: int = 0
    confirmed_count: int = 0
    cooldown_events: int = 0
    candidate_updates: int = 0
    latest_candidate: Optional[str] = None
    latest_streak: int = 0
    latest_progress_ratio: float = 0.0
    is_in_cooldown: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProcedureDiagnosticSection:
    """SequenceValidatorFSM authoritative procedure state diagnostics (Gate B9)."""
    current_step_index: int = 0
    current_step_id: Optional[str] = None
    current_expected_action: Optional[str] = None
    current_expected_object: Optional[str] = None
    fsm_state: str = "IDLE"
    valid_transitions: int = 0
    skipped_steps: int = 0
    out_of_order_events: int = 0
    invalid_object_events: int = 0
    unrecognized_events: int = 0
    low_confidence_events: int = 0
    completed_steps: int = 0
    is_completed: bool = False
    latest_validation_status: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecoveryDiagnosticSection:
    """Procedural recovery handling diagnostics (Gate B12)."""
    total_recoveries: int = 0
    skipped_recoveries: int = 0
    out_of_order_recoveries: int = 0
    invalid_object_recoveries: int = 0
    unrecognized_recoveries: int = 0
    active_state: str = "IDLE"
    latest_event: Optional[Dict[str, Any]] = None
    latest_instruction: Optional[str] = None
    latest_timeout_s: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LifecycleTelemetrySection:
    """Telemetry session lifecycle tracking."""
    session_id: Optional[str] = None
    session_state: str = "IDLE"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    total_cycles: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TelemetrySnapshot:
    """
    Complete, immutable diagnostic and telemetry snapshot.
    Deterministic, structured, and JSON-serializable.
    """
    timestamp: str
    lifecycle: LifecycleTelemetrySection
    pipeline: PipelineTelemetrySection
    stage_latencies: StageLatenciesSection
    buffer: BufferTelemetrySection
    consistency: ConsistencyDiagnosticSection
    uncertainty: UncertaintyDiagnosticSection
    temporal: TemporalDiagnosticSection
    procedure: ProcedureDiagnosticSection
    recovery: RecoveryDiagnosticSection

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "lifecycle": self.lifecycle.to_dict(),
            "pipeline": self.pipeline.to_dict(),
            "stage_latencies": self.stage_latencies.to_dict(),
            "buffer": self.buffer.to_dict(),
            "consistency": self.consistency.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "temporal": self.temporal.to_dict(),
            "procedure": self.procedure.to_dict(),
            "recovery": self.recovery.to_dict(),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class TelemetryDiagnosticAggregator:
    """
    Deterministic, thread-safe telemetry and diagnostic aggregator.
    
    Acts as a passive observer collecting performance metrics, latencies,
    and diagnostic events across all Workstream B perception stages.
    """

    def __init__(self, window_size: int = 200, rolling_fps_window_sec: float = 2.0):
        """
        Args:
            window_size: Maximum capacity for rolling latency and timestamp deques.
            rolling_fps_window_sec: Window in seconds for computing rolling frequency.
        """
        self._window_size = max(10, int(window_size))
        self._fps_window_sec = max(0.1, float(rolling_fps_window_sec))
        self._lock = threading.Lock()

        # Session Lifecycle State
        self._session_id: Optional[str] = None
        self._session_state: str = "IDLE"  # IDLE, RUNNING, PAUSED, STOPPED, COMPLETED
        self._session_start_iso: Optional[str] = None
        self._session_end_iso: Optional[str] = None
        self._session_start_mono: Optional[float] = None
        self._session_accumulated_pause_sec: float = 0.0
        self._pause_start_mono: Optional[float] = None
        self._total_cycles: int = 0

        # Rolling Storage (bounded memory)
        self._submission_times: Deque[float] = collections.deque(maxlen=self._window_size)
        self._processing_times: Deque[float] = collections.deque(maxlen=self._window_size)
        self._inference_durations_ms: Deque[float] = collections.deque(maxlen=self._window_size)
        self._frame_ages_ms: Deque[float] = collections.deque(maxlen=self._window_size)
        self._dispatch_latencies_ms: Deque[float] = collections.deque(maxlen=self._window_size)
        self._end_to_end_latencies_ms: Deque[float] = collections.deque(maxlen=self._window_size)

        # Stage Latency Storage (ms)
        self._stage_a_ms: Deque[float] = collections.deque(maxlen=self._window_size)
        self._stage_b_ms: Deque[float] = collections.deque(maxlen=self._window_size)
        self._stage_c_ms: Deque[float] = collections.deque(maxlen=self._window_size)
        self._stage_d_ms: Deque[float] = collections.deque(maxlen=self._window_size)
        self._stage_e_ms: Deque[float] = collections.deque(maxlen=self._window_size)
        self._stage_f_ms: Deque[float] = collections.deque(maxlen=self._window_size)
        self._total_pipe_ms: Deque[float] = collections.deque(maxlen=self._window_size)

        # Latest Raw Metrics
        self._latest_infer_ms: Optional[float] = None
        self._latest_frame_age_ms: Optional[float] = None
        self._latest_dispatch_ms: Optional[float] = None
        self._latest_e2e_ms: Optional[float] = None
        self._latest_stage_latencies: Dict[str, float] = {}

        # Pipeline Telemetry Counters & Config
        self._frames_submitted: int = 0
        self._frames_replaced: int = 0
        self._frames_processed: int = 0
        self._error_count: int = 0
        self._results_emitted: int = 0
        self._buffer_occupancy: int = 0
        self._rate_strategy: str = "OPPORTUNISTIC_LATEST"
        self._min_ai_interval_sec: float = 0.0
        self._target_ai_fps: Optional[float] = None

        # Multimodal Consistency Diagnostics (B11.2)
        self._consistency_total: int = 0
        self._cat_reliable_aligned: int = 0
        self._cat_cross_modal_conflict: int = 0
        self._cat_missing_object: int = 0
        self._cat_low_object_confidence: int = 0
        self._cat_static_object: int = 0
        self._cat_object_flicker: int = 0
        self._cat_action_flicker: int = 0
        self._cat_spatial_contradiction: int = 0
        self._cat_idle: int = 0
        self._latest_consistency_category: Optional[str] = None
        self._latest_consistency_score: Optional[float] = None
        self._latest_consistency_is_reliable: Optional[bool] = None

        # Uncertainty Diagnostics (B11.1)
        self._uncertainty_total: int = 0
        self._unc_confident: int = 0
        self._unc_marginal: int = 0
        self._unc_low_conf: int = 0
        self._unc_conflict: int = 0
        self._unc_resolved: int = 0
        self._unc_confirmed_anomaly: int = 0
        self._unc_idle: int = 0
        self._latest_unc_state: Optional[str] = None
        self._latest_unc_conf: Optional[float] = None
        self._latest_unc_margin: Optional[float] = None
        self._latest_unc_is_reliable: Optional[bool] = None
        self._unc_active_window_size: int = 0

        # Temporal Confirmation Diagnostics (B10)
        self._temporal_total: int = 0
        self._temporal_confirmed: int = 0
        self._temporal_cooldown_events: int = 0
        self._temporal_candidate_updates: int = 0
        self._latest_temporal_candidate: Optional[str] = None
        self._latest_temporal_streak: int = 0
        self._latest_temporal_progress: float = 0.0
        self._temporal_in_cooldown: bool = False

        # Procedure Validation Diagnostics (B9 FSM)
        self._fsm_current_step_index: int = 0
        self._fsm_current_step_id: Optional[str] = None
        self._fsm_expected_action: Optional[str] = None
        self._fsm_expected_object: Optional[str] = None
        self._fsm_state: str = "IDLE"
        self._fsm_valid_transitions: int = 0
        self._fsm_skipped_steps: int = 0
        self._fsm_out_of_order_events: int = 0
        self._fsm_invalid_object_events: int = 0
        self._fsm_unrecognized_events: int = 0
        self._fsm_low_conf_events: int = 0
        self._fsm_completed_steps: int = 0
        self._fsm_is_completed: bool = False
        self._fsm_latest_status: Optional[str] = None

        # Recovery Diagnostics (B12)
        self._recovery_total: int = 0
        self._rec_skipped_count: int = 0
        self._rec_out_of_order_count: int = 0
        self._rec_invalid_object_count: int = 0
        self._rec_unrecognized_count: int = 0
        self._recovery_active_state: str = "IDLE"
        self._latest_recovery_event: Optional[Dict[str, Any]] = None
        self._latest_recovery_instruction: Optional[str] = None
        self._latest_recovery_timeout_s: Optional[float] = None

    # -------------------------------------------------------------------------
    # Helper Computations (Pure / Static)
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_rolling_fps(timestamps: Sequence[float], window_sec: float = 2.0) -> float:
        """Computes rolling frequency from sequence of monotonic timestamps."""
        if not timestamps or len(timestamps) < 2:
            return 0.0
        t_latest = timestamps[-1]
        recent = [t for t in timestamps if (t_latest - t) <= window_sec]
        if len(recent) < 2:
            return 0.0
        duration = recent[-1] - recent[0]
        if duration <= 0.0:
            return 0.0
        return float((len(recent) - 1) / duration)

    @staticmethod
    def _compute_mean(values: Sequence[float]) -> float:
        """Computes mean of numeric sequence or 0.0 if empty."""
        if not values or len(values) == 0:
            return 0.0
        return float(np.mean(values))

    @staticmethod
    def _compute_p50(values: Sequence[float]) -> float:
        """Computes 50th percentile (median) of numeric sequence or 0.0 if empty."""
        if not values or len(values) == 0:
            return 0.0
        return float(np.percentile(values, 50))

    @staticmethod
    def _compute_p95(values: Sequence[float]) -> float:
        """Computes 95th percentile of numeric sequence or 0.0 if empty."""
        if not values or len(values) == 0:
            return 0.0
        return float(np.percentile(values, 95))

    @staticmethod
    def _compute_p99(values: Sequence[float]) -> float:
        """Computes 99th percentile of numeric sequence or 0.0 if empty."""
        if not values or len(values) == 0:
            return 0.0
        return float(np.percentile(values, 99))

    # -------------------------------------------------------------------------
    # Lifecycle Management API
    # -------------------------------------------------------------------------

    def start_session(
        self,
        session_id: Optional[str] = None,
        start_time_iso: Optional[str] = None,
    ) -> None:
        """
        Starts a new clean telemetry monitoring session.
        """
        now = datetime.now()
        mono_now = time.monotonic()
        with self._lock:
            self._session_id = session_id or f"SESSION_{now.strftime('%Y%m%d_%H%M%S')}"
            self._session_state = "RUNNING"
            self._session_start_iso = start_time_iso or now.isoformat()
            self._session_end_iso = None
            self._session_start_mono = mono_now
            self._session_accumulated_pause_sec = 0.0
            self._pause_start_mono = None
            self._total_cycles = 0

    def pause_session(self) -> None:
        """
        Pauses the telemetry session without losing valid accumulated data.
        """
        with self._lock:
            if self._session_state == "RUNNING":
                self._session_state = "PAUSED"
                self._pause_start_mono = time.monotonic()

    def resume_session(self) -> None:
        """
        Resumes telemetry session and adjusts duration to ignore paused time.
        """
        with self._lock:
            if self._session_state == "PAUSED":
                self._session_state = "RUNNING"
                if self._pause_start_mono is not None:
                    paused_dur = time.monotonic() - self._pause_start_mono
                    self._session_accumulated_pause_sec += max(0.0, paused_dur)
                    self._pause_start_mono = None

    def stop_session(self) -> None:
        """
        Stops active monitoring and timestamps the session termination.
        """
        with self._lock:
            if self._session_state in ("RUNNING", "PAUSED"):
                self._session_state = "STOPPED"
                self._session_end_iso = datetime.now().isoformat()
                if self._pause_start_mono is not None:
                    paused_dur = time.monotonic() - self._pause_start_mono
                    self._session_accumulated_pause_sec += max(0.0, paused_dur)
                    self._pause_start_mono = None

    def complete_session(self) -> None:
        """
        Marks procedure experiment session as completed.
        """
        with self._lock:
            self._session_state = "COMPLETED"
            self._session_end_iso = datetime.now().isoformat()
            if self._pause_start_mono is not None:
                paused_dur = time.monotonic() - self._pause_start_mono
                self._session_accumulated_pause_sec += max(0.0, paused_dur)
                self._pause_start_mono = None

    def reset(self) -> None:
        """
        Clears all telemetry counters, rolling windows, and diagnostic state.
        """
        with self._lock:
            self._session_id = None
            self._session_state = "IDLE"
            self._session_start_iso = None
            self._session_end_iso = None
            self._session_start_mono = None
            self._session_accumulated_pause_sec = 0.0
            self._pause_start_mono = None
            self._total_cycles = 0

            # Clear rolling buffers
            self._submission_times.clear()
            self._processing_times.clear()
            self._inference_durations_ms.clear()
            self._frame_ages_ms.clear()
            self._dispatch_latencies_ms.clear()
            self._end_to_end_latencies_ms.clear()

            self._stage_a_ms.clear()
            self._stage_b_ms.clear()
            self._stage_c_ms.clear()
            self._stage_d_ms.clear()
            self._stage_e_ms.clear()
            self._stage_f_ms.clear()
            self._total_pipe_ms.clear()

            self._latest_infer_ms = None
            self._latest_frame_age_ms = None
            self._latest_dispatch_ms = None
            self._latest_e2e_ms = None
            self._latest_stage_latencies.clear()

            self._frames_submitted = 0
            self._frames_replaced = 0
            self._frames_processed = 0
            self._error_count = 0
            self._results_emitted = 0
            self._buffer_occupancy = 0

            # Consistency
            self._consistency_total = 0
            self._cat_reliable_aligned = 0
            self._cat_cross_modal_conflict = 0
            self._cat_missing_object = 0
            self._cat_low_object_confidence = 0
            self._cat_static_object = 0
            self._cat_object_flicker = 0
            self._cat_action_flicker = 0
            self._cat_spatial_contradiction = 0
            self._cat_idle = 0
            self._latest_consistency_category = None
            self._latest_consistency_score = None
            self._latest_consistency_is_reliable = None

            # Uncertainty
            self._uncertainty_total = 0
            self._unc_confident = 0
            self._unc_marginal = 0
            self._unc_low_conf = 0
            self._unc_conflict = 0
            self._unc_resolved = 0
            self._unc_confirmed_anomaly = 0
            self._unc_idle = 0
            self._latest_unc_state = None
            self._latest_unc_conf = None
            self._latest_unc_margin = None
            self._latest_unc_is_reliable = None
            self._unc_active_window_size = 0

            # Temporal
            self._temporal_total = 0
            self._temporal_confirmed = 0
            self._temporal_cooldown_events = 0
            self._temporal_candidate_updates = 0
            self._latest_temporal_candidate = None
            self._latest_temporal_streak = 0
            self._latest_temporal_progress = 0.0
            self._temporal_in_cooldown = False

            # Procedure
            self._fsm_current_step_index = 0
            self._fsm_current_step_id = None
            self._fsm_expected_action = None
            self._fsm_expected_object = None
            self._fsm_state = "IDLE"
            self._fsm_valid_transitions = 0
            self._fsm_skipped_steps = 0
            self._fsm_out_of_order_events = 0
            self._fsm_invalid_object_events = 0
            self._fsm_unrecognized_events = 0
            self._fsm_low_conf_events = 0
            self._fsm_completed_steps = 0
            self._fsm_is_completed = False
            self._fsm_latest_status = None

            # Recovery
            self._recovery_total = 0
            self._rec_skipped_count = 0
            self._rec_out_of_order_count = 0
            self._rec_invalid_object_count = 0
            self._rec_unrecognized_count = 0
            self._recovery_active_state = "IDLE"
            self._latest_recovery_event = None
            self._latest_recovery_instruction = None
            self._latest_recovery_timeout_s = None

    # -------------------------------------------------------------------------
    # Telemetry Ingestion API (Thread-Safe)
    # -------------------------------------------------------------------------

    def record_frame_submission(self, timestamp_mono: Optional[float] = None) -> None:
        """Record upstream camera frame submission to buffer."""
        t = timestamp_mono if timestamp_mono is not None else time.monotonic()
        with self._lock:
            self._frames_submitted += 1
            self._submission_times.append(t)

    def record_buffer_status(
        self,
        submission_count: int,
        replacement_count: int,
        has_pending: bool = False,
    ) -> None:
        """Update buffer state metrics from LatestFrameBuffer."""
        with self._lock:
            self._frames_submitted = max(self._frames_submitted, submission_count)
            self._frames_replaced = replacement_count
            self._buffer_occupancy = 1 if has_pending else 0

    def record_worker_timings(
        self,
        process_end_mono: float,
        infer_dur_ms: float,
        frame_age_ms: Optional[float] = None,
        dispatch_ms: Optional[float] = None,
        e2e_ms: Optional[float] = None,
    ) -> None:
        """Record worker loop processing times and latencies."""
        with self._lock:
            self._frames_processed += 1
            self._processing_times.append(process_end_mono)
            self._inference_durations_ms.append(infer_dur_ms)
            self._latest_infer_ms = infer_dur_ms
            if frame_age_ms is not None:
                self._frame_ages_ms.append(frame_age_ms)
                self._latest_frame_age_ms = frame_age_ms
            if dispatch_ms is not None:
                self._dispatch_latencies_ms.append(dispatch_ms)
                self._latest_dispatch_ms = dispatch_ms
            if e2e_ms is not None:
                self._end_to_end_latencies_ms.append(e2e_ms)
                self._latest_e2e_ms = e2e_ms

    def record_stage_latencies(
        self,
        stage_a_rectification_ms: float = 0.0,
        stage_b_detection_ms: float = 0.0,
        stage_c_consistency_ms: float = 0.0,
        stage_d_temporal_ms: float = 0.0,
        stage_e_fsm_ms: float = 0.0,
        stage_f_adapter_ms: float = 0.0,
        total_pipeline_ms: Optional[float] = None,
    ) -> None:
        """Record stage-level timing breakdown in milliseconds."""
        tot = (
            total_pipeline_ms
            if total_pipeline_ms is not None
            else (
                stage_a_rectification_ms
                + stage_b_detection_ms
                + stage_c_consistency_ms
                + stage_d_temporal_ms
                + stage_e_fsm_ms
                + stage_f_adapter_ms
            )
        )
        with self._lock:
            self._stage_a_ms.append(stage_a_rectification_ms)
            self._stage_b_ms.append(stage_b_detection_ms)
            self._stage_c_ms.append(stage_c_consistency_ms)
            self._stage_d_ms.append(stage_d_temporal_ms)
            self._stage_e_ms.append(stage_e_fsm_ms)
            self._stage_f_ms.append(stage_f_adapter_ms)
            self._total_pipe_ms.append(tot)

            self._latest_stage_latencies = {
                "stage_a_rectification_ms": round(stage_a_rectification_ms, 3),
                "stage_b_detection_ms": round(stage_b_detection_ms, 3),
                "stage_c_consistency_ms": round(stage_c_consistency_ms, 3),
                "stage_d_temporal_ms": round(stage_d_temporal_ms, 3),
                "stage_e_fsm_ms": round(stage_e_fsm_ms, 3),
                "stage_f_adapter_ms": round(stage_f_adapter_ms, 3),
                "total_pipeline_ms": round(tot, 3),
            }

    def record_rate_config(
        self,
        strategy: str,
        min_ai_interval_sec: float = 0.0,
        target_ai_fps: Optional[float] = None,
    ) -> None:
        """Record worker frame-rate matching configuration."""
        with self._lock:
            self._rate_strategy = strategy
            self._min_ai_interval_sec = min_ai_interval_sec
            self._target_ai_fps = target_ai_fps

    def record_error(self) -> None:
        """Increment inference worker exception count."""
        with self._lock:
            self._error_count += 1

    def record_result_emitted(self) -> None:
        """Increment public result dispatch counter."""
        with self._lock:
            self._results_emitted += 1

    # -------------------------------------------------------------------------
    # Diagnostic Observations Ingestion API
    # -------------------------------------------------------------------------

    def record_consistency_result(self, consistency_result: Optional[Dict[str, Any]]) -> None:
        """Record multimodal consistency evaluation outcome."""
        if consistency_result is None:
            return
        category = consistency_result.get("category", "IDLE")
        score = consistency_result.get("consistency_score", 0.0)
        is_reliable = consistency_result.get("is_reliable", True)

        with self._lock:
            self._consistency_total += 1
            self._latest_consistency_category = category
            self._latest_consistency_score = score
            self._latest_consistency_is_reliable = is_reliable

            if category == "RELIABLE_ALIGNED":
                self._cat_reliable_aligned += 1
            elif category == "CROSS_MODAL_CONFLICT":
                self._cat_cross_modal_conflict += 1
            elif category == "UNCERTAIN_MISSING_OBJECT":
                self._cat_missing_object += 1
            elif category == "UNCERTAIN_LOW_OBJECT_CONFIDENCE":
                self._cat_low_object_confidence += 1
            elif category == "UNCERTAIN_STATIC_OBJECT":
                self._cat_static_object += 1
            elif category == "INSTABILITY_OBJECT_FLICKER":
                self._cat_object_flicker += 1
            elif category == "INSTABILITY_ACTION_FLICKER":
                self._cat_action_flicker += 1
            elif category == "UNCERTAIN_SPATIAL_CONTRADICTION":
                self._cat_spatial_contradiction += 1
            elif category == "IDLE":
                self._cat_idle += 1

    def record_uncertainty_result(self, uncertainty_result: Optional[Dict[str, Any]]) -> None:
        """Record uncertainty handler outcome."""
        if uncertainty_result is None:
            return
        state = uncertainty_result.get("state", "IDLE")
        conf = uncertainty_result.get("confidence", 0.0)
        margin = uncertainty_result.get("margin", 0.0)
        is_reliable = uncertainty_result.get("is_reliable", True)
        win_size = uncertainty_result.get("evidence_count", 0)

        with self._lock:
            self._uncertainty_total += 1
            self._latest_unc_state = state
            self._latest_unc_conf = conf
            self._latest_unc_margin = margin
            self._latest_unc_is_reliable = is_reliable
            self._unc_active_window_size = win_size

            if state == "CONFIDENT":
                self._unc_confident += 1
            elif state == "MARGINAL" or state == "UNCERTAIN_EVIDENCE_ACCUMULATING":
                self._unc_marginal += 1
            elif state == "UNCERTAIN_LOW_CONFIDENCE":
                self._unc_low_conf += 1
            elif state == "UNCERTAIN_CONFLICT":
                self._unc_conflict += 1
            elif state == "RESOLVED":
                self._unc_resolved += 1
            elif state == "CONFIRMED_ANOMALY":
                self._unc_confirmed_anomaly += 1
            elif state == "IDLE":
                self._unc_idle += 1

    def record_temporal_result(self, temporal_result: Optional[Dict[str, Any]]) -> None:
        """Record temporal confirmation filter outcome."""
        if temporal_result is None:
            return
        confirmed = temporal_result.get("confirmed", False)
        candidate = temporal_result.get("candidate")
        streak = temporal_result.get("streak", 0)
        progress = temporal_result.get("confirmation_progress", 0.0)
        in_cooldown = temporal_result.get("in_cooldown", False)

        with self._lock:
            self._temporal_total += 1
            if confirmed:
                self._temporal_confirmed += 1
            if in_cooldown and not self._temporal_in_cooldown:
                self._temporal_cooldown_events += 1
            if candidate != self._latest_temporal_candidate and candidate is not None:
                self._temporal_candidate_updates += 1

            self._latest_temporal_candidate = candidate
            self._latest_temporal_streak = streak
            self._latest_temporal_progress = progress
            self._temporal_in_cooldown = in_cooldown

    def record_fsm_result(self, fsm_result: Optional[Dict[str, Any]], validator: Optional[Any] = None) -> None:
        """Record authoritative SequenceValidatorFSM outcome and procedure status."""
        with self._lock:
            if validator is not None:
                self._fsm_current_step_index = getattr(validator, "current_step_index", 0)
                self._fsm_state = getattr(validator, "state", "IDLE")
                curr_step = validator.get_current_expected_step() if hasattr(validator, "get_current_expected_step") else None
                if curr_step is not None:
                    self._fsm_current_step_id = getattr(curr_step, "step_id", None)
                    self._fsm_expected_action = getattr(curr_step, "expected_action", None)
                    self._fsm_expected_object = getattr(curr_step, "required_object", getattr(curr_step, "expected_object", None))
                if hasattr(validator, "validated_steps"):
                    self._fsm_completed_steps = len(validator.validated_steps)
                self._fsm_is_completed = (self._fsm_state == "COMPLETED")

            if fsm_result is not None:
                status = fsm_result.get("status", "UNKNOWN")
                val_status = fsm_result.get("validation_status")
                err_type = fsm_result.get("error_type") or val_status
                self._fsm_latest_status = val_status or status

                if status == "VALID":
                    if val_status != "IDLE":
                        self._fsm_valid_transitions += 1
                elif status == "SKIPPED" or err_type in ("SKIPPED", "SKIPPED_STEP"):
                    self._fsm_skipped_steps += 1
                elif err_type == "INVALID_OBJECT":
                    self._fsm_invalid_object_events += 1
                elif err_type == "UNRECOGNIZED":
                    self._fsm_unrecognized_events += 1
                elif err_type == "LOW_CONFIDENCE":
                    self._fsm_low_conf_events += 1
                elif status in ("OUT_OF_SEQUENCE", "OUT_OF_ORDER") or err_type == "OUT_OF_ORDER":
                    self._fsm_out_of_order_events += 1

    def record_recovery_event(self, recovery_event: Optional[Union[Dict[str, Any], Any]]) -> None:
        """Record procedural recovery event emitted by RecoveryManager."""
        if recovery_event is None:
            return
        rec_dict = recovery_event.to_dict() if hasattr(recovery_event, "to_dict") else recovery_event
        if not isinstance(rec_dict, dict):
            return

        ev_type = rec_dict.get("event_type", "PROCEDURAL_RECOVERY")
        status = rec_dict.get("procedural_status", "")
        instr = rec_dict.get("recovery_instruction")
        timeout_s = rec_dict.get("timeout_s")

        with self._lock:
            self._recovery_total += 1
            self._latest_recovery_event = rec_dict
            self._latest_recovery_instruction = instr
            self._latest_recovery_timeout_s = timeout_s
            self._recovery_active_state = "RECOVERY_ACTIVE"

            if status == "SKIPPED" or "SKIPPED" in ev_type:
                self._rec_skipped_count += 1
            elif status in ("OUT_OF_ORDER", "OUT_OF_SEQUENCE") or "OUT_OF_ORDER" in ev_type:
                self._rec_out_of_order_count += 1
            elif "INVALID_OBJECT" in str(rec_dict.get("explanation", "")) or "OBJECT" in ev_type:
                self._rec_invalid_object_count += 1
            elif status == "UNRECOGNIZED":
                self._rec_unrecognized_count += 1

    def record_cycle(
        self,
        consistency_result: Optional[Dict[str, Any]] = None,
        uncertainty_result: Optional[Dict[str, Any]] = None,
        temporal_result: Optional[Dict[str, Any]] = None,
        fsm_result: Optional[Dict[str, Any]] = None,
        recovery_event: Optional[Union[Dict[str, Any], Any]] = None,
        validator: Optional[Any] = None,
        stage_latencies: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Convenience composite ingestion updating all diagnostic domains for one frame cycle.
        """
        with self._lock:
            self._total_cycles += 1

        if consistency_result is not None:
            self.record_consistency_result(consistency_result)
        if uncertainty_result is not None:
            self.record_uncertainty_result(uncertainty_result)
        if temporal_result is not None:
            self.record_temporal_result(temporal_result)
        if fsm_result is not None or validator is not None:
            self.record_fsm_result(fsm_result, validator=validator)
        if recovery_event is not None:
            self.record_recovery_event(recovery_event)
        if stage_latencies is not None:
            self.record_stage_latencies(
                stage_a_rectification_ms=stage_latencies.get("stage_a_rectification_ms", 0.0),
                stage_b_detection_ms=stage_latencies.get("stage_b_detection_ms", 0.0),
                stage_c_consistency_ms=stage_latencies.get("stage_c_consistency_ms", 0.0),
                stage_d_temporal_ms=stage_latencies.get("stage_d_temporal_ms", 0.0),
                stage_e_fsm_ms=stage_latencies.get("stage_e_fsm_ms", 0.0),
                stage_f_adapter_ms=stage_latencies.get("stage_f_adapter_ms", 0.0),
                total_pipeline_ms=stage_latencies.get("total_pipeline_ms"),
            )

    # -------------------------------------------------------------------------
    # Snapshot Generation API (Thread-Safe, Deterministic)
    # -------------------------------------------------------------------------

    def get_snapshot(self) -> TelemetrySnapshot:
        """
        Generates an immutable, thread-safe TelemetrySnapshot.
        """
        with self._lock:
            now_iso = datetime.now().isoformat()
            mono_now = time.monotonic()

            # Session duration calculation
            duration_s = 0.0
            if self._session_start_mono is not None:
                if self._session_state == "RUNNING":
                    elapsed = mono_now - self._session_start_mono - self._session_accumulated_pause_sec
                    duration_s = max(0.0, elapsed)
                elif self._session_state == "PAUSED" and self._pause_start_mono is not None:
                    elapsed = self._pause_start_mono - self._session_start_mono - self._session_accumulated_pause_sec
                    duration_s = max(0.0, elapsed)
                elif self._session_state in ("STOPPED", "COMPLETED"):
                    elapsed = mono_now - self._session_start_mono - self._session_accumulated_pause_sec
                    duration_s = max(0.0, elapsed)

            # Copy deques for statistics computation
            sub_times = list(self._submission_times)
            proc_times = list(self._processing_times)
            infer_durs = list(self._inference_durations_ms)
            frame_ages = list(self._frame_ages_ms)
            disp_lats = list(self._dispatch_latencies_ms)
            e2e_lats = list(self._end_to_end_latencies_ms)

            stg_a = list(self._stage_a_ms)
            stg_b = list(self._stage_b_ms)
            stg_c = list(self._stage_c_ms)
            stg_d = list(self._stage_d_ms)
            stg_e = list(self._stage_e_ms)
            stg_f = list(self._stage_f_ms)
            stg_tot = list(self._total_pipe_ms)

            # Snapshot Section 1: Lifecycle
            lifecycle_sec = LifecycleTelemetrySection(
                session_id=self._session_id,
                session_state=self._session_state,
                start_time=self._session_start_iso,
                end_time=self._session_end_iso,
                duration_seconds=round(duration_s, 2),
                total_cycles=self._total_cycles,
            )

            # Snapshot Section 2: Pipeline
            ingest_fps = self._compute_rolling_fps(sub_times, window_sec=self._fps_window_sec)
            ai_fps = self._compute_rolling_fps(proc_times, window_sec=self._fps_window_sec)
            sub_cnt = self._frames_submitted
            rep_cnt = self._frames_replaced
            drop_rate = (rep_cnt / sub_cnt) if sub_cnt > 0 else 0.0

            pipeline_sec = PipelineTelemetrySection(
                frames_submitted=sub_cnt,
                frames_replaced=rep_cnt,
                frames_processed=self._frames_processed,
                error_count=self._error_count,
                results_emitted=self._results_emitted,
                camera_ingest_fps=round(ingest_fps, 2),
                ai_processing_fps=round(ai_fps, 2),
                frame_drop_rate=round(drop_rate, 4),
                buffer_occupancy=self._buffer_occupancy,
                mean_inference_ms=round(self._compute_mean(infer_durs), 3),
                p50_inference_ms=round(self._compute_p50(infer_durs), 3),
                p95_inference_ms=round(self._compute_p95(infer_durs), 3),
                p99_inference_ms=round(self._compute_p99(infer_durs), 3),
                mean_frame_age_ms=round(self._compute_mean(frame_ages), 3),
                p95_frame_age_ms=round(self._compute_p95(frame_ages), 3),
                mean_dispatch_latency_ms=round(self._compute_mean(disp_lats), 3),
                p95_dispatch_latency_ms=round(self._compute_p95(disp_lats), 3),
                mean_end_to_end_ms=round(self._compute_mean(e2e_lats), 3),
                p50_end_to_end_ms=round(self._compute_p50(e2e_lats), 3),
                p95_end_to_end_ms=round(self._compute_p95(e2e_lats), 3),
                p99_end_to_end_ms=round(self._compute_p99(e2e_lats), 3),
                latest_inference_duration_ms=round(self._latest_infer_ms, 3) if self._latest_infer_ms is not None else None,
                latest_frame_age_ms=round(self._latest_frame_age_ms, 3) if self._latest_frame_age_ms is not None else None,
                latest_end_to_end_ms=round(self._latest_e2e_ms, 3) if self._latest_e2e_ms is not None else None,
                rate_strategy=self._rate_strategy,
                min_ai_interval_sec=round(self._min_ai_interval_sec, 4),
                target_ai_fps=round(self._target_ai_fps, 2) if self._target_ai_fps is not None else None,
            )

            # Snapshot Section 3: Stage Latencies
            mean_stages = {
                "stage_a_rectification_ms": round(self._compute_mean(stg_a), 3),
                "stage_b_detection_ms": round(self._compute_mean(stg_b), 3),
                "stage_c_consistency_ms": round(self._compute_mean(stg_c), 3),
                "stage_d_temporal_ms": round(self._compute_mean(stg_d), 3),
                "stage_e_fsm_ms": round(self._compute_mean(stg_e), 3),
                "stage_f_adapter_ms": round(self._compute_mean(stg_f), 3),
                "total_pipeline_ms": round(self._compute_mean(stg_tot), 3),
            }
            stage_sec = StageLatenciesSection(
                stage_a_rectification_ms=mean_stages["stage_a_rectification_ms"],
                stage_b_detection_ms=mean_stages["stage_b_detection_ms"],
                stage_c_consistency_ms=mean_stages["stage_c_consistency_ms"],
                stage_d_temporal_ms=mean_stages["stage_d_temporal_ms"],
                stage_e_fsm_ms=mean_stages["stage_e_fsm_ms"],
                stage_f_adapter_ms=mean_stages["stage_f_adapter_ms"],
                total_pipeline_ms=mean_stages["total_pipeline_ms"],
                mean=mean_stages,
                latest=dict(self._latest_stage_latencies),
            )

            # Snapshot Section 4: Buffer
            buffer_sec = BufferTelemetrySection(
                submission_count=sub_cnt,
                replacement_count=rep_cnt,
                has_pending=(self._buffer_occupancy > 0),
                drop_rate=round(drop_rate, 4),
            )

            # Snapshot Section 5: Consistency
            consistency_sec = ConsistencyDiagnosticSection(
                total_evaluations=self._consistency_total,
                reliable_aligned_count=self._cat_reliable_aligned,
                cross_modal_conflict_count=self._cat_cross_modal_conflict,
                missing_object_count=self._cat_missing_object,
                low_object_confidence_count=self._cat_low_object_confidence,
                static_object_count=self._cat_static_object,
                object_flicker_count=self._cat_object_flicker,
                action_flicker_count=self._cat_action_flicker,
                spatial_contradiction_count=self._cat_spatial_contradiction,
                idle_count=self._cat_idle,
                latest_category=self._latest_consistency_category,
                latest_score=round(self._latest_consistency_score, 4) if self._latest_consistency_score is not None else None,
                latest_is_reliable=self._latest_consistency_is_reliable,
            )

            # Snapshot Section 6: Uncertainty
            uncertainty_sec = UncertaintyDiagnosticSection(
                total_evaluations=self._uncertainty_total,
                confident_count=self._unc_confident,
                marginal_count=self._unc_marginal,
                low_confidence_count=self._unc_low_conf,
                conflict_count=self._unc_conflict,
                resolved_count=self._unc_resolved,
                confirmed_anomaly_count=self._unc_confirmed_anomaly,
                idle_count=self._unc_idle,
                latest_state=self._latest_unc_state,
                latest_confidence=round(self._latest_unc_conf, 4) if self._latest_unc_conf is not None else None,
                latest_margin=round(self._latest_unc_margin, 4) if self._latest_unc_margin is not None else None,
                latest_is_reliable=self._latest_unc_is_reliable,
                active_window_size=self._unc_active_window_size,
            )

            # Snapshot Section 7: Temporal
            temporal_sec = TemporalDiagnosticSection(
                total_evaluations=self._temporal_total,
                confirmed_count=self._temporal_confirmed,
                cooldown_events=self._temporal_cooldown_events,
                candidate_updates=self._temporal_candidate_updates,
                latest_candidate=self._latest_temporal_candidate,
                latest_streak=self._latest_temporal_streak,
                latest_progress_ratio=round(self._latest_temporal_progress, 4),
                is_in_cooldown=self._temporal_in_cooldown,
            )

            # Snapshot Section 8: Procedure
            procedure_sec = ProcedureDiagnosticSection(
                current_step_index=self._fsm_current_step_index,
                current_step_id=self._fsm_current_step_id,
                current_expected_action=self._fsm_expected_action,
                current_expected_object=self._fsm_expected_object,
                fsm_state=self._fsm_state,
                valid_transitions=self._fsm_valid_transitions,
                skipped_steps=self._fsm_skipped_steps,
                out_of_order_events=self._fsm_out_of_order_events,
                invalid_object_events=self._fsm_invalid_object_events,
                unrecognized_events=self._fsm_unrecognized_events,
                low_confidence_events=self._fsm_low_conf_events,
                completed_steps=self._fsm_completed_steps,
                is_completed=self._fsm_is_completed,
                latest_validation_status=self._fsm_latest_status,
            )

            # Snapshot Section 9: Recovery
            recovery_sec = RecoveryDiagnosticSection(
                total_recoveries=self._recovery_total,
                skipped_recoveries=self._rec_skipped_count,
                out_of_order_recoveries=self._rec_out_of_order_count,
                invalid_object_recoveries=self._rec_invalid_object_count,
                unrecognized_recoveries=self._rec_unrecognized_count,
                active_state=self._recovery_active_state,
                latest_event=self._latest_recovery_event,
                latest_instruction=self._latest_recovery_instruction,
                latest_timeout_s=self._latest_recovery_timeout_s,
            )

            return TelemetrySnapshot(
                timestamp=now_iso,
                lifecycle=lifecycle_sec,
                pipeline=pipeline_sec,
                stage_latencies=stage_sec,
                buffer=buffer_sec,
                consistency=consistency_sec,
                uncertainty=uncertainty_sec,
                temporal=temporal_sec,
                procedure=procedure_sec,
                recovery=recovery_sec,
            )

    def get_snapshot_dict(self) -> Dict[str, Any]:
        """Convenience method returning snapshot dictionary."""
        return self.get_snapshot().to_dict()

    def get_snapshot_json(self, indent: Optional[int] = None) -> str:
        """Convenience method returning JSON-serialized snapshot string."""
        return self.get_snapshot().to_json(indent=indent)
