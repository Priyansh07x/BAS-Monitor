"""
session_metrics_exporter.py — Session Metrics Exporter & Diagnostic Summary
ISRO SIH26174 BAS Experiment Monitor — Workstream B (Gate B13.2)

Combines TelemetryDiagnosticAggregator snapshots with ExperimentLogger audit trails
to produce deterministic, serializable session-level metrics, mathematically safe
derived indicators, and human-readable execution reports.

Architectural Guarantees:
1. Pure Observer & Reporter: Zero mutation of FSM, temporal filter, uncertainty handler,
   or recovery state.
2. Source of Truth Integrity: Consumes authoritative data from TelemetryDiagnosticAggregator
   and ExperimentLogger without duplicating internal calculations.
3. Mathematical Safety: All derived ratios implement explicit zero-denominator protection.
4. Epistemic Separation: Procedural validity and execution indicators are strictly
   distinguished from neural model classification accuracy.
5. Contract Isolation: Completely out-of-band; zero impact on the frozen 8-field public AI contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.ai.telemetry_aggregator import (
    TelemetryDiagnosticAggregator,
    TelemetrySnapshot,
)
from backend.logging.experiment_logger import ExperimentLogger


@dataclass
class SessionMetrics:
    """
    Complete, deterministic session-level metrics representation.
    """
    session_id: str
    experiment_id: str
    experiment_name: str
    operator_id: str
    session_state: str
    start_time: Optional[str]
    end_time: Optional[str]
    duration_seconds: float
    total_cycles: int
    notes: str
    pipeline: Dict[str, Any]
    latencies: Dict[str, Any]
    stage_latencies: Dict[str, Any]
    buffer: Dict[str, Any]
    consistency: Dict[str, Any]
    uncertainty: Dict[str, Any]
    temporal: Dict[str, Any]
    procedure: Dict[str, Any]
    recovery: Dict[str, Any]
    derived_metrics: Dict[str, Any]
    steps_audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    recoveries: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session metrics to dictionary with JSON-serializable primitives."""
        return asdict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize session metrics to structured JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        """Generate human-readable ASCII summary report."""
        lines = [
            "=" * 78,
            "ISRO BHARATIYA ANTARIKSH STATION (BAS) — EXPERIMENT SESSION METRICS REPORT",
            "=" * 78,
            f"Session ID       : {self.session_id}",
            f"Experiment       : {self.experiment_id} — {self.experiment_name}",
            f"Operator ID      : {self.operator_id}",
            f"Session State    : {self.session_state}",
            f"Start Time       : {self.start_time or 'N/A'}",
            f"End Time         : {self.end_time or 'N/A'}",
            f"Active Duration  : {self.duration_seconds:.2f} s",
            f"Total Cycles     : {self.total_cycles}",
            "-" * 78,
            "1. PIPELINE THROUGHPUT & BUFFER UTILIZATION:",
            f"  Frames Submitted : {self.pipeline.get('frames_submitted', 0)}",
            f"  Frames Processed : {self.pipeline.get('frames_processed', 0)}",
            f"  Frames Replaced  : {self.pipeline.get('frames_replaced', 0)}",
            f"  Frame Drop Rate  : {self.pipeline.get('frame_drop_rate', 0.0) * 100:.2f}%",
            f"  Camera Ingest FPS: {self.pipeline.get('camera_ingest_fps', 0.0):.2f} Hz",
            f"  AI Processing FPS: {self.pipeline.get('ai_processing_fps', 0.0):.2f} Hz",
            f"  Rate Strategy    : {self.pipeline.get('rate_strategy', 'N/A')}",
            "-" * 78,
            "2. LATENCY PROFILES (ms):",
            f"  Mean Inference Latency   : {self.latencies.get('mean_inference_ms', 0.0):.2f} ms",
            f"  P95 Inference Latency    : {self.latencies.get('p95_inference_ms', 0.0):.2f} ms",
            f"  Mean Frame Age           : {self.latencies.get('mean_frame_age_ms', 0.0):.2f} ms",
            f"  P95 Frame Age            : {self.latencies.get('p95_frame_age_ms', 0.0):.2f} ms",
            f"  Mean Dispatch Latency    : {self.latencies.get('mean_dispatch_latency_ms', 0.0):.2f} ms",
            f"  Mean End-to-End Latency  : {self.latencies.get('mean_end_to_end_ms', 0.0):.2f} ms",
            f"  P95 End-to-End Latency   : {self.latencies.get('p95_end_to_end_ms', 0.0):.2f} ms",
            "-" * 78,
            "3. STAGE LATENCIES (ms):",
            f"  Stage A (Rectification)  : {self.stage_latencies.get('stage_a_rectification_ms', 0.0):.2f} ms",
            f"  Stage B (Detection)      : {self.stage_latencies.get('stage_b_detection_ms', 0.0):.2f} ms",
            f"  Stage C (Consistency)    : {self.stage_latencies.get('stage_c_consistency_ms', 0.0):.2f} ms",
            f"  Stage D (Temporal)       : {self.stage_latencies.get('stage_d_temporal_ms', 0.0):.2f} ms",
            f"  Stage E (FSM Validation) : {self.stage_latencies.get('stage_e_fsm_ms', 0.0):.2f} ms",
            f"  Stage F (Public Adapter) : {self.stage_latencies.get('stage_f_adapter_ms', 0.0):.2f} ms",
            f"  Total Pipeline Latency   : {self.stage_latencies.get('total_pipeline_ms', 0.0):.2f} ms",
            "-" * 78,
            "4. DIAGNOSTICS SUMMARY:",
            f"  Multimodal Consistency   : {self.consistency.get('total_evaluations', 0)} evaluations "
            f"(Aligned: {self.consistency.get('reliable_aligned_count', 0)}, "
            f"Conflicts: {self.consistency.get('cross_modal_conflict_count', 0)}, "
            f"Missing Object: {self.consistency.get('missing_object_count', 0)})",
            f"  Uncertainty Accumulation : {self.uncertainty.get('total_evaluations', 0)} evaluations "
            f"(Confident: {self.uncertainty.get('confident_count', 0)}, "
            f"Marginal: {self.uncertainty.get('marginal_count', 0)}, "
            f"Resolved: {self.uncertainty.get('resolved_count', 0)})",
            f"  Temporal Confirmation    : {self.temporal.get('total_evaluations', 0)} evaluations "
            f"(Confirmed: {self.temporal.get('confirmed_count', 0)}, "
            f"Cooldown Events: {self.temporal.get('cooldown_events', 0)})",
            f"  Procedural Transitions   : Valid: {self.procedure.get('valid_transitions', 0)}, "
            f"Skipped: {self.procedure.get('skipped_steps', 0)}, "
            f"Out-of-Order: {self.procedure.get('out_of_order_events', 0)}, "
            f"Completed Steps: {self.procedure.get('completed_steps', 0)}",
            f"  Recovery Handling        : Total: {self.recovery.get('total_recoveries', 0)}, "
            f"State: {self.recovery.get('active_state', 'IDLE')}",
            "-" * 78,
            "5. DERIVED PERFORMANCE & PROCEDURE INDICATORS:",
            f"  Processed Frame Ratio    : {self.derived_metrics.get('processed_frame_ratio', 0.0) * 100:.2f}%",
            f"  Effective Drop Rate      : {self.derived_metrics.get('effective_drop_rate', 0.0) * 100:.2f}%",
            f"  Procedural Validity Ratio: {self.derived_metrics.get('procedural_validity_ratio', 0.0) * 100:.2f}%",
            f"  Anomaly Ratio            : {self.derived_metrics.get('anomaly_ratio', 0.0) * 100:.2f}%",
            f"  Recovery Trigger Ratio   : {self.derived_metrics.get('recovery_ratio', 0.0) * 100:.2f}%",
            f"  Step Completion Ratio    : {self.derived_metrics.get('step_completion_ratio', 0.0) * 100:.2f}%",
            f"  Mean Action Confidence   : {self.derived_metrics.get('average_confidence', 0.0):.2f}",
            "=" * 78,
            "NOTE: Procedural validity and execution indicators measure protocol adherence.",
            "They do NOT represent neural model classification accuracy on absent EXP-001 checkpoints.",
            "=" * 78,
        ]
        return "\n".join(lines)


class SessionMetricsExporter:
    """
    Standardized exporter transforming active and completed session telemetry into
    structured JSON, dictionaries, and ASCII telemetry logs.
    """

    @staticmethod
    def calculate_derived_metrics(
        pipeline_data: Dict[str, Any],
        procedure_data: Dict[str, Any],
        recovery_data: Dict[str, Any],
        logger_summary: Optional[Dict[str, Any]] = None,
        total_canonical_steps: int = 5,
    ) -> Dict[str, float]:
        """
        Calculates mathematically safe derived session performance and protocol indicators.
        Guarantees zero-division safety.
        """
        frames_submitted = pipeline_data.get("frames_submitted", 0)
        frames_processed = pipeline_data.get("frames_processed", 0)
        frames_replaced = pipeline_data.get("frames_replaced", 0)

        # 1. Processed Frame Ratio
        proc_ratio = (frames_processed / frames_submitted) if frames_submitted > 0 else 0.0

        # 2. Effective Drop Rate
        drop_rate = (frames_replaced / frames_submitted) if frames_submitted > 0 else 0.0

        # 3. Procedural Validity Ratio (Valid transitions / Total procedural actions evaluated)
        valid_trans = procedure_data.get("valid_transitions", 0)
        skipped = procedure_data.get("skipped_steps", 0)
        ooo = procedure_data.get("out_of_order_events", 0)
        invalid_obj = procedure_data.get("invalid_object_events", 0)
        unrecognized = procedure_data.get("unrecognized_events", 0)
        total_evals = valid_trans + skipped + ooo + invalid_obj + unrecognized

        if total_evals > 0:
            val_ratio = valid_trans / total_evals
        elif valid_trans > 0:
            val_ratio = 1.0
        else:
            val_ratio = 0.0

        # 4. Anomaly Ratio (from logger summary or procedural counts)
        total_steps_logged = (logger_summary or {}).get("total_steps", valid_trans + skipped + ooo)
        total_anomalies = skipped + ooo + invalid_obj + unrecognized
        anomaly_ratio = (total_anomalies / total_steps_logged) if total_steps_logged > 0 else 0.0

        # 5. Recovery Ratio (Recoveries triggered / Total anomalies observed)
        total_recoveries = recovery_data.get("total_recoveries", (logger_summary or {}).get("recoveries_triggered", 0))
        if total_anomalies > 0:
            rec_ratio = total_recoveries / total_anomalies
        else:
            rec_ratio = 1.0 if total_recoveries == 0 else 0.0

        # 6. Step Completion Ratio
        completed_steps = procedure_data.get("completed_steps", (logger_summary or {}).get("valid_steps", 0))
        tot_steps = max(1, total_canonical_steps)
        completion_ratio = min(1.0, completed_steps / tot_steps)

        # 7. Average Confidence
        avg_conf = (logger_summary or {}).get("average_confidence", 0.0)

        return {
            "processed_frame_ratio": round(proc_ratio, 4),
            "effective_drop_rate": round(drop_rate, 4),
            "procedural_validity_ratio": round(val_ratio, 4),
            "anomaly_ratio": round(anomaly_ratio, 4),
            "recovery_ratio": round(rec_ratio, 4),
            "step_completion_ratio": round(completion_ratio, 4),
            "average_confidence": round(avg_conf, 2),
        }

    @classmethod
    def export_session(
        cls,
        telemetry_aggregator: Optional[TelemetryDiagnosticAggregator] = None,
        experiment_logger: Optional[ExperimentLogger] = None,
        session_data: Optional[Dict[str, Any]] = None,
        snapshot: Optional[Union[TelemetrySnapshot, Dict[str, Any]]] = None,
        experiment_id: str = "EXP-001",
        experiment_name: str = "Microgravity Sample Transfer and Containment Procedure",
        operator_id: str = "Astronaut-01",
        notes: str = "",
    ) -> SessionMetrics:
        """
        Assembles a comprehensive SessionMetrics instance from available telemetry and logger sources.
        """
        # Resolve Snapshot Data
        if snapshot is not None:
            snap_dict = snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(snapshot)
        elif telemetry_aggregator is not None:
            snap_dict = telemetry_aggregator.get_snapshot_dict()
        else:
            # Fallback empty snapshot
            snap_dict = TelemetryDiagnosticAggregator().get_snapshot_dict()

        # Resolve Session Logger Data
        active_sess = session_data
        if active_sess is None and experiment_logger is not None:
            active_sess = getattr(experiment_logger, "active_session", None)

        sess_id = (
            (active_sess or {}).get("session_id")
            or snap_dict.get("lifecycle", {}).get("session_id")
            or f"SESSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        exp_id = (active_sess or {}).get("experiment_id") or experiment_id
        exp_name = (active_sess or {}).get("experiment_name") or experiment_name
        op_id = (active_sess or {}).get("operator_id") or operator_id
        state = (active_sess or {}).get("status") or snap_dict.get("lifecycle", {}).get("session_state", "IDLE")
        start_t = (active_sess or {}).get("start_time") or snap_dict.get("lifecycle", {}).get("start_time")
        end_t = (active_sess or {}).get("end_time") or snap_dict.get("lifecycle", {}).get("end_time")
        dur_s = (active_sess or {}).get("duration_seconds") or snap_dict.get("lifecycle", {}).get("duration_seconds", 0.0)
        tot_cycles = snap_dict.get("lifecycle", {}).get("total_cycles", 0)
        sess_notes = (active_sess or {}).get("notes") or notes

        pipeline_sec = snap_dict.get("pipeline", {})
        stage_sec = snap_dict.get("stage_latencies", {})
        buffer_sec = snap_dict.get("buffer", {})
        consistency_sec = snap_dict.get("consistency", {})
        uncertainty_sec = snap_dict.get("uncertainty", {})
        temporal_sec = snap_dict.get("temporal", {})
        procedure_sec = snap_dict.get("procedure", {})
        recovery_sec = snap_dict.get("recovery", {})

        latencies_sec = {
            "mean_inference_ms": pipeline_sec.get("mean_inference_ms", 0.0),
            "p95_inference_ms": pipeline_sec.get("p95_inference_ms", 0.0),
            "mean_frame_age_ms": pipeline_sec.get("mean_frame_age_ms", 0.0),
            "p95_frame_age_ms": pipeline_sec.get("p95_frame_age_ms", 0.0),
            "mean_dispatch_latency_ms": pipeline_sec.get("mean_dispatch_latency_ms", 0.0),
            "p95_dispatch_latency_ms": pipeline_sec.get("p95_dispatch_latency_ms", 0.0),
            "mean_end_to_end_ms": pipeline_sec.get("mean_end_to_end_ms", 0.0),
            "p95_end_to_end_ms": pipeline_sec.get("p95_end_to_end_ms", 0.0),
            "latest_inference_duration_ms": pipeline_sec.get("latest_inference_duration_ms"),
            "latest_frame_age_ms": pipeline_sec.get("latest_frame_age_ms"),
            "latest_end_to_end_ms": pipeline_sec.get("latest_end_to_end_ms"),
        }

        # Calculate Derived Metrics
        logger_sum = (active_sess or {}).get("summary")
        derived = cls.calculate_derived_metrics(
            pipeline_data=pipeline_sec,
            procedure_data=procedure_sec,
            recovery_data=recovery_sec,
            logger_summary=logger_sum,
        )

        steps_trail = list((active_sess or {}).get("steps_conducted", []))
        anomalies_list = list((active_sess or {}).get("anomalies", []))
        recoveries_list = list((active_sess or {}).get("recoveries", []))

        return SessionMetrics(
            session_id=sess_id,
            experiment_id=exp_id,
            experiment_name=exp_name,
            operator_id=op_id,
            session_state=state,
            start_time=start_t,
            end_time=end_t,
            duration_seconds=float(dur_s),
            total_cycles=int(tot_cycles),
            notes=sess_notes,
            pipeline=pipeline_sec,
            latencies=latencies_sec,
            stage_latencies=stage_sec,
            buffer=buffer_sec,
            consistency=consistency_sec,
            uncertainty=uncertainty_sec,
            temporal=temporal_sec,
            procedure=procedure_sec,
            recovery=recovery_sec,
            derived_metrics=derived,
            steps_audit_trail=steps_trail,
            anomalies=anomalies_list,
            recoveries=recoveries_list,
        )

    @classmethod
    def export_to_file(
        cls,
        metrics: SessionMetrics,
        output_dir: Union[Path, str],
        base_name: Optional[str] = None,
        export_text: bool = True,
    ) -> Dict[str, Path]:
        """
        Exports session metrics to JSON and optional ASCII text report files.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        name = base_name or metrics.session_id
        json_path = out_dir / f"{name}_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(metrics.to_json(indent=2))

        results = {"json_path": json_path}

        if export_text:
            txt_path = out_dir / f"{name}_summary.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(metrics.to_text())
            results["txt_path"] = txt_path

        return results
