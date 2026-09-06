"""
Structured Experiment Run Logger
ISRO SIH26174 BAS Experiment Monitor

Generates lightweight, timestamped, structured records of conducted experiment steps,
validation outcomes, and operational anomalies according to ISRO SIH requirements.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any


class ExperimentLogger:
    """
    Records and exports structured experiment session data.
    Outputs:
      1. Machine-readable JSON summary (`EXP_YYYYMMDD_HHMMSS_<id>.json`)
      2. Lightweight formatted text log (`EXP_YYYYMMDD_HHMMSS_<id>.log`)
    """

    def __init__(self, log_dir: Optional[Path] = None):
        if log_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.log_dir = base_dir / "data" / "logs"
        else:
            self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.active_session: Optional[Dict[str, Any]] = None

    def start_session(
        self,
        experiment_id: str,
        experiment_name: str,
        operator_id: str = "Astronaut-01",
        notes: str = ""
    ) -> Dict[str, Any]:
        """Initialize a new structured experiment session."""
        now = datetime.now()
        timestamp_slug = now.strftime("%Y%m%d_%H%M%S")
        session_id = f"RUN_{timestamp_slug}_{experiment_id}"

        self.active_session = {
            "session_id": session_id,
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "operator_id": operator_id,
            "notes": notes,
            "start_time": now.isoformat(),
            "end_time": None,
            "duration_seconds": 0.0,
            "status": "IN_PROGRESS",
            "steps_conducted": [],
            "anomalies": [],
            "summary": {
                "total_steps": 0,
                "valid_steps": 0,
                "out_of_order_steps": 0,
                "skipped_steps": 0,
                "average_confidence": 0.0
            }
        }
        return self.active_session

    def log_step(
        self,
        step_id: int,
        step_title: str,
        expected_action: str,
        detected_action: str,
        confidence: float,
        validation_status: str,  # 'VALID', 'OUT_OF_ORDER', 'SKIPPED', 'UNRECOGNIZED'
        details: Optional[str] = None
    ) -> Dict[str, Any]:
        """Record an individual step event during the experiment."""
        if not self.active_session:
            raise RuntimeError("Cannot log step: No active experiment session.")

        now = datetime.now()
        event = {
            "step_id": step_id,
            "step_title": step_title,
            "timestamp": now.isoformat(),
            "expected_action": expected_action,
            "detected_action": detected_action,
            "confidence": round(float(confidence), 2),
            "validation_status": validation_status.upper(),
            "details": details or ""
        }

        self.active_session["steps_conducted"].append(event)
        self._recalculate_summary()
        return event

    def log_anomaly(
        self,
        anomaly_type: str,  # 'SKIPPED_STEP', 'OUT_OF_ORDER', 'TIMEOUT', 'LOW_CONFIDENCE'
        message: str,
        step_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Record an anomaly or procedure deviation."""
        if not self.active_session:
            raise RuntimeError("Cannot log anomaly: No active experiment session.")

        now = datetime.now()
        anomaly = {
            "timestamp": now.isoformat(),
            "type": anomaly_type.upper(),
            "step_id": step_id,
            "message": message
        }
        self.active_session["anomalies"].append(anomaly)
        return anomaly

    def _recalculate_summary(self) -> None:
        """Update metrics in the active session."""
        steps = self.active_session["steps_conducted"]
        total = len(steps)
        if total == 0:
            return

        valid_count = sum(1 for s in steps if s["validation_status"] == "VALID")
        ooo_count = sum(1 for s in steps if s["validation_status"] == "OUT_OF_ORDER")
        skipped_count = sum(1 for s in steps if s["validation_status"] == "SKIPPED")
        avg_conf = sum(s["confidence"] for s in steps) / total

        self.active_session["summary"] = {
            "total_steps": total,
            "valid_steps": valid_count,
            "out_of_order_steps": ooo_count,
            "skipped_steps": skipped_count,
            "average_confidence": round(avg_conf, 2)
        }

    def end_session(self, status: str = "COMPLETED") -> Dict[str, Path]:
        """
        Finalize the experiment session and save both JSON and formatted TXT logs.
        Returns paths to the generated files.
        """
        if not self.active_session:
            raise RuntimeError("Cannot end session: No active experiment session.")

        now = datetime.now()
        start = datetime.fromisoformat(self.active_session["start_time"])
        duration = (now - start).total_seconds()

        self.active_session["end_time"] = now.isoformat()
        self.active_session["duration_seconds"] = round(duration, 2)
        self.active_session["status"] = status.upper()
        self._recalculate_summary()

        session_id = self.active_session["session_id"]
        json_path = self.log_dir / f"{session_id}.json"
        txt_path = self.log_dir / f"{session_id}.log"

        # 1. Write structured JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.active_session, f, indent=2)

        # 2. Write formatted lightweight text log
        self._write_formatted_text_log(txt_path)

        completed_session = self.active_session
        self.active_session = None

        return {
            "json_path": json_path,
            "txt_path": txt_path,
            "session_data": completed_session
        }

    def _write_formatted_text_log(self, file_path: Path) -> None:
        """Create a clean human-readable telemetry log."""
        s = self.active_session
        summary = s["summary"]

        lines = [
            "=" * 72,
            f"ISRO BHARATIYA ANTARIKSH STATION (BAS) - EXPERIMENT TELEMETRY LOG",
            "=" * 72,
            f"Session ID      : {s['session_id']}",
            f"Experiment ID   : {s['experiment_id']} - {s['experiment_name']}",
            f"Operator        : {s['operator_id']}",
            f"Start Time      : {s['start_time']}",
            f"End Time        : {s['end_time']}",
            f"Total Duration  : {s['duration_seconds']}s",
            f"Final Status    : {s['status']}",
            "-" * 72,
            f"SUMMARY METRICS:",
            f"  Total Steps Logged : {summary['total_steps']}",
            f"  Valid Actions      : {summary['valid_steps']}",
            f"  Out of Order       : {summary['out_of_order_steps']}",
            f"  Skipped Steps      : {summary['skipped_steps']}",
            f"  Mean AI Confidence : {summary['average_confidence']}%",
            "-" * 72,
            "CONDUCTED PROCEDURE AUDIT TRAIL:",
            f"{'TIME':<20} | {'STEP':<6} | {'ACTION (EXP / DET)':<30} | {'CONF':<6} | {'STATUS'}",
            "-" * 72
        ]

        for step in s["steps_conducted"]:
            t_str = step["timestamp"].split("T")[-1][:8]
            step_id_str = f"S{step['step_id']}"
            action_str = f"{step['expected_action'][:13]}/{step['detected_action'][:13]}"
            conf_str = f"{step['confidence']}%"
            status_str = step["validation_status"]
            lines.append(f"{t_str:<20} | {step_id_str:<6} | {action_str:<30} | {conf_str:<6} | {status_str}")

        if s["anomalies"]:
            lines.extend([
                "-" * 72,
                "PROCEDURE ANOMALIES & ALERTS:"
            ])
            for a in s["anomalies"]:
                t_str = a["timestamp"].split("T")[-1][:8]
                step_ref = f"[Step {a['step_id']}] " if a.get("step_id") else ""
                lines.append(f"  [{t_str}] [{a['type']}] {step_ref}{a['message']}")

        lines.extend([
            "=" * 72,
            f"END OF LOG FILE - {s['session_id']}",
            "=" * 72,
            ""
        ])

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
