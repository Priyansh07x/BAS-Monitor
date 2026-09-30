"""
app_state.py -- Runtime application state & Authoritative FSM Owner.

This is NOT saved to disk. It resets every time the app restarts.
Permanent data (experiments) lives in experiments.json via storage.py.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path

from .video.camera_manager import get_camera_manager
from .video.camera import Camera
from .ai.inference_worker import InferenceWorker
from .experiment.sequence_validator import SequenceValidatorFSM, get_sequence_validator
from .experiment.procedure_manager import ProcedureManager
from .experiment.recovery_manager import RecoveryManager

class AppState:
    """
    Application-level state manager.
    Authoritatively owns the single SequenceValidatorFSM instance, RecoveryManager, and InferenceWorker.
    """

    def __init__(
        self,
        inference_worker: Optional[InferenceWorker] = None,
        sequence_validator: Optional[SequenceValidatorFSM] = None,
        procedure_manager: Optional[ProcedureManager] = None,
        recovery_manager: Optional[RecoveryManager] = None,
    ):
        self.current_experiment_id: Optional[str] = None
        self.selected_video_source: Optional[str] = None
        self.selected_video_path: Optional[str] = None

        # Phase A2: delegate camera management to CameraManager
        self.camera_manager = get_camera_manager()
        self.camera_manager.load_from_config()

        # 1. Authoritative SequenceValidatorFSM instance for active session
        if sequence_validator is not None:
            self.sequence_validator = sequence_validator
        elif procedure_manager is not None:
            self.sequence_validator = SequenceValidatorFSM(procedure_manager=procedure_manager)
        else:
            self.sequence_validator = get_sequence_validator()

        # 2. Authoritative RecoveryManager instance
        if recovery_manager is not None:
            self.recovery_manager = recovery_manager
        else:
            self.recovery_manager = RecoveryManager(procedure_manager=self.sequence_validator.pm)

        # 3. Background InferenceWorker wired to the exact same authoritative validator and recovery manager
        if inference_worker is not None:
            self.inference_worker = inference_worker
            if self.inference_worker.validator is None:
                self.inference_worker.validator = self.sequence_validator
                if hasattr(self.inference_worker.pipeline, "validator"):
                    self.inference_worker.pipeline.validator = self.sequence_validator
            if getattr(self.inference_worker.pipeline, "recovery_manager", None) is None:
                self.inference_worker.pipeline.recovery_manager = self.recovery_manager
        else:
            self.inference_worker = InferenceWorker(
                validator=self.sequence_validator,
                recovery_manager=self.recovery_manager,
            )

    @property
    def camera(self):
        return self.camera_manager.get_active()

    @property
    def current_step_index(self) -> int:
        """Authoritative step index directly from SequenceValidatorFSM."""
        return self.sequence_validator.current_step_index

    @current_step_index.setter
    def current_step_index(self, value: int) -> None:
        self.sequence_validator.current_step_index = int(value)

    @property
    def monitoring_status(self) -> str:
        """Authoritative monitoring status directly from SequenceValidatorFSM state."""
        return self.sequence_validator.state

    @monitoring_status.setter
    def monitoring_status(self, value: str) -> None:
        self.sequence_validator.state = str(value)

    def start_monitoring(self, operator_id: str = "Astronaut-01", notes: str = "") -> Dict[str, Any]:
        """Starts procedure monitoring on the authoritative FSM and resets recovery manager."""
        self.recovery_manager.reset()
        return self.sequence_validator.start(operator_id=operator_id, notes=notes)

    def pause_monitoring(self) -> None:
        """Pauses procedure monitoring on the authoritative FSM and pauses recovery manager."""
        self.sequence_validator.pause()
        self.recovery_manager.pause()

    def resume_monitoring(self) -> None:
        """Resumes procedure monitoring on the authoritative FSM and resumes recovery manager."""
        self.sequence_validator.resume()
        self.recovery_manager.resume()

    def stop_monitoring(self) -> None:
        """Pauses/stops procedure monitoring on the authoritative FSM and resets recovery."""
        self.sequence_validator.pause()
        self.recovery_manager.reset()

    def reset_monitoring(self) -> None:
        """Resets the authoritative FSM, recovery manager, and session state."""
        self.sequence_validator.reset()
        self.recovery_manager.reset()

    def advance_step(self) -> None:
        """Advances current step index on FSM."""
        self.sequence_validator.current_step_index += 1

    def set_experiment(self, experiment_id: str) -> None:
        self.current_experiment_id = experiment_id
        self.reset_monitoring()

    def set_video_source(self, source: Optional[str], path: Optional[str] = None) -> None:
        self.selected_video_source = source
        self.selected_video_path = path

    def start_camera(self):
        """Select default USB camera and connect."""
        sources = self.camera_manager.list_sources()
        if not sources:
            self.camera_manager.enumerate_usb_cameras(max_index=3)
            sources = self.camera_manager.list_sources()
        if sources:
            active = self.camera_manager.get_active_id()
            if active and self.camera_manager.get_active() and self.camera_manager.get_active().is_connected():
                return True
            return self.camera_manager.select(sources[0]["source_id"])
        return False

    def read_camera_frame(self):
        return self.camera_manager.read()

    def stop_camera(self):
        active = self.camera_manager.get_active()
        if active:
            active.disconnect()

    def to_dict(self):
        # Include camera status in state dict
        cam_status = None
        active = self.camera_manager.get_active()
        if active:
            cam_status = active.get_status().to_dict()

        exp_step = self.sequence_validator.get_current_expected_step()
        
        return {
            "current_experiment_id": self.current_experiment_id or getattr(self.sequence_validator.pm, "experiment_id", "EXP-001"),
            "current_step_index": self.sequence_validator.current_step_index,
            "current_step_id": exp_step.step_id if exp_step else None,
            "monitoring_status": self.sequence_validator.state,
            "selected_video_source": self.selected_video_source,
            "selected_video_path": self.selected_video_path,
            "camera_status": cam_status,
        }
