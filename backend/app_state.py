"""
app_state.py -- Runtime application state.

This is NOT saved to disk. It resets every time the app restarts.
Permanent data (experiments) lives in experiments.json via storage.py.
"""
from .video.camera_manager import get_camera_manager


class AppState:
    def __init__(self):
        self.current_experiment_id = None
        self.current_step_index = 0
        self.monitoring_status = "IDLE"
        self.selected_video_source = None
        self.selected_video_path = None

        # Phase A2: delegate camera management to CameraManager
        self.camera_manager = get_camera_manager()
        self.camera_manager.load_from_config()

    # Legacy property — returns the active CameraSource so old code
    # referencing self.camera still works transparently
    @property
    def camera(self):
        return self.camera_manager.get_active()

    def reset_monitoring(self):
        self.current_step_index = 0
        self.monitoring_status = "IDLE"

    def start_monitoring(self):
        self.monitoring_status = "RUNNING"

    def pause_monitoring(self):
        self.monitoring_status = "PAUSED"

    def stop_monitoring(self):
        self.monitoring_status = "STOPPED"

    def advance_step(self):
        self.current_step_index += 1

    def set_experiment(self, experiment_id):
        self.current_experiment_id = experiment_id
        self.reset_monitoring()

    def set_video_source(self, source, path=None):
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

        return {
            "current_experiment_id": self.current_experiment_id,
            "current_step_index": self.current_step_index,
            "monitoring_status": self.monitoring_status,
            "selected_video_source": self.selected_video_source,
            "selected_video_path": self.selected_video_path,
            "camera_status": cam_status,
        }