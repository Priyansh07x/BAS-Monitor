"""
app_state.py -- Runtime application state.

This is NOT saved to disk. It resets every time the app restarts.
Permanent data (experiments) lives in experiments.json via storage.py.
"""
from .video.camera import Camera

class AppState:
    def __init__(self):
        self.current_experiment_id = None
        self.current_step_index = 0
        self.monitoring_status = "IDLE"
        self.selected_video_source = None
        self.selected_video_path = None
        self.camera = Camera()

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
        return self.camera.open()

    def read_camera_frame(self):
        return self.camera.read()

    def stop_camera(self):
        self.camera.release()

    def to_dict(self):
        return {
            "current_experiment_id": self.current_experiment_id,
            "current_step_index": self.current_step_index,
            "monitoring_status": self.monitoring_status,
            "selected_video_source": self.selected_video_source,
            "selected_video_path": self.selected_video_path,
        }