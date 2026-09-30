import threading
import time
import json
import psutil
from typing import Callable, Optional, Dict, Any

class DiagnosticsPoller:
    """
    Background worker that polls hardware metrics (CPU, RAM, Temp) at a fixed interval
    and emits a merged payload with application state (FPS, AI Model, Camera) via callback.
    """
    def __init__(self, interval_sec: float = 1.0):
        self.interval_sec = interval_sec
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[str], None]] = None
        
        # Application State References to merge into payload
        self.ai_mode = "CPU Edge Mode"
        self.camera_state = "STANDBY"
        self.fps = 0.0

    def set_callback(self, callback: Callable[[str], None]):
        self._callback = callback

    def update_app_state(self, ai_mode: Optional[str] = None, camera_state: Optional[str] = None, fps: Optional[float] = None):
        if ai_mode is not None:
            self.ai_mode = ai_mode
        if camera_state is not None:
            self.camera_state = camera_state
        if fps is not None:
            self.fps = round(fps, 1)

    def start(self):
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="DiagnosticsPoller")
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _poll_loop(self):
        # Initial psutil calls to seed CPU stats
        psutil.cpu_percent()
        while self._running.is_set():
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            
            # Fetch temp if available (often not exposed natively on macOS without third party, but we'll try)
            temp = 0.0
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps and "coretemp" in temps:
                    temp = temps["coretemp"][0].current
            
            payload = {
                "cpu_percent": cpu,
                "ram_percent": ram,
                "temp_celsius": temp,
                "ai_mode": self.ai_mode,
                "camera_state": self.camera_state,
                "fps": self.fps
            }
            
            if self._callback:
                try:
                    self._callback(json.dumps(payload))
                except Exception:
                    pass
                    
            time.sleep(self.interval_sec)
