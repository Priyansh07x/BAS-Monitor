"""
Voice Alert and Guidance Service
ISRO SIH26174 BAS Experiment Monitor

Provides non-blocking, offline audible alerts and next-step spoken instructions
to astronauts during microgravity experiment execution.
"""

import sys
import queue
import threading
import subprocess
import time
from typing import Optional, Dict, Any


class VoiceAlertService:
    """
    Thread-safe, non-blocking offline text-to-speech service with priority queuing.
    Supports pyttsx3 offline engine, native macOS `say` fallback, and headless mock mode.
    """

    def __init__(
        self,
        enabled: bool = True,
        rate: int = 175,
        volume: float = 1.0,
        voice_id: Optional[str] = None
    ):
        self.enabled = enabled
        self.rate = rate
        self.volume = max(0.0, min(1.0, volume))
        self.voice_id = voice_id

        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._current_process: Optional[subprocess.Popen] = None
        self._engine = None
        self._engine_type = "mock"  # 'pyttsx3', 'macos_say', 'mock'

        self._init_tts_engine()

        # Start background worker thread
        self._worker_thread = threading.Thread(
            target=self._speech_worker, daemon=True, name="VoiceAlertWorker"
        )
        self._worker_thread.start()

    def _init_tts_engine(self) -> None:
        """Initialize the local speech engine with fallbacks."""
        # 1. Try pyttsx3 first
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            if self.voice_id:
                engine.setProperty("voice", self.voice_id)
            self._engine = engine
            self._engine_type = "pyttsx3"
            return
        except Exception:
            pass

        # 2. Fallback to native macOS `say` command if running on Darwin
        if sys.platform == "darwin":
            self._engine_type = "macos_say"
            return

        # 3. Headless/Mock fallback
        self._engine_type = "mock"

    def set_enabled(self, enabled: bool) -> None:
        """Toggle voice guidance on or off."""
        self.enabled = enabled
        if not enabled:
            self.stop_current_speech()

    def set_rate(self, rate: int) -> None:
        """Set speech rate in words per minute (typically 120-220)."""
        self.rate = rate
        if self._engine_type == "pyttsx3" and self._engine:
            try:
                self._engine.setProperty("rate", self.rate)
            except Exception:
                pass

    def set_volume(self, volume: float) -> None:
        """Set volume (0.0 to 1.0)."""
        self.volume = max(0.0, min(1.0, volume))
        if self._engine_type == "pyttsx3" and self._engine:
            try:
                self._engine.setProperty("volume", self.volume)
            except Exception:
                pass

    def speak(self, text: str, priority: bool = False) -> None:
        """
        Enqueue a speech alert.
        If priority=True, any non-urgent pending messages in the queue are flushed immediately.
        """
        if not self.enabled or not text or not text.strip():
            return

        clean_text = text.strip()

        if priority:
            # Clear pending queue for immediate high-priority alerts
            self.stop_current_speech()
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    break

        self._queue.put((clean_text, priority))

    def stop_current_speech(self) -> None:
        """Interrupt currently playing audio if supported."""
        if self._current_process and self._current_process.poll() is None:
            try:
                self._current_process.terminate()
            except Exception:
                pass
        if self._engine_type == "pyttsx3" and self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass

    def _speech_worker(self) -> None:
        """Background thread pulling messages from queue and speaking them."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            text, is_priority = item
            if not self.enabled:
                self._queue.task_done()
                continue

            try:
                self._dispatch_speech(text)
            except Exception:
                pass
            finally:
                self._queue.task_done()

    def _dispatch_speech(self, text: str) -> None:
        """Execute speech on the chosen TTS engine."""
        if self._engine_type == "pyttsx3" and self._engine:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
                return
            except Exception:
                # If pyttsx3 encounters a driver error, fallback to macos say
                if sys.platform == "darwin":
                    self._engine_type = "macos_say"
                else:
                    self._engine_type = "mock"

        if self._engine_type == "macos_say":
            try:
                # Use macOS built-in command line speech synthesizer
                cmd = ["say", "-r", str(self.rate), text]
                self._current_process = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                self._current_process.wait()
                return
            except Exception:
                self._engine_type = "mock"

        # Mock fallback (non-blocking sleep to simulate speech duration)
        duration = max(0.4, len(text.split()) * (60.0 / self.rate))
        time.sleep(duration)

    # --- Predefined Astronaut Alert Helpers ---

    def alert_next_step(self, step_id: int, step_title: str) -> None:
        """Audible next-step guidance."""
        self.speak(f"Proceed to Step {step_id}: {step_title}", priority=False)

    def alert_out_of_order(self, expected_title: str, detected_action: str) -> None:
        """Spoken caution for out-of-order action."""
        self.speak(
            f"Caution. Detected {detected_action}, but expecting {expected_title}.",
            priority=True
        )

    def alert_skipped_step(self, step_id: int, step_title: str) -> None:
        """Spoken warning for skipped step."""
        self.speak(
            f"Warning. Step {step_id}, {step_title}, was skipped. Please verify.",
            priority=True
        )

    def alert_procedure_completed(self, experiment_name: str = "Experiment") -> None:
        """Procedure completion confirmation."""
        self.speak(
            f"Procedure complete. All steps for {experiment_name} have been successfully verified.",
            priority=True
        )

    def shutdown(self) -> None:
        """Gracefully stop the worker thread."""
        self._stop_event.set()
        self.stop_current_speech()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)


# Module-level singleton
_voice_service_instance: Optional[VoiceAlertService] = None


def get_voice_service() -> VoiceAlertService:
    """Get or initialize the global VoiceAlertService singleton."""
    global _voice_service_instance
    if _voice_service_instance is None:
        _voice_service_instance = VoiceAlertService()
    return _voice_service_instance
