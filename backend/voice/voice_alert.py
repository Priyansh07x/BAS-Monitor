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
        voice_id: Optional[str] = None,
        cooldown_seconds: float = 5.0,
        announce_steps: bool = True,
        announce_warnings: bool = True,
    ):
        self.enabled = enabled
        self.rate = rate
        self.volume = max(0.0, min(1.0, volume))
        self.voice_id = voice_id
        self.cooldown_seconds = cooldown_seconds
        self.announce_steps = announce_steps
        self.announce_warnings = announce_warnings

        # Debounce tracking: maps message text -> last spoken timestamp
        self._last_spoken: Dict[str, float] = {}

        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._current_process: Optional[subprocess.Popen] = None
        self._engine = None
        self._engine_type = "mock"  # 'pyttsx3', 'macos_say', 'mock'

        self._init_tts_engine()

        # Debounce tracking for recovery alerts
        self._debounce_lock = threading.Lock()
        self._last_recovery_key: Optional[tuple] = None
        self._last_recovery_time: float = 0.0

        # Start background worker thread
        self._worker_thread = threading.Thread(
            target=self._speech_worker, daemon=True, name="VoiceAlertWorker"
        )
        self._worker_thread.start()

    def _init_tts_engine(self) -> None:
        """Initialize the local speech engine with fallbacks."""
        # 1. On macOS, prefer native `say` command (reliable in threads)
        if sys.platform == "darwin":
            self._engine_type = "macos_say"
            return

        # 2. Try pyttsx3 on other platforms
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
        Duplicate messages within cooldown_seconds are silently dropped (debounced).
        """
        if not self.enabled or not text or not text.strip():
            return

        clean_text = text.strip()

        # Debounce: skip if same message was spoken within cooldown window
        now = time.time()
        if clean_text in self._last_spoken:
            elapsed = now - self._last_spoken[clean_text]
            if elapsed < self.cooldown_seconds:
                return
        self._last_spoken[clean_text] = now

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
                cmd = ["say", "-r", str(self.rate)]
                if self.voice_id:
                    cmd.extend(["-v", self.voice_id])
                cmd.append(text)
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
        if not self.announce_steps:
            return
        self.speak(f"Proceed to Step {step_id}: {step_title}", priority=False)

    def alert_out_of_order(self, expected_title: str, detected_action: str) -> None:
        """Spoken caution for out-of-order action."""
        if not self.announce_warnings:
            return
        self.speak(
            f"Caution. Detected {detected_action}, but expecting {expected_title}.",
            priority=True
        )

    def alert_skipped_step(self, step_id: int, step_title: str) -> None:
        """Spoken warning for skipped step."""
        if not self.announce_warnings:
            return
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

    def alert_recovery_guidance(
        self,
        step_id: Any,
        recovery_text: str,
        debounce_seconds: float = 3.0,
        priority: bool = True,
    ) -> bool:
        """
        Audible corrective procedural guidance for astronaut recovery.
        Debounces repeated identical recovery alerts within debounce_seconds window.

        Returns True if speech was enqueued, False if debounced or disabled.
        """
        if not self.enabled or not recovery_text or not str(recovery_text).strip():
            return False

        clean_text = str(recovery_text).strip()
        norm_step = str(step_id).strip() if step_id is not None else ""
        key = (norm_step, clean_text)
        now = time.monotonic()

        with self._debounce_lock:
            if self._last_recovery_key == key and (now - self._last_recovery_time) < debounce_seconds:
                return False
            self._last_recovery_key = key
            self._last_recovery_time = now

        spoken_text = f"Recovery guidance for Step {norm_step}. {clean_text}" if norm_step else clean_text
        self.speak(spoken_text, priority=priority)
        return True

    def reset_debounce(self) -> None:
        """Reset the debounce cache for recovery alerts."""
        with self._debounce_lock:
            self._last_recovery_key = None
            self._last_recovery_time = 0.0

    def shutdown(self) -> None:
        """Gracefully stop the worker thread."""
        self._stop_event.set()
        self.stop_current_speech()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)


# Module-level singleton
_voice_service_instance: Optional[VoiceAlertService] = None


def get_voice_service() -> VoiceAlertService:
    """Get or initialize the global VoiceAlertService singleton, loading config from settings.json."""
    global _voice_service_instance
    if _voice_service_instance is None:
        import json
        from pathlib import Path

        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "settings.json"
        kwargs: Dict[str, Any] = {}

        try:
            with open(config_path, "r") as f:
                settings = json.load(f)
            voice_cfg = settings.get("voice", {})
            kwargs["enabled"] = voice_cfg.get("enabled", True)
            kwargs["rate"] = voice_cfg.get("rate", 175)
            kwargs["volume"] = voice_cfg.get("volume", 1.0)
            kwargs["voice_id"] = voice_cfg.get("voice_id", None)
            kwargs["cooldown_seconds"] = voice_cfg.get("cooldown_seconds", 5.0)
            kwargs["announce_steps"] = voice_cfg.get("announce_steps", True)
            kwargs["announce_warnings"] = voice_cfg.get("announce_warnings", True)
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # Fall back to defaults

        _voice_service_instance = VoiceAlertService(**kwargs)
    return _voice_service_instance

