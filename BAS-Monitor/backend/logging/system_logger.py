"""
System Diagnostics and Application Logger
ISRO SIH26174 BAS Experiment Monitor
"""

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from datetime import datetime
from typing import Callable, List, Optional


class SystemLogger:
    """
    Manages system-level diagnostic logs with dual output:
    1. Rotating log files on local disk (data/logs/system_YYYYMMDD.log)
    2. Real-time in-memory event callbacks (for GUI terminal feed)
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        max_bytes: int = 5 * 1024 * 1024,  # 5 MB
        backup_count: int = 5,
        log_level: int = logging.INFO
    ):
        if log_dir is None:
            # Default to BAS-Monitor/data/logs
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.log_dir = base_dir / "data" / "logs"
        else:
            self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.listeners: List[Callable[[dict], None]] = []

        # Setup standard Python logger
        self.logger = logging.getLogger("BAS_System")
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        # Clear any existing handlers to prevent duplicate lines
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 1. Console Handler
        console_fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(console_fmt)
        ch.setLevel(log_level)
        self.logger.addHandler(ch)

        # 2. Daily Rotating File Handler
        date_str = datetime.now().strftime("%Y%m%d")
        file_path = self.log_dir / f"system_{date_str}.log"
        fh = RotatingFileHandler(
            file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(file_fmt)
        fh.setLevel(log_level)
        self.logger.addHandler(fh)

    def add_listener(self, callback: Callable[[dict], None]) -> None:
        """Register a callback to receive real-time log payloads."""
        if callback not in self.listeners:
            self.listeners.append(callback)

    def remove_listener(self, callback: Callable[[dict], None]) -> None:
        """Unregister a previously added callback."""
        if callback in self.listeners:
            self.listeners.remove(callback)

    def _notify_listeners(self, level_tag: str, message: str) -> None:
        now = datetime.now()
        payload = {
            "timestamp": now.strftime("%H:%M:%S"),
            "iso_time": now.isoformat(),
            "level": level_tag,
            "message": message
        }
        for cb in self.listeners:
            try:
                cb(payload)
            except Exception:
                # Callback failures must not interrupt the logging system
                pass

    def info(self, msg: str) -> None:
        """Standard informational system message."""
        self.logger.info(msg)
        self._notify_listeners("SYS", msg)

    def warn(self, msg: str) -> None:
        """Warning / caution message."""
        self.logger.warning(msg)
        self._notify_listeners("WARN", msg)

    def error(self, msg: str) -> None:
        """System / hardware error message."""
        self.logger.error(msg)
        self._notify_listeners("ERR", msg)

    def debug(self, msg: str) -> None:
        """Verbose debugging message."""
        self.logger.debug(msg)
        self._notify_listeners("DBG", msg)

    def ai(self, msg: str) -> None:
        """AI inference specific log event."""
        self.logger.info(f"[AI] {msg}")
        self._notify_listeners("AI", msg)

    def stream(self, msg: str) -> None:
        """Network and streaming specific log event."""
        self.logger.info(f"[STREAM] {msg}")
        self._notify_listeners("STREAM", msg)


# Module-level singleton
_system_logger_instance: Optional[SystemLogger] = None


def get_system_logger() -> SystemLogger:
    """Get or initialize the global SystemLogger singleton."""
    global _system_logger_instance
    if _system_logger_instance is None:
        _system_logger_instance = SystemLogger()
    return _system_logger_instance
