"""
Voice Alert and Guidance Subsystem
Offline Speech Feedback for BAS Experiment Monitor
"""

from .voice_alert import VoiceAlertService, get_voice_service

__all__ = ["VoiceAlertService", "get_voice_service"]
