"""
Unit tests for VoiceAlertService using unittest.
"""

import time
import unittest
from backend.voice.voice_alert import VoiceAlertService


class TestVoiceSubsystem(unittest.TestCase):

    def test_voice_service_queue_and_mute(self):
        service = VoiceAlertService(enabled=False)  # Disabled to avoid sound in automated runs

        service.speak("Routine step alert 1")
        service.speak("Routine step alert 2")
        service.alert_next_step(3, "Analyzer Chamber Insertion")

        time.sleep(0.3)

        # Set speech parameters
        service.set_rate(180)
        service.set_volume(0.8)

        self.assertEqual(service.rate, 180)
        self.assertEqual(service.volume, 0.8)

        service.set_enabled(False)
        self.assertFalse(service.enabled)

        service.shutdown()

    def test_voice_service_helper_methods(self):
        service = VoiceAlertService(enabled=False)

        service.alert_next_step(1, "Container Retrieval")
        service.alert_out_of_order("Sample Pipetting", "PICK_TOOL")
        service.alert_skipped_step(2, "Sample Pipetting")
        service.alert_procedure_completed("EXP-01")

        service.shutdown()


if __name__ == "__main__":
    unittest.main()
