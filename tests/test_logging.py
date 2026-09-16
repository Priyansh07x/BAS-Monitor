"""
Unit tests for SystemLogger and ExperimentLogger using unittest.
"""

import json
import unittest
from pathlib import Path
import tempfile
import shutil

from backend.logging.system_logger import SystemLogger
from backend.logging.experiment_logger import ExperimentLogger


class TestLoggingSubsystem(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_system_logger_event_notifications(self):
        logger = SystemLogger(log_dir=self.temp_dir)
        events = []

        def on_log(event):
            events.append(event)

        logger.add_listener(on_log)

        logger.info("Test system initialization")
        logger.ai("Detected container box with confidence 0.95")
        logger.warn("Sample container out of sequence")
        logger.error("Camera disconnection warning")

        self.assertEqual(len(events), 4)
        self.assertEqual(events[0]["level"], "SYS")
        self.assertIn("Test system initialization", events[0]["message"])
        self.assertEqual(events[1]["level"], "AI")
        self.assertEqual(events[2]["level"], "WARN")
        self.assertEqual(events[3]["level"], "ERR")

        # Test removing listener
        logger.remove_listener(on_log)
        logger.info("Ignored by listener")
        self.assertEqual(len(events), 4)

    def test_experiment_logger_lifecycle(self):
        exp_logger = ExperimentLogger(log_dir=self.temp_dir)

        # 1. Start Session
        session = exp_logger.start_session(
            experiment_id="EXP-01",
            experiment_name="Microgravity Fluid Transfer",
            operator_id="Astronaut-01"
        )
        self.assertEqual(session["status"], "IN_PROGRESS")
        self.assertEqual(session["experiment_id"], "EXP-01")

        # 2. Log Steps
        exp_logger.log_step(
            step_id=1,
            step_title="Container Retrieval",
            expected_action="PICK_CONTAINER",
            detected_action="PICK_CONTAINER",
            confidence=95.5,
            validation_status="VALID"
        )

        exp_logger.log_step(
            step_id=2,
            step_title="Sample Pipetting",
            expected_action="PIPETTE_TRANSFER",
            detected_action="INSERT_ANALYZER",
            confidence=91.0,
            validation_status="OUT_OF_ORDER"
        )

        exp_logger.log_anomaly(
            anomaly_type="OUT_OF_ORDER",
            message="Action PIPETTE_TRANSFER was skipped before INSERT_ANALYZER",
            step_id=2
        )

        # 3. End Session
        results = exp_logger.end_session(status="COMPLETED")

        json_path = results["json_path"]
        txt_path = results["txt_path"]
        session_data = results["session_data"]

        self.assertTrue(json_path.exists())
        self.assertTrue(txt_path.exists())
        self.assertEqual(session_data["status"], "COMPLETED")

        # Verify JSON Schema & Data
        with open(json_path, "r", encoding="utf-8") as f:
            loaded_json = json.load(f)

        self.assertEqual(loaded_json["experiment_id"], "EXP-01")
        self.assertEqual(len(loaded_json["steps_conducted"]), 2)
        self.assertEqual(len(loaded_json["anomalies"]), 1)
        self.assertEqual(loaded_json["summary"]["total_steps"], 2)
        self.assertEqual(loaded_json["summary"]["valid_steps"], 1)
        self.assertEqual(loaded_json["summary"]["out_of_order_steps"], 1)
        self.assertEqual(loaded_json["summary"]["average_confidence"], 93.25)

        # Verify formatted text log contents
        with open(txt_path, "r", encoding="utf-8") as f:
            txt_content = f.read()

        self.assertIn("ISRO BHARATIYA ANTARIKSH STATION (BAS) - EXPERIMENT TELEMETRY LOG", txt_content)
        self.assertIn("EXP-01 - Microgravity Fluid Transfer", txt_content)
        self.assertIn("OUT_OF_ORDER", txt_content)


if __name__ == "__main__":
    unittest.main()
