import os
import sys
import json
from pathlib import Path

backend_dir = Path("backend")
config_dir = Path("config")
data_dir = Path("data")

def check_file(path):
    return path.exists() and path.is_file()

def report(tc, desc, passed, notes=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {tc} | {desc} {notes}")

print("=== Input & Ingestion ===")
report("TC-IN-01", "Live camera module (camera.py)", check_file(backend_dir / "video" / "camera.py"))
report("TC-IN-02", "Uploaded video loader (video_loader.py)", check_file(backend_dir / "video" / "video_loader.py"))
report("TC-IN-03", "Frame processor (frame_processor.py)", check_file(backend_dir / "video" / "frame_processor.py"))

print("\n=== AI Perception ===")
report("TC-AI-01", "Object detector (object_detector.py)", check_file(backend_dir / "ai" / "object_detector.py"))
report("TC-AI-02", "Pose detector (pose_detector.py)", check_file(backend_dir / "ai" / "pose_detector.py"))
report("TC-AI-03", "Hand detector (hand_detector.py)", check_file(backend_dir / "ai" / "hand_detector.py"))
report("TC-AI-04", "Interaction logic (interaction_logic.py)", check_file(backend_dir / "experiment" / "interaction_logic.py"))
report("TC-AI-05", "Action recognizer (action_recognizer.py)", check_file(backend_dir / "ai" / "action_recognizer.py"))
report("TC-AI-06", "Inference pipeline (inference_pipeline.py)", check_file(backend_dir / "ai" / "inference_pipeline.py"))

print("\n=== Sequence Validation & FSM ===")
report("TC-SQ-01", "Procedure manager (procedure_manager.py)", check_file(backend_dir / "experiment" / "procedure_manager.py"))
report("TC-SQ-02..05", "Sequence validator FSM (sequence_validator.py)", check_file(backend_dir / "experiment" / "sequence_validator.py"))

print("\n=== Voice Alerts (TTS) ===")
report("TC-VO-01", "Offline voice engine (voice_alert.py)", check_file(backend_dir / "voice" / "voice_alert.py"))
settings_json = config_dir / "settings.json"
has_voice_settings = False
if settings_json.exists():
    try:
        with open(settings_json, 'r') as f:
            cfg = json.load(f)
            has_voice_settings = "voice" in cfg
    except: pass
report("TC-VO-04", "Voice settings in config/settings.json", has_voice_settings)

print("\n=== Video Recording & IP Streaming ===")
report("TC-RC-01", "Local recorder module (recorder.py)", check_file(backend_dir / "video" / "recorder.py"))
report("TC-ST-01", "IP Stream module (ip_stream.py or streamer.py)", check_file(backend_dir / "network" / "streamer.py") or check_file(backend_dir / "network" / "ip_stream.py"))
has_network_cfg = check_file(config_dir / "network.json")
report("TC-ST-01_cfg", "Streaming destination in config/network.json", has_network_cfg)

print("\n=== Structured Logging ===")
report("TC-LG-01", "Experiment logger (experiment_logger.py)", check_file(backend_dir / "logging" / "experiment_logger.py"))
report("TC-LG-03", "System logger (system_logger.py)", check_file(backend_dir / "logging" / "system_logger.py"))

print("\n=== GUI Dashboard & Others ===")
report("TC-UI", "Desktop GUI (main.py, frontend/)", check_file(Path("main.py")) and check_file(Path("frontend/index.html")))
report("TC-STORAGE", "Storage manager (storage_manager.py)", check_file(backend_dir / "storage" / "storage_manager.py") or check_file(backend_dir / "storage.py"))

