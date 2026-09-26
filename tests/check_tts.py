"""
check_tts.py — Live TTS Verification Script
Proves that VoiceAlertService produces audible speech and that debouncing works.
"""

import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)

from backend.voice.voice_alert import VoiceAlertService


def test_tts():
    print("=" * 60)
    print("TTS LIVE VERIFICATION")
    print("=" * 60)

    # 1. Initialize with a short cooldown so we can test debouncing quickly
    service = VoiceAlertService(enabled=True, rate=175, volume=1.0, cooldown_seconds=3.0)
    print(f"Engine type: {service._engine_type}")
    print(f"Cooldown: {service.cooldown_seconds}s")
    print()

    # --- TEST 1: Basic speech ---
    print("[Test 1] Basic speech — you should hear 'Hello, this is BAS Monitor'")
    service.speak("Hello, this is BAS Monitor")
    time.sleep(3)

    # --- TEST 2: Debounce protection ---
    print("[Test 2] Debounce — sending same message 5 times rapidly...")
    print("         You should hear it only ONCE (the rest are debounced)")
    for i in range(5):
        service.speak("Warning. Step 2 was skipped.")
        print(f"         Sent attempt {i + 1}")
    time.sleep(4)

    # --- TEST 3: Priority interrupt ---
    print("[Test 3] Priority interrupt — queuing 2 low-priority, then 1 high-priority")
    print("         You should hear only the priority message")
    service.speak("Low priority message one")
    service.speak("Low priority message two")
    time.sleep(0.1)
    service.speak("DANGER. Critical safety alert!", priority=True)
    time.sleep(4)

    # --- TEST 4: Different messages pass through ---
    print("[Test 4] Different messages — you should hear 2 distinct alerts")
    service.speak("Step 3. Insert the sample container.")
    time.sleep(0.1)
    service.speak("Step 4. Close the analyzer lid.")
    time.sleep(5)

    # --- RESULTS ---
    print()
    print("=" * 60)
    print("✅ TTS verification complete.")
    print("   If you heard audio for Tests 1, 3, and 4 — TTS is working correctly.")
    print("   If Test 2 played only once — debouncing is working correctly.")
    print("=" * 60)

    service.shutdown()


if __name__ == "__main__":
    test_tts()
