# TTS (Text-to-Speech) Implementation & Optimization Plan

> **Objective:** Safely integrate Text-to-Speech offline voice alerts into the `BAS-Monitor` pipeline without causing AI frame drops, UI freezing, or audio queue spamming.

## 1. Identified Issues with Standard TTS Integration

When attempting to integrate TTS libraries like `pyttsx3` or system TTS into a real-time computer vision pipeline (running at 30 FPS), several critical issues emerge:

1. **Blocking Delays (The Main Issue):** Standard TTS commands (like `pyttsx3.runAndWait()`) block the main execution thread. If it takes 2 seconds to speak a warning, your AI video feed will completely freeze for 2 seconds, dropping crucial frames.
2. **Event Spamming & Queue Overload:** If a user commits an error, the AI might detect that error for 15 consecutive frames. A naive TTS implementation will queue the warning 15 times, resulting in a robotic backlog ("Error... Error... Error...") that plays long after the user has fixed the mistake.
3. **OS-Level Threading Crashes:** On macOS, `pyttsx3` relies on `NSSpeechSynthesizer`, which can crash or hang if called from non-main Python threads without a proper event loop.
4. **Initialization Latency:** Loading the TTS engine for the very first time causes a noticeable stutter/lag in the application.

---

## 2. The Architectural Solution

To resolve these issues, we must decouple the TTS engine from the AI inference pipeline using a **Dedicated Background Worker** with **Debouncing Logic**.

### A. Non-Blocking Audio Thread
Instead of speaking directly, the AI pipeline will place string messages into a `queue.Queue()`. A separate daemon thread will monitor this queue and execute the speech commands. This allows the AI to continue processing frames at 30 FPS instantly.

### B. Cooldown / Debouncing Mechanism
We will introduce a `cooldown_timer`. If the TTS engine speaks "Skipped step 2", it will ignore identical requests for the next `X` seconds (e.g., 5 seconds).

### C. macOS Native Subprocess Fallback
Because macOS handles background python GUI threads poorly, the safest and most robust non-blocking TTS on Mac is to use `subprocess.Popen(["say", text])`. This completely detaches the audio generation from the Python GIL (Global Interpreter Lock).

---

## 3. Step-by-Step Implementation Plan

### Step 1: Create `backend/voice/voice_alert.py`
We will author a `VoiceAlertService` class that utilizes the singleton pattern and a background queue.

```python
# Example Conceptual Logic
import subprocess
import threading
import queue
import time

class VoiceAlertService:
    def __init__(self):
        self.q = queue.Queue()
        self.last_spoken = {}
        self.cooldown = 5.0 # seconds
        
        # Start background worker
        threading.Thread(target=self._worker, daemon=True).start()

    def speak(self, text: str):
        # 1. Debounce logic
        if time.time() - self.last_spoken.get(text, 0) < self.cooldown:
            return 
            
        self.last_spoken[text] = time.time()
        
        # 2. Add to non-blocking queue
        self.q.put(text)

    def _worker(self):
        while True:
            text = self.q.get()
            # 3. OS-level non-blocking execution (macOS specific)
            subprocess.run(["say", text]) 
            self.q.task_done()
```

### Step 2: Wire TTS to the FSM (`sequence_validator.py`)
Modify the Finite State Machine so that when an illegal state transition occurs, it triggers the TTS singleton.

```python
# Inside sequence_validator.py
if not is_valid_step:
    self.voice_service.speak(f"Warning. Expected {expected_action}, but detected {detected_action}.")
```

### Step 3: Implement Queue Clearing (Interrupts)
If a critical safety alert triggers (e.g., "DANGER!"), we must add a `.clear()` method to the queue to immediately stop any pending, low-priority speech in favor of the critical warning.

### Step 4: Add Configuration Toggles
Update `config/settings.json` to include:
```json
"voice": {
    "enabled": true,
    "cooldown_seconds": 5.0,
    "voice_type": "Samantha" 
}
```

---

## 4. Next Actions
If approved, we will:
1. Write the `voice_alert.py` module containing the threaded debouncer.
2. Hook it into the FSM logic.
3. Add a decision log `[DECISION-019]` tracking this architectural solution.
