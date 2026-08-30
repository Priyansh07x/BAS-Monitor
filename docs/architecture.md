# BAS Experiment Monitor -- Architecture Contract (Phase 0)

This document is the single source of truth for the seam between
Part 1 (AI model / dataset) and Part 2 (GUI + backend). Anyone building
either side should read this before writing code.

## 1. Ownership split

| Area | Owns |
|---|---|
| Part 1 (dataset/model team) | training/, final content of models/*.pt |
| Part 2 (this work -- GUI + backend) | backend/, frontend/, config/, main.py |

## 2. The AI-result contract

backend/ai/inference_pipeline.py MUST always return results in this
exact shape, whether it's coming from a real model or the placeholder:

{
  "timestamp": "2026-08-29T10:00:05",
  "action": "PICK_RED",
  "object": "RED_SAMPLE",
  "confidence": 0.94,
  "expected_step": "S1",
  "detected_step": "S1",
  "status": "VALID",
  "next_step": "S2"
}

status is always one of: VALID, SKIPPED, OUT_OF_SEQUENCE.

Everything downstream (sequence validator, GUI, logger, voice alert)
only ever consumes this dict. Nothing downstream should ever import
YOLO/pose/SlowFast code directly -- only inference_pipeline.py does.

## 3. Placeholder mode (current phase)

Until Part 1 delivers a trained model:

- data/generate_sample_run.py produces data/sample_run.pkl -- a list
  of result dicts simulating one full experiment run, including one
  deliberate OUT_OF_SEQUENCE case.
- inference_pipeline.py runs in PLACEHOLDER mode: it loads the
  pickle and yields one result at a time (on a timer, mimicking a live
  feed) via get_next_result().

## 4. Swap-in day (later)

When Part 1 delivers models/object_detector.pt and models/action_model.pt:

- Only inference_pipeline.py changes internally.
- The public method get_next_result() keeps the same signature and
  return shape.
- No other file in backend/ or frontend/ should need to change.

## 5. experiment.json contract

config/experiment.json defines the ordered step list the validator
checks against. Step id values (S1, S2, ...) must match what
inference_pipeline.py puts in expected_step/detected_step.

## 6. Do not build yet (explicitly out of scope for this phase)

- Real YOLO/pose/hand/SlowFast inference
- 3D HMR
- Login/auth
- Cloud backend
