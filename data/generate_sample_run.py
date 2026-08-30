"""
Generates data/sample_run.pkl -- a placeholder list of AI-result dicts that
mimics what backend/ai/inference_pipeline.py will eventually receive from
the real Part 1 model (YOLO + pose + SlowFast).

This lets GUI + backend (validator, logger, voice, GUI updates) be built
and tested end-to-end WITHOUT a trained model.

Run this once:  python data/generate_sample_run.py
"""

import pickle
from datetime import datetime, timedelta

EXPECTED_SEQUENCE = ["PICK_RED", "PLACE_RED", "PICK_BLUE", "PLACE_BLUE", "CLOSE_LID"]

SIMULATED_DETECTED_ACTIONS = [
    "PICK_RED",
    "PLACE_RED",
    "PICK_BLUE",
    "CLOSE_LID",
]

OBJECT_MAP = {
    "PICK_RED": "RED_SAMPLE",
    "PLACE_RED": "RED_SAMPLE",
    "PICK_BLUE": "BLUE_SAMPLE",
    "PLACE_BLUE": "BLUE_SAMPLE",
    "CLOSE_LID": "CONTAINER_LID",
}


def build_sample_run():
    results = []
    t0 = datetime(2026, 8, 29, 10, 0, 0)
    expected_idx = 0

    for i, detected_action in enumerate(SIMULATED_DETECTED_ACTIONS):
        expected_action = EXPECTED_SEQUENCE[expected_idx] if expected_idx < len(EXPECTED_SEQUENCE) else None
        status = "VALID" if detected_action == expected_action else "OUT_OF_SEQUENCE"

        result = {
            "timestamp": (t0 + timedelta(seconds=i * 5)).isoformat(),
            "action": detected_action,
            "object": OBJECT_MAP[detected_action],
            "confidence": 0.90 + (i * 0.01),
            "expected_step": f"S{expected_idx + 1}" if expected_idx < len(EXPECTED_SEQUENCE) else None,
            "detected_step": f"S{EXPECTED_SEQUENCE.index(detected_action) + 1}",
            "status": status,
            "next_step": f"S{expected_idx + 2}" if status == "VALID" and expected_idx + 1 < len(EXPECTED_SEQUENCE) else None,
        }
        results.append(result)

        if status == "VALID":
            expected_idx += 1

    return results


if __name__ == "__main__":
    data = build_sample_run()
    with open("data/sample_run.pkl", "wb") as f:
        pickle.dump(data, f)
    print(f"Wrote {len(data)} placeholder AI results to data/sample_run.pkl")
    for r in data:
        print(r)
