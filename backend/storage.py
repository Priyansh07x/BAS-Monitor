"""
storage.py -- Raw JSON persistence layer.

Only responsibility: read/write data/experiments.json.
No experiment "business logic" (validation, id generation, etc.) lives here --
that belongs in experiment_manager.py. Keeping this separation means we can
swap JSON for SQLite later by rewriting only this file.
"""

import json
import os

DATA_FILE = os.path.join("data", "experiments.json")

DEFAULT_DATA = {"experiments": []}


def _ensure_data_file():
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_DATA, f, indent=2)


def load_experiments():
    _ensure_data_file()
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        data = DEFAULT_DATA.copy()
    return data.get("experiments", [])


def save_experiments(experiments):
    _ensure_data_file()
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump({"experiments": experiments}, f, indent=2)