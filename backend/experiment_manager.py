"""
experiment_manager.py -- Experiment business logic.

Responsibilities: validation, id generation, CRUD operations.
Talks to storage.py for actual persistence -- never touches the JSON
file directly.
"""

import uuid

from . import storage


def get_experiments():
    return storage.load_experiments()


def get_experiment(experiment_id):
    experiments = storage.load_experiments()
    for exp in experiments:
        if exp.get("id") == experiment_id:
            return exp
    return None


def create_experiment(name, description="", steps=None):
    if not name or not name.strip():
        raise ValueError("Experiment name is required")

    experiments = storage.load_experiments()

    new_experiment = {
        "id": f"exp_{uuid.uuid4().hex[:8]}",
        "name": name.strip(),
        "description": description.strip() if description else "",
        "steps": _normalize_steps(steps or []),
    }

    experiments.append(new_experiment)
    storage.save_experiments(experiments)
    return new_experiment


def update_experiment(experiment_id, name=None, description=None, steps=None):
    experiments = storage.load_experiments()
    for exp in experiments:
        if exp.get("id") == experiment_id:
            if name is not None:
                exp["name"] = name.strip()
            if description is not None:
                exp["description"] = description.strip()
            if steps is not None:
                exp["steps"] = _normalize_steps(steps)
            storage.save_experiments(experiments)
            return exp
    return None


def delete_experiment(experiment_id):
    experiments = storage.load_experiments()
    filtered = [e for e in experiments if e.get("id") != experiment_id]
    if len(filtered) == len(experiments):
        return False
    storage.save_experiments(filtered)
    return True


def _normalize_steps(steps):
    normalized = []
    for i, step in enumerate(steps, start=1):
        normalized.append({
            "id": i,
            "instruction": step.get("instruction", "").strip(),
            "expected_action": step.get("expected_action", "").strip(),
        })
    return normalized