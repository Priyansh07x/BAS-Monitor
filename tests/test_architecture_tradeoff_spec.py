"""
tests/test_architecture_tradeoff_spec.py
Gate B5.1 — Architecture Trade-off Specification Tests
ISRO SIH26174 BAS Experiment Monitor — Workstream B

Validates:
1. docs/workstream-b-architecture-tradeoff.md exists and is readable.
2. Required sections and headings are present.
3. Feature vector dimension (111) in documentation matches FrameProcessor implementation.
4. Temporal window size (30) in documentation matches ActionClassifier implementation.
5. Tensor contracts (B, C, T, H, W) and (B, T, D) are explicitly defined.
6. Public AI result contract keys from architecture.md are preserved and referenced.
7. Architectural neutrality is maintained (no winner declared).
8. Current blockers (dataset, checkpoints) are explicitly documented.
9. Decision rule section is present with multi-criteria requirements.
"""

from pathlib import Path
import numpy as np
import pytest

from backend.video.frame_processor import FrameProcessor
from backend.ai.action_classifier import ActionClassifier

SPEC_DOC_PATH = Path("docs/workstream-b-architecture-tradeoff.md")

REQUIRED_SECTIONS = [
    "## 1. Purpose & Scope",
    "## 2. Candidate Architecture A: End-to-End Spatio-Temporal Video Model (R(2+1)D-18)",
    "## 3. Candidate Architecture B: Compact Feature-Based Temporal Model (1D-TCN)",
    "## 4. Side-by-Side Comparison Table",
    "## 5. Epistemic Status: Fact vs. Estimate vs. Future Measurement",
    "## 6. Formal Benchmark Metrics for Gate B5",
    "## 7. Controlled Benchmark Protocol & Invariants",
    "## 8. Current Blockers for Real Model Evaluation",
    "## 9. Model Input Contracts & Frozen Public AI Schema",
    "## 10. Decision Rule for Gate B5",
    "## 11. Conclusion & Handoff to Gate B5.2",
]


def test_specification_document_exists():
    """Verify that docs/workstream-b-architecture-tradeoff.md exists and is non-empty."""
    assert SPEC_DOC_PATH.exists(), "docs/workstream-b-architecture-tradeoff.md does not exist"
    assert SPEC_DOC_PATH.is_file(), "docs/workstream-b-architecture-tradeoff.md is not a file"
    content = SPEC_DOC_PATH.read_text(encoding="utf-8")
    assert len(content) > 1000, "Specification document is unexpectedly short"


def test_required_sections_present():
    """Verify all 11 required sections exist in the document."""
    content = SPEC_DOC_PATH.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in content, f"Missing required section heading: '{section}'"


def test_feature_vector_dimension_matches_code():
    """Verify the 111-dim feature vector documented in spec matches FrameProcessor output."""
    content = SPEC_DOC_PATH.read_text(encoding="utf-8")
    assert "111" in content, "Specification must document 111-dimensional feature vector"
    assert "99" in content, "Specification must document 99 pose landmark features (33x3)"
    assert "12" in content, "Specification must document 12 object centroid features (4x3)"

    # Verify real FrameProcessor output vector dimension
    dummy_pose = [{"x": 0.1, "y": 0.2, "z": 0.3} for _ in range(33)]
    dummy_boxes = [
        {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2, "confidence": 0.9}
        for _ in range(4)
    ]
    vec = FrameProcessor.extract_keypoint_vector(dummy_pose, dummy_boxes)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (111,), f"Expected vector shape (111,), got {vec.shape}"

    # Also test empty/padded fallback
    empty_vec = FrameProcessor.extract_keypoint_vector(None, None)
    assert empty_vec.shape == (111,), f"Expected padded vector shape (111,), got {empty_vec.shape}"


def test_temporal_window_matches_code():
    """Verify the 30-frame temporal window documented in spec matches ActionClassifier."""
    content = SPEC_DOC_PATH.read_text(encoding="utf-8")
    assert "30" in content, "Specification must document 30-frame temporal window"

    classifier = ActionClassifier()
    assert classifier.window_size == 30, f"Expected default window_size 30, got {classifier.window_size}"


def test_tensor_contracts_defined():
    """Verify both (B, C, T, H, W) and (B, T, D) tensor shapes are explicitly documented."""
    content = SPEC_DOC_PATH.read_text(encoding="utf-8")
    assert "(B, C, T, H, W)" in content or "(B, 3, T, H, W)" in content, "R(2+1)D tensor shape missing"
    assert "(B, T, D)" in content or "(B, T, 111)" in content, "1D-TCN tensor shape missing"


def test_frozen_public_contract_preserved():
    """Verify all 8 keys of the frozen public AI contract are referenced in the spec."""
    content = SPEC_DOC_PATH.read_text(encoding="utf-8")
    required_keys = [
        "timestamp",
        "action",
        "object",
        "confidence",
        "expected_step",
        "detected_step",
        "status",
        "next_step",
    ]
    for k in required_keys:
        assert f'"{k}"' in content or f"`{k}`" in content, f"Public contract key '{k}' missing from spec"


def test_architectural_neutrality():
    """Verify specification does not declare a winner or select an architecture prematurely."""
    content = SPEC_DOC_PATH.read_text(encoding="utf-8")
    assert "NO WINNER DECLARED" in content or "does NOT declare a winner" in content
    assert "SPECIFIED" in content


def test_blockers_explicitly_documented():
    """Verify that current data and model checkpoint blockers are explicitly stated."""
    content = SPEC_DOC_PATH.read_text(encoding="utf-8")
    assert "EXP-001" in content
    assert "NOT COLLECTED" in content or "not collected" in content
    assert "NOT PRESENT" in content or "not present" in content or "no trained checkpoint" in content.lower()


def test_decision_rule_multi_criteria():
    """Verify the decision rule requires multi-criteria evidence, not accuracy alone."""
    content = SPEC_DOC_PATH.read_text(encoding="utf-8")
    assert "Decision Rule" in content or "decision rule" in content
    assert "accuracy alone" in content.lower() or "not accuracy alone" in content.lower()
