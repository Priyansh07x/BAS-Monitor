"""
tests/test_architecture_decision.py
Gate B5.4 — Architecture Decision Record Tests
ISRO SIH26174 BAS Experiment Monitor — Workstream B

Validates:
1. docs/workstream-b-architecture-decision.md exists and is readable.
2. Required sections and governance headings are present.
3. Executive status states final neural model is DEFERRED / NOT SELECTED.
4. Part-2 engineering architectural direction is Candidate B (Decoupled Temporal Pipeline).
5. Epistemic separation (verified measurements vs theoretical estimates vs blocked metrics) is enforced.
6. Total Candidate B runtime includes upstream detector/pose costs (not just vector extraction).
7. Frozen public AI result contract from architecture.md is preserved.
8. Multi-criteria decision rule requires evidence beyond accuracy alone.
9. Prerequisites to unblock final model selection are explicitly registered.
"""

from pathlib import Path
import pytest

DECISION_DOC_PATH = Path("docs/workstream-b-architecture-decision.md")

REQUIRED_SECTIONS = [
    "## 1. Executive Status & Gate Outcome",
    "## 2. Side-by-Side Evidence Table",
    "## 3. Detailed Rationale for Candidate B Architectural Direction",
    "## 4. Final Model-Selection Gate & Acceptance Protocol",
    "## 5. Deferred Items Register",
    "## 6. Relationship to Gate B6 (Real Inference Pipeline)",
]


def test_decision_document_exists():
    """Verify that docs/workstream-b-architecture-decision.md exists and is non-empty."""
    assert DECISION_DOC_PATH.exists(), "docs/workstream-b-architecture-decision.md does not exist"
    assert DECISION_DOC_PATH.is_file(), "docs/workstream-b-architecture-decision.md is not a file"
    content = DECISION_DOC_PATH.read_text(encoding="utf-8")
    assert len(content) > 1000, "Decision document is unexpectedly short"


def test_required_sections_present():
    """Verify all 6 required governance sections exist in the document."""
    content = DECISION_DOC_PATH.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in content, f"Missing required section heading: '{section}'"


def test_final_model_selection_deferred():
    """Verify final neural model selection is explicitly marked as DEFERRED (no fake winner)."""
    content = DECISION_DOC_PATH.read_text(encoding="utf-8")
    assert "DEFERRED" in content
    assert "Final Neural" in content
    assert "DEFERRED" in content and "NO" in content
    assert "NOT" in content and "established" in content


def test_part2_architecture_direction_candidate_b():
    """Verify Candidate B is adopted as the Part-2 engineering direction."""
    content = DECISION_DOC_PATH.read_text(encoding="utf-8")
    assert "Candidate B" in content
    assert "Decoupled" in content or "Feature-Based" in content


def test_epistemic_separation_enforced():
    """Verify the document strictly separates verified measurements from blocked metrics."""
    content = DECISION_DOC_PATH.read_text(encoding="utf-8")
    assert "VERIFIED REALITY" in content or "VERIFIED MEASUREMENT" in content
    assert "THEORETICAL ESTIMATE" in content
    assert "BLOCKED" in content


def test_candidate_b_full_pipeline_costs_acknowledged():
    """Verify that upstream detector and pose preprocessing costs are explicitly acknowledged."""
    content = DECISION_DOC_PATH.read_text(encoding="utf-8")
    assert "Detector" in content or "detector" in content
    assert "Pose" in content or "pose" in content
    assert "29.9" in content or "30" in content  # Pose normalization timing reference


def test_frozen_public_contract_preserved():
    """Verify the 8 frozen public AI contract keys are preserved and referenced."""
    content = DECISION_DOC_PATH.read_text(encoding="utf-8")
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
        assert f'"{k}"' in content or f"`{k}`" in content, f"Public contract key '{k}' missing from decision doc"


def test_multi_criteria_selection_rule():
    """Verify final model selection requires multi-metric evidence table rather than arbitrary score."""
    content = DECISION_DOC_PATH.read_text(encoding="utf-8")
    assert "Not Arbitrary Weighted Score" in content or "evidence table" in content.lower()
    assert "Classification Quality" in content or "classification quality" in content.lower()
    assert "Inference Latency" in content or "latency" in content.lower()
    assert "Throughput" in content or "throughput" in content.lower()
    assert "Memory Footprint" in content or "memory" in content.lower()
    assert "Robustness" in content or "robustness" in content.lower()
    assert "Edge Feasibility" in content or "edge feasibility" in content.lower()
    assert "Complexity" in content or "complexity" in content.lower()
    assert "Auditability" in content or "auditability" in content.lower()


def test_engineering_direction_distinguished_from_empirical_proof():
    """Verify Candidate B adoption is explicitly stated as engineering architecture direction, not empirical proof."""
    content = DECISION_DOC_PATH.read_text(encoding="utf-8").lower()
    assert "engineering architecture" in content
    assert "empirical proof" in content
    assert "outperform" in content or "superiority" in content


def test_composite_score_is_non_binding():
    """Verify any composite score is explicitly stated as non-binding / illustrative."""
    content = DECISION_DOC_PATH.read_text(encoding="utf-8")
    assert "NON-BINDING" in content or "non-binding" in content.lower()
    assert "illustrative" in content.lower() or "conceptual" in content.lower()

