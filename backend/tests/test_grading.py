"""
backend/tests/test_grading.py

Pytest suite for the deterministic grading engine.
Run from the backend/ directory:
    pytest tests/ -v

These tests verify:
  - Every grading rule fires correctly
  - Boundary conditions (confidence=0.5 accepted, 0.49 rejected)
  - Empty-batch handling
  - Percentage rounding invariant (sum = 100.0)
  - Pydantic rejects invalid / low-quality records
  - grade_reason is always present and non-empty
"""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Make sure the app package is importable from tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.calculations import grade_batch
from app.core.grading_rules import GradeLabel, classify_onion
from app.core.schemas import OnionObservation


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make(
    size="Medium",
    color="Golden-brown",
    sprouting=False,
    defect_present=False,
    defect_description=None,
    confidence=0.9,
    onion_id="onion_01",
) -> OnionObservation:
    return OnionObservation(
        onion_id=onion_id,
        size=size,
        color=color,
        sprouting=sprouting,
        defect_present=defect_present,
        defect_description=defect_description,
        confidence=confidence,
    )


# ── Rule tests ────────────────────────────────────────────────────────────────

class TestClassifyOnion:

    def test_sprouting_is_defect(self):
        obs = _make(sprouting=True)
        grade, reason = classify_onion(obs)
        assert grade == GradeLabel.defect
        assert "sprouting" in reason.lower()

    def test_rot_keyword_is_defect(self):
        obs = _make(defect_present=True, defect_description="visible rot on surface")
        grade, reason = classify_onion(obs)
        assert grade == GradeLabel.defect
        assert "rot" in reason.lower() or "major" in reason.lower()

    def test_decay_keyword_is_defect(self):
        obs = _make(defect_present=True, defect_description="decay visible near stem")
        grade, reason = classify_onion(obs)
        assert grade == GradeLabel.defect

    def test_mould_keyword_is_defect(self):
        obs = _make(defect_present=True, defect_description="mould patches on skin")
        grade, reason = classify_onion(obs)
        assert grade == GradeLabel.defect

    def test_major_keyword_is_defect(self):
        obs = _make(defect_present=True, defect_description="major damage to outer layers")
        grade, reason = classify_onion(obs)
        assert grade == GradeLabel.defect

    def test_minor_blemish_is_urs(self):
        obs = _make(defect_present=True, defect_description="minor surface blemish")
        grade, reason = classify_onion(obs)
        assert grade == GradeLabel.urs

    def test_defect_present_no_description_is_urs(self):
        obs = _make(defect_present=True, defect_description=None)
        grade, reason = classify_onion(obs)
        assert grade == GradeLabel.urs

    def test_small_clean_is_urs(self):
        obs = _make(size="Small")
        grade, reason = classify_onion(obs)
        assert grade == GradeLabel.urs
        assert "small" in reason.lower()

    def test_medium_clean_is_grade_a(self):
        obs = _make(size="Medium")
        grade, reason = classify_onion(obs)
        assert grade == GradeLabel.grade_a

    def test_large_clean_is_grade_a(self):
        obs = _make(size="Large")
        grade, reason = classify_onion(obs)
        assert grade == GradeLabel.grade_a

    def test_sprouting_overrides_no_defect(self):
        """Sprouting alone is always Defect, even if defect_present is False."""
        obs = _make(sprouting=True, defect_present=False)
        grade, _ = classify_onion(obs)
        assert grade == GradeLabel.defect

    def test_reason_is_always_non_empty(self):
        """Every path through classify_onion must return a non-empty reason."""
        cases = [
            _make(sprouting=True),
            _make(defect_present=True, defect_description="rot"),
            _make(defect_present=True, defect_description="minor blemish"),
            _make(size="Small"),
            _make(size="Medium"),
            _make(size="Large"),
        ]
        for obs in cases:
            _, reason = classify_onion(obs)
            assert isinstance(reason, str) and len(reason) > 0, (
                f"Empty reason for {obs}"
            )


# ── Batch calculation tests ───────────────────────────────────────────────────

class TestGradeBatch:

    def test_empty_batch_returns_zeros(self):
        result = grade_batch([])
        assert result["total_validated"] == 0
        assert result["grade_a_count"] == 0
        assert result["urs_count"] == 0
        assert result["defect_count"] == 0
        assert result["grade_a_pct"] == 0.0
        assert result["urs_pct"] == 0.0
        assert result["defect_pct"] == 0.0
        assert result["per_onion_grades"] == []

    def test_single_grade_a(self):
        result = grade_batch([_make()])
        assert result["grade_a_count"] == 1
        assert result["urs_count"] == 0
        assert result["defect_count"] == 0
        assert result["grade_a_pct"] == 100.0

    def test_percentage_sum_is_100(self):
        """grade_a_pct + urs_pct + defect_pct must equal 100.0 for any batch."""
        batch = [
            _make(size="Large", onion_id="onion_01"),
            _make(size="Small", onion_id="onion_02"),
            _make(defect_present=True, defect_description="rot", onion_id="onion_03"),
        ]
        result = grade_batch(batch)
        total = round(result["grade_a_pct"] + result["urs_pct"] + result["defect_pct"], 1)
        assert total == 100.0, f"Percentages sum to {total}, expected 100.0"

    def test_percentage_sum_is_100_for_awkward_batch_size(self):
        """Specifically tests the size=3 case which triggers 33.3+33.3+33.3 drift."""
        batch = [
            _make(size="Medium", onion_id="onion_01"),
            _make(size="Small", onion_id="onion_02"),
            _make(sprouting=True, onion_id="onion_03"),
        ]
        result = grade_batch(batch)
        total = round(result["grade_a_pct"] + result["urs_pct"] + result["defect_pct"], 1)
        assert total == 100.0

    def test_mixed_batch_counts(self):
        batch = [
            _make(size="Large", onion_id="onion_01"),
            _make(size="Large", onion_id="onion_02"),
            _make(size="Small", onion_id="onion_03"),
            _make(sprouting=True, onion_id="onion_04"),
        ]
        result = grade_batch(batch)
        assert result["total_validated"] == 4
        assert result["grade_a_count"] == 2
        assert result["urs_count"] == 1
        assert result["defect_count"] == 1

    def test_per_onion_grades_has_grade_reason(self):
        batch = [_make()]
        result = grade_batch(batch)
        record = result["per_onion_grades"][0]
        assert "grade_reason" in record
        assert isinstance(record["grade_reason"], str)
        assert len(record["grade_reason"]) > 0

    def test_counts_sum_to_total(self):
        batch = [
            _make(size="Large", onion_id=f"onion_{i:02d}")
            for i in range(7)
        ] + [
            _make(size="Small", onion_id=f"onion_{i:02d}")
            for i in range(7, 10)
        ]
        result = grade_batch(batch)
        total = result["grade_a_count"] + result["urs_count"] + result["defect_count"]
        assert total == result["total_validated"]


# ── Pydantic validation tests ─────────────────────────────────────────────────

class TestPydanticValidation:

    def test_confidence_at_floor_accepted(self):
        """confidence=0.5 is exactly at the floor — must be accepted."""
        obs = _make(confidence=0.5)
        assert obs.confidence == 0.5

    def test_confidence_below_floor_rejected(self):
        """confidence=0.49 is below the floor — must be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            _make(confidence=0.49)
        assert "too low" in str(exc_info.value).lower() or "confidence" in str(exc_info.value).lower()

    def test_placeholder_color_rejected(self):
        with pytest.raises(ValidationError):
            _make(color="n/a")

    def test_unknown_color_rejected(self):
        with pytest.raises(ValidationError):
            _make(color="unknown")

    def test_valid_color_accepted(self):
        obs = _make(color="Golden-brown")
        assert obs.color == "Golden-brown"

    def test_valid_sizes_accepted(self):
        for size in ("Small", "Medium", "Large"):
            obs = _make(size=size)
            assert obs.size.value == size

    def test_invalid_size_rejected(self):
        with pytest.raises(ValidationError):
            _make(size="Jumbo")
