"""
core/calculations.py

Transparent, testable percentage calculation. This is the only place
Grade A % / URS % / Defect % are produced -- Gemini never sees or guesses
these numbers, so every figure on the dashboard can be traced back to a
per-onion record.

Percentage rounding invariant:
    grade_a_pct + urs_pct + defect_pct == 100.0 for any non-empty batch.
    Enforced by computing defect_pct as the residual (100 − a − u) to
    eliminate floating-point drift from independent rounding.
"""
from typing import List

from .grading_rules import GradeLabel, classify_onion, compute_diameter_mm
from .schemas import OnionObservation


def grade_batch(validated: List[OnionObservation]) -> dict:
    total = len(validated)
    if total == 0:
        return {
            "total_validated": 0,
            "grade_a_count": 0,
            "urs_count": 0,
            "defect_count": 0,
            "grade_a_pct": 0.0,
            "urs_pct": 0.0,
            "defect_pct": 0.0,
            "per_onion_grades": [],
        }

    # compute_diameter_mm returns None if calibration is unavailable (honest fallback)
    labelled = [
        (o.onion_id, *classify_onion(o, diameter_mm=compute_diameter_mm(o)))
        for o in validated
    ]

    grade_a_count = sum(1 for _, label, _ in labelled if label == GradeLabel.grade_a)
    urs_count = sum(1 for _, label, _ in labelled if label == GradeLabel.urs)
    defect_count = sum(1 for _, label, _ in labelled if label == GradeLabel.defect)

    # Compute first two independently, derive third as residual so the three
    # always sum to exactly 100.0 regardless of batch size.
    grade_a_pct = round(grade_a_count / total * 100, 1)
    urs_pct = round(urs_count / total * 100, 1)
    defect_pct = round(100.0 - grade_a_pct - urs_pct, 1)

    return {
        "total_validated": total,
        "grade_a_count": grade_a_count,
        "urs_count": urs_count,
        "defect_count": defect_count,
        "grade_a_pct": grade_a_pct,
        "urs_pct": urs_pct,
        "defect_pct": defect_pct,
        "per_onion_grades": [
            {
                "onion_id": onion_id,
                "grade": label.value,
                "grade_reason": reason,
            }
            for onion_id, label, reason in labelled
        ],
    }
