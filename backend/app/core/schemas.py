"""
Pydantic schemas for OnionGrade AI.

These models describe exactly what Gemini Vision is allowed to perceive
about an onion (size, colour, sprouting, defects) -- never a grade or a
percentage. Grades and percentages are produced later, deterministically,
by core/grading_rules.py and core/calculations.py.
"""
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class SizeClass(str, Enum):
    small = "Small"
    medium = "Medium"
    large = "Large"


class BoundingBox(BaseModel):
    """Normalized bounding box for the complete visible onion.

    x/y are the top-left corner; width/height are normalized to 0..1.
    """

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


class OnionObservation(BaseModel):
    """Raw, validated perception of a single onion."""

    onion_id: str = Field(..., min_length=1, description="e.g. onion_01")
    size: SizeClass
    color: str = Field(..., min_length=2, max_length=80)
    sprouting: bool
    defect_present: bool
    defect_description: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    bbox: Optional[BoundingBox] = None
    # AGMARK diameter calibration (Feature 4)
    # If the caller supplies pixels_per_mm (from a reference object in the image),
    # a physical diameter can be computed. Otherwise diameter_mm is None and the
    # grading rule falls back to the size-class method.
    pixels_per_mm: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Calibration: pixels per mm from a known reference object. "
            "Required to compute a physical diameter. "
            "If None, physical diameter is unavailable and grading falls back to size class."
        ),
    )

    @field_validator("color")
    @classmethod
    def color_not_placeholder(cls, v: str) -> str:
        if v.strip().lower() in {"", "n/a", "unknown", "-"}:
            raise ValueError("color observation missing or unusable")
        return v.strip()

    @field_validator("confidence")
    @classmethod
    def confidence_floor(cls, v: float) -> float:
        # Low-confidence perceptions (e.g. an occluded onion) are treated
        # as not reliably observed, so validation rejects them.
        if v < 0.5:
            raise ValueError("perception confidence too low to validate")
        return v


class BatchObservation(BaseModel):
    batch_id: str
    onions: List[OnionObservation]


class FlaggedRecord(BaseModel):
    raw: dict
    reason: str


class GradeResult(BaseModel):
    total_validated: int
    grade_a_count: int
    urs_count: int
    defect_count: int
    grade_a_pct: float
    urs_pct: float
    defect_pct: float
    per_onion_grades: List[dict]


class AnalyzeResponse(BaseModel):
    batch_id: str
    total_observed: int
    validated: List[OnionObservation]
    flagged: List[FlaggedRecord]
    grade_result: GradeResult
    report: str
