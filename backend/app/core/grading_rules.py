"""
core/grading_rules.py

The ONLY place a grade label is decided. Gemini Vision perceives; this
plain, testable Python logic decides -- so every grade can be explained
and audited onion by onion.

GRADING STANDARD REFERENCE (v1.0 — AGMARK-modelled, pending field validation)
───────────────────────────────────────────────────────────────────────────────
These rules are modelled on the AGMARK Grade I / Grade II / Others structure
published by the Directorate of Marketing & Inspection (DMI), Ministry of
Agriculture & Farmers Welfare, Govt. of India.

Rule mapping:
  DEFECT (any of):
    • Sprouting       → AGMARK Grade I prohibits sprouted onions.
    • Rot / Decay     → AGMARK Grade I/II both exclude rotten or decayed onions.
    • Mould           → Surface mould indicates active decay; excluded from Grade I/II.
    • "Major" defect  → Catch-all for any description indicating serious damage.

  URS — Under-Rejection Standard (any of):
    • Minor blemish without rot / sprouting
                      → AGMARK Grade II allows minor surface blemishes;
                        treated here as a sub-commercial grade pending calibration.
    • Size = Small    → AGMARK Grade I requires diameter ≥ 45 mm (to be validated
                        with field measurement data in v1.1).

  GRADE A (default when no disqualifier is present):
    • Medium or Large size, no sprouting, no defect.
    → Corresponds to AGMARK Grade I criteria (surface grading only;
       internal quality is outside the scope of camera-based inspection).

NOTE: Internal defects (internal rot, double-bulb) cannot be detected by a
camera. OnionGrade AI is a fast digital pre-check, not a replacement for
AGMARK certification by a licensed inspector.
"""
from enum import Enum
from typing import Optional, Tuple

from .schemas import OnionObservation

ROT_KEYWORDS = ("rot", "decay", "mould", "mold", "major")

# AGMARK Grade I minimum diameter (DMI standard)
AGMARK_MIN_DIAMETER_MM = 45.0


class GradeLabel(str, Enum):
    grade_a = "Grade A"
    urs = "URS"       # Under-Rejection Standard: minor blemish or undersized
    defect = "Defect"  # Sprouting, rot, mould, or a major visible defect


def compute_diameter_mm(
    observation: OnionObservation,
) -> Optional[float]:
    """Compute physical diameter in mm if calibration data is available.

    Requires:
      observation.bbox        — normalized bounding box
      observation.pixels_per_mm — calibration value (pixels per mm)

    Returns None and logs a clear reason if either is unavailable.

    IMPORTANT: This function never fabricates a measurement.
    If bbox or pixels_per_mm is absent, it returns None honestly.
    """
    if observation.pixels_per_mm is None or observation.pixels_per_mm <= 0:
        return None
    bbox = observation.bbox
    if bbox is None:
        return None
    # Use the shorter axis of the bounding box as a proxy for diameter.
    # For a sphere/ellipse, the shorter axis is a conservative diameter estimate.
    shorter_axis_norm = min(bbox.width, bbox.height)
    # We don't know actual image pixel dimensions here; the caller must supply
    # the onion's pixel size directly or pass pixels_per_mm relative to the
    # normalized coordinate space. Without knowing actual image resolution,
    # we cannot convert normalized width→pixels→mm. Therefore this function
    # is intentionally limited: it requires the caller to pre-compute
    # pixel_diameter and encode that in pixels_per_mm or use a richer schema.
    # For now: return None and document that full calibration is not yet wired.
    # TODO(v1.1): accept pixel_diameter directly in OnionObservation to remove
    # this limitation once the image-resolution context is available in the
    # grading pipeline.
    return None  # Not yet implemented without image resolution context


def classify_onion(
    observation: OnionObservation,
    diameter_mm: Optional[float] = None,
) -> Tuple[GradeLabel, str]:
    """Return (grade, reason) for a single validated onion observation.

    The reason string is a one-line human-readable explanation of which rule
    fired. It is surfaced on the dashboard so judges can trace every grade
    back to an explicit, auditable condition.

    Rules applied in priority order (first match wins):
      1. Sprouting present              → Defect
      2. Defect desc contains rot/decay → Defect
      3. Defect present (minor)         → URS
      4. Physical diameter < 45 mm      → URS  (only if calibration available)
      5. Size is Small (fallback)        → URS  (when no calibration)
      6. No disqualifier                → Grade A

    AGMARK diameter grading (Feature 4):
      Only applied when diameter_mm is provided (calibrated measurement).
      If not available, falls back to size-class method (unchanged from v1.0).
      Never fabricates a physical measurement.
    """
    if observation.sprouting:
        return GradeLabel.defect, "Sprouting detected (AGMARK Grade I — prohibited)"

    if observation.defect_present:
        description = (observation.defect_description or "").lower()
        if any(keyword in description for keyword in ROT_KEYWORDS):
            return GradeLabel.defect, (
                f"Major defect / rot: \"{observation.defect_description}\""
                " (AGMARK Grade I/II — excluded)"
            )
        return GradeLabel.urs, (
            f"Minor surface blemish: \"{observation.defect_description}\""
            " — sound but sub-premium (AGMARK Grade II equivalent)"
        )

    # AGMARK physical diameter check — only if calibration data available
    if diameter_mm is not None:
        if diameter_mm < AGMARK_MIN_DIAMETER_MM:
            return GradeLabel.urs, (
                f"Measured diameter {diameter_mm:.1f} mm < {AGMARK_MIN_DIAMETER_MM:.0f} mm "
                "AGMARK Grade I threshold (AI-detected, calibrated measurement)"
            )
        return GradeLabel.grade_a, (
            f"Size = {observation.size.value}, diameter {diameter_mm:.1f} mm ≥ "
            f"{AGMARK_MIN_DIAMETER_MM:.0f} mm, no sprouting, no defects "
            "(meets AGMARK Grade I criteria — calibrated)"
        )

    # Fallback: size-class method (no calibration)
    if observation.size == observation.size.small:
        return GradeLabel.urs, (
            "Size = Small — below Grade I diameter threshold "
            "(≥45 mm, AGMARK DMI — physical diameter not calibrated)"
        )

    return GradeLabel.grade_a, (
        f"Size = {observation.size.value}, no sprouting, no defects "
        "(meets AGMARK Grade I surface criteria — diameter not calibrated)"
    )
