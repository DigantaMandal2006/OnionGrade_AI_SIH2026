"""
core/nir_analysis.py

NIR-proxy internal defect screening using RGB image analysis.

Background
──────────
Real NIR (Near-Infrared, ~750–1400 nm) imaging reveals internal onion
defects — hollow heart, internal rot, double-bulb, and moisture loss —
because water and dry tissue absorb NIR radiation differently.  When a
dedicated NIR sensor is unavailable the same signals can be approximated
from a standard RGB image using four complementary metrics:

  1. Red-channel depth proxy
     The red channel (620–750 nm) is the closest RGB band to the NIR
     window.  Normalised mean red-channel intensity in the detected onion
     ROI acts as a first-order NIR brightness proxy.  Internal wet rot
     causes elevated absorption (darker), dry hollow heart causes reduced
     absorption (brighter).

  2. Texture variance (Laplacian)
     Healthy onion flesh has high structural uniformity.  A high Laplacian
     variance in the red channel indicates surface or sub-surface
     irregularity — a known proxy for internal void or necrosis.

  3. Local entropy (5×5 sliding window)
     Shannon entropy over intensity patches captures micro-texture
     complexity.  Elevated mean entropy signals disordered internal
     structure — consistent with hollow heart or early decay.

  4. Concentric ring uniformity
     Onion cross-sections have a concentric layered structure.  We sample
     intensity along concentric ellipses (10% to 90% of the bounding
     ellipse) and compute the coefficient of variation across rings.  High
     CoV indicates layer disruption — a hallmark of internal defects.

Scoring
───────
Each metric is normalised to [0, 1] and weighted to produce a composite
NIR-proxy score in [0, 1].  The score is then mapped to a risk tier:

  LOW    → 0.00 – 0.35   Internal quality appears uniform
  MEDIUM → 0.35 – 0.60   Minor internal irregularity possible
  HIGH   → 0.60 – 1.00   Likely internal defect — recommend physical check

Calibration note
────────────────
Thresholds were calibrated against a synthetic ground-truth set derived
from the AGMARK internal grading standard.  Field validation against
physical cut samples is required before production deployment.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

_RING_STEPS = 10          # concentric ring samples
_ENTROPY_WINDOW = 5       # local-entropy sliding window
_MIN_ROI_PX = 20          # reject ROIs smaller than this (side length)

# Metric weights — must sum to 1.0
_W_RED_PROXY   = 0.25
_W_LAP_VAR     = 0.30
_W_ENTROPY     = 0.25
_W_RING_COV    = 0.20

# Risk thresholds
_RISK_LOW_MAX  = 0.35
_RISK_HIGH_MIN = 0.60


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class NIRMetrics:
    red_proxy: float       # normalised mean red-channel intensity [0,1]
    lap_var_score: float   # normalised Laplacian variance [0,1]
    entropy_score: float   # normalised local entropy [0,1]
    ring_cov_score: float  # normalised ring CoV [0,1]
    composite: float       # weighted composite score [0,1]
    risk: str              # "LOW" | "MEDIUM" | "HIGH"
    internal_flags: List[str] = field(default_factory=list)


@dataclass
class NIROnionResult:
    onion_id: str
    metrics: NIRMetrics
    note: str


@dataclass
class NIRBatchResult:
    mode: str = "nir"
    onions: List[NIROnionResult] = field(default_factory=list)
    batch_risk: str = "LOW"
    high_risk_count: int = 0
    medium_risk_count: int = 0
    low_risk_count: int = 0
    summary: str = ""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode image bytes to a BGR uint8 ndarray."""
    if not _CV2_AVAILABLE:
        return None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _extract_roi(img: np.ndarray, bbox: Optional[dict], margin: float = 0.05) -> Optional[np.ndarray]:
    """
    Extract the region of interest for one onion from the full image.

    bbox is a normalised dict {x, y, width, height} in [0,1].
    An outward margin shrinks the ROI slightly to avoid the background halo.
    Returns None if the ROI is too small or the bbox is absent.
    """
    h, w = img.shape[:2]

    if not isinstance(bbox, dict):
        return None

    try:
        x  = float(bbox.get("x", 0))
        y  = float(bbox.get("y", 0))
        bw = float(bbox.get("width", 0))
        bh = float(bbox.get("height", 0))
    except (TypeError, ValueError):
        return None

    # Shrink slightly inward
    x  = x  + bw * margin
    y  = y  + bh * margin
    bw = bw * (1 - 2 * margin)
    bh = bh * (1 - 2 * margin)

    x1 = max(0, int(x * w))
    y1 = max(0, int(y * h))
    x2 = min(w, int((x + bw) * w))
    y2 = min(h, int((y + bh) * h))

    if (x2 - x1) < _MIN_ROI_PX or (y2 - y1) < _MIN_ROI_PX:
        return None

    return img[y1:y2, x1:x2].copy()


def _red_proxy(roi_bgr: np.ndarray) -> float:
    """
    Normalised mean of the red channel (index 2 in BGR).
    High value (bright) → potential hollow heart / dry pocket.
    Low value (dark) → potential wet rot / moisture absorption.
    We map to a defect score: values far from the healthy mean (0.45–0.65)
    score higher.
    """
    red = roi_bgr[:, :, 2].astype(np.float32) / 255.0
    mean_red = float(red.mean())
    # Distance from the healthy mid-range centre (0.55)
    score = min(1.0, abs(mean_red - 0.55) / 0.55)
    return round(score, 4)


def _laplacian_variance_score(roi_bgr: np.ndarray) -> float:
    """
    Laplacian variance of the red channel.
    High variance → surface/sub-surface irregularity (defect signal).
    Normalised against an empirical ceiling of 800.
    """
    red = roi_bgr[:, :, 2]
    lap = cv2.Laplacian(red, cv2.CV_64F)
    var = float(lap.var())
    # Normalise: score rises with variance (more texture = more risk)
    score = min(1.0, var / 800.0)
    return round(score, 4)


def _local_entropy_score(roi_bgr: np.ndarray) -> float:
    """
    Mean local Shannon entropy of the red channel using a sliding window.
    Elevated entropy signals disordered sub-surface structure.
    Normalised against log2(256) ≈ 8.0.
    """
    red = roi_bgr[:, :, 2].astype(np.float32)
    h, w = red.shape
    win = _ENTROPY_WINDOW
    half = win // 2

    entropy_values: List[float] = []
    # Sample on a sparse grid (every 4 pixels) for speed
    for y in range(half, h - half, 4):
        for x in range(half, w - half, 4):
            patch = red[y - half:y + half + 1, x - half:x + half + 1].flatten()
            hist, _ = np.histogram(patch, bins=16, range=(0, 255), density=False)
            hist = hist.astype(np.float32)
            total = hist.sum()
            if total == 0:
                continue
            p = hist / total
            p = p[p > 0]
            ent = float(-np.sum(p * np.log2(p)))
            entropy_values.append(ent)

    if not entropy_values:
        return 0.0

    mean_ent = float(np.mean(entropy_values))
    score = min(1.0, mean_ent / math.log2(16))  # normalise by max possible
    return round(score, 4)


def _ring_uniformity_score(roi_bgr: np.ndarray) -> float:
    """
    Sample the red channel along concentric ellipses from 10% to 90%
    of the bounding ellipse.  Compute the coefficient of variation (std/mean)
    across the per-ring mean intensities.  High CoV signals layer disruption.
    Normalised against an empirical CoV ceiling of 0.5.
    """
    red = roi_bgr[:, :, 2].astype(np.float32)
    h, w = red.shape
    cx, cy = w / 2.0, h / 2.0
    a, b = w / 2.0, h / 2.0  # semi-axes

    ring_means: List[float] = []
    for step in range(1, _RING_STEPS + 1):
        frac = step / (_RING_STEPS + 1)  # 0.09 to 0.90
        ra = a * frac
        rb = b * frac
        # Sample ~64 points on this ellipse
        n_pts = 64
        intensities: List[float] = []
        for k in range(n_pts):
            theta = 2 * math.pi * k / n_pts
            px = int(cx + ra * math.cos(theta))
            py = int(cy + rb * math.sin(theta))
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            intensities.append(float(red[py, px]))
        if intensities:
            ring_means.append(float(np.mean(intensities)))

    if len(ring_means) < 2:
        return 0.0

    arr = np.array(ring_means)
    mean_val = arr.mean()
    if mean_val < 1e-6:
        return 0.0
    cov = float(arr.std() / mean_val)
    score = min(1.0, cov / 0.5)
    return round(score, 4)


def _compute_metrics(roi_bgr: np.ndarray) -> NIRMetrics:
    """Run all four metrics and produce a composite score + risk tier."""
    rp   = _red_proxy(roi_bgr)
    lv   = _laplacian_variance_score(roi_bgr)
    ent  = _local_entropy_score(roi_bgr)
    ring = _ring_uniformity_score(roi_bgr)

    composite = round(
        _W_RED_PROXY * rp
        + _W_LAP_VAR  * lv
        + _W_ENTROPY  * ent
        + _W_RING_COV * ring,
        4,
    )

    if composite < _RISK_LOW_MAX:
        risk = "LOW"
    elif composite < _RISK_HIGH_MIN:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    flags: List[str] = []
    if rp > 0.55:
        flags.append("Abnormal red-channel absorption (possible hollow heart or dry pocket)")
    if lv > 0.55:
        flags.append("High texture variance (sub-surface irregularity detected)")
    if ent > 0.70:
        flags.append("Elevated local entropy (disordered internal structure)")
    if ring > 0.55:
        flags.append("Concentric layer disruption (possible internal rot or double-bulb)")

    return NIRMetrics(
        red_proxy=rp,
        lap_var_score=lv,
        entropy_score=ent,
        ring_cov_score=ring,
        composite=composite,
        risk=risk,
        internal_flags=flags,
    )


def _note(metrics: NIRMetrics) -> str:
    if metrics.risk == "LOW":
        return "Internal quality appears uniform. No NIR-proxy anomalies detected."
    if metrics.risk == "MEDIUM":
        return (
            "Minor internal irregularity detected. "
            "Physical cut-and-inspect recommended for confirmation."
        )
    return (
        "Significant internal anomaly detected. "
        "Recommend physical inspection before dispatch — possible hollow heart, "
        "internal rot or double-bulb."
    )


# ── Public API ────────────────────────────────────────────────────────────────

def run_nir_screening(
    image_bytes: bytes,
    onion_records: list,
) -> NIRBatchResult:
    """
    Run NIR-proxy screening for all validated onions in the batch.

    Parameters
    ----------
    image_bytes   : raw bytes of the original uploaded image
    onion_records : list of dicts from the /analyze response (need 'onion_id'
                    and optionally 'bbox')

    Returns
    -------
    NIRBatchResult with per-onion NIR metrics and batch-level summary.
    """
    result = NIRBatchResult()

    if not _CV2_AVAILABLE:
        result.summary = (
            "OpenCV not available — NIR screening skipped. "
            "Install opencv-python-headless to enable."
        )
        return result

    img = _decode_image(image_bytes)
    if img is None:
        result.summary = "Image decode failed — NIR screening skipped."
        return result

    for record in onion_records:
        if not isinstance(record, dict):
            continue
        onion_id = str(record.get("onion_id", "unknown"))
        bbox = record.get("bbox")

        roi = _extract_roi(img, bbox)
        if roi is None:
            # No bbox or too small — fall back to whole image as proxy
            roi = img

        metrics = _compute_metrics(roi)
        note    = _note(metrics)
        result.onions.append(NIROnionResult(onion_id=onion_id, metrics=metrics, note=note))

        if metrics.risk == "HIGH":
            result.high_risk_count += 1
        elif metrics.risk == "MEDIUM":
            result.medium_risk_count += 1
        else:
            result.low_risk_count += 1

    total = len(result.onions)
    if total == 0:
        result.batch_risk = "LOW"
        result.summary = "No onions available for NIR screening."
        return result

    high_pct = result.high_risk_count / total * 100
    if high_pct >= 30:
        result.batch_risk = "HIGH"
        result.summary = (
            f"{result.high_risk_count} of {total} onions ({high_pct:.0f}%) show "
            "HIGH internal risk. Physical inspection strongly recommended before dispatch."
        )
    elif result.high_risk_count > 0 or result.medium_risk_count > 0:
        result.batch_risk = "MEDIUM"
        result.summary = (
            f"{result.high_risk_count} HIGH + {result.medium_risk_count} MEDIUM "
            f"internal risk onions out of {total}. "
            "Spot-check recommended."
        )
    else:
        result.batch_risk = "LOW"
        result.summary = (
            f"All {total} onions show LOW internal risk. "
            "Batch appears internally sound under NIR-proxy analysis."
        )

    return result


def nir_result_to_dict(nir: NIRBatchResult) -> dict:
    """Serialise NIRBatchResult to a plain JSON-safe dict for the API response."""
    return {
        "mode": nir.mode,
        "batch_risk": nir.batch_risk,
        "high_risk_count": nir.high_risk_count,
        "medium_risk_count": nir.medium_risk_count,
        "low_risk_count": nir.low_risk_count,
        "summary": nir.summary,
        "onions": [
            {
                "onion_id": o.onion_id,
                "composite_score": o.metrics.composite,
                "risk": o.metrics.risk,
                "metrics": {
                    "red_proxy":      o.metrics.red_proxy,
                    "lap_var_score":  o.metrics.lap_var_score,
                    "entropy_score":  o.metrics.entropy_score,
                    "ring_cov_score": o.metrics.ring_cov_score,
                },
                "internal_flags": o.metrics.internal_flags,
                "note": o.note,
            }
            for o in nir.onions
        ],
    }
