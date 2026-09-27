"""Gemini Vision perception layer for OnionGrade AI.

The number and location of onions are determined from the uploaded image.
Gemini performs perception only; deterministic Python performs grading.
"""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional


def _load_env_file() -> None:
    """Load API keys from backend/.env without requiring python-dotenv."""
    here = Path(__file__).resolve()
    candidates = [here.parents[2] / ".env", here.parents[3] / ".env"]
    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            for raw in env_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            pass
        break


_load_env_file()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
DEMO_MODE = not bool(GOOGLE_API_KEY)
# ── Model fallback list ───────────────────────────────────────────────────────
# GEMINI_VISION_MODELS  (comma-separated, tried in order):
#   Model 1 → quota/error → Model 2 → quota/error → … → offline CV fallback
#
# GEMINI_VISION_MODEL   (single model, legacy env var, still supported):
#   If set, it is prepended to the fallback list.
#
# Default fallback chain - real model IDs accepted by the Google AI SDK (2025):
#
#   Tier 1 - Gemini 3.x Flash (latest, highest capability):
#     gemini-3.8-flash               Newest, best accuracy
#     gemini-3.7-flash               Strong accuracy, thinking support
#     gemini-3.6-flash               Fast, capable
#     gemini-3.5-flash               Solid free-tier option
#     gemini-3.5-flash-lite          Lightweight 3.5 variant
#     gemini-3.1-flash-lite          Ultra-light 3.x variant
#
#   Tier 2 - Gemini 2.5 Flash (best confirmed free-tier accuracy):
#     gemini-2.5-flash               GA release, best confirmed free model
#     gemini-2.5-flash-preview-05-20 Preview variant, strong accuracy
#
#   Tier 3 - Gemini 2.0 Flash (fast, low quota cost):
#     gemini-2.0-flash               Full gen-2 Flash, good accuracy
#     gemini-2.0-flash-lite          Ultra-fast, lowest quota cost
#
#   Tier 4 - Stable 1.5 Flash (reliable free-tier fallbacks):
#     gemini-1.5-flash               Pinned 1.5-Flash, stable fallback
#     gemini-1.5-flash-8b            8B variant, extra quota headroom
#
#   Tier 5 - Pro models (highest accuracy, limited free quota):
#     gemini-1.5-pro                 Best accuracy, last resort
#
# Override the whole chain via GEMINI_VISION_MODELS env var if needed.
_DEFAULT_VISION_MODELS = [
    # Tier 1: 3.x Flash - latest generation
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    # Tier 2: 2.5 Flash - best confirmed free-tier accuracy
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    # Tier 3: 2.0 Flash - fast, low quota cost
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    # Tier 4: stable 1.5 Flash - reliable free-tier fallbacks
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    # Tier 5: Pro models - highest accuracy, last resort
    "gemini-1.5-pro",
]


def _build_model_list() -> list:
    raw = os.getenv("GEMINI_VISION_MODELS", "").strip()
    if raw:
        models = [m.strip() for m in raw.split(",") if m.strip()]
    else:
        models = list(_DEFAULT_VISION_MODELS)

    # Honour legacy GEMINI_VISION_MODEL: always promote it to position 0.
    # This ensures that if a user explicitly sets a single model, it is tried
    # first — even if it already appears later in the default list.
    legacy = os.getenv("GEMINI_VISION_MODEL", "").strip()
    if legacy:
        if legacy in models:
            models.remove(legacy)   # remove from wherever it is
        models.insert(0, legacy)    # always first
    return models

VISION_MODELS: list = _build_model_list()
# Expose first model as VISION_MODEL for backwards-compat (used in /health)
VISION_MODEL: str = VISION_MODELS[0] if VISION_MODELS else "gemini-2.0-flash-lite"

# "synthetic" (default) -> fixed 12-onion demo batch, mixed grades.
# "cv" -> run OpenCV blob detection on the actual uploaded image.
# Only matters when DEMO_MODE is True (no API key set).
# Strip any accidental whitespace so "= synthetic" (with a leading space)
# and the common typo "synthethic" both degrade gracefully to the default.
_raw_demo_mode = os.getenv("DEMO_DATA_MODE", "synthetic").strip().lower()
DEMO_DATA_MODE = _raw_demo_mode if _raw_demo_mode in ("synthetic", "cv") else "synthetic"

PERCEPTION_PROMPT = r"""You are the object-detection and visual-perception stage of an onion quality-grading system.

IMPORTANT: The uploaded image is the source of truth. DO NOT use fixed/demo onion counts.

Your job is to inspect THIS IMAGE and return one record for EVERY DISTINCT PHYSICAL WHOLE ONION
that is actually visible.

DETECTION RULES:
1. Count the physical onions in THIS IMAGE first.
2. Return exactly that many onion records. If the image contains 1 onion, return 1. If it contains 4,
   return 4. If it contains 12, return 12. Never use a fixed count.
3. One physical onion = one record. Never create multiple records for the same onion because of its
   stem, roots, skin layers, highlights, shadows, damaged areas, or visible segments.
4. Do not detect background objects, shadows, stems alone, roots alone, labels, text, watermarks,
   or empty regions as onions.
5. For every onion, draw a tight bounding box around the visible onion body. The box should cover the
   onion itself, not a large surrounding area.
6. For overlapping onions, return a separate box for each physical onion. It is acceptable for boxes
   to overlap when the onions overlap.
7. Carefully distinguish touching/overlapping onions by their visible contours, not by arbitrary regions.
8. IDs must be assigned AFTER counting and in visual reading order: top-to-bottom, then left-to-right.
   Use onion_01, onion_02, onion_03, ... with no gaps.
9. Confidence is confidence in the visual observation/detection, not a grade.
10. Do not assign Grade A, URS, Defect, quality score, recommendation, or final grade.

BOUNDING BOX:
Use normalized coordinates from 0 to 1 relative to the complete uploaded image:
- x = left edge / image width
- y = top edge / image height
- width = box width / image width
- height = box height / image height

Return JSON only in exactly this structure:
{
  "batch_id": "B-0412",
  "onions": [
    {
      "onion_id": "onion_01",
      "size": "Small" | "Medium" | "Large",
      "color": "short visible colour description",
      "sprouting": true | false,
      "defect_present": true | false,
      "defect_description": "short visible description or null",
      "confidence": 0.0,
      "bbox": {
        "x": 0.0,
        "y": 0.0,
        "width": 0.0,
        "height": 0.0
      }
    }
  ]
}
"""


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


# How long (seconds) to wait for a single Gemini model response.
# Worst-case chain: 3 models × 3 attempts × this value must stay under the
# frontend request timeout (300 s).  30 s × 9 = 270 s < 300 s.
_GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))
# How many times to retry the SAME model on transient errors before moving on.
_GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))


def _try_one_model(
    model_name: str,
    image_bytes: bytes,
    mime_type: str,
) -> Dict[str, Any]:
    """Attempt Gemini Vision with a single named model.

    Returns the parsed JSON dict on success.
    Raises:
      RuntimeError  — quota exceeded or bad model name (do NOT retry/fallback)
      _TransientError — timeout or server error (caller should try next model)
      json.JSONDecodeError — malformed response (caller should try next model)
    """
    import time

    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(model_name)

    last_exc: Exception = RuntimeError(f"Unknown error with model {model_name}")

    for attempt in range(1, _GEMINI_MAX_RETRIES + 2):  # first try + retries
        try:
            response = model.generate_content(
                [PERCEPTION_PROMPT, {"mime_type": mime_type, "data": image_bytes}],
                generation_config={"temperature": 0},
                request_options={"timeout": _GEMINI_TIMEOUT},
            )
            return _extract_json(response.text)

        except google_exceptions.ResourceExhausted as exc:
            # Quota exhausted on this model — move to next model immediately.
            raise _QuotaError(
                f"Model '{model_name}' quota exceeded"
            ) from exc

        except (google_exceptions.Unauthenticated,
                google_exceptions.PermissionDenied) as exc:
            # Invalid / revoked API key — no point trying other models,
            # they will all fail for the same reason. Raise immediately so
            # the FastAPI layer surfaces a clear "check your API key" message.
            raise _AuthError(
                f"API key rejected (check GOOGLE_API_KEY in your .env). "
                f"Keys must start with 'AIza'. Detail: {exc}"
            ) from exc

        except google_exceptions.InvalidArgument as exc:
            # Could be a bad model name OR a malformed request.
            # Treat as a model-level error so the next model is tried.
            raise _ModelError(
                f"Model '{model_name}' rejected — bad name or request: {exc}"
            ) from exc

        except google_exceptions.NotFound as exc:
            # Model does not exist — move to next model.
            raise _ModelError(
                f"Model '{model_name}' not found: {exc}"
            ) from exc

        except google_exceptions.DeadlineExceeded as exc:
            last_exc = exc
            if attempt <= _GEMINI_MAX_RETRIES:
                time.sleep(attempt * 3)
            continue

        except (google_exceptions.ServiceUnavailable,
                google_exceptions.InternalServerError) as exc:
            last_exc = exc
            if attempt <= _GEMINI_MAX_RETRIES:
                time.sleep(attempt * 2)
            continue

        except json.JSONDecodeError as exc:
            last_exc = exc
            if attempt <= _GEMINI_MAX_RETRIES:
                time.sleep(1)
            continue

    # All retries exhausted for this model — let caller try the next one.
    raise _TransientError(
        f"Model '{model_name}' timed out after {_GEMINI_MAX_RETRIES + 1} attempt(s): "
        f"{last_exc}"
    ) from last_exc


class _TransientError(Exception):
    """Timeout / server error — safe to try the next model in the list."""


class _QuotaError(_TransientError):
    """Quota exhausted on this model — try next model."""


class _ModelError(Exception):
    """Bad model name / unsupported — try next model but log a warning."""


class _AuthError(Exception):
    """Invalid / revoked API key — stop immediately, do not try other models."""


def _call_gemini_vision(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    """Try each model in VISION_MODELS in order until one succeeds.

    Stops immediately on an auth error (bad API key) — no point trying the
    rest of the chain if the key itself is invalid.
    Configure with GEMINI_VISION_MODELS=model1,model2,... in .env.

    Raises RuntimeError only when ALL models have been exhausted, so the
    FastAPI layer can return a clean HTTP 503 with an actionable message.
    """
    import logging
    logger = logging.getLogger("oniongrade.vision")

    model_errors: list = []

    for model_name in VISION_MODELS:
        try:
            result = _try_one_model(model_name, image_bytes, mime_type)
            if model_name != VISION_MODELS[0]:
                logger.info("Gemini Vision succeeded using fallback model '%s'", model_name)
            return result

        except _AuthError as exc:
            # Bad API key — all models will fail for the same reason, stop now.
            logger.error("Authentication failed — stopping fallback chain: %s", exc)
            raise RuntimeError(str(exc)) from exc

        except _ModelError as exc:
            # Bad model name — skip, try next.
            logger.warning("Skipping model '%s': %s", model_name, exc)
            model_errors.append(f"{model_name}: not found/unsupported")
            continue

        except _TransientError as exc:
            # Timeout or quota — try next model.
            logger.warning("Model '%s' unavailable: %s — trying next model", model_name, exc)
            model_errors.append(str(exc))
            continue

    # Every model in the list failed.
    summary = "; ".join(model_errors) if model_errors else "all models failed"
    raise RuntimeError(
        f"All Gemini Vision models failed: [{summary}]. "
        f"Tried: {', '.join(VISION_MODELS)}. "
        f"If the problem persists, remove GOOGLE_API_KEY to use offline OpenCV mode."
    )


# ---------------------------------------------------------------------------
# DEMO dataset (no API key, DEMO_DATA_MODE=synthetic): fixed 12-onion batch.
#
# 5 clean (-> Grade A), 4 minor-blemish/small (-> URS), 3 sprouting/rot
# (-> Defect). Used only for pitching/demoing the dashboard end-to-end
# without hitting the live Gemini API. Grades are STILL decided by
# grading_rules.py / calculations.py, exactly like with a real Gemini
# response -- this function only fakes the raw perception layer.
# ---------------------------------------------------------------------------

_DEMO_SIZES = ["Medium", "Large", "Medium", "Large", "Medium",
               "Small", "Medium", "Small", "Medium",
               "Medium", "Small", "Large"]

_DEMO_COLORS = ["Golden-brown", "Reddish-purple", "Yellow-brown", "White",
                "Golden-brown", "Yellow-brown", "Reddish-purple", "White",
                "Golden-brown", "Dark brown", "Yellow-brown", "Reddish-purple"]


def _generate_demo_batch(count: int = 12) -> Dict[str, Any]:
    onions = []
    cols, rows = 4, 3  # 4x3 grid layout, matches a typical batch photo
    cell_w, cell_h = 1.0 / cols, 1.0 / rows

    for i in range(count):
        row, col = divmod(i, cols)

        if i < 5:                       # onions 1-5: clean -> Grade A
            sprouting = False
            defect_present = False
            defect_description = None
            confidence = 0.94
        elif i < 9:                     # onions 6-9: minor issue -> URS
            sprouting = False
            defect_present = True
            defect_description = "Minor surface blemish, otherwise sound"
            confidence = 0.89
        else:                           # onions 10-12: major issue -> Defect
            sprouting = (i == 10)
            defect_present = not sprouting
            defect_description = None if sprouting else "Rot/mould visible on surface - major defect"
            confidence = 0.9

        x = round(col * cell_w + 0.02, 4)
        y = round(row * cell_h + 0.02, 4)
        onions.append({
            "onion_id": f"onion_{i + 1:02d}",
            "size": _DEMO_SIZES[i % len(_DEMO_SIZES)],
            "color": _DEMO_COLORS[i % len(_DEMO_COLORS)],
            "sprouting": sprouting,
            "defect_present": defect_present,
            "defect_description": defect_description,
            "confidence": confidence,
            "bbox": {
                "x": x,
                "y": y,
                "width": round(cell_w - 0.04, 4),
                "height": round(cell_h - 0.04, 4),
            },
        })

    return {"batch_id": "DEMO-BATCH-12", "onions": onions}


# ---------------------------------------------------------------------------
# Offline (no-API-key, DEMO_DATA_MODE=cv) detection fallback.
#
# Runs classic computer-vision directly on the ACTUAL uploaded image:
# it estimates the background color from the image corners, then treats
# any pixels that differ enough from that background as "object" (onion)
# pixels (Otsu threshold on the color-distance map). This works regardless
# of whether the onion is brown, red, purple or white, and regardless of
# whether the background is white, wood, cardboard, etc. -- unlike a fixed
# hue range, it never mistakes a plain light background for a "white onion".
# ---------------------------------------------------------------------------


def _classify_size(area: float, avg_area: float) -> str:
    ratio = area / avg_area if avg_area else 1.0
    if ratio < 0.75:
        return "Small"
    if ratio > 1.3:
        return "Large"
    return "Medium"


def _normalize_illumination(hsv_roi):
    """Flatten shading/shadow gradients across the onion's own V channel.

    A shadow under an onion and a rot patch on its skin can both read as
    "dark" in raw pixel values -- the difference is that a shadow is a
    smooth gradient while a defect is a *local* anomaly relative to its
    neighbourhood. Dividing V by a heavily-blurred copy of itself removes
    the smooth gradient and leaves local anomalies standing out.
    """
    import cv2
    import numpy as np

    v = hsv_roi[:, :, 2].astype(np.float32)
    h, w = v.shape[:2]
    sigma = max(h, w) / 4.0 or 1.0
    illum = cv2.GaussianBlur(v, (0, 0), sigmaX=sigma, sigmaY=sigma)
    norm_v = np.clip((v / (illum + 1e-6)) * 128.0, 0, 255).astype(np.uint8)
    return norm_v


def _masked_pixels(roi, mask):
    """Return only the onion's own pixels (mask==255), never the
    background or a neighbouring onion that happened to share the
    bounding box."""
    import numpy as np

    if mask is None:
        return roi.reshape(-1, roi.shape[-1])
    flat_mask = mask.reshape(-1) > 0
    flat_pixels = roi.reshape(-1, roi.shape[-1])
    selected = flat_pixels[flat_mask]
    return selected if selected.size else flat_pixels


def _dominant_color_name(roi, mask=None) -> str:
    import cv2
    import numpy as np

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    px = _masked_pixels(hsv_roi, mask)
    if px.size == 0:
        return "Yellow-brown"
    h, s, v = (int(np.median(px[:, i])) for i in range(3))

    if v < 60:
        return "Dark brown"
    if s < 35 and v > 150:
        return "White"
    if h < 20 or h > 165:
        return "Reddish-purple"
    if h < 35:
        return "Golden-brown"
    return "Yellow-brown"


def _looks_sprouting(roi, mask=None) -> bool:
    import cv2
    import numpy as np

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv_roi, np.array([35, 40, 40]), np.array([85, 255, 255]))
    if mask is not None:
        green_mask = cv2.bitwise_and(green_mask, mask)
        denom = cv2.countNonZero(mask)
    else:
        denom = roi.shape[0] * roi.shape[1]
    green_ratio = cv2.countNonZero(green_mask) / float(denom or 1)
    return green_ratio > 0.05


def _has_defect(roi, mask=None) -> bool:
    import cv2
    import numpy as np

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    norm_v = _normalize_illumination(hsv_roi)
    dark_mask = cv2.inRange(norm_v, 0, 95)
    if mask is not None:
        dark_mask = cv2.bitwise_and(dark_mask, mask)
        denom = cv2.countNonZero(mask)
    else:
        denom = roi.shape[0] * roi.shape[1]
    dark_ratio = cv2.countNonZero(dark_mask) / float(denom or 1)
    return dark_ratio > 0.08


def _defect_hint(roi, mask=None) -> Optional[str]:
    return "Dark/discolored patch detected on surface" if _has_defect(roi, mask) else None


def _detection_confidence(mask, w: int, h: int, from_split: bool) -> float:
    """Rough, honest confidence score instead of a hardcoded constant.

    Based on how "onion-shaped" the segmented blob actually is: a clean
    round blob fills roughly pi/4 (~0.785) of its own bounding box. A low
    fill ratio usually means partial occlusion, a merged/under-split
    blob, or noisy segmentation -- all reasons to trust the reading less.
    """
    import cv2
    import numpy as np

    rect_area = float(w * h) or 1.0
    mask_area = cv2.countNonZero(mask) if mask is not None else rect_area
    fill_ratio = mask_area / rect_area
    roundness_score = 1.0 - min(abs(fill_ratio - 0.785) / 0.785, 1.0)

    confidence = 0.55 + 0.35 * roundness_score
    if from_split:
        # Watershed-split blobs (touching/overlapping onions) are
        # inherently less certain than a single clean contour.
        confidence -= 0.08
    return round(float(np.clip(confidence, 0.45, 0.93)), 2)


def _estimate_background_color(img):
    import numpy as np

    img_h, img_w = img.shape[:2]
    corner = max(6, min(img_h, img_w) // 20)
    patches = [
        img[0:corner, 0:corner],
        img[0:corner, img_w - corner:img_w],
        img[img_h - corner:img_h, 0:corner],
        img[img_h - corner:img_h, img_w - corner:img_w],
    ]
    stacked = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    return np.median(stacked, axis=0)


def _build_foreground_mask(img):
    import cv2
    import numpy as np

    bg_color = _estimate_background_color(img)
    diff = np.linalg.norm(img.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _split_touching_blobs(mask, img_area: float):
    """Split touching/overlapping onions inside one foreground blob.

    Simple external contours treat two touching onions as a single blob,
    which under-counts the batch. This uses a distance-transform +
    watershed pass: each onion's center sits far from every edge (its own
    boundary AND the seam where it touches its neighbour), so the local
    peaks of the distance transform are reliable one-peak-per-onion seeds,
    even when onions overlap or touch.
    """
    import cv2
    import numpy as np

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []

    # Peaks of the distance map = onion centers. A relative threshold
    # (vs. this image's own max distance) makes it work for both small
    # and large onions instead of one fixed pixel radius.
    sure_fg = np.uint8((dist > 0.45 * dist.max()) * 255)

    n_labels, markers = cv2.connectedComponents(sure_fg)
    if n_labels <= 1:
        return []

    # Watershed needs: background = 1, unknown = 0, each seed a unique
    # label >= 2. It carves the shared mask along the seam between seeds.
    unknown = cv2.subtract(mask, sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_bgr_for_ws = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_bgr_for_ws, markers)

    boxes = []
    for label in range(2, markers.max() + 1):
        region = np.uint8(markers == label) * 255
        area = cv2.countNonZero(region)
        if area < img_area * 0.0015:
            continue
        ys, xs = np.where(region > 0)
        x, y, w, h = xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        # Keep the full-image-sized mask (not just the bbox) so the caller
        # can crop out this onion's own pixels later, never a neighbour's.
        boxes.append((int(x), int(y), int(w), int(h), float(area), region))
    return boxes


def _detect_onions_offline(image_bytes: bytes) -> Dict[str, Any]:
    """Detect onions in the ACTUAL uploaded image without calling any AI API."""
    import cv2
    import numpy as np

    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Could not decode the uploaded image for offline detection.")

    img_h, img_w = img.shape[:2]
    mask = _build_foreground_mask(img)

    img_area = img_w * img_h
    min_area = img_area * 0.003
    max_area = img_area * 0.92

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Each entry: (x, y, w, h, area, onion_mask, from_split)
    # onion_mask is a full-image-sized binary mask of ONLY this onion's own
    # pixels -- carried through so later steps never sample background or
    # a neighbouring onion just because it shares the same bounding box.
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h) if h else 0
        if aspect < 0.25 or aspect > 4.0:
            continue

        # A blob shaped much wider/taller than a single round onion, or
        # noticeably less round than a circle of the same area, is very
        # likely 2+ touching onions merged into one contour -- try to
        # split it with watershed instead of reporting it as one onion.
        perimeter = cv2.arcLength(c, True)
        circularity = (4 * np.pi * area / (perimeter * perimeter)) if perimeter else 1.0
        looks_merged = circularity < 0.72 or max(w, h) / float(min(w, h) or 1) > 1.6

        if looks_merged and area > min_area * 2.2:
            local_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(local_mask, [c], -1, 255, thickness=cv2.FILLED)
            split_boxes = _split_touching_blobs(local_mask, img_area)
            if len(split_boxes) >= 2:
                boxes.extend((sx, sy, sw, sh, sarea, smask, True) for sx, sy, sw, sh, sarea, smask in split_boxes)
                continue

        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [c], -1, 255, thickness=cv2.FILLED)
        boxes.append((x, y, w, h, area, contour_mask, False))

    # Fallback: nothing passed the filters (e.g. onion fills ~the whole
    # frame with almost no border) -- treat the largest non-background
    # blob as a single onion instead of failing outright.
    if not boxes and contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area >= img_area * 0.01:
            x, y, w, h = cv2.boundingRect(c)
            fallback_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(fallback_mask, [c], -1, 255, thickness=cv2.FILLED)
            boxes.append((x, y, w, h, area, fallback_mask, False))

    boxes.sort(key=lambda b: (b[1], b[0]))  # top-to-bottom, then left-to-right

    if not boxes:
        return {"batch_id": "OFFLINE-SCAN", "onions": []}

    avg_area = sum(b[4] for b in boxes) / len(boxes)

    onions = []
    for i, (x, y, w, h, area, onion_mask, from_split) in enumerate(boxes, start=1):
        roi = img[y:y + h, x:x + w]
        roi_mask = onion_mask[y:y + h, x:x + w]
        onions.append({
            "onion_id": f"onion_{i:02d}",
            "size": _classify_size(area, avg_area),
            "color": _dominant_color_name(roi, roi_mask),
            "sprouting": _looks_sprouting(roi, roi_mask),
            "defect_present": _has_defect(roi, roi_mask),
            "defect_description": _defect_hint(roi, roi_mask),
            "confidence": _detection_confidence(roi_mask, w, h, from_split),
            "bbox": {
                "x": round(x / img_w, 4),
                "y": round(y / img_h, 4),
                "width": round(min(w / img_w, 1.0), 4),
                "height": round(min(h / img_h, 1.0), 4),
            },
        })

    return {"batch_id": "OFFLINE-SCAN", "onions": onions}


def analyze_batch_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> Dict[str, Any]:
    """Analyze the uploaded image. In DEMO_MODE, return a fixed 12-onion demo
    batch (mixed Grade A / URS / Defect) unless DEMO_DATA_MODE=cv is set."""
    if DEMO_MODE:
        if DEMO_DATA_MODE == "cv":
            result = _detect_onions_offline(image_bytes)
        else:
            result = _generate_demo_batch(12)
    else:
        result = _call_gemini_vision(image_bytes, mime_type)

    onions = result.get("onions")
    if not isinstance(onions, list):
        raise RuntimeError("Vision analysis returned an invalid onion detection response.")

    # Keep only object records. The backend will validate all fields and normalize IDs.
    result["onions"] = [item for item in onions if isinstance(item, dict)]
    return result
