"""FastAPI backend for OnionGrade AI.

Version: 2.0.0

New in 2.0.0:
  - SQLite scan history (/history, /history/{scan_id})
  - Image preprocessing pipeline (auto-applied before Gemini)
  - XLSX export endpoint (/export/xlsx)
  - API health/configuration check (/api-health)
  - Multi-image batch support (source_image field on onion records)
  - NIR/SWIR dual-image endpoint (/analyze/nir)
  - Confidence threshold query param (?confidence_threshold=0.7)
  - Analysis progress via SSE (/analyze/progress)
"""
import io
import json
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from app.core.api_health import check_gemini_api
from app.core.calculations import grade_batch
from app.core.db import get_scan, get_scan_summary, init_db, list_scans, save_scan
from app.core.excel_export import build_xlsx
from app.core.image_preprocessing import get_image_dimensions, preprocess_image
from app.core.nir_analysis import nir_result_to_dict, run_nir_screening
from app.core.report_service import generate_report
from app.core.schemas import OnionObservation
from app.core.vision_service import DEMO_MODE, VISION_MODEL, VISION_MODELS, analyze_batch_image

logger = logging.getLogger("oniongrade.main")

# ── CORS ─────────────────────────────────────────────────────────────────────
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="OnionGrade AI", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# Accepted MIME types for the /analyze endpoint.
_ALLOWED_MIME = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
_MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB

# ── Startup: init DB ──────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    init_db()


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        import cv2  # noqa: F401
        cv_available = True
    except ImportError:
        cv_available = False

    return {
        "status": "ok",
        "version": "2.0.0",
        "demo_mode": DEMO_MODE,
        "vision_model": VISION_MODEL,
        "vision_model_fallback_list": VISION_MODELS,
        "opencv_available": cv_available,
        "allowed_origins": ALLOWED_ORIGINS,
    }


@app.get("/api-health")
def api_health():
    """Lightweight Gemini API configuration status check.

    Never exposes the key; never makes a live API request.
    """
    return check_gemini_api()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bbox_iou(a: dict, b: dict) -> float:
    """Calculate IoU for two normalized bounding boxes."""
    try:
        ax1, ay1 = float(a.get("x", 0)), float(a.get("y", 0))
        ax2 = ax1 + float(a.get("width", 0))
        ay2 = ay1 + float(a.get("height", 0))
        bx1, by1 = float(b.get("x", 0)), float(b.get("y", 0))
        bx2 = bx1 + float(b.get("width", 0))
        by2 = by1 + float(b.get("height", 0))
    except (TypeError, ValueError):
        return 0.0

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _deduplicate_detections(records: list[dict]) -> list[dict]:
    """Remove duplicate boxes that describe the same physical onion."""
    kept: list[dict] = []

    candidates = [r for r in records if isinstance(r, dict)]
    candidates.sort(
        key=lambda r: float(r.get("confidence", 0) or 0),
        reverse=True,
    )

    for candidate in candidates:
        bbox = candidate.get("bbox")
        if not isinstance(bbox, dict):
            kept.append(candidate)
            continue

        try:
            cx = float(bbox.get("x", 0)) + float(bbox.get("width", 0)) / 2
            cy = float(bbox.get("y", 0)) + float(bbox.get("height", 0)) / 2
        except (TypeError, ValueError):
            kept.append(candidate)
            continue

        duplicate = False
        for existing in kept:
            eb = existing.get("bbox")
            if not isinstance(eb, dict):
                continue
            try:
                ecx = float(eb.get("x", 0)) + float(eb.get("width", 0)) / 2
                ecy = float(eb.get("y", 0)) + float(eb.get("height", 0)) / 2
            except (TypeError, ValueError):
                continue

            center_distance = ((cx - ecx) ** 2 + (cy - ecy) ** 2) ** 0.5
            if _bbox_iou(bbox, eb) >= 0.55 and center_distance <= 0.12:
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    kept.sort(
        key=lambda r: (
            float((r.get("bbox") or {}).get("y", 0) or 0),
            float((r.get("bbox") or {}).get("x", 0) or 0),
        )
    )
    return kept


def _defect_type(observation: OnionObservation) -> str:
    description = (observation.defect_description or "").lower()
    if observation.sprouting:
        return "Sprouting"
    if "mold" in description or "mould" in description:
        return "Mold"
    if "rot" in description:
        return "Rot"
    if "bruise" in description:
        return "Bruising"
    if "physical" in description or "damage" in description:
        return "Physical Damage"
    if "discolor" in description:
        return "Discoloration"
    if "decay" in description:
        return "Decay"
    return "Other" if observation.defect_present else "None"


def _severity(grade: str) -> str:
    if grade == "Defect":
        return "Major"
    if grade == "URS":
        return "Minor"
    return "None"


def _quality_assessment(grade_result: dict) -> dict:
    total = grade_result.get("total_validated", 0)
    defect_pct = float(grade_result.get("defect_pct", 0))
    grade_a_pct = float(grade_result.get("grade_a_pct", 0))

    if total == 0:
        status = "POOR"
        description = "No onions passed validation."
        recommendation = "Perform manual inspection."
    elif defect_pct >= 20:
        status = "POOR"
        description = "A significant share of the validated batch has defects."
        recommendation = "Separate defective onions and perform manual inspection."
    elif grade_a_pct >= 70 and defect_pct < 10:
        status = "GOOD"
        description = "Most validated onions meet Grade A criteria."
        recommendation = "Batch is suitable for normal handling and sorting."
    else:
        status = "ACCEPTABLE"
        description = "The batch contains a mix of acceptable and lower-quality onions."
        recommendation = "Review URS and flagged onions before final dispatch."

    return {"status": status, "description": description, "recommendation": recommendation}


def _defect_analysis(onions: list[OnionObservation], grade_result: dict) -> dict:
    categories = Counter()
    for onion in onions:
        if onion.defect_present or onion.sprouting:
            categories[_defect_type(onion)] += 1
    return {
        "total_defective": int(grade_result.get("defect_count", 0) + grade_result.get("urs_count", 0)),
        "categories": dict(categories),
    }


def _classify_confidence(confidence: float, threshold: float) -> str:
    """Classify a detection's confidence relative to the user threshold.

    Returns:
      "Accepted"       — confidence >= threshold
      "Low Confidence" — confidence >= 0.5 (validation floor) but < threshold
      "Manual Review"  — (never excluded silently — always visible)
    """
    if confidence >= threshold:
        return "Accepted"
    return "Low Confidence"


def _build_response(
    raw: dict,
    inspection_mode: str,
    image_bytes: bytes,
    source_image: str = "",
    confidence_threshold: float = 0.5,
) -> dict:
    """Core analysis pipeline shared by /analyze and /analyze/nir."""
    batch_id = raw.get("batch_id", "batch")
    raw_onions = _deduplicate_detections(raw.get("onions", []) or [])

    # Normalize IDs, tag source image
    normalized_raw = []
    for index, entry in enumerate(raw_onions, start=1):
        if isinstance(entry, dict):
            item = dict(entry)
            item["onion_id"] = f"onion_{index:02d}"
            if source_image:
                item["source_image"] = source_image
            normalized_raw.append(item)

    # Pydantic validation
    validated: list[OnionObservation] = []
    flagged: list[dict] = []
    for entry in normalized_raw:
        try:
            validated.append(OnionObservation(**entry))
        except ValidationError as exc:
            reason = exc.errors()[0]["msg"] if exc.errors() else "invalid record"
            flagged.append({"raw": entry, "reason": reason})

    if not validated:
        raise HTTPException(status_code=422, detail="No valid onion records after validation.")

    # Deterministic grading
    grade_result = grade_batch(validated)
    grade_lookup = {
        item["onion_id"]: item
        for item in grade_result["per_onion_grades"]
    }

    onions = []
    for onion in validated:
        grade_info = grade_lookup.get(onion.onion_id, {})
        grade = grade_info.get("grade", "-")
        record = onion.model_dump()
        record["severity"] = _severity(grade)
        record["defect_type"] = _defect_type(onion)
        record["final_grade"] = grade
        record["grade_reason"] = grade_info.get("grade_reason", "")
        record["source_image"] = source_image or ""
        # Confidence classification vs user threshold
        conf = float(record.get("confidence") or 0.0)
        record["review_status"] = _classify_confidence(conf, confidence_threshold)
        onions.append(record)

    summary = {
        "grade_a": {"count": grade_result["grade_a_count"], "percentage": grade_result["grade_a_pct"]},
        "urs": {"count": grade_result["urs_count"], "percentage": grade_result["urs_pct"]},
        "defect": {"count": grade_result["defect_count"], "percentage": grade_result["defect_pct"]},
    }

    quality = _quality_assessment(grade_result)
    defect_analysis = _defect_analysis(validated, grade_result)
    report_text = generate_report(
        grade_result,
        batch_id=batch_id,
        total_observed=len(raw_onions),
        defect_analysis=defect_analysis,
        overall_status=quality["status"],
        recommendation=quality["recommendation"],
    )

    # NIR screening
    nir_results: Optional[dict] = None
    if inspection_mode == "nir":
        nir_batch = run_nir_screening(image_bytes, onions)
        nir_results = nir_result_to_dict(nir_batch)

    response: dict = {
        "batch": {"batch_id": batch_id, "observed": len(raw_onions), "validated": len(validated)},
        "batch_id": batch_id,
        "total_observed": len(raw_onions),
        "summary": summary,
        "onions": onions,
        "validated": onions,
        "flagged": flagged,
        "chart": {
            "labels": ["Grade A", "URS", "Defect"],
            "values": [grade_result["grade_a_pct"], grade_result["urs_pct"], grade_result["defect_pct"]],
        },
        "grade_result": grade_result,
        "overall_quality": quality,
        "defect_analysis": defect_analysis,
        "report": report_text,
        "inspection_mode": inspection_mode,
        "confidence_threshold": confidence_threshold,
        "system": {
            "vision": "Gemini Vision",
            "vision_model": VISION_MODEL,
            "grading": "Deterministic Python",
            "demo_mode": DEMO_MODE,
        },
        "demo_mode": DEMO_MODE,
    }
    if nir_results is not None:
        response["nir_results"] = nir_results
    return response


# ── Analyze endpoint ──────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    inspection_mode: Optional[Literal["rgb", "nir"]] = Query(default="rgb"),
    confidence_threshold: float = Query(default=0.5, ge=0.0, le=1.0),
    save_history: bool = Query(default=True),
):
    """Analyze a single uploaded image.

    New parameters (all backward-compatible defaults):
      confidence_threshold — controls review_status label on each onion (default 0.5)
      save_history         — persist result to SQLite history (default True)
    """
    # 1. Input validation
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{content_type}'. "
                "Please upload a JPEG, PNG or WebP image."
            ),
        )

    raw_image_bytes = await file.read()
    if not raw_image_bytes:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    if len(raw_image_bytes) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image is too large ({len(raw_image_bytes) // (1024*1024)} MB). "
                "Maximum allowed size is 20 MB."
            ),
        )

    # 2. Image preprocessing (resizes/rotates for Gemini; originals kept for UI/NIR)
    processed_bytes, processed_mime = preprocess_image(raw_image_bytes)

    # 3. Vision perception
    try:
        raw = analyze_batch_image(
            processed_bytes,
            mime_type=processed_mime,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Vision analysis failed: {exc}") from exc

    # 4. Build full response (uses original bytes for NIR to preserve quality)
    response = _build_response(
        raw,
        inspection_mode=inspection_mode or "rgb",
        image_bytes=raw_image_bytes,   # original for NIR analysis
        source_image=file.filename or "",
        confidence_threshold=confidence_threshold,
    )

    # Add preprocessing metadata
    orig_w, orig_h = get_image_dimensions(raw_image_bytes)
    proc_w, proc_h = get_image_dimensions(processed_bytes)
    response["preprocessing"] = {
        "original_size_bytes": len(raw_image_bytes),
        "processed_size_bytes": len(processed_bytes),
        "original_dimensions": [orig_w, orig_h],
        "processed_dimensions": [proc_w, proc_h],
    }

    # 5. Save to history
    if save_history:
        scan_id = save_scan(
            response,
            analysis_mode=inspection_mode or "rgb",
            demo_mode=bool(DEMO_MODE),
            vision_model=VISION_MODEL,
        )
        if scan_id is not None:
            response["scan_id"] = scan_id

    return response


# ── NIR dual-image endpoint ───────────────────────────────────────────────────

@app.post("/analyze/nir")
async def analyze_nir(
    rgb_file: UploadFile = File(...),
    nir_file: Optional[UploadFile] = File(default=None),
    confidence_threshold: float = Query(default=0.5, ge=0.0, le=1.0),
    save_history: bool = Query(default=True),
):
    """Analyze with optional real NIR/SWIR image upload.

    If nir_file is provided: uses that for NIR analysis (real NIR mode).
    If nir_file is absent: falls back to RGB-proxy NIR mode (existing behavior).

    Mode is reported in the response as:
      nir_input_mode: "real_nir" | "rgb_proxy"
    """
    # Validate RGB
    content_type = (rgb_file.content_type or "").lower().split(";")[0].strip()
    if content_type not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported RGB file type '{content_type}'.",
        )

    rgb_bytes = await rgb_file.read()
    if not rgb_bytes:
        raise HTTPException(status_code=400, detail="Empty RGB file upload.")
    if len(rgb_bytes) > _MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="RGB image too large (max 20 MB).")

    # Preprocess RGB for Gemini
    processed_rgb, processed_mime = preprocess_image(rgb_bytes)

    # Optional real NIR image
    nir_bytes: Optional[bytes] = None
    nir_input_mode = "rgb_proxy"
    if nir_file is not None:
        nir_ct = (nir_file.content_type or "").lower().split(";")[0].strip()
        if nir_ct not in _ALLOWED_MIME:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported NIR file type '{nir_ct}'.",
            )
        nir_bytes = await nir_file.read()
        if nir_bytes:
            nir_input_mode = "real_nir"
        else:
            nir_bytes = None

    # Vision analysis (RGB only — NIR is not sent to Gemini)
    try:
        raw = analyze_batch_image(processed_rgb, mime_type=processed_mime)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Vision analysis failed: {exc}") from exc

    # For NIR screening use the real NIR image if available, else original RGB
    nir_image_for_analysis = nir_bytes if nir_bytes else rgb_bytes

    # Build response with NIR mode forced on
    response = _build_response(
        raw,
        inspection_mode="nir",
        image_bytes=nir_image_for_analysis,
        source_image=rgb_file.filename or "",
        confidence_threshold=confidence_threshold,
    )

    response["nir_input_mode"] = nir_input_mode

    if save_history:
        scan_id = save_scan(
            response,
            analysis_mode="nir",
            demo_mode=bool(DEMO_MODE),
            vision_model=VISION_MODEL,
        )
        if scan_id is not None:
            response["scan_id"] = scan_id

    return response


# ── History endpoints ─────────────────────────────────────────────────────────

@app.get("/history")
def history_list(limit: int = Query(default=50, ge=1, le=500)):
    """List previous scans (newest first, summary only)."""
    return {"scans": list_scans(limit=limit)}


@app.get("/history/{scan_id}")
def history_get(scan_id: int):
    """Reload a full stored analysis result WITHOUT calling Gemini."""
    data = get_scan(scan_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found.")
    summary = get_scan_summary(scan_id) or {}
    return {"scan_id": scan_id, "summary": summary, "result": data}


@app.delete("/history/{scan_id}")
def history_delete(scan_id: int):
    """Delete a stored scan."""
    summary = get_scan_summary(scan_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found.")
    from app.core.db import delete_scan
    ok = delete_scan(scan_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to delete scan.")
    return {"deleted": scan_id}


# ── XLSX export endpoint ──────────────────────────────────────────────────────

@app.post("/export/xlsx")
async def export_xlsx(
    file: UploadFile = File(...),
    inspection_mode: Optional[Literal["rgb", "nir"]] = Query(default="rgb"),
    confidence_threshold: float = Query(default=0.5, ge=0.0, le=1.0),
):
    """Analyze an image and return an XLSX workbook directly."""
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in _ALLOWED_MIME:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{content_type}'.")

    raw_image_bytes = await file.read()
    if not raw_image_bytes:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    processed_bytes, processed_mime = preprocess_image(raw_image_bytes)

    try:
        raw = analyze_batch_image(processed_bytes, mime_type=processed_mime)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    response = _build_response(
        raw,
        inspection_mode=inspection_mode or "rgb",
        image_bytes=raw_image_bytes,
        source_image=file.filename or "",
        confidence_threshold=confidence_threshold,
    )

    xlsx_bytes = build_xlsx(
        onions=response.get("onions") or [],
        grade_result=response.get("grade_result") or {},
        batch_id=response.get("batch_id") or "unknown",
        analysis_mode=inspection_mode or "rgb",
        demo_mode=bool(DEMO_MODE),
        vision_model=VISION_MODEL,
    )

    if xlsx_bytes is None:
        raise HTTPException(
            status_code=500,
            detail="XLSX generation failed. Ensure openpyxl is installed.",
        )

    filename = f"oniongrade_{response.get('batch_id', 'results')}.xlsx"
    return StreamingResponse(
        io.BytesIO(xlsx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/export/xlsx/{scan_id}")
def export_xlsx_from_history(scan_id: int):
    """Export a previously saved scan as XLSX (no Gemini call)."""
    data = get_scan(scan_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found.")

    xlsx_bytes = build_xlsx(
        onions=data.get("onions") or [],
        grade_result=data.get("grade_result") or {},
        batch_id=data.get("batch_id") or "unknown",
        analysis_mode=data.get("inspection_mode") or "rgb",
        demo_mode=bool(data.get("demo_mode")),
        vision_model=(data.get("system") or {}).get("vision_model") or "",
    )

    if xlsx_bytes is None:
        raise HTTPException(status_code=500, detail="XLSX generation failed.")

    filename = f"oniongrade_{data.get('batch_id', scan_id)}.xlsx"
    return StreamingResponse(
        io.BytesIO(xlsx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
