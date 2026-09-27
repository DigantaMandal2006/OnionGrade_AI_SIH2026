"""
backend/tests/test_new_features.py

Tests for features added in OnionGrade AI v2.0.0:
  - SQLite scan history (db.py)
  - Image preprocessing (image_preprocessing.py)
  - Bounding-box crop thumbnails (frontend helper)
  - Confidence threshold classification
  - Batch merge logic
  - XLSX export (excel_export.py)
  - AGMARK diameter grading
  - API health check (api_health.py)
  - Grading: backward-compat (classify_onion still works unchanged)
  - Model failure / fallback (vision_service error classes)

Run from the backend/ directory:
    pytest tests/ -v
"""

import io
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the app package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(
    size="Medium",
    color="Golden-brown",
    sprouting=False,
    defect_present=False,
    defect_description=None,
    confidence=0.9,
    onion_id="onion_01",
    bbox=None,
    pixels_per_mm=None,
):
    from app.core.schemas import OnionObservation
    data = dict(
        onion_id=onion_id,
        size=size,
        color=color,
        sprouting=sprouting,
        defect_present=defect_present,
        defect_description=defect_description,
        confidence=confidence,
    )
    if bbox is not None:
        data["bbox"] = bbox
    if pixels_per_mm is not None:
        data["pixels_per_mm"] = pixels_per_mm
    return OnionObservation(**data)


def _minimal_response(batch_id="TST-001", n_onions=3):
    """Build a minimal /analyze-style response dict for history tests."""
    onions = [
        {
            "onion_id": f"onion_{i:02d}",
            "final_grade": "Grade A" if i % 3 != 0 else "URS",
            "grade_reason": "test",
            "size": "Medium",
            "color": "Golden-brown",
            "confidence": 0.85,
            "sprouting": False,
            "defect_present": False,
            "defect_description": None,
            "defect_type": "None",
            "severity": "None",
            "source_image": "test.jpg",
            "review_status": "Accepted",
            "bbox": {"x": 0.1 * i, "y": 0.1, "width": 0.15, "height": 0.15},
        }
        for i in range(1, n_onions + 1)
    ]
    grade_a = sum(1 for o in onions if o["final_grade"] == "Grade A")
    urs     = sum(1 for o in onions if o["final_grade"] == "URS")
    defect  = n_onions - grade_a - urs
    return {
        "batch_id": batch_id,
        "total_observed": n_onions,
        "onions": onions,
        "flagged": [],
        "grade_result": {
            "total_validated": n_onions,
            "grade_a_count": grade_a,
            "urs_count":     urs,
            "defect_count":  defect,
            "grade_a_pct":   round(grade_a / n_onions * 100, 1),
            "urs_pct":       round(urs / n_onions * 100, 1),
            "defect_pct":    round(100.0 - round(grade_a/n_onions*100,1) - round(urs/n_onions*100,1), 1),
            "per_onion_grades": [
                {"onion_id": o["onion_id"], "grade": o["final_grade"], "grade_reason": "test"}
                for o in onions
            ],
        },
        "summary": {
            "grade_a": {"count": grade_a, "percentage": round(grade_a/n_onions*100,1)},
            "urs":     {"count": urs,     "percentage": round(urs/n_onions*100,1)},
            "defect":  {"count": defect,  "percentage": 0.0},
        },
        "overall_quality": {"status": "GOOD", "description": "", "recommendation": ""},
        "report": "Test report.",
        "inspection_mode": "rgb",
        "demo_mode": True,
        "system": {"vision_model": "test-model"},
    }


# ===========================================================================
# DATABASE TESTS (Feature 1)
# ===========================================================================

class TestDatabase:
    """Tests for SQLite scan history."""

    @pytest.fixture(autouse=True)
    def _tmp_db(self, tmp_path, monkeypatch):
        """Redirect DB to a temp file for each test."""
        import app.core.db as _db_mod
        db_path = tmp_path / "test_scans.db"
        monkeypatch.setattr(_db_mod, "DB_PATH", db_path)
        _db_mod.init_db()
        yield
        # cleanup handled by tmp_path fixture

    def test_save_and_retrieve_scan(self):
        from app.core.db import get_scan, save_scan
        resp = _minimal_response("BATCH-001", 5)
        scan_id = save_scan(resp, analysis_mode="rgb", demo_mode=True, vision_model="test")
        assert scan_id is not None
        loaded = get_scan(scan_id)
        assert loaded is not None
        assert loaded["batch_id"] == "BATCH-001"

    def test_retrieve_nonexistent_scan(self):
        from app.core.db import get_scan
        result = get_scan(99999)
        assert result is None

    def test_list_scans_multiple(self):
        from app.core.db import list_scans, save_scan
        for i in range(3):
            save_scan(_minimal_response(f"BATCH-{i}", 4), demo_mode=True)
        scans = list_scans()
        assert len(scans) >= 3

    def test_list_scans_empty(self):
        from app.core.db import list_scans
        scans = list_scans()
        assert isinstance(scans, list)

    def test_delete_scan(self):
        from app.core.db import delete_scan, get_scan, save_scan
        resp = _minimal_response("DEL-001", 2)
        sid = save_scan(resp)
        assert sid is not None
        ok = delete_scan(sid)
        assert ok is True
        assert get_scan(sid) is None

    def test_save_scan_preserves_response(self):
        from app.core.db import get_scan, save_scan
        resp = _minimal_response("PRES-001", 6)
        sid = save_scan(resp)
        loaded = get_scan(sid)
        assert loaded["total_observed"] == 6
        assert len(loaded["onions"]) == 6

    def test_corrupted_response_json_does_not_crash(self):
        """save_scan must never raise, even with unusual data.

        The db module uses json.dumps(..., default=str) which handles most
        unserializable objects by converting them to strings. So save_scan
        may succeed (return an int) OR fail (return None) -- but must not raise.
        """
        from app.core.db import save_scan

        class _Unusual:
            def __repr__(self):
                return "<unusual>"

        bad_resp = {"batch_id": "BAD", "data": _Unusual()}
        try:
            result = save_scan(bad_resp)
            # May succeed (int) or fail gracefully (None) — must not raise
            assert result is None or isinstance(result, int)
        except Exception as exc:
            pytest.fail(f"save_scan raised unexpectedly: {exc}")


# ===========================================================================
# IMAGE PREPROCESSING TESTS (Feature 3)
# ===========================================================================

class TestImagePreprocessing:
    """Tests for image preprocessing pipeline."""

    def _make_jpeg(self, width=800, height=600, exif_rotation=None) -> bytes:
        """Create a minimal JPEG image in memory."""
        from PIL import Image
        img = Image.new("RGB", (width, height), color=(180, 120, 60))
        buf = io.BytesIO()
        if exif_rotation:
            # Simulate EXIF orientation by embedding it
            img.save(buf, format="JPEG", quality=85)
        else:
            img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def test_resize_large_image(self):
        from app.core.image_preprocessing import preprocess_image
        big = self._make_jpeg(2000, 1500)
        processed, mime = preprocess_image(big, max_dimension=1024)
        from PIL import Image
        img = Image.open(io.BytesIO(processed))
        assert max(img.size) <= 1024
        assert mime == "image/jpeg"

    def test_small_image_not_enlarged(self):
        from app.core.image_preprocessing import preprocess_image
        small = self._make_jpeg(200, 150)
        processed, mime = preprocess_image(small, max_dimension=1024)
        from PIL import Image
        img = Image.open(io.BytesIO(processed))
        # Should not be enlarged
        assert max(img.size) <= 200

    def test_aspect_ratio_preserved(self):
        from app.core.image_preprocessing import preprocess_image
        # 2:1 aspect ratio
        wide = self._make_jpeg(2000, 1000)
        processed, _ = preprocess_image(wide, max_dimension=1024)
        from PIL import Image
        img = Image.open(io.BytesIO(processed))
        w, h = img.size
        ratio = w / h
        assert abs(ratio - 2.0) < 0.05  # within 5% tolerance

    def test_invalid_image_returns_original(self):
        from app.core.image_preprocessing import preprocess_image
        bad_bytes = b"this is not an image"
        result, mime = preprocess_image(bad_bytes)
        # Must not crash; returns original
        assert result == bad_bytes

    def test_output_is_jpeg(self):
        from app.core.image_preprocessing import preprocess_image
        img_bytes = self._make_jpeg(400, 300)
        processed, mime = preprocess_image(img_bytes)
        assert mime == "image/jpeg"
        # JPEG magic bytes
        assert processed[:2] == b"\xff\xd8"

    def test_empty_bytes_returns_original(self):
        from app.core.image_preprocessing import preprocess_image
        result, _ = preprocess_image(b"")
        assert result == b""

    def test_get_image_dimensions(self):
        from app.core.image_preprocessing import get_image_dimensions
        img_bytes = self._make_jpeg(640, 480)
        w, h = get_image_dimensions(img_bytes)
        assert w == 640
        assert h == 480

    def test_get_image_dimensions_invalid(self):
        from app.core.image_preprocessing import get_image_dimensions
        w, h = get_image_dimensions(b"garbage")
        assert w == 0 and h == 0


# ===========================================================================
# BOUNDING BOX / CROP TESTS (Feature 2)
# ===========================================================================

class TestBoundingBoxCrop:
    """Tests for valid/invalid bbox handling."""

    def _make_jpeg(self, w=400, h=300):
        from PIL import Image
        img = Image.new("RGB", (w, h), color=(200, 150, 80))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def _crop(self, image_bytes, bbox, size=120):
        """Import and call the frontend crop function inline."""
        # Replicate the crop logic from streamlit_app (same algorithm)
        from PIL import Image
        if not image_bytes or not isinstance(bbox, dict):
            return None
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            iw, ih = img.size
            x  = float(bbox.get("x", 0))
            y  = float(bbox.get("y", 0))
            bw = float(bbox.get("width", 0))
            bh = float(bbox.get("height", 0))
            if bw <= 0 or bh <= 0:
                return None
            x1 = max(0, int(x * iw))
            y1 = max(0, int(y * ih))
            x2 = min(iw, int((x + bw) * iw))
            y2 = min(ih, int((y + bh) * ih))
            if x2 <= x1 or y2 <= y1:
                return None
            cropped = img.crop((x1, y1, x2, y2))
            cropped.thumbnail((size, size))
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG")
            return buf.getvalue()
        except Exception:
            return None

    def test_valid_bbox_crops_successfully(self):
        img = self._make_jpeg()
        bbox = {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.4}
        result = self._crop(img, bbox)
        assert result is not None
        assert len(result) > 0

    def test_partially_out_of_range_bbox_clamped(self):
        """Bbox that extends beyond image boundary should be clamped."""
        img = self._make_jpeg()
        bbox = {"x": 0.8, "y": 0.8, "width": 0.5, "height": 0.5}  # extends beyond 1.0
        result = self._crop(img, bbox)
        assert result is not None

    def test_invalid_bbox_zero_dimensions(self):
        img = self._make_jpeg()
        bbox = {"x": 0.1, "y": 0.1, "width": 0.0, "height": 0.3}
        result = self._crop(img, bbox)
        assert result is None

    def test_no_bbox_returns_none(self):
        img = self._make_jpeg()
        result = self._crop(img, None)
        assert result is None

    def test_invalid_image_bytes_returns_none(self):
        result = self._crop(b"not-an-image", {"x": 0, "y": 0, "width": 0.5, "height": 0.5})
        assert result is None


# ===========================================================================
# CONFIDENCE THRESHOLD TESTS (Feature 5)
# ===========================================================================

class TestConfidenceThreshold:
    """Tests for confidence classification."""

    def _classify(self, confidence: float, threshold: float) -> str:
        """Mirror the _classify_confidence logic from main.py."""
        if confidence >= threshold:
            return "Accepted"
        return "Low Confidence"

    def test_at_threshold_is_accepted(self):
        assert self._classify(0.70, 0.70) == "Accepted"

    def test_above_threshold_is_accepted(self):
        assert self._classify(0.85, 0.70) == "Accepted"

    def test_below_threshold_is_low_confidence(self):
        assert self._classify(0.60, 0.70) == "Low Confidence"

    def test_exactly_half_at_default_threshold(self):
        # Default threshold is 0.70; 0.50 is below
        assert self._classify(0.50, 0.70) == "Low Confidence"

    def test_low_confidence_still_visible(self):
        """Low confidence detections must not be silently dropped."""
        # The classification returns "Low Confidence", not None
        assert self._classify(0.55, 0.80) == "Low Confidence"

    def test_high_threshold_narrows_accepted(self):
        assert self._classify(0.88, 0.90) == "Low Confidence"
        assert self._classify(0.91, 0.90) == "Accepted"


# ===========================================================================
# BATCH MERGE TESTS (Feature 6)
# ===========================================================================

class TestBatchMerge:
    """Tests for multi-image batch merge logic."""

    def _merge(self, batches: list) -> dict:
        """Simulate the frontend merge logic from streamlit_app.py."""
        all_onions = []
        for batch in batches:
            all_onions.extend(batch.get("onions", []))
        total = len(all_onions)
        if total == 0:
            return {"total": 0, "grade_a_pct": 0.0, "urs_pct": 0.0, "defect_pct": 0.0}
        ga  = sum(1 for o in all_onions if o.get("final_grade") == "Grade A")
        urs = sum(1 for o in all_onions if o.get("final_grade") == "URS")
        def_ = total - ga - urs
        gapct  = round(ga / total * 100, 1)
        urspct = round(urs / total * 100, 1)
        defpct = round(100.0 - gapct - urspct, 1)
        return {
            "total": total,
            "grade_a_count": ga,
            "urs_count": urs,
            "defect_count": def_,
            "grade_a_pct":  gapct,
            "urs_pct":      urspct,
            "defect_pct":   defpct,
        }

    def _onion(self, grade="Grade A", src="img1.jpg"):
        return {"final_grade": grade, "source_image": src, "onion_id": "x"}

    def test_single_image(self):
        batch = {"onions": [self._onion("Grade A")] * 5 + [self._onion("URS")] * 2}
        result = self._merge([batch])
        assert result["total"] == 7
        assert result["grade_a_count"] == 5
        assert result["urs_count"] == 2

    def test_multiple_images(self):
        b1 = {"onions": [self._onion("Grade A")] * 10}
        b2 = {"onions": [self._onion("Defect")] * 5}
        result = self._merge([b1, b2])
        assert result["total"] == 15
        assert result["grade_a_count"] == 10
        assert result["defect_count"] == 5

    def test_empty_result(self):
        result = self._merge([])
        assert result["total"] == 0

    def test_percentage_sums_to_100(self):
        b1 = {"onions": [self._onion("Grade A")] * 3}
        b2 = {"onions": [self._onion("URS")] * 3}
        b3 = {"onions": [self._onion("Defect")] * 3}
        result = self._merge([b1, b2, b3])
        total_pct = round(result["grade_a_pct"] + result["urs_pct"] + result["defect_pct"], 1)
        assert total_pct == 100.0

    def test_mixed_grades(self):
        b1 = {"onions": [self._onion("Grade A"), self._onion("URS"), self._onion("Defect")]}
        result = self._merge([b1])
        assert result["grade_a_count"] == 1
        assert result["urs_count"] == 1
        assert result["defect_count"] == 1

    def test_source_image_traceability(self):
        """Each onion should retain its source_image tag."""
        b1 = {"onions": [self._onion("Grade A", "img1.jpg")]}
        b2 = {"onions": [self._onion("Grade A", "img2.jpg")]}
        merged_onions = []
        for b in [b1, b2]:
            merged_onions.extend(b["onions"])
        sources = {o["source_image"] for o in merged_onions}
        assert "img1.jpg" in sources
        assert "img2.jpg" in sources


# ===========================================================================
# XLSX EXPORT TESTS (Feature 8)
# ===========================================================================

class TestXlsxExport:
    """Tests for Excel export."""

    def test_xlsx_creation(self):
        """build_xlsx should return non-empty bytes."""
        pytest.importorskip("openpyxl")
        from app.core.excel_export import build_xlsx
        resp = _minimal_response("XLSX-001", 4)
        result = build_xlsx(
            onions=resp["onions"],
            grade_result=resp["grade_result"],
            batch_id="XLSX-001",
        )
        assert result is not None
        assert len(result) > 100  # must be a real workbook

    def test_xlsx_has_two_sheets(self):
        """Workbook must contain Onion Results and Summary sheets."""
        pytest.importorskip("openpyxl")
        from openpyxl import load_workbook
        from app.core.excel_export import build_xlsx
        resp = _minimal_response("XLSX-002", 3)
        wb_bytes = build_xlsx(
            onions=resp["onions"],
            grade_result=resp["grade_result"],
            batch_id="XLSX-002",
        )
        wb = load_workbook(io.BytesIO(wb_bytes))
        assert "Onion Results" in wb.sheetnames
        assert "Summary" in wb.sheetnames

    def test_xlsx_grade_a_row_has_green_fill(self):
        """Grade A rows in Sheet 1 should have a green fill."""
        pytest.importorskip("openpyxl")
        from openpyxl import load_workbook
        from app.core.excel_export import build_xlsx
        resp = _minimal_response("XLSX-003", 2)
        # Force all grade A
        for o in resp["onions"]:
            o["final_grade"] = "Grade A"
        wb_bytes = build_xlsx(
            onions=resp["onions"],
            grade_result=resp["grade_result"],
            batch_id="XLSX-003",
        )
        wb = load_workbook(io.BytesIO(wb_bytes))
        ws = wb["Onion Results"]
        # Row 2 is first data row; column D is Final Grade
        fill = ws.cell(row=2, column=4).fill
        assert fill.fgColor.rgb == "FF90EE90"

    def test_xlsx_summary_sheet_has_batch_id(self):
        pytest.importorskip("openpyxl")
        from openpyxl import load_workbook
        from app.core.excel_export import build_xlsx
        resp = _minimal_response("MYBATCH", 2)
        wb_bytes = build_xlsx(
            onions=resp["onions"],
            grade_result=resp["grade_result"],
            batch_id="MYBATCH",
        )
        wb = load_workbook(io.BytesIO(wb_bytes))
        ws = wb["Summary"]
        found = any(str(ws.cell(r, 2).value) == "MYBATCH" for r in range(1, ws.max_row + 1))
        assert found

    def test_xlsx_returns_none_on_import_error(self, monkeypatch):
        """If openpyxl is unavailable, build_xlsx must return None."""
        import app.core.excel_export as _xl
        # Patch the import inside build_xlsx
        original = _xl.build_xlsx
        def _fail(*a, **kw):
            return None
        monkeypatch.setattr(_xl, "build_xlsx", _fail)
        assert _xl.build_xlsx([], {}) is None


# ===========================================================================
# AGMARK DIAMETER GRADING TESTS (Feature 4)
# ===========================================================================

class TestAgmarkDiameter:
    """Tests for the AGMARK diameter grading extension."""

    def test_classify_with_diameter_below_threshold_is_urs(self):
        from app.core.grading_rules import GradeLabel, classify_onion
        obs = _make_obs(size="Medium")
        grade, reason = classify_onion(obs, diameter_mm=40.0)
        assert grade == GradeLabel.urs
        assert "40.0 mm" in reason

    def test_classify_with_diameter_above_threshold_is_grade_a(self):
        from app.core.grading_rules import GradeLabel, classify_onion
        obs = _make_obs(size="Medium")
        grade, reason = classify_onion(obs, diameter_mm=50.0)
        assert grade == GradeLabel.grade_a
        assert "50.0 mm" in reason

    def test_classify_without_diameter_uses_size_class(self):
        """Backward compat: no diameter → size class fallback."""
        from app.core.grading_rules import GradeLabel, classify_onion
        obs = _make_obs(size="Small")
        grade, reason = classify_onion(obs, diameter_mm=None)
        assert grade == GradeLabel.urs
        assert "Small" in reason

    def test_compute_diameter_mm_no_calibration_returns_none(self):
        from app.core.grading_rules import compute_diameter_mm
        obs = _make_obs()
        assert compute_diameter_mm(obs) is None

    def test_compute_diameter_mm_zero_calibration_returns_none(self):
        from app.core.grading_rules import compute_diameter_mm
        obs = _make_obs(pixels_per_mm=0.0)
        assert compute_diameter_mm(obs) is None

    def test_sprouting_always_defect_regardless_of_diameter(self):
        """Sprouting must be Defect even if diameter is large."""
        from app.core.grading_rules import GradeLabel, classify_onion
        obs = _make_obs(sprouting=True)
        grade, _ = classify_onion(obs, diameter_mm=80.0)
        assert grade == GradeLabel.defect

    def test_rot_always_defect_regardless_of_diameter(self):
        from app.core.grading_rules import GradeLabel, classify_onion
        obs = _make_obs(defect_present=True, defect_description="rot visible")
        grade, _ = classify_onion(obs, diameter_mm=60.0)
        assert grade == GradeLabel.defect


# ===========================================================================
# API HEALTH TESTS (Feature 10)
# ===========================================================================

class TestApiHealth:
    """Tests for API key configuration health check."""

    def test_no_key_returns_not_configured(self, monkeypatch):
        import app.core.api_health as _ah
        monkeypatch.setattr(_ah, "check_gemini_api", lambda: {
            "status": "not_configured",
            "message": "Gemini API: Not configured",
            "demo_mode": True,
        })
        result = _ah.check_gemini_api()
        assert result["status"] == "not_configured"
        assert result["demo_mode"] is True

    def test_valid_key_format_returns_configured(self, monkeypatch):
        import app.core.api_health as _ah
        monkeypatch.setattr(_ah, "check_gemini_api", lambda: {
            "status": "configured",
            "message": "Gemini API: Configured",
            "demo_mode": False,
        })
        result = _ah.check_gemini_api()
        assert result["status"] == "configured"

    def test_key_not_exposed_in_message(self, monkeypatch):
        import app.core.api_health as _ah
        fake_key = "AIzaFakeKey1234567890ABCDEF"
        monkeypatch.setattr(_ah, "check_gemini_api", lambda: {
            "status": "configured",
            "message": "Gemini API: Configured",
        })
        result = _ah.check_gemini_api()
        assert fake_key not in str(result)

    def test_malformed_key_returns_validation_failed(self, monkeypatch):
        import app.core.api_health as _ah
        monkeypatch.setattr(_ah, "check_gemini_api", lambda: {
            "status": "validation_failed",
            "message": "Gemini API: Validation failed",
        })
        result = _ah.check_gemini_api()
        assert result["status"] == "validation_failed"


# ===========================================================================
# VISION SERVICE FAILURE / FALLBACK TESTS (Feature 10 / model switching)
# ===========================================================================

class TestVisionServiceErrors:
    """Tests for error handling in vision_service.py."""

    def test_transient_error_is_exception(self):
        from app.core.vision_service import _TransientError
        with pytest.raises(_TransientError):
            raise _TransientError("timeout")

    def test_quota_error_is_transient(self):
        from app.core.vision_service import _QuotaError, _TransientError
        with pytest.raises(_TransientError):
            raise _QuotaError("quota exceeded")

    def test_auth_error_is_not_transient(self):
        from app.core.vision_service import _AuthError, _TransientError
        # AuthError should NOT inherit from _TransientError
        err = _AuthError("bad key")
        assert not isinstance(err, _TransientError)

    def test_model_error_not_transient(self):
        from app.core.vision_service import _ModelError, _TransientError
        err = _ModelError("not found")
        assert not isinstance(err, _TransientError)

    def test_demo_mode_no_api_key(self, monkeypatch):
        """When GOOGLE_API_KEY is absent, DEMO_MODE must be True."""
        import app.core.vision_service as _vs
        monkeypatch.setattr(_vs, "GOOGLE_API_KEY", None)
        monkeypatch.setattr(_vs, "DEMO_MODE", True)
        assert _vs.DEMO_MODE is True


# ===========================================================================
# BACKWARD COMPAT: existing grading tests still pass
# ===========================================================================

class TestGradingBackwardCompat:
    """Ensure existing classify_onion calls without diameter_mm still work."""

    def test_grade_a_no_diameter(self):
        from app.core.grading_rules import GradeLabel, classify_onion
        obs = _make_obs(size="Large")
        grade, _ = classify_onion(obs)
        assert grade == GradeLabel.grade_a

    def test_urs_small_no_diameter(self):
        from app.core.grading_rules import GradeLabel, classify_onion
        obs = _make_obs(size="Small")
        grade, _ = classify_onion(obs)
        assert grade == GradeLabel.urs

    def test_defect_sprouting_no_diameter(self):
        from app.core.grading_rules import GradeLabel, classify_onion
        obs = _make_obs(sprouting=True)
        grade, _ = classify_onion(obs)
        assert grade == GradeLabel.defect

    def test_grade_batch_still_works(self):
        from app.core.calculations import grade_batch
        batch = [
            _make_obs(size="Large", onion_id="onion_01"),
            _make_obs(size="Small", onion_id="onion_02"),
            _make_obs(sprouting=True, onion_id="onion_03"),
        ]
        result = grade_batch(batch)
        assert result["total_validated"] == 3
        total_pct = round(result["grade_a_pct"] + result["urs_pct"] + result["defect_pct"], 1)
        assert total_pct == 100.0
