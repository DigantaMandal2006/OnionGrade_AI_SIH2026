"""
Streamlit frontend for OnionGrade AI.

Version: 1.0.0

Features:
    - Batch image upload
    - FastAPI backend connection
    - Grade A / URS / Defect dashboard cards
    - Overall batch-quality assessment
    - Defect-category analysis
    - Per-onion grading table
    - AI confidence percentage
    - Flagged records
    - Bar chart
    - Digital report
    - CSV export
    - JSON export
    - PDF export

Architecture:
    Streamlit
        -> FastAPI /analyze
        -> Gemini Vision perception
        -> Pydantic validation
        -> Deterministic Python grading
        -> Batch assessment
        -> LangChain report
"""

import csv
import io
import json
import math
import os
from datetime import datetime
from html import escape

import plotly.graph_objects as go
import requests
import streamlit as st

from PIL import Image, ImageDraw, ImageFont
import re

try:
    import openpyxl as _openpyxl_avail
    _XLSX_AVAILABLE = True
except ImportError:
    _XLSX_AVAILABLE = False

from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000",
)

st.set_page_config(
    page_title="OnionGrade AI",
    page_icon="\U0001F9C5",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

THEME = {
    "plum": "#5B2333",
    "plum_deep": "#431A26",
    "copper": "#B5651D",
    "copper_deep": "#8F4D15",
    "gold": "#D9A441",
    "sage": "#4E7A52",
    "sage_soft": "#E4EEE1",
    "rust": "#A23E2E",
    "rust_soft": "#F6E4DF",
    "line": "#E2D5C0",
    "ink": "#2B1B12",
    "ink_soft": "#6B5A4C",
}

# ---------------------------------------------------------------------------
# Language / i18n
# ---------------------------------------------------------------------------

_TRANSLATIONS = {
    "en": {
        "app_subtitle":       "AI perceives, Python decides — onion quality grading pipeline",
        "upload_heading":     "Upload onion batch photo",
        "upload_caption":     "Upload a clear batch image (JPEG, PNG or WebP, max 20 MB). Gemini Vision perceives size, colour, sprouting and visible defects.",
        "camera_heading":     "Take a photo",
        "camera_caption":     "Capture directly from camera — no file save needed.",
        "batch_id_label":     "Batch ID (optional)",
        "batch_id_help":      "Enter your own batch number, e.g. LOT-MH-001. Leave blank to auto-generate.",
        "analyse_btn":        "Analyse batch →",
        "new_analysis_btn":   "🔄 New analysis",
        "grade_a":            "Grade A",
        "urs":                "URS",
        "defect":             "Defect",
        "sprouting":          "Sprouting",
        "rot":                "Rot",
        "mould":              "Mould",
        "bruising":           "Bruising",
        "discoloration":      "Discoloration",
        "decay":              "Decay",
        "physical_damage":    "Physical Damage",
        "none":               "None",
        "other":              "Other",
        "onions":             "onions",
        "per_onion_heading":  "Per-onion grading record",
        "per_onion_caption":  "The number shown in the image corresponds directly to the onion ID in this table.",
        "override_col":       "Override Grade",
        "override_note":      "Change a grade — recalculation happens instantly. Overridden rows are marked in the PDF.",
        "rule_engine_title":  "⚙️ How are grades decided? — View the rule engine",
        "defect_heading":     "Defect analysis",
        "quality_heading":    "Overall batch assessment",
        "confidence_heading": "AI perception confidence — per onion",
        "confidence_caption": "Each dot is one onion. Dashed line = validation floor (50%). Below floor = flagged.",
        "report_heading":     "Digital report",
        "export_heading":     "Export results",
        "export_csv":         "⬇️ Export CSV",
        "export_json":        "⬇️ Export JSON",
        "export_pdf":         "📄 Export PDF",
        "history_heading":    "Batch history",
        "history_empty":      "No previous batches in this session.",
        "compare_heading":    "Batch comparison",
        "trend_heading":      "Grade A% trend",
        "img_quality_warn":   "⚠️ Image quality may be low (blurry or dark). Results may be less accurate.",
        "status_good":        "GOOD",
        "status_acceptable":  "ACCEPTABLE",
        "status_poor":        "POOR",
        "recommendation":     "Recommendation",
        "batch_quality":      "Batch quality status",
        "flagged_title":      "flagged & excluded from grading",
        "flagged_caption":    "Flagged records are excluded from deterministic grading because their observations did not pass validation.",
        "no_defects":         "✓ No defective onions detected",
        "no_defects_sub":     "No defective onion records were identified in the validated batch.",
        "results_placeholder":"Results will appear here",
        "results_sub":        "Upload a batch photo and select Analyse batch. Grade distribution, quality assessment, defect analysis and per-onion results will appear here.",
        "grade_reason_col":   "Grade Reason",
        "image_num_col":      "#",
        "onion_id_col":       "Onion ID",
        "size_col":           "Size",
        "colour_col":         "Colour",
        "severity_col":       "Severity",
        "defect_col":         "Defect",
        "defect_desc_col":    "Defect Description",
        "confidence_col":     "Confidence",
        "spinner_msg":        "Gemini Vision perceiving → validating → grading → analysing defects → generating report…",
        "final_grade_col":    "Final Grade",
        "inspection_mode_label": "Inspection Mode",
        "mode_rgb":           "🔴 External RGB grading",
        "mode_nir":           "🔵 Internal NIR screening",
        "mode_rgb_desc":      "Grades onions on external surface quality — size, colour, skin condition, sprouting and visible defects captured under standard visible light.",
        "mode_nir_desc":      "Screens onions for internal defects — hollow heart, internal rot and moisture loss invisible to the naked eye (simulated from image texture when a real NIR sensor is absent).",
        # stepper labels
        "step_upload":        "Upload",
        "step_perceive":      "Perceive",
        "step_validate":      "Validate",
        "step_grade":         "Grade",
        "step_report":        "Report",
        # processing progress
        "prog_uploading":     "⬆ Uploading",
        "prog_processing":    "⚙ Processing",
        "prog_analyzing":     "🔍 Analyzing",
        "prog_preparing":     "📊 Preparing",
        "prog_complete":      "✓ Complete",
        # CV mode
        "cv_mode_heading":    "CV Summary",
        "cv_batch_info":      "Batch Information",
        "cv_grade_breakdown": "Grade Breakdown",
        "cv_top_defects":     "Top Defects",
        "cv_ai_confidence":   "AI Confidence",
        "cv_quick_actions":   "Quick Actions",
        "cv_avg_confidence":  "Average confidence",
        "cv_highest_conf":    "Highest confidence onion",
        "cv_lowest_conf":     "Lowest confidence onion",
        "cv_flagged_count":   "Flagged records",
        "cv_batch_status":    "Batch status",
        "cv_total_onions":    "Total onions",
        "cv_export_hint":     "Use the export buttons below to download CSV, JSON or PDF.",
        # camera UI
        "camera_closed_title":"Camera is closed",
        "camera_closed_sub":  "Nothing is activated yet — click below to turn on your camera and capture a photo.",
        "camera_open_btn":    "📷 Open Camera",
        "camera_close_btn":   "✕ Close Camera",
        # health messages
        "health_offline":     "**Offline mode** — no Gemini API key detected. Running local OpenCV analysis.",
        "health_online":      "**Gemini Vision active** — AI-powered per-onion perception is enabled.",
        "health_error":       "Cannot reach the backend. Start it with `uvicorn app.main:app --reload`.",
        # misc
        "no_validated_records": "No validated onion records available.",
        "select_rgb":         "Select RGB",
        "rgb_selected":       "✓ RGB selected",
        "select_nir":         "Select NIR",
        "nir_selected":       "✓ NIR selected",
        "excluded_label":     "Excluded",
        "overridden_note":    "grade(s) manually overridden",
    },
    "hi": {
        "app_subtitle":       "AI देखता है, Python तय करता है — प्याज गुणवत्ता ग्रेडिंग",
        "upload_heading":     "प्याज बैच की फ़ोटो अपलोड करें",
        "upload_caption":     "स्पष्ट बैच फ़ोटो अपलोड करें (JPEG, PNG या WebP, अधिकतम 20 MB)।",
        "camera_heading":     "फ़ोटो लें",
        "camera_caption":     "सीधे कैमरे से फ़ोटो लें — फ़ाइल सेव करने की ज़रूरत नहीं।",
        "batch_id_label":     "बैच ID (वैकल्पिक)",
        "batch_id_help":      "अपना बैच नंबर डालें, जैसे LOT-MH-001। खाली छोड़ें तो स्वचालित बनेगा।",
        "analyse_btn":        "बैच विश्लेषण करें →",
        "new_analysis_btn":   "🔄 नया विश्लेषण",
        "grade_a":            "श्रेणी A",
        "urs":                "URS",
        "defect":             "दोषपूर्ण",
        "sprouting":          "अंकुरण",
        "rot":                "सड़न",
        "mould":              "फफूंद",
        "bruising":           "चोट के निशान",
        "discoloration":      "रंग परिवर्तन",
        "decay":              "क्षय",
        "physical_damage":    "भौतिक क्षति",
        "none":               "कोई नहीं",
        "other":              "अन्य",
        "onions":             "प्याज",
        "per_onion_heading":  "प्रति प्याज ग्रेडिंग रिकॉर्ड",
        "per_onion_caption":  "छवि में दिखाया नंबर इस तालिका की पंक्ति से मेल खाता है।",
        "override_col":       "ग्रेड बदलें",
        "override_note":      "ग्रेड बदलें — प्रतिशत तुरंत पुनः गणना होगी। बदले गए रिकॉर्ड PDF में चिह्नित होंगे।",
        "rule_engine_title":  "⚙️ ग्रेड कैसे तय होता है? — नियम इंजन देखें",
        "defect_heading":     "दोष विश्लेषण",
        "quality_heading":    "समग्र बैच मूल्यांकन",
        "confidence_heading": "AI धारणा विश्वास — प्रति प्याज",
        "confidence_caption": "प्रत्येक बिंदु एक प्याज है। 50% से नीचे = अमान्य।",
        "report_heading":     "डिजिटल रिपोर्ट",
        "export_heading":     "परिणाम निर्यात करें",
        "export_csv":         "⬇️ CSV निर्यात",
        "export_json":        "⬇️ JSON निर्यात",
        "export_pdf":         "📄 PDF निर्यात",
        "history_heading":    "बैच इतिहास",
        "history_empty":      "इस सत्र में कोई पिछला बैच नहीं।",
        "compare_heading":    "बैच तुलना",
        "trend_heading":      "श्रेणी A% प्रवृत्ति",
        "img_quality_warn":   "⚠️ छवि गुणवत्ता कम हो सकती है (धुंधली या अंधेरी)। परिणाम कम सटीक हो सकते हैं।",
        "status_good":        "अच्छा",
        "status_acceptable":  "स्वीकार्य",
        "status_poor":        "खराब",
        "recommendation":     "सिफारिश",
        "batch_quality":      "बैच गुणवत्ता स्थिति",
        "flagged_title":      "अमान्य रिकॉर्ड",
        "flagged_caption":    "ये रिकॉर्ड सत्यापन पास न करने के कारण ग्रेडिंग से बाहर हैं।",
        "no_defects":         "✓ कोई दोषपूर्ण प्याज नहीं",
        "no_defects_sub":     "सत्यापित बैच में कोई दोषपूर्ण रिकॉर्ड नहीं मिला।",
        "results_placeholder":"परिणाम यहाँ दिखेंगे",
        "results_sub":        "बैच फ़ोटो अपलोड करें और विश्लेषण करें चुनें।",
        "grade_reason_col":   "ग्रेड कारण",
        "image_num_col":      "#",
        "onion_id_col":       "प्याज ID",
        "size_col":           "आकार",
        "colour_col":         "रंग",
        "severity_col":       "गंभीरता",
        "defect_col":         "दोष",
        "defect_desc_col":    "दोष विवरण",
        "confidence_col":     "विश्वास",
        "spinner_msg":        "Gemini Vision विश्लेषण → सत्यापन → ग्रेडिंग → रिपोर्ट…",
        "final_grade_col":    "अंतिम श्रेणी",
        "inspection_mode_label": "निरीक्षण मोड",
        "mode_rgb":           "🔴 बाह्य RGB ग्रेडिंग",
        "mode_nir":           "🔵 आंतरिक NIR स्क्रीनिंग",
        "mode_rgb_desc":      "बाहरी सतह गुणवत्ता के आधार पर प्याज को ग्रेड करता है — आकार, रंग, छिलका, अंकुरण और दृश्य दोष।",
        "mode_nir_desc":      "आंतरिक दोषों की जांच — खोखला दिल, आंतरिक सड़न और नमी की कमी जो नंगी आंखों से नहीं दिखती।",
        "step_upload":        "अपलोड",
        "step_perceive":      "पहचान",
        "step_validate":      "सत्यापन",
        "step_grade":         "ग्रेडिंग",
        "step_report":        "रिपोर्ट",
        "prog_uploading":     "⬆ अपलोड हो रहा है",
        "prog_processing":    "⚙ प्रक्रिया हो रही है",
        "prog_analyzing":     "🔍 विश्लेषण हो रहा है",
        "prog_preparing":     "📊 तैयार किया जा रहा है",
        "prog_complete":      "✓ पूर्ण",
        "cv_mode_heading":    "सारांश दृश्य",
        "cv_batch_info":      "बैच जानकारी",
        "cv_grade_breakdown": "श्रेणी विवरण",
        "cv_top_defects":     "प्रमुख दोष",
        "cv_ai_confidence":   "AI विश्वास स्तर",
        "cv_quick_actions":   "त्वरित कार्य",
        "cv_avg_confidence":  "औसत विश्वास",
        "cv_highest_conf":    "सर्वाधिक विश्वस्त प्याज",
        "cv_lowest_conf":     "न्यूनतम विश्वस्त प्याज",
        "cv_flagged_count":   "अमान्य रिकॉर्ड",
        "cv_batch_status":    "बैच स्थिति",
        "cv_total_onions":    "कुल प्याज",
        "cv_export_hint":     "परिणाम डाउनलोड करने के लिए नीचे दिए निर्यात बटन का उपयोग करें।",
        "camera_closed_title":"कैमरा बंद है",
        "camera_closed_sub":  "अभी कुछ सक्रिय नहीं है — कैमरा चालू करने के लिए नीचे दबाएं।",
        "camera_open_btn":    "📷 कैमरा खोलें",
        "camera_close_btn":   "✕ कैमरा बंद करें",
        "health_offline":     "**ऑफलाइन मोड** — Gemini API कुंजी नहीं मिली। स्थानीय विश्लेषण चल रहा है।",
        "health_online":      "**Gemini Vision सक्रिय** — AI-संचालित प्रति-प्याज पहचान सक्षम है।",
        "health_error":       "बैकएंड से कनेक्ट नहीं हो सका। कृपया सर्वर शुरू करें।",
        "no_validated_records": "कोई सत्यापित प्याज रिकॉर्ड उपलब्ध नहीं।",
        "select_rgb":         "RGB चुनें",
        "rgb_selected":       "✓ RGB चयनित",
        "select_nir":         "NIR चुनें",
        "nir_selected":       "✓ NIR चयनित",
        "excluded_label":     "बाहर",
        "overridden_note":    "ग्रेड मैन्युअल रूप से बदले गए",
    },
    "mr": {
        "app_subtitle":       "AI पाहतो, Python ठरवतो — कांदा गुणवत्ता ग्रेडिंग",
        "upload_heading":     "कांदा बॅचचा फोटो अपलोड करा",
        "upload_caption":     "स्पष्ट बॅच फोटो अपलोड करा (JPEG, PNG किंवा WebP, कमाल 20 MB).",
        "camera_heading":     "फोटो काढा",
        "camera_caption":     "थेट कॅमेऱ्यातून फोटो काढा — फाइल सेव्ह करण्याची गरज नाही.",
        "batch_id_label":     "बॅच ID (पर्यायी)",
        "batch_id_help":      "तुमचा बॅच नंबर टाका, उदा. LOT-MH-001. रिकामे ठेवल्यास स्वयंचलित तयार होईल.",
        "analyse_btn":        "बॅच विश्लेषण करा →",
        "new_analysis_btn":   "🔄 नवीन विश्लेषण",
        "grade_a":            "श्रेणी A",
        "urs":                "URS",
        "defect":             "दोषपूर्ण",
        "sprouting":          "अंकुरण",
        "rot":                "कुजणे",
        "mould":              "बुरशी",
        "bruising":           "जखम",
        "discoloration":      "रंग बदल",
        "decay":              "क्षय",
        "physical_damage":    "शारीरिक नुकसान",
        "none":               "काहीही नाही",
        "other":              "इतर",
        "onions":             "कांदे",
        "per_onion_heading":  "प्रति कांदा ग्रेडिंग रेकॉर्ड",
        "per_onion_caption":  "प्रतिमेतील क्रमांक या तक्त्यातील पंक्तीशी जुळतो.",
        "override_col":       "श्रेणी बदला",
        "override_note":      "श्रेणी बदला — टक्केवारी लगेच पुन्हा मोजली जाईल. बदललेल्या नोंदी PDF मध्ये चिन्हांकित होतील.",
        "rule_engine_title":  "⚙️ श्रेणी कशी ठरवली जाते? — नियम इंजिन पहा",
        "defect_heading":     "दोष विश्लेषण",
        "quality_heading":    "एकूण बॅच मूल्यांकन",
        "confidence_heading": "AI धारणा विश्वास — प्रति कांदा",
        "confidence_caption": "प्रत्येक बिंदू एक कांदा आहे. 50% खाली = अवैध.",
        "report_heading":     "डिजिटल अहवाल",
        "export_heading":     "निकाल निर्यात करा",
        "export_csv":         "⬇️ CSV निर्यात",
        "export_json":        "⬇️ JSON निर्यात",
        "export_pdf":         "📄 PDF निर्यात",
        "history_heading":    "बॅच इतिहास",
        "history_empty":      "या सत्रात कोणताही मागील बॅच नाही.",
        "compare_heading":    "बॅच तुलना",
        "trend_heading":      "श्रेणी A% कल",
        "img_quality_warn":   "⚠️ प्रतिमा गुणवत्ता कमी असू शकते (धुक्याची किंवा गडद). निकाल कमी अचूक असू शकतात.",
        "status_good":        "उत्तम",
        "status_acceptable":  "स्वीकारार्ह",
        "status_poor":        "खराब",
        "recommendation":     "शिफारस",
        "batch_quality":      "बॅच गुणवत्ता स्थिती",
        "flagged_title":      "अवैध नोंदी",
        "flagged_caption":    "या नोंदी सत्यापन पास न केल्यामुळे ग्रेडिंगमधून वगळल्या आहेत.",
        "no_defects":         "✓ कोणतेही दोषपूर्ण कांदे आढळले नाहीत",
        "no_defects_sub":     "सत्यापित बॅचमध्ये कोणतीही दोषपूर्ण नोंद आढळली नाही.",
        "results_placeholder":"निकाल येथे दिसतील",
        "results_sub":        "बॅच फोटो अपलोड करा आणि विश्लेषण करा निवडा.",
        "grade_reason_col":   "श्रेणी कारण",
        "image_num_col":      "#",
        "onion_id_col":       "कांदा ID",
        "size_col":           "आकार",
        "colour_col":         "रंग",
        "severity_col":       "तीव्रता",
        "defect_col":         "दोष",
        "defect_desc_col":    "दोष वर्णन",
        "confidence_col":     "विश्वास",
        "spinner_msg":        "Gemini Vision विश्लेषण → सत्यापन → ग्रेडिंग → अहवाल…",
        "final_grade_col":    "अंतिम श्रेणी",
        "inspection_mode_label": "तपासणी मोड",
        "mode_rgb":           "🔴 बाह्य RGB ग्रेडिंग",
        "mode_nir":           "🔵 अंतर्गत NIR स्क्रीनिंग",
        "mode_rgb_desc":      "बाह्य पृष्ठभाग गुणवत्तेनुसार कांदा ग्रेड करतो — आकार, रंग, साल, अंकुरण आणि दृश्य दोष.",
        "mode_nir_desc":      "अंतर्गत दोष तपासतो — पोकळ मध्य, आतील कुजणे आणि ओलावा कमी जे उघड्या डोळ्यांनी दिसत नाही.",
        "step_upload":        "अपलोड",
        "step_perceive":      "ओळख",
        "step_validate":      "सत्यापन",
        "step_grade":         "श्रेणीकरण",
        "step_report":        "अहवाल",
        "prog_uploading":     "⬆ अपलोड होत आहे",
        "prog_processing":    "⚙ प्रक्रिया होत आहे",
        "prog_analyzing":     "🔍 विश्लेषण होत आहे",
        "prog_preparing":     "📊 तयार केले जात आहे",
        "prog_complete":      "✓ पूर्ण",
        "cv_mode_heading":    "सारांश दृश्य",
        "cv_batch_info":      "बॅच माहिती",
        "cv_grade_breakdown": "श्रेणी तपशील",
        "cv_top_defects":     "प्रमुख दोष",
        "cv_ai_confidence":   "AI विश्वास पातळी",
        "cv_quick_actions":   "त्वरित कृती",
        "cv_avg_confidence":  "सरासरी विश्वास",
        "cv_highest_conf":    "सर्वाधिक विश्वासार्ह कांदा",
        "cv_lowest_conf":     "किमान विश्वासार्ह कांदा",
        "cv_flagged_count":   "अवैध नोंदी",
        "cv_batch_status":    "बॅच स्थिती",
        "cv_total_onions":    "एकूण कांदे",
        "cv_export_hint":     "निकाल डाउनलोड करण्यासाठी खालील निर्यात बटणे वापरा।",
        "camera_closed_title":"कॅमेरा बंद आहे",
        "camera_closed_sub":  "अद्याप काहीही सक्रिय नाही — कॅमेरा सुरू करण्यासाठी खाली दाबा.",
        "camera_open_btn":    "📷 कॅमेरा उघडा",
        "camera_close_btn":   "✕ कॅमेरा बंद करा",
        "health_offline":     "**ऑफलाइन मोड** — Gemini API की आढळली नाही. स्थानिक विश्लेषण चालू आहे.",
        "health_online":      "**Gemini Vision सक्रिय** — AI-चालित प्रति-कांदा ओळख सक्षम आहे.",
        "health_error":       "बॅकएंडशी कनेक्ट होता आले नाही. कृपया सर्व्हर सुरू करा.",
        "no_validated_records": "कोणतेही सत्यापित कांदा नोंदी उपलब्ध नाहीत.",
        "select_rgb":         "RGB निवडा",
        "rgb_selected":       "✓ RGB निवडले",
        "select_nir":         "NIR निवडा",
        "nir_selected":       "✓ NIR निवडले",
        "excluded_label":     "वगळलेले",
        "overridden_note":    "श्रेणी मॅन्युअली बदलल्या",
    },
    "gu": {
        "app_subtitle":       "AI જુએ છે, Python નક્કી કરે છે — ડુંગળી ગુણવત્તા ગ્રેડિંગ",
        "upload_heading":     "ડુંગળી બૅચનો ફોટો અપલોડ કરો",
        "upload_caption":     "સ્પષ્ટ બૅચ ફોટો અપલોડ કરો (JPEG, PNG અથવા WebP, મહત્તમ 20 MB).",
        "camera_heading":     "ફોટો લો",
        "camera_caption":     "સીધા કૅમેરાથી ફોટો લો — ફાઇલ સેવ કરવાની જરૂર નથી.",
        "batch_id_label":     "બૅચ ID (વૈકલ્પિક)",
        "batch_id_help":      "તમારો બૅચ નંબર દાખલ કરો, દા.ત. LOT-GJ-001. ખાલી છોડો તો આપોઆપ બનશે.",
        "analyse_btn":        "બૅચ વિશ્લેષણ કરો →",
        "new_analysis_btn":   "🔄 નવું વિશ્લેષણ",
        "grade_a":            "ગ્રેડ A",
        "urs":                "URS",
        "defect":             "ખામીવાળું",
        "sprouting":          "અંકુરણ",
        "rot":                "સડો",
        "mould":              "ફૂગ",
        "bruising":           "ઈજા",
        "discoloration":      "રંગ ફેરફાર",
        "decay":              "ક્ષય",
        "physical_damage":    "ભૌતિક નુકસાન",
        "none":               "કોઈ નહીં",
        "other":              "અન્ય",
        "onions":             "ડુંગળી",
        "per_onion_heading":  "પ્રત્યેક ડુંગળી ગ્રેડિંગ રેકોર્ડ",
        "per_onion_caption":  "છબીમાં દર્શાવેલ નંબર આ કોષ્ટકની પંક્તિ સાથે મળે છે.",
        "override_col":       "ગ્રેડ બદલો",
        "override_note":      "ગ્રેડ બદલો — ટકાવારી તરત ફરી ગણાશે. બદલેલ રેકોર્ડ PDF માં ચિહ્નિત થશે.",
        "rule_engine_title":  "⚙️ ગ્રેડ કેવી રીતે નક્કી થાય? — નિયમ એન્જિન જુઓ",
        "defect_heading":     "ખામી વિશ્લેષણ",
        "quality_heading":    "એકંદર બૅચ મૂલ્યાંકન",
        "confidence_heading": "AI ધારણા વિશ્વાસ — પ્રત્યેક ડુંગળી",
        "confidence_caption": "દરેક બિંદુ એક ડુંગળી છે. 50% ની નીચે = અમાન્ય.",
        "report_heading":     "ડિજિટલ રિપોર્ટ",
        "export_heading":     "પરિણામ નિકાસ કરો",
        "export_csv":         "⬇️ CSV નિકાસ",
        "export_json":        "⬇️ JSON નિકાસ",
        "export_pdf":         "📄 PDF નિકાસ",
        "history_heading":    "બૅચ ઇતિહાસ",
        "history_empty":      "આ સત્રમાં કોઈ પાછલો બૅચ નથી.",
        "compare_heading":    "બૅચ સરખામણી",
        "trend_heading":      "ગ્રેડ A% વલણ",
        "img_quality_warn":   "⚠️ છબીની ગુણવત્તા ઓછી હોઈ શકે (ઝાંખી અથવા અંધારી). પરિણામ ઓછા ચોક્કસ હોઈ શકે.",
        "status_good":        "સારું",
        "status_acceptable":  "સ્વીકાર્ય",
        "status_poor":        "નબળું",
        "recommendation":     "ભલામણ",
        "batch_quality":      "બૅચ ગુણવત્તા સ્થિતિ",
        "flagged_title":      "અમાન્ય રેકોર્ડ",
        "flagged_caption":    "આ રેકોર્ડ ચકાસણી પાસ ન કરવાથી ગ્રેડિંગમાંથી બાકાત છે.",
        "no_defects":         "✓ કોઈ ખામીવાળી ડુંગળી મળી નથી",
        "no_defects_sub":     "ચકાસાયેલ બૅચમાં કોઈ ખામીવાળો રેકોર્ড મળ્યો નથી.",
        "results_placeholder":"પરિણામ અહીં દેખાશે",
        "results_sub":        "બૅચ ફોટો અપલોડ કરો અને વિશ્લેષણ કરો પસંદ કરો.",
        "grade_reason_col":   "ગ્રેડ કારણ",
        "image_num_col":      "#",
        "onion_id_col":       "ડુંગળી ID",
        "size_col":           "કદ",
        "colour_col":         "રંગ",
        "severity_col":       "તીવ્રતા",
        "defect_col":         "ખામી",
        "defect_desc_col":    "ખામી વર્ણન",
        "confidence_col":     "વિશ્વાસ",
        "spinner_msg":        "Gemini Vision વિશ્લેષણ → ચકાસણી → ગ્રેડિંગ → અહેવાલ…",
        "final_grade_col":    "અંતિમ ગ્રેડ",
        "inspection_mode_label": "તપાસ મોડ",
        "mode_rgb":           "🔴 બાહ્ય RGB ગ્રેડિંગ",
        "mode_nir":           "🔵 આંતરિક NIR સ્ક્રીનિંગ",
        "mode_rgb_desc":      "બાહ્ય સપાટી ગુણવત્તા આધારે ડુંગળીને ગ્રેડ આપે — કદ, રંગ, છાલ, અંકુરણ અને દૃશ્ય ખામીઓ.",
        "mode_nir_desc":      "આંતરિક ખામીઓ તપાસે — હોલો હાર્ટ, આંતરિક સડો અને ભેજ ઘટ, જે નરી આંખે દેખાતા નથી.",
        "step_upload":        "અપલોડ",
        "step_perceive":      "ઓળખ",
        "step_validate":      "ચકાસણી",
        "step_grade":         "ગ્રેડિંગ",
        "step_report":        "અહેવાલ",
        "prog_uploading":     "⬆ અપલોડ થઈ રહ્યું છે",
        "prog_processing":    "⚙ પ્રક્રિયા ચાલી રહી છે",
        "prog_analyzing":     "🔍 વિશ્લેષણ ચાલી રહ્યું છે",
        "prog_preparing":     "📊 તૈયાર કરાઈ રહ્યું છે",
        "prog_complete":      "✓ પૂર્ણ",
        "cv_mode_heading":    "સારાંશ દૃશ્ય",
        "cv_batch_info":      "બૅચ માહિતી",
        "cv_grade_breakdown": "ગ્રેડ વિગત",
        "cv_top_defects":     "મુખ્ય ખામીઓ",
        "cv_ai_confidence":   "AI વિશ્વાસ સ્તર",
        "cv_quick_actions":   "ઝડપી ક્રિયાઓ",
        "cv_avg_confidence":  "સરેરાશ વિશ્વાસ",
        "cv_highest_conf":    "સૌથી વધુ વિશ્વસ્ત ડુંગળી",
        "cv_lowest_conf":     "સૌથી ઓછી વિશ્વસ્ત ડુંગળી",
        "cv_flagged_count":   "અમાન્ય રેકોર્ડ",
        "cv_batch_status":    "બૅચ સ્થિતિ",
        "cv_total_onions":    "કુલ ડુંગળી",
        "cv_export_hint":     "પરિણામ ડાઉનલોડ કરવા માટે નીચેના નિકાસ બટનો વાપરો.",
        "camera_closed_title":"કૅમેરા બંધ છે",
        "camera_closed_sub":  "હજી કંઈ સક્રિય નથી — કૅમેરો ચાલુ કરવા નીચે દબાવો.",
        "camera_open_btn":    "📷 કૅમેરો ખોલો",
        "camera_close_btn":   "✕ કૅમેરો બંધ કરો",
        "health_offline":     "**ઑફલાઇન મોડ** — Gemini API ચાવી મળી નથી. સ્થાનિક વિશ્લેષણ ચાલી રહ્યું છે.",
        "health_online":      "**Gemini Vision સક્રિય** — AI-સંચાલિત પ્રતિ-ડુંગળી ઓળખ સક્ષમ છે.",
        "health_error":       "બૅકએન્ડ સાથે કનેક્ટ થઈ શકાયું નહીં. કૃપા કરી સર્વર શરૂ કરો.",
        "no_validated_records": "કોઈ સત્યાપિત ડુંગળી રેકોર્ડ ઉપલ્બ્ધ નથી.",
        "select_rgb":         "RGB પસંદ કરો",
        "rgb_selected":       "✓ RGB પસંદ",
        "select_nir":         "NIR પસંદ કરો",
        "nir_selected":       "✓ NIR પસંદ",
        "excluded_label":     "બાકાત",
        "overridden_note":    "ગ્રેડ મૅન્યુઅલી બદલ્યા",
    },
    "pa": {
        "app_subtitle":       "AI ਦੇਖਦਾ ਹੈ, Python ਫੈਸਲਾ ਕਰਦਾ ਹੈ — ਪਿਆਜ਼ ਗੁਣਵੱਤਾ ਗ੍ਰੇਡਿੰਗ",
        "upload_heading":     "ਪਿਆਜ਼ ਬੈਚ ਦੀ ਫੋਟੋ ਅਪਲੋਡ ਕਰੋ",
        "upload_caption":     "ਸਾਫ਼ ਬੈਚ ਫੋਟੋ ਅਪਲੋਡ ਕਰੋ (JPEG, PNG ਜਾਂ WebP, ਵੱਧ ਤੋਂ ਵੱਧ 20 MB).",
        "camera_heading":     "ਫੋਟੋ ਲਓ",
        "camera_caption":     "ਸਿੱਧਾ ਕੈਮਰੇ ਤੋਂ ਫੋਟੋ ਲਓ — ਫਾਈਲ ਸੇਵ ਕਰਨ ਦੀ ਲੋੜ ਨਹੀਂ।",
        "batch_id_label":     "ਬੈਚ ID (ਵਿਕਲਪਿਕ)",
        "batch_id_help":      "ਆਪਣਾ ਬੈਚ ਨੰਬਰ ਦਾਖਲ ਕਰੋ, ਜਿਵੇਂ LOT-PB-001. ਖਾਲੀ ਛੱਡੋ ਤਾਂ ਆਪਣੇ ਆਪ ਬਣੇਗਾ।",
        "analyse_btn":        "ਬੈਚ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰੋ →",
        "new_analysis_btn":   "🔄 ਨਵਾਂ ਵਿਸ਼ਲੇਸ਼ਣ",
        "grade_a":            "ਗ੍ਰੇਡ A",
        "urs":                "URS",
        "defect":             "ਨੁਕਸਦਾਰ",
        "sprouting":          "ਉਗਾਈ",
        "rot":                "ਸੜਾਂਦ",
        "mould":              "ਉੱਲੀ",
        "bruising":           "ਸੱਟ",
        "discoloration":      "ਰੰਗ ਬਦਲਾਅ",
        "decay":              "ਖਰਾਬੀ",
        "physical_damage":    "ਸਰੀਰਕ ਨੁਕਸਾਨ",
        "none":               "ਕੋਈ ਨਹੀਂ",
        "other":              "ਹੋਰ",
        "onions":             "ਪਿਆਜ਼",
        "per_onion_heading":  "ਪ੍ਰਤੀ ਪਿਆਜ਼ ਗ੍ਰੇਡਿੰਗ ਰਿਕਾਰਡ",
        "per_onion_caption":  "ਤਸਵੀਰ ਵਿੱਚ ਦਿਖਾਇਆ ਨੰਬਰ ਇਸ ਸਾਰਣੀ ਦੀ ਕਤਾਰ ਨਾਲ ਮੇਲ ਖਾਂਦਾ ਹੈ।",
        "override_col":       "ਗ੍ਰੇਡ ਬਦਲੋ",
        "override_note":      "ਗ੍ਰੇਡ ਬਦਲੋ — ਪ੍ਰਤੀਸ਼ਤ ਤੁਰੰਤ ਮੁੜ ਗਣਨਾ ਹੋਵੇਗੀ. ਬਦਲੇ ਰਿਕਾਰਡ PDF ਵਿੱਚ ਦਰਸਾਏ ਜਾਣਗੇ।",
        "rule_engine_title":  "⚙️ ਗ੍ਰੇਡ ਕਿਵੇਂ ਤੈਅ ਹੁੰਦਾ ਹੈ? — ਨਿਯਮ ਇੰਜਣ ਦੇਖੋ",
        "defect_heading":     "ਨੁਕਸ ਵਿਸ਼ਲੇਸ਼ਣ",
        "quality_heading":    "ਸਮੁੱਚਾ ਬੈਚ ਮੁਲਾਂਕਣ",
        "confidence_heading": "AI ਧਾਰਨਾ ਭਰੋਸਾ — ਪ੍ਰਤੀ ਪਿਆਜ਼",
        "confidence_caption": "ਹਰੇਕ ਬਿੰਦੂ ਇੱਕ ਪਿਆਜ਼ ਹੈ. 50% ਤੋਂ ਘੱਟ = ਅਵੈਧ।",
        "report_heading":     "ਡਿਜੀਟਲ ਰਿਪੋਰਟ",
        "export_heading":     "ਨਤੀਜੇ ਨਿਰਯਾਤ ਕਰੋ",
        "export_csv":         "⬇️ CSV ਨਿਰਯਾਤ",
        "export_json":        "⬇️ JSON ਨਿਰਯਾਤ",
        "export_pdf":         "📄 PDF ਨਿਰਯਾਤ",
        "history_heading":    "ਬੈਚ ਇਤਿਹਾਸ",
        "history_empty":      "ਇਸ ਸੈਸ਼ਨ ਵਿੱਚ ਕੋਈ ਪਿਛਲਾ ਬੈਚ ਨਹੀਂ।",
        "compare_heading":    "ਬੈਚ ਤੁਲਨਾ",
        "trend_heading":      "ਗ੍ਰੇਡ A% ਰੁਝਾਨ",
        "img_quality_warn":   "⚠️ ਤਸਵੀਰ ਦੀ ਗੁਣਵੱਤਾ ਘੱਟ ਹੋ ਸਕਦੀ ਹੈ (ਧੁੰਦਲੀ ਜਾਂ ਹਨੇਰੀ). ਨਤੀਜੇ ਘੱਟ ਸਹੀ ਹੋ ਸਕਦੇ ਹਨ।",
        "status_good":        "ਚੰਗਾ",
        "status_acceptable":  "ਸਵੀਕਾਰਯੋਗ",
        "status_poor":        "ਮਾੜਾ",
        "recommendation":     "ਸਿਫਾਰਸ਼",
        "batch_quality":      "ਬੈਚ ਗੁਣਵੱਤਾ ਸਥਿਤੀ",
        "flagged_title":      "ਅਵੈਧ ਰਿਕਾਰਡ",
        "flagged_caption":    "ਇਹ ਰਿਕਾਰਡ ਪ੍ਰਮਾਣਿਕਤਾ ਪਾਸ ਨਾ ਕਰਨ ਕਾਰਨ ਗ੍ਰੇਡਿੰਗ ਤੋਂ ਬਾਹਰ ਹਨ।",
        "no_defects":         "✓ ਕੋਈ ਨੁਕਸਦਾਰ ਪਿਆਜ਼ ਨਹੀਂ ਮਿਲਿਆ",
        "no_defects_sub":     "ਪ੍ਰਮਾਣਿਤ ਬੈਚ ਵਿੱਚ ਕੋਈ ਨੁਕਸਦਾਰ ਰਿਕਾਰਡ ਨਹੀਂ ਮਿਲਿਆ।",
        "results_placeholder":"ਨਤੀਜੇ ਇੱਥੇ ਦਿਖਾਈ ਦੇਣਗੇ",
        "results_sub":        "ਬੈਚ ਫੋਟੋ ਅਪਲੋਡ ਕਰੋ ਅਤੇ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰੋ ਚੁਣੋ।",
        "grade_reason_col":   "ਗ੍ਰੇਡ ਕਾਰਨ",
        "image_num_col":      "#",
        "onion_id_col":       "ਪਿਆਜ਼ ID",
        "size_col":           "ਆਕਾਰ",
        "colour_col":         "ਰੰਗ",
        "severity_col":       "ਗੰਭੀਰਤਾ",
        "defect_col":         "ਨੁਕਸ",
        "defect_desc_col":    "ਨੁਕਸ ਵੇਰਵਾ",
        "confidence_col":     "ਭਰੋਸਾ",
        "spinner_msg":        "Gemini Vision ਵਿਸ਼ਲੇਸ਼ਣ → ਪ੍ਰਮਾਣਿਕਤਾ → ਗ੍ਰੇਡਿੰਗ → ਰਿਪੋਰਟ…",
        "final_grade_col":    "ਅੰਤਿਮ ਗ੍ਰੇਡ",
        "inspection_mode_label": "ਨਿਰੀਖਣ ਮੋਡ",
        "mode_rgb":           "🔴 ਬਾਹਰੀ RGB ਗ੍ਰੇਡਿੰਗ",
        "mode_nir":           "🔵 ਅੰਦਰੂਨੀ NIR ਸਕ੍ਰੀਨਿੰਗ",
        "mode_rgb_desc":      "ਬਾਹਰੀ ਸਤਹ ਗੁਣਵੱਤਾ ਦੇ ਅਧਾਰ 'ਤੇ ਪਿਆਜ਼ ਨੂੰ ਗ੍ਰੇਡ ਕਰਦਾ ਹੈ — ਆਕਾਰ, ਰੰਗ, ਛਿੱਲ, ਉਗਾਈ ਅਤੇ ਦਿੱਖਣ ਵਾਲੇ ਨੁਕਸ।",
        "mode_nir_desc":      "ਅੰਦਰੂਨੀ ਨੁਕਸਾਂ ਦੀ ਜਾਂਚ — ਖੋਖਲਾ ਕੇਂਦਰ, ਅੰਦਰੂਨੀ ਸੜਾਂਦ ਅਤੇ ਨਮੀ ਘਟਾਓ ਜੋ ਨੰਗੀ ਅੱਖ ਨਾਲ ਨਹੀਂ ਦਿੱਖਦੇ।",
        "step_upload":        "ਅਪਲੋਡ",
        "step_perceive":      "ਪਛਾਣ",
        "step_validate":      "ਤਸਦੀਕ",
        "step_grade":         "ਗ੍ਰੇਡਿੰਗ",
        "step_report":        "ਰਿਪੋਰਟ",
        "prog_uploading":     "⬆ ਅਪਲੋਡ ਹੋ ਰਿਹਾ ਹੈ",
        "prog_processing":    "⚙ ਪ੍ਰਕਿਰਿਆ ਚੱਲ ਰਹੀ ਹੈ",
        "prog_analyzing":     "🔍 ਵਿਸ਼ਲੇਸ਼ਣ ਚੱਲ ਰਿਹਾ ਹੈ",
        "prog_preparing":     "📊 ਤਿਆਰ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ",
        "prog_complete":      "✓ ਮੁਕੰਮਲ",
        "cv_mode_heading":    "ਸੰਖੇਪ ਦ੍ਰਿਸ਼",
        "cv_batch_info":      "ਬੈਚ ਜਾਣਕਾਰੀ",
        "cv_grade_breakdown": "ਗ੍ਰੇਡ ਵੇਰਵਾ",
        "cv_top_defects":     "ਮੁੱਖ ਨੁਕਸ",
        "cv_ai_confidence":   "AI ਭਰੋਸਾ ਪੱਧਰ",
        "cv_quick_actions":   "ਤੁਰੰਤ ਕਾਰਵਾਈਆਂ",
        "cv_avg_confidence":  "ਔਸਤ ਭਰੋਸਾ",
        "cv_highest_conf":    "ਸਭ ਤੋਂ ਵੱਧ ਭਰੋਸੇਯੋਗ ਪਿਆਜ਼",
        "cv_lowest_conf":     "ਸਭ ਤੋਂ ਘੱਟ ਭਰੋਸੇਯੋਗ ਪਿਆਜ਼",
        "cv_flagged_count":   "ਅਵੈਧ ਰਿਕਾਰਡ",
        "cv_batch_status":    "ਬੈਚ ਸਥਿਤੀ",
        "cv_total_onions":    "ਕੁੱਲ ਪਿਆਜ਼",
        "cv_export_hint":     "ਨਤੀਜੇ ਡਾਊਨਲੋਡ ਕਰਨ ਲਈ ਹੇਠਾਂ ਨਿਰਯਾਤ ਬਟਨ ਵਰਤੋ।",
        "camera_closed_title":"ਕੈਮਰਾ ਬੰਦ ਹੈ",
        "camera_closed_sub":  "ਅਜੇ ਕੁਝ ਵੀ ਸਰਗਰਮ ਨਹੀਂ — ਕੈਮਰਾ ਚਾਲੂ ਕਰਨ ਲਈ ਹੇਠਾਂ ਦਬਾਓ।",
        "camera_open_btn":    "📷 ਕੈਮਰਾ ਖੋਲ੍ਹੋ",
        "camera_close_btn":   "✕ ਕੈਮਰਾ ਬੰਦ ਕਰੋ",
        "health_offline":     "**ਆਫਲਾਈਨ ਮੋਡ** — Gemini API ਕੁੰਜੀ ਨਹੀਂ ਮਿਲੀ। ਸਥਾਨਕ ਵਿਸ਼ਲੇਸ਼ਣ ਚੱਲ ਰਿਹਾ ਹੈ।",
        "health_online":      "**Gemini Vision ਸਰਗਰਮ** — AI-ਸੰਚਾਲਿਤ ਪ੍ਰਤੀ-ਪਿਆਜ਼ ਪਛਾਣ ਸਮਰੱਥ ਹੈ।",
        "health_error":       "ਬੈਕਐਂਡ ਨਾਲ ਕਨੈਕਟ ਨਹੀਂ ਹੋ ਸਕਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਸਰਵਰ ਸ਼ੁਰੂ ਕਰੋ।",
        "no_validated_records": "ਕੋਈ ਪ੍ਰਮਾਣਿਤ ਪਿਆਜ਼ ਰਿਕਾਰਡ ਉਪਲਬਧ ਨਹੀਂ।",
        "select_rgb":         "RGB ਚੁਣੋ",
        "rgb_selected":       "✓ RGB ਚੁਣਿਆ",
        "select_nir":         "NIR ਚੁਣੋ",
        "nir_selected":       "✓ NIR ਚੁਣਿਆ",
        "excluded_label":     "ਬਾਹਰ ਕੱਢਿਆ",
        "overridden_note":    "ਗ੍ਰੇਡ ਮੈਨੂਅਲੀ ਬਦਲੇ ਗਏ",
    },
    "ta": {
        "app_subtitle":       "AI பார்க்கிறது, Python தீர்மானிக்கிறது — வெங்காய தர நிர்ணயம்",
        "upload_heading":     "வெங்காய தொகுதி புகைப்படத்தை பதிவேற்றுக",
        "upload_caption":     "தெளிவான தொகுதி புகைப்படத்தை பதிவேற்றுக (JPEG, PNG அல்லது WebP, அதிகபட்சம் 20 MB).",
        "camera_heading":     "புகைப்படம் எடு",
        "camera_caption":     "நேரடியாக கேமராவிலிருந்து புகைப்படம் எடு — கோப்பு சேமிக்க தேவையில்லை.",
        "batch_id_label":     "தொகுதி ID (விருப்பமானது)",
        "batch_id_help":      "உங்கள் தொகுதி எண்ணை உள்ளிடுக, எ.கா. LOT-TN-001. காலியாக விட்டால் தானாக உருவாகும்.",
        "analyse_btn":        "தொகுதியை பகுப்பாய்வு செய் →",
        "new_analysis_btn":   "🔄 புதிய பகுப்பாய்வு",
        "grade_a":            "தரம் A",
        "urs":                "URS",
        "defect":             "குறைபாடுள்ளது",
        "sprouting":          "முளைப்பு",
        "rot":                "அழுகல்",
        "mould":              "பூஞ்சை",
        "bruising":           "காயம்",
        "discoloration":      "நிற மாற்றம்",
        "decay":              "சிதைவு",
        "physical_damage":    "உடல் சேதம்",
        "none":               "எதுவுமில்லை",
        "other":              "மற்றவை",
        "onions":             "வெங்காயங்கள்",
        "per_onion_heading":  "ஒவ்வொரு வெங்காய தர பதிவு",
        "per_onion_caption":  "படத்தில் காட்டப்பட்ட எண் இந்த அட்டவணையின் வரிசையுடன் பொருந்துகிறது.",
        "override_col":       "தரம் மாற்று",
        "override_note":      "தரத்தை மாற்றுக — சதவீதம் உடனடியாக மறுகணக்கிடப்படும். மாற்றப்பட்ட பதிவுகள் PDF இல் குறிக்கப்படும்.",
        "rule_engine_title":  "⚙️ தரங்கள் எவ்வாறு தீர்மானிக்கப்படுகின்றன? — விதி இயந்திரத்தை காண்க",
        "defect_heading":     "குறைபாடு பகுப்பாய்வு",
        "quality_heading":    "ஒட்டுமொத்த தொகுதி மதிப்பீடு",
        "confidence_heading": "AI உணர்வு நம்பிக்கை — ஒவ்வொரு வெங்காயம்",
        "confidence_caption": "ஒவ்வொரு புள்ளியும் ஒரு வெங்காயம். 50% க்கு கீழே = தவறானது.",
        "report_heading":     "டிஜிட்டல் அறிக்கை",
        "export_heading":     "முடிவுகளை ஏற்றுமதி செய்",
        "export_csv":         "⬇️ CSV ஏற்றுமதி",
        "export_json":        "⬇️ JSON ஏற்றுமதி",
        "export_pdf":         "📄 PDF ஏற்றுமதி",
        "history_heading":    "தொகுதி வரலாறு",
        "history_empty":      "இந்த அமர்வில் முந்தைய தொகுதிகள் இல்லை.",
        "compare_heading":    "தொகுதி ஒப்பீடு",
        "trend_heading":      "தரம் A% போக்கு",
        "img_quality_warn":   "⚠️ படத்தின் தரம் குறைவாக இருக்கலாம் (மங்கலான அல்லது இருட்டான). முடிவுகள் குறைவான துல்லியமாக இருக்கலாம்.",
        "status_good":        "நல்லது",
        "status_acceptable":  "ஏற்றுக்கொள்ளத்தக்கது",
        "status_poor":        "மோசம்",
        "recommendation":     "பரிந்துரை",
        "batch_quality":      "தொகுதி தர நிலை",
        "flagged_title":      "தவறான பதிவுகள்",
        "flagged_caption":    "இந்த பதிவுகள் சரிபார்ப்பை கடக்காததால் தர நிர்ணயத்தில் இருந்து விலக்கப்பட்டுள்ளன.",
        "no_defects":         "✓ குறைபாடுள்ள வெங்காயங்கள் எதுவும் இல்லை",
        "no_defects_sub":     "சரிபார்க்கப்பட்ட தொகுதியில் குறைபாடுள்ள பதிவுகள் எதுவும் இல்லை.",
        "results_placeholder":"முடிவுகள் இங்கே தோன்றும்",
        "results_sub":        "தொகுதி புகைப்படத்தை பதிவேற்றி பகுப்பாய்வை தேர்ந்தெடுக்கவும்.",
        "grade_reason_col":   "தர காரணம்",
        "image_num_col":      "#",
        "onion_id_col":       "வெங்காய ID",
        "size_col":           "அளவு",
        "colour_col":         "நிறம்",
        "severity_col":       "தீவிரம்",
        "defect_col":         "குறைபாடு",
        "defect_desc_col":    "குறைபாடு விளக்கம்",
        "confidence_col":     "நம்பிக்கை",
        "spinner_msg":        "Gemini Vision பகுப்பாய்வு → சரிபார்ப்பு → தர நிர்ணயம் → அறிக்கை…",
        "final_grade_col":    "இறுதி தரம்",
        "inspection_mode_label": "ஆய்வு முறை",
        "mode_rgb":           "🔴 வெளிப்புற RGB தரம் நிர்ணயம்",
        "mode_nir":           "🔵 உள்ளக NIR திரையிடல்",
        "mode_rgb_desc":      "வெளிப்புற மேற்பரப்பு தரம் — அளவு, நிறம், தோல், முளைப்பு மற்றும் தெரியும் குறைபாடுகளை வகைப்படுத்துகிறது.",
        "mode_nir_desc":      "உள்ளக குறைபாடுகளை திரையிடுகிறது — உள் அழுகல் மற்றும் ஈரப்பதம் குறைபாடு, கண்ணுக்கு தெரியாதவை.",
        "step_upload":        "பதிவேற்றம்",
        "step_perceive":      "உணர்வு",
        "step_validate":      "சரிபார்ப்பு",
        "step_grade":         "தர நிர்ணயம்",
        "step_report":        "அறிக்கை",
        "prog_uploading":     "⬆ பதிவேற்றம் நடக்கிறது",
        "prog_processing":    "⚙ செயலாக்கம் நடக்கிறது",
        "prog_analyzing":     "🔍 பகுப்பாய்வு நடக்கிறது",
        "prog_preparing":     "📊 தயாராகிறது",
        "prog_complete":      "✓ முடிந்தது",
        "cv_mode_heading":    "சுருக்க பார்வை",
        "cv_batch_info":      "தொகுதி தகவல்",
        "cv_grade_breakdown": "தர விவரம்",
        "cv_top_defects":     "முக்கிய குறைபாடுகள்",
        "cv_ai_confidence":   "AI நம்பிக்கை அளவு",
        "cv_quick_actions":   "விரைவு செயல்கள்",
        "cv_avg_confidence":  "சராசரி நம்பிக்கை",
        "cv_highest_conf":    "அதிக நம்பிக்கையுள்ள வெங்காயம்",
        "cv_lowest_conf":     "குறைந்த நம்பிக்கையுள்ள வெங்காயம்",
        "cv_flagged_count":   "தவறான பதிவுகள்",
        "cv_batch_status":    "தொகுதி நிலை",
        "cv_total_onions":    "மொத்த வெங்காயங்கள்",
        "cv_export_hint":     "முடிவுகளை பதிவிறக்கம் செய்ய கீழே உள்ள ஏற்றுமதி பொத்தான்களை பயன்படுத்துங்கள்.",
        "camera_closed_title":"கேமரா மூடப்பட்டுள்ளது",
        "camera_closed_sub":  "இன்னும் எதுவும் செயல்படவில்லை — கேமராவை இயக்க கீழே அழுத்துங்கள்.",
        "camera_open_btn":    "📷 கேமராவை திற",
        "camera_close_btn":   "✕ கேமராவை மூடு",
        "health_offline":     "**ஆஃப்லைன் பயன்முறை** — Gemini API திறவுகோல் இல்லை. உள்ளூர் பகுப்பாய்வு இயங்குகிறது.",
        "health_online":      "**Gemini Vision செயலில்** — AI-சக்தியுள்ள ஒவ்வொரு வெங்காய பகுப்பாய்வு இயக்கப்பட்டுள்ளது.",
        "health_error":       "பின்தள சேவையகத்துடன் இணைக்க முடியவில்லை. சேவையகத்தை தொடங்குங்கள்.",
        "no_validated_records": "சரிபார்க்கப்பட்ட வெங்காய பதிவுகள் எதுவும் இல்லை.",
        "select_rgb":         "RGB தேர்வு செய்",
        "rgb_selected":       "✓ RGB தேர்ந்தது",
        "select_nir":         "NIR தேர்வு செய்",
        "nir_selected":       "✓ NIR தேர்ந்தது",
        "excluded_label":     "விலக்கப்பட்டது",
        "overridden_note":    "தரங்கள் கைமுறையாக மாற்றப்பட்டன",
    },
    "te": {
        "app_subtitle":       "AI చూస్తుంది, Python నిర్ణయిస్తుంది — ఉల్లిపాయ నాణ్యత గ్రేడింగ్",
        "upload_heading":     "ఉల్లిపాయ బ్యాచ్ ఫోటో అప్‌లోడ్ చేయండి",
        "upload_caption":     "స్పష్టమైన బ్యాచ్ ఫోటో అప్‌లోడ్ చేయండి (JPEG, PNG లేదా WebP, గరిష్ఠం 20 MB).",
        "camera_heading":     "ఫోటో తీయండి",
        "camera_caption":     "నేరుగా కెమెరా నుండి ఫోటో తీయండి — ఫైల్ సేవ్ చేయవలసిన అవసరం లేదు.",
        "batch_id_label":     "బ్యాచ్ ID (ఐచ్ఛికం)",
        "batch_id_help":      "మీ బ్యాచ్ నంబర్ నమోదు చేయండి, ఉదా. LOT-AP-001. ఖాళీగా వదిలితే స్వయంచాలకంగా రూపొందుతుంది.",
        "analyse_btn":        "బ్యాచ్ విశ్లేషించండి →",
        "new_analysis_btn":   "🔄 కొత్త విశ్లేషణ",
        "grade_a":            "గ్రేడ్ A",
        "urs":                "URS",
        "defect":             "లోపభూయిష్ఠమైనది",
        "sprouting":          "మొలకెత్తడం",
        "rot":                "కుళ్ళు",
        "mould":              "అచ్చు",
        "bruising":           "గాయం",
        "discoloration":      "రంగు మారడం",
        "decay":              "క్షయం",
        "physical_damage":    "శారీరక నష్టం",
        "none":               "ఏదీ లేదు",
        "other":              "ఇతరాలు",
        "onions":             "ఉల్లిపాయలు",
        "per_onion_heading":  "ప్రతి ఉల్లిపాయ గ్రేడింగ్ రికార్డు",
        "per_onion_caption":  "చిత్రంలో చూపిన నంబర్ ఈ పట్టిక వరుసతో సరిపోతుంది.",
        "override_col":       "గ్రేడ్ మార్చండి",
        "override_note":      "గ్రేడ్ మార్చండి — శాతం వెంటనే మళ్ళీ లెక్కించబడుతుంది. మార్చిన రికార్డులు PDF లో గుర్తించబడతాయి.",
        "rule_engine_title":  "⚙️ గ్రేడ్‌లు ఎలా నిర్ణయించబడతాయి? — నియమ ఇంజిన్ చూడండి",
        "defect_heading":     "లోపం విశ్లేషణ",
        "quality_heading":    "మొత్తం బ్యాచ్ అంచనా",
        "confidence_heading": "AI అవగాహన విశ్వాసం — ప్రతి ఉల్లిపాయ",
        "confidence_caption": "ప్రతి చుక్క ఒక ఉల్లిపాయ. 50% కంటే తక్కువ = చెల్లదు.",
        "report_heading":     "డిజిటల్ నివేదిక",
        "export_heading":     "ఫలితాలు ఎగుమతి చేయండి",
        "export_csv":         "⬇️ CSV ఎగుమతి",
        "export_json":        "⬇️ JSON ఎగుమతి",
        "export_pdf":         "📄 PDF ఎగుమతి",
        "history_heading":    "బ్యాచ్ చరిత్ర",
        "history_empty":      "ఈ సెషన్‌లో మునుపటి బ్యాచ్‌లు లేవు.",
        "compare_heading":    "బ్యాచ్ పోలిక",
        "trend_heading":      "గ్రేడ్ A% ధోరణి",
        "img_quality_warn":   "⚠️ చిత్రం నాణ్యత తక్కువగా ఉండవచ్చు (మసకగా లేదా చీకటిగా). ఫలితాలు తక్కువ ఖచ్చితంగా ఉండవచ్చు.",
        "status_good":        "మంచిది",
        "status_acceptable":  "ఆమోదయోగ్యం",
        "status_poor":        "పేలవంగా",
        "recommendation":     "సిఫారసు",
        "batch_quality":      "బ్యాచ్ నాణ్యత స్థితి",
        "flagged_title":      "చెల్లని రికార్డులు",
        "flagged_caption":    "ఈ రికార్డులు ధృవీకరణ పాస్ కాకపోవడంతో గ్రేడింగ్ నుండి మినహాయించబడ్డాయి.",
        "no_defects":         "✓ లోపభూయిష్ఠమైన ఉల్లిపాయలు కనుగొనబడలేదు",
        "no_defects_sub":     "ధృవీకరించబడిన బ్యాచ్‌లో లోపభూయిష్ఠమైన రికార్డులు కనుగొనబడలేదు.",
        "results_placeholder":"ఫలితాలు ఇక్కడ కనిపిస్తాయి",
        "results_sub":        "బ్యాచ్ ఫోటో అప్‌లోడ్ చేసి విశ్లేషించు ఎంచుకోండి.",
        "grade_reason_col":   "గ్రేడ్ కారణం",
        "image_num_col":      "#",
        "onion_id_col":       "ఉల్లిపాయ ID",
        "size_col":           "పరిమాణం",
        "colour_col":         "రంగు",
        "severity_col":       "తీవ్రత",
        "defect_col":         "లోపం",
        "defect_desc_col":    "లోపం వివరణ",
        "confidence_col":     "విశ్వాసం",
        "spinner_msg":        "Gemini Vision విశ్లేషణ → ధృవీకరణ → గ్రేడింగ్ → నివేదిక…",
        "final_grade_col":    "తుది గ్రేడ్",
        "inspection_mode_label": "తనిఖీ మోడ్",
        "mode_rgb":           "🔴 బాహ్య RGB గ్రేడింగ్",
        "mode_nir":           "🔵 అంతర్గత NIR స్క్రీనింగ్",
        "mode_rgb_desc":      "బాహ్య ఉపరితల నాణ్యత ఆధారంగా ఉల్లిపాయలను గ్రేడ్ చేస్తుంది — పరిమాణం, రంగు, చర్మం, మొలకెత్తడం మరియు కనిపించే లోపాలు.",
        "mode_nir_desc":      "అంతర్గత లోపాలను స్క్రీన్ చేస్తుంది — లోపలి కుళ్ళు మరియు తేమ తగ్గుదల, నగ్నంగా కనిపించవు.",
        "step_upload":        "అప్‌లోడ్",
        "step_perceive":      "గుర్తింపు",
        "step_validate":      "ధృవీకరణ",
        "step_grade":         "గ్రేడింగ్",
        "step_report":        "నివేదిక",
        "prog_uploading":     "⬆ అప్‌లోడ్ అవుతోంది",
        "prog_processing":    "⚙ ప్రాసెస్ అవుతోంది",
        "prog_analyzing":     "🔍 విశ్లేషించబడుతోంది",
        "prog_preparing":     "📊 సిద్ధమవుతోంది",
        "prog_complete":      "✓ పూర్తయింది",
        "cv_mode_heading":    "సారాంశ దృశ్యం",
        "cv_batch_info":      "బ్యాచ్ సమాచారం",
        "cv_grade_breakdown": "గ్రేడ్ వివరాలు",
        "cv_top_defects":     "ప్రధాన లోపాలు",
        "cv_ai_confidence":   "AI విశ్వాస స్థాయి",
        "cv_quick_actions":   "త్వరిత చర్యలు",
        "cv_avg_confidence":  "సగటు విశ్వాసం",
        "cv_highest_conf":    "అత్యధిక విశ్వాసం గల ఉల్లిపాయ",
        "cv_lowest_conf":     "అత్యల్ప విశ్వాసం గల ఉల్లిపాయ",
        "cv_flagged_count":   "చెల్లని రికార్డులు",
        "cv_batch_status":    "బ్యాచ్ స్థితి",
        "cv_total_onions":    "మొత్తం ఉల్లిపాయలు",
        "cv_export_hint":     "ఫలితాలు డౌన్‌లోడ్ చేయడానికి దిగువ ఎగుమతి బటన్‌లను ఉపయోగించండి.",
        "camera_closed_title":"కెమెరా మూసివేయబడింది",
        "camera_closed_sub":  "ఇంకా ఏమీ సక్రియంగా లేదు — కెమెరాను ఆన్ చేయడానికి దిగువ నొక్కండి.",
        "camera_open_btn":    "📷 కెమెరా తెరవండి",
        "camera_close_btn":   "✕ కెమెరా మూయండి",
        "health_offline":     "**ఆఫ్‌లైన్ మోడ్** — Gemini API కీ కనుగొనబడలేదు. స్థానిక విశ్లేషణ నడుస్తోంది.",
        "health_online":      "**Gemini Vision సక్రియం** — AI-ఆధారిత ప్రతి-ఉల్లిపాయ గుర్తింపు ప్రారంభించబడింది.",
        "health_error":       "బ్యాకెండ్‌కు కనెక్ట్ అవడం సాధ్యం కాలేదు. సర్వర్‌ను ప్రారంభించండి.",
        "no_validated_records": "ధృవీకరించబడిన ఉల్లిపాయ రికార్డులు అందుబాటులో లేవు.",
        "select_rgb":         "RGB ఎంచుకోండి",
        "rgb_selected":       "✓ RGB ఎంచుకున్నారు",
        "select_nir":         "NIR ఎంచుకోండి",
        "nir_selected":       "✓ NIR ఎంచుకున్నారు",
        "excluded_label":     "మినహాయించబడింది",
        "overridden_note":    "గ్రేడ్‌లు మాన్యువల్‌గా మార్చబడ్డాయి",
    },
    "kn": {
        "app_subtitle":       "AI ನೋಡುತ್ತದೆ, Python ನಿರ್ಧರಿಸುತ್ತದೆ — ಈರುಳ್ಳಿ ಗುಣಮಟ್ಟ ಶ್ರೇಣೀಕರಣ",
        "upload_heading":     "ಈರುಳ್ಳಿ ಬ್ಯಾಚ್ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "upload_caption":     "ಸ್ಪಷ್ಟ ಬ್ಯಾಚ್ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ (JPEG, PNG ಅಥವಾ WebP, ಗರಿಷ್ಠ 20 MB).",
        "camera_heading":     "ಫೋಟೋ ತೆಗೆಯಿರಿ",
        "camera_caption":     "ನೇರವಾಗಿ ಕ್ಯಾಮೆರಾದಿಂದ ಫೋಟೋ ತೆಗೆಯಿರಿ — ಫೈಲ್ ಉಳಿಸುವ ಅಗತ್ಯವಿಲ್ಲ.",
        "batch_id_label":     "ಬ್ಯಾಚ್ ID (ಐಚ್ಛಿಕ)",
        "batch_id_help":      "ನಿಮ್ಮ ಬ್ಯಾಚ್ ಸಂಖ್ಯೆ ನಮೂದಿಸಿ, ಉದಾ. LOT-KA-001. ಖಾಲಿ ಬಿಟ್ಟರೆ ಸ್ವಯಂಚಾಲಿತವಾಗಿ ಉತ್ಪತ್ತಿಯಾಗುತ್ತದೆ.",
        "analyse_btn":        "ಬ್ಯಾಚ್ ವಿಶ್ಲೇಷಿಸಿ →",
        "new_analysis_btn":   "🔄 ಹೊಸ ವಿಶ್ಲೇಷಣೆ",
        "grade_a":            "ಶ್ರೇಣಿ A",
        "urs":                "URS",
        "defect":             "ದೋಷಪೂರಿತ",
        "sprouting":          "ಮೊಳಕೆಯೊಡೆಯುವಿಕೆ",
        "rot":                "ಕೊಳೆತ",
        "mould":              "ಶಿಲೀಂಧ್ರ",
        "bruising":           "ಗಾಯ",
        "discoloration":      "ಬಣ್ಣ ಬದಲಾವಣೆ",
        "decay":              "ಕ್ಷಯ",
        "physical_damage":    "ದೈಹಿಕ ಹಾನಿ",
        "none":               "ಯಾವುದೂ ಇಲ್ಲ",
        "other":              "ಇತರ",
        "onions":             "ಈರುಳ್ಳಿಗಳು",
        "per_onion_heading":  "ಪ್ರತಿ ಈರುಳ್ಳಿ ಶ್ರೇಣೀಕರಣ ದಾಖಲೆ",
        "per_onion_caption":  "ಚಿತ್ರದಲ್ಲಿ ತೋರಿಸಿದ ಸಂಖ್ಯೆ ಈ ಕೋಷ್ಟಕದ ಸಾಲಿಗೆ ಹೊಂದಿಕೆಯಾಗುತ್ತದೆ.",
        "override_col":       "ಶ್ರೇಣಿ ಬದಲಿಸಿ",
        "override_note":      "ಶ್ರೇಣಿ ಬದಲಿಸಿ — ಶೇಕಡಾ ತಕ್ಷಣ ಮರು ಲೆಕ್ಕಿಸಲಾಗುತ್ತದೆ. ಬದಲಾದ ದಾಖಲೆಗಳು PDF ನಲ್ಲಿ ಗುರುತಿಸಲಾಗುತ್ತದೆ.",
        "rule_engine_title":  "⚙️ ಶ್ರೇಣಿಗಳು ಹೇಗೆ ನಿರ್ಧರಿಸಲಾಗುತ್ತದೆ? — ನಿಯಮ ಎಂಜಿನ್ ನೋಡಿ",
        "defect_heading":     "ದೋಷ ವಿಶ್ಲೇಷಣೆ",
        "quality_heading":    "ಒಟ್ಟಾರೆ ಬ್ಯಾಚ್ ಮೌಲ್ಯಮಾಪನ",
        "confidence_heading": "AI ಗ್ರಹಿಕೆ ವಿಶ್ವಾಸ — ಪ್ರತಿ ಈರುಳ್ಳಿ",
        "confidence_caption": "ಪ್ರತಿ ಚುಕ್ಕೆ ಒಂದು ಈರುಳ್ಳಿ. 50% ಗಿಂತ ಕಡಿಮೆ = ಅಮಾನ್ಯ.",
        "report_heading":     "ಡಿಜಿಟಲ್ ವರದಿ",
        "export_heading":     "ಫಲಿತಾಂಶಗಳನ್ನು ರಫ್ತು ಮಾಡಿ",
        "export_csv":         "⬇️ CSV ರಫ್ತು",
        "export_json":        "⬇️ JSON ರಫ್ತು",
        "export_pdf":         "📄 PDF ರಫ್ತು",
        "history_heading":    "ಬ್ಯಾಚ್ ಇತಿಹಾಸ",
        "history_empty":      "ಈ ಅಧಿವೇಶನದಲ್ಲಿ ಹಿಂದಿನ ಬ್ಯಾಚ್‌ಗಳಿಲ್ಲ.",
        "compare_heading":    "ಬ್ಯಾಚ್ ಹೋಲಿಕೆ",
        "trend_heading":      "ಶ್ರೇಣಿ A% ಪ್ರವೃತ್ತಿ",
        "img_quality_warn":   "⚠️ ಚಿತ್ರದ ಗುಣಮಟ್ಟ ಕಡಿಮೆಯಿರಬಹುದು (ಮಬ್ಬಾದ ಅಥವಾ ಕತ್ತಲೆ). ಫಲಿತಾಂಶಗಳು ಕಡಿಮೆ ನಿಖರವಾಗಿರಬಹುದು.",
        "status_good":        "ಉತ್ತಮ",
        "status_acceptable":  "ಸ್ವೀಕಾರಾರ್ಹ",
        "status_poor":        "ಕಳಪೆ",
        "recommendation":     "ಶಿಫಾರಸು",
        "batch_quality":      "ಬ್ಯಾಚ್ ಗುಣಮಟ್ಟ ಸ್ಥಿತಿ",
        "flagged_title":      "ಅಮಾನ್ಯ ದಾಖಲೆಗಳು",
        "flagged_caption":    "ಈ ದಾಖಲೆಗಳು ಮೌಲ್ಯೀಕರಣ ಪಾಸ್ ಮಾಡದ ಕಾರಣ ಶ್ರೇಣೀಕರಣದಿಂದ ಹೊರಗಿಡಲಾಗಿದೆ.",
        "no_defects":         "✓ ದೋಷಪೂರಿತ ಈರುಳ್ಳಿಗಳು ಪತ್ತೆಯಾಗಿಲ್ಲ",
        "no_defects_sub":     "ಮೌಲ್ಯೀಕೃತ ಬ್ಯಾಚ್‌ನಲ್ಲಿ ದೋಷಪೂರಿತ ದಾಖಲೆಗಳು ಪತ್ತೆಯಾಗಿಲ್ಲ.",
        "results_placeholder":"ಫಲಿತಾಂಶಗಳು ಇಲ್ಲಿ ಕಾಣಿಸುತ್ತವೆ",
        "results_sub":        "ಬ್ಯಾಚ್ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ಮತ್ತು ವಿಶ್ಲೇಷಿಸಿ ಆಯ್ಕೆ ಮಾಡಿ.",
        "grade_reason_col":   "ಶ್ರೇಣಿ ಕಾರಣ",
        "image_num_col":      "#",
        "onion_id_col":       "ಈರುಳ್ಳಿ ID",
        "size_col":           "ಗಾತ್ರ",
        "colour_col":         "ಬಣ್ಣ",
        "severity_col":       "ತೀವ್ರತೆ",
        "defect_col":         "ದೋಷ",
        "defect_desc_col":    "ದೋಷ ವಿವರಣೆ",
        "confidence_col":     "ವಿಶ್ವಾಸ",
        "spinner_msg":        "Gemini Vision ವಿಶ್ಲೇಷಣೆ → ಮೌಲ್ಯೀಕರಣ → ಶ್ರೇಣೀಕರಣ → ವರದಿ…",
        "final_grade_col":    "ಅಂತಿಮ ಶ್ರೇಣಿ",
        "inspection_mode_label": "ತಪಾಸಣೆ ಮೋಡ್",
        "mode_rgb":           "🔴 ಬಾಹ್ಯ RGB ಶ್ರೇಣೀಕರಣ",
        "mode_nir":           "🔵 ಆಂತರಿಕ NIR ಸ್ಕ್ರೀನಿಂಗ್",
        "mode_rgb_desc":      "ಬಾಹ್ಯ ಮೇಲ್ಮೈ ಗುಣಮಟ್ಟ ಆಧಾರದ ಮೇಲೆ ಈರುಳ್ಳಿ ಶ್ರೇಣೀಕರಿಸುತ್ತದೆ — ಗಾತ್ರ, ಬಣ್ಣ, ಚರ್ಮ, ಮೊಳಕೆ ಮತ್ತು ಗೋಚರ ದೋಷಗಳು.",
        "mode_nir_desc":      "ಆಂತರಿಕ ದೋಷಗಳನ್ನು ಸ್ಕ್ರೀನ್ ಮಾಡುತ್ತದೆ — ಒಳ ಕೊಳೆತ ಮತ್ತು ತೇವಾಂಶ ಕಡಿಮೆ, ಬರಿಗಣ್ಣಿಗೆ ಕಾಣಿಸದ್ದು.",
        "step_upload":        "ಅಪ್‌ಲೋಡ್",
        "step_perceive":      "ಗ್ರಹಿಕೆ",
        "step_validate":      "ಮೌಲ್ಯೀಕರಣ",
        "step_grade":         "ಶ್ರೇಣೀಕರಣ",
        "step_report":        "ವರದಿ",
        "prog_uploading":     "⬆ ಅಪ್‌ಲೋಡ್ ಆಗುತ್ತಿದೆ",
        "prog_processing":    "⚙ ಪ್ರಕ್ರಿಯೆ ನಡೆಯುತ್ತಿದೆ",
        "prog_analyzing":     "🔍 ವಿಶ್ಲೇಷಣೆ ನಡೆಯುತ್ತಿದೆ",
        "prog_preparing":     "📊 ಸಿದ್ಧಪಡಿಸಲಾಗುತ್ತಿದೆ",
        "prog_complete":      "✓ ಪೂರ್ಣಗೊಂಡಿದೆ",
        "cv_mode_heading":    "ಸಾರಾಂಶ ದೃಶ್ಯ",
        "cv_batch_info":      "ಬ್ಯಾಚ್ ಮಾಹಿತಿ",
        "cv_grade_breakdown": "ಶ್ರೇಣಿ ವಿವರ",
        "cv_top_defects":     "ಪ್ರಮುಖ ದೋಷಗಳು",
        "cv_ai_confidence":   "AI ವಿಶ್ವಾಸ ಮಟ್ಟ",
        "cv_quick_actions":   "ತ್ವರಿತ ಕ್ರಿಯೆಗಳು",
        "cv_avg_confidence":  "ಸರಾಸರಿ ವಿಶ್ವಾಸ",
        "cv_highest_conf":    "ಹೆಚ್ಚು ವಿಶ್ವಾಸಾರ್ಹ ಈರುಳ್ಳಿ",
        "cv_lowest_conf":     "ಕಡಿಮೆ ವಿಶ್ವಾಸಾರ್ಹ ಈರುಳ್ಳಿ",
        "cv_flagged_count":   "ಅಮಾನ್ಯ ದಾಖಲೆಗಳು",
        "cv_batch_status":    "ಬ್ಯಾಚ್ ಸ್ಥಿತಿ",
        "cv_total_onions":    "ಒಟ್ಟು ಈರುಳ್ಳಿಗಳು",
        "cv_export_hint":     "ಫಲಿತಾಂಶಗಳನ್ನು ಡೌನ್‌ಲೋಡ್ ಮಾಡಲು ಕೆಳಗಿನ ರಫ್ತು ಬಟನ್‌ಗಳನ್ನು ಬಳಸಿ.",
        "camera_closed_title":"ಕ್ಯಾಮೆರಾ ಮುಚ್ಚಲಾಗಿದೆ",
        "camera_closed_sub":  "ಇನ್ನೂ ಏನೂ ಸಕ್ರಿಯಗೊಂಡಿಲ್ಲ — ಕ್ಯಾಮೆರಾ ಆನ್ ಮಾಡಲು ಕೆಳಗೆ ಒತ್ತಿ.",
        "camera_open_btn":    "📷 ಕ್ಯಾಮೆರಾ ತೆರೆಯಿರಿ",
        "camera_close_btn":   "✕ ಕ್ಯಾಮೆರಾ ಮುಚ್ಚಿರಿ",
        "health_offline":     "**ಆಫ್‌ಲೈನ್ ಮೋಡ್** — Gemini API ಕೀ ಪತ್ತೆಯಾಗಿಲ್ಲ. ಸ್ಥಳೀಯ ವಿಶ್ಲೇಷಣೆ ಚಾಲ್ತಿಯಲ್ಲಿದೆ.",
        "health_online":      "**Gemini Vision ಸಕ್ರಿಯ** — AI-ಚಾಲಿತ ಪ್ರತಿ-ಈರುಳ್ಳಿ ಗ್ರಹಿಕೆ ಸಕ್ರಿಯಗೊಂಡಿದೆ.",
        "health_error":       "ಬ್ಯಾಕೆಂಡ್‌ಗೆ ಸಂಪರ್ಕ ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ಸರ್ವರ್ ಪ್ರಾರಂಭಿಸಿ.",
        "no_validated_records": "ಯಾವುದೇ ಮೌಲ್ಯೀಕೃತ ಈರುಳ್ಳಿ ದಾಖಲೆಗಳು ಲಭ್ಯವಿಲ್ಲ.",
        "select_rgb":         "RGB ಆಯ್ಕೆ ಮಾಡಿ",
        "rgb_selected":       "✓ RGB ಆಯ್ಕೆ",
        "select_nir":         "NIR ಆಯ್ಕೆ ಮಾಡಿ",
        "nir_selected":       "✓ NIR ಆಯ್ಕೆ",
        "excluded_label":     "ಹೊರಗಿಡಲಾಗಿದೆ",
        "overridden_note":    "ಶ್ರೇಣಿಗಳನ್ನು ಕೈಯಾರೆ ಬದಲಿಸಲಾಗಿದೆ",
    },
    "bn": {
        "app_subtitle":       "AI দেখে, Python সিদ্ধান্ত নেয় — পেঁয়াজ গুণমান গ্রেডিং",
        "upload_heading":     "পেঁয়াজ ব্যাচের ছবি আপলোড করুন",
        "upload_caption":     "স্পষ্ট ব্যাচ ছবি আপলোড করুন (JPEG, PNG বা WebP, সর্বোচ্চ 20 MB)।",
        "camera_heading":     "ছবি তুলুন",
        "camera_caption":     "সরাসরি ক্যামেরা থেকে ছবি তুলুন — ফাইল সেভ করার দরকার নেই।",
        "batch_id_label":     "ব্যাচ ID (ঐচ্ছিক)",
        "batch_id_help":      "আপনার ব্যাচ নম্বর লিখুন, যেমন LOT-WB-001। খালি রাখলে স্বয়ংক্রিয়ভাবে তৈরি হবে।",
        "analyse_btn":        "ব্যাচ বিশ্লেষণ করুন →",
        "new_analysis_btn":   "🔄 নতুন বিশ্লেষণ",
        "grade_a":            "গ্রেড A",
        "urs":                "URS",
        "defect":             "ত্রুটিপূর্ণ",
        "sprouting":          "অঙ্কুরোদগম",
        "rot":                "পচন",
        "mould":              "ছত্রাক",
        "bruising":           "আঘাত",
        "discoloration":      "রঙ পরিবর্তন",
        "decay":              "ক্ষয়",
        "physical_damage":    "শারীরিক ক্ষতি",
        "none":               "কোনোটি নয়",
        "other":              "অন্যান্য",
        "onions":             "পেঁয়াজ",
        "per_onion_heading":  "প্রতিটি পেঁয়াজ গ্রেডিং রেকর্ড",
        "per_onion_caption":  "ছবিতে দেখানো নম্বর এই টেবিলের সারির সাথে মেলে।",
        "override_col":       "গ্রেড পরিবর্তন করুন",
        "override_note":      "গ্রেড পরিবর্তন করুন — শতাংশ তাৎক্ষণিকভাবে পুনরায় গণনা হবে। পরিবর্তিত রেকর্ড PDF-এ চিহ্নিত হবে।",
        "rule_engine_title":  "⚙️ গ্রেড কীভাবে নির্ধারণ হয়? — নিয়ম ইঞ্জিন দেখুন",
        "defect_heading":     "ত্রুটি বিশ্লেষণ",
        "quality_heading":    "সামগ্রিক ব্যাচ মূল্যায়ন",
        "confidence_heading": "AI উপলব্ধি আস্থা — প্রতিটি পেঁয়াজ",
        "confidence_caption": "প্রতিটি বিন্দু একটি পেঁয়াজ। 50%-এর নিচে = অবৈধ।",
        "report_heading":     "ডিজিটাল প্রতিবেদন",
        "export_heading":     "ফলাফল রপ্তানি করুন",
        "export_csv":         "⬇️ CSV রপ্তানি",
        "export_json":        "⬇️ JSON রপ্তানি",
        "export_pdf":         "📄 PDF রপ্তানি",
        "history_heading":    "ব্যাচ ইতিহাস",
        "history_empty":      "এই সেশনে কোনো পূর্ববর্তী ব্যাচ নেই।",
        "compare_heading":    "ব্যাচ তুলনা",
        "trend_heading":      "গ্রেড A% প্রবণতা",
        "img_quality_warn":   "⚠️ ছবির মান কম হতে পারে (ঝাপসা বা অন্ধকার)। ফলাফল কম নির্ভুল হতে পারে।",
        "status_good":        "ভালো",
        "status_acceptable":  "গ্রহণযোগ্য",
        "status_poor":        "খারাপ",
        "recommendation":     "সুপারিশ",
        "batch_quality":      "ব্যাচ মান অবস্থা",
        "flagged_title":      "অবৈধ রেকর্ড",
        "flagged_caption":    "এই রেকর্ডগুলি যাচাইকরণ পাস না করায় গ্রেডিং থেকে বাদ দেওয়া হয়েছে।",
        "no_defects":         "✓ কোনো ত্রুটিপূর্ণ পেঁয়াজ পাওয়া যায়নি",
        "no_defects_sub":     "যাচাইকৃত ব্যাচে কোনো ত্রুটিপূর্ণ রেকর্ড পাওয়া যায়নি।",
        "results_placeholder":"ফলাফল এখানে প্রদর্শিত হবে",
        "results_sub":        "ব্যাচ ছবি আপলোড করুন এবং বিশ্লেষণ করুন নির্বাচন করুন।",
        "grade_reason_col":   "গ্রেড কারণ",
        "image_num_col":      "#",
        "onion_id_col":       "পেঁয়াজ ID",
        "size_col":           "আকার",
        "colour_col":         "রঙ",
        "severity_col":       "তীব্রতা",
        "defect_col":         "ত্রুটি",
        "defect_desc_col":    "ত্রুটি বিবরণ",
        "confidence_col":     "আস্থা",
        "spinner_msg":        "Gemini Vision বিশ্লেষণ → যাচাইকরণ → গ্রেডিং → প্রতিবেদন…",
        "final_grade_col":    "চূড়ান্ত গ্রেড",
        "inspection_mode_label": "পরিদর্শন মোড",
        "mode_rgb":           "🔴 বাহ্যিক RGB গ্রেডিং",
        "mode_nir":           "🔵 অভ্যন্তরীণ NIR স্ক্রিনিং",
        "mode_rgb_desc":      "বাহ্যিক পৃষ্ঠের মান অনুযায়ী পেঁয়াজ গ্রেড করে — আকার, রঙ, খোসা, অঙ্কুরোদগম এবং দৃশ্যমান ত্রুটি।",
        "mode_nir_desc":      "অভ্যন্তরীণ ত্রুটি স্ক্রিন করে — ভেতরের পচন এবং আর্দ্রতা হ্রাস যা খালি চোখে দেখা যায় না।",
        "step_upload":        "আপলোড",
        "step_perceive":      "শনাক্তকরণ",
        "step_validate":      "যাচাইকরণ",
        "step_grade":         "গ্রেডিং",
        "step_report":        "প্রতিবেদন",
        "prog_uploading":     "⬆ আপলোড হচ্ছে",
        "prog_processing":    "⚙ প্রক্রিয়াকরণ চলছে",
        "prog_analyzing":     "🔍 বিশ্লেষণ চলছে",
        "prog_preparing":     "📊 প্রস্তুত করা হচ্ছে",
        "prog_complete":      "✓ সম্পন্ন",
        "cv_mode_heading":    "সারসংক্ষেপ দৃশ্য",
        "cv_batch_info":      "ব্যাচ তথ্য",
        "cv_grade_breakdown": "গ্রেড বিবরণ",
        "cv_top_defects":     "প্রধান ত্রুটি",
        "cv_ai_confidence":   "AI বিশ্বাসযোগ্যতা স্তর",
        "cv_quick_actions":   "দ্রুত পদক্ষেপ",
        "cv_avg_confidence":  "গড় বিশ্বাসযোগ্যতা",
        "cv_highest_conf":    "সর্বোচ্চ বিশ্বাসযোগ্য পেঁয়াজ",
        "cv_lowest_conf":     "সর্বনিম্ন বিশ্বাসযোগ্য পেঁয়াজ",
        "cv_flagged_count":   "অবৈধ রেকর্ড",
        "cv_batch_status":    "ব্যাচ অবস্থা",
        "cv_total_onions":    "মোট পেঁয়াজ",
        "cv_export_hint":     "ফলাফল ডাউনলোড করতে নিচের রপ্তানি বোতামগুলি ব্যবহার করুন।",
        "camera_closed_title":"ক্যামেরা বন্ধ আছে",
        "camera_closed_sub":  "এখনও কিছুই সক্রিয় নয় — ক্যামেরা চালু করতে নিচে চাপুন।",
        "camera_open_btn":    "📷 ক্যামেরা খুলুন",
        "camera_close_btn":   "✕ ক্যামেরা বন্ধ করুন",
        "health_offline":     "**অফলাইন মোড** — Gemini API কী পাওয়া যায়নি। স্থানীয় বিশ্লেষণ চলছে।",
        "health_online":      "**Gemini Vision সক্রিয়** — AI-চালিত প্রতি-পেঁয়াজ শনাক্তকরণ সক্ষম।",
        "health_error":       "ব্যাকএন্ডে সংযোগ করা যাচ্ছে না। সার্ভার চালু করুন।",
        "no_validated_records": "কোনো যাচাইকৃত পেঁয়াজ রেকর্ড পাওয়া যায়নি।",
        "select_rgb":         "RGB বেছে নিন",
        "rgb_selected":       "✓ RGB বেছে নেওয়া হয়েছে",
        "select_nir":         "NIR বেছে নিন",
        "nir_selected":       "✓ NIR বেছে নেওয়া হয়েছে",
        "excluded_label":     "বাদ দেওয়া হয়েছে",
        "overridden_note":    "গ্রেড ম্যানুয়ালি পরিবর্তন করা হয়েছে",
    },
}

def T(key: str) -> str:
    """Return translated string for current language."""
    lang = st.session_state.get("lang", "en")
    return _TRANSLATIONS.get(lang, _TRANSLATIONS["en"]).get(key, key)


def md(html: str) -> None:
    """Render raw HTML safely using Streamlit's HTML renderer."""
    st.html(html)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def pct(value) -> float:
    """Safely convert a value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_text(value, default="-") -> str:
    """Convert values to displayable text."""
    if value is None or value == "":
        return default
    return str(value)


def get_batch_id(data: dict) -> str:
    """
    Support the new structured response.

    New:
        data["batch"]["batch_id"]

    Fallback:
        data["batch_id"]
    """
    return (
        data.get("batch", {}).get("batch_id")
        or data.get("batch_id")
        or "batch"
    )


def get_report_text(data: dict) -> str:
    """
    Support:
        report: "string"

    or:
        report: {"text": "..."}
    """
    report = data.get("report", "")

    if isinstance(report, dict):
        return str(
            report.get("text")
            or report.get("content")
            or report.get("report")
            or ""
        )

    return str(report)


def get_onion_records(data: dict) -> list:
    """Get validated onions from the new response structure."""
    return data.get("onions") or data.get("validated") or []


# ---------------------------------------------------------------------------
# Image quality check (blur + brightness)
# ---------------------------------------------------------------------------

def check_image_quality(image_bytes: bytes) -> str | None:
    """Return a warning string if image is too dark or blurry, else None."""
    try:
        import numpy as np
        from PIL import Image as _PILImage
        img = _PILImage.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
        arr = np.array(img, dtype=np.float32)
        # Brightness: mean pixel value (0-255). Below 40 = very dark.
        brightness = float(arr.mean())
        # Blur: variance of Laplacian. Below 40 = blurry.
        import PIL.ImageFilter as _F
        lap = img.filter(_F.FIND_EDGES)
        blur_score = float(np.array(lap, dtype=np.float32).var())
        if brightness < 40:
            return T("img_quality_warn") + f" (brightness={brightness:.0f})"
        if blur_score < 40:
            return T("img_quality_warn") + f" (sharpness={blur_score:.0f})"
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Onion image marker helpers
# ---------------------------------------------------------------------------

def get_onion_number(onion_id: str, fallback: int) -> int:
    """
    Extract the numeric part from onion IDs.

    Examples:
        onion_01 -> 1
        onion_02 -> 2
        onion_15 -> 15

    The fallback keeps the function safe if the ID has no number.
    """
    if not onion_id:
        return fallback

    match = re.search(r"(\d+)$", str(onion_id))

    if match:
        return int(match.group(1))

    return fallback


def draw_onion_markers(
    image_bytes: bytes,
    onions: list,
) -> bytes:
    """
    Draw numbered bounding boxes on the uploaded onion image.

    The number comes from onion_id, not enumerate(), so:

        onion_01 -> #1
        onion_02 -> #2
        onion_03 -> #3

    This prevents numbering from changing when an onion is flagged.
    """

    if not image_bytes:
        return image_bytes

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:
        return image_bytes

    draw = ImageDraw.Draw(image)

    image_width, image_height = image.size

    # Try to use a readable font.
    try:
        font = ImageFont.truetype(
            "arial.ttf",
            max(16, int(image_width * 0.025)),
        )
    except Exception:
        font = ImageFont.load_default()

    try:
        small_font = ImageFont.truetype(
            "arial.ttf",
            max(13, int(image_width * 0.018)),
        )
    except Exception:
        small_font = ImageFont.load_default()

    for fallback_index, onion in enumerate(
        onions,
        start=1,
    ):

        if not isinstance(onion, dict):
            continue

        onion_id = str(
            onion.get(
                "onion_id",
                "",
            )
        )

        bbox = onion.get("bbox")

        # If Gemini did not return a bounding box,
        # keep the table record but don't draw a fake box.
        if not isinstance(bbox, dict):
            continue

        try:
            x = float(bbox.get("x", 0))
            y = float(bbox.get("y", 0))
            width = float(
                bbox.get("width", 0)
            )
            height = float(
                bbox.get("height", 0)
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        # Ignore invalid boxes.
        if width <= 0 or height <= 0:
            continue

        # Coordinates are normalized 0-1.
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        left = int(
            x * image_width
        )
        top = int(
            y * image_height
        )

        right = int(
            (x + width) * image_width
        )
        bottom = int(
            (y + height) * image_height
        )

        # Keep box inside image.
        left = max(
            0,
            min(image_width - 1, left),
        )

        top = max(
            0,
            min(image_height - 1, top),
        )

        right = max(
            left + 1,
            min(image_width - 1, right),
        )

        bottom = max(
            top + 1,
            min(image_height - 1, bottom),
        )

        # The image marker is always sequential: #1, #2, #3, ...
        # Backend IDs are normalized to onion_01, onion_02, onion_03, ...
        number = fallback_index

        label = f"#{number}"

        # ---------------------------------------------------------------
        # Bounding box
        # ---------------------------------------------------------------

        draw.rectangle(
            [
                left,
                top,
                right,
                bottom,
            ],
            outline="#D9A441",
            width=max(
                3,
                int(image_width * 0.006),
            ),
        )

        # ---------------------------------------------------------------
        # Marker position
        # ---------------------------------------------------------------

        marker_radius = max(
            15,
            int(image_width * 0.025),
        )

        # Put the number at the upper-left corner of the actual box.
        # If there is not enough room above the box, keep it just inside the image.
        marker_center_x = left + marker_radius
        marker_center_y = top + marker_radius

        # Keep marker inside image.
        marker_center_x = max(
            marker_radius,
            min(
                image_width - marker_radius,
                marker_center_x,
            ),
        )

        marker_center_y = max(
            marker_radius,
            min(
                image_height - marker_radius,
                marker_center_y,
            ),
        )

        # ---------------------------------------------------------------
        # Marker circle
        # ---------------------------------------------------------------

        draw.ellipse(
            [
                marker_center_x - marker_radius,
                marker_center_y - marker_radius,
                marker_center_x + marker_radius,
                marker_center_y + marker_radius,
            ],
            fill="#5B2333",
            outline="#FFFFFF",
            width=max(
                2,
                int(image_width * 0.003),
            ),
        )

        # ---------------------------------------------------------------
        # Marker text
        # ---------------------------------------------------------------

        try:
            text_box = draw.textbbox(
                (0, 0),
                label,
                font=font,
            )

            text_width = (
                text_box[2] - text_box[0]
            )

            text_height = (
                text_box[3] - text_box[1]
            )

        except Exception:
            text_width = 12
            text_height = 12

        draw.text(
            (
                marker_center_x - text_width / 2,
                marker_center_y - text_height / 2 - 2,
            ),
            label,
            fill="#FFFFFF",
            font=font,
        )

        # ---------------------------------------------------------------
        # Small ID label
        # ---------------------------------------------------------------

        if onion_id:

            id_label = onion_id

            try:
                id_box = draw.textbbox(
                    (0, 0),
                    id_label,
                    font=small_font,
                )

                id_width = (
                    id_box[2] - id_box[0]
                )

                id_height = (
                    id_box[3] - id_box[1]
                )

            except Exception:
                id_width = 60
                id_height = 14

            label_x = left
            label_y = bottom + 5

            if (
                label_y + id_height + 8
                > image_height
            ):
                label_y = max(
                    2,
                    top - id_height - 12,
                )

            # Background rectangle.
            draw.rounded_rectangle(
                [
                    label_x,
                    label_y,
                    min(
                        image_width - 2,
                        label_x + id_width + 12,
                    ),
                    min(
                        image_height - 2,
                        label_y + id_height + 8,
                    ),
                ],
                radius=5,
                fill="#431A26",
            )

            draw.text(
                (
                    label_x + 6,
                    label_y + 3,
                ),
                id_label,
                fill="#FFFFFF",
                font=small_font,
            )

    output = io.BytesIO()

    image.save(
        output,
        format="PNG",
    )

    return output.getvalue()

def get_summary(data: dict) -> dict:
    """
    Return summary from the new structured API.

    Falls back to grade_result for compatibility.
    """

    summary = data.get("summary")

    if summary:
        return summary

    gr = data.get("grade_result", {})

    return {
        "grade_a": {
            "count": gr.get("grade_a_count", 0),
            "percentage": gr.get("grade_a_pct", 0),
        },
        "urs": {
            "count": gr.get("urs_count", 0),
            "percentage": gr.get("urs_pct", 0),
        },
        "defect": {
            "count": gr.get("defect_count", 0),
            "percentage": gr.get("defect_pct", 0),
        },
    }


def get_overall_quality(data: dict) -> dict:
    """Get batch-level quality assessment."""
    quality = data.get("overall_quality")

    if isinstance(quality, dict):
        return quality

    return {
        "status": "NOT AVAILABLE",
        "description": "Batch assessment was not provided.",
        "recommendation": "Perform manual quality inspection.",
    }


# ---------------------------------------------------------------------------
# Progress ring
# ---------------------------------------------------------------------------

def ring_svg(
    percentage: float,
    color: str,
    track: str = "var(--ring-track,#EFE6D6)",
    size: int = 118,
    stroke: int = 11,
) -> str:

    percentage = min(max(pct(percentage), 0), 100)

    radius = (size - stroke) / 2
    circumference = 2 * math.pi * radius

    offset = circumference * (
        1 - percentage / 100
    )

    cx = cy = size / 2

    return f"""
    <svg width="{size}" height="{size}"
         viewBox="0 0 {size} {size}"
         style="transform:rotate(-90deg);">

      <circle
        cx="{cx}"
        cy="{cy}"
        r="{radius}"
        fill="none"
        stroke="{track}"
        stroke-width="{stroke}"
      />

      <circle
        cx="{cx}"
        cy="{cy}"
        r="{radius}"
        fill="none"
        stroke="{color}"
        stroke-width="{stroke}"
        stroke-dasharray="{circumference:.2f}"
        stroke-dashoffset="{offset:.2f}"
        stroke-linecap="round"
      />

    </svg>
    """


def metric_ring_card(
    label: str,
    percentage: float,
    count: int,
    color: str,
    background: str,
) -> str:

    ring = ring_svg(
        percentage,
        color,
    )

    return f"""
    <div style="
        flex:1;
        background:var(--surface);
        border:1px solid var(--line);
        border-radius:18px;
        padding:22px 16px 20px;
        text-align:center;
        box-shadow:0 2px 12px var(--shadow);
        transition:box-shadow 0.22s ease;
    ">
      <div style="
          position:relative;
          width:118px;
          height:118px;
          margin:0 auto;
      ">
        {ring}
        <div style="
            position:absolute;
            inset:0;
            display:flex;
            align-items:center;
            justify-content:center;
        ">
          <div style="
              font-family:'Fraunces',Georgia,serif;
              font-weight:700;
              font-size:22px;
              color:{color};
              letter-spacing:-0.02em;
          ">
              {pct(percentage):.1f}%
          </div>
        </div>
      </div>

      <div style="
          margin-top:12px;
          font-size:11px;
          font-weight:700;
          text-transform:uppercase;
          letter-spacing:0.07em;
          color:var(--ink-soft);
      ">
          {escape(label)}
      </div>

      <div style="
          font-size:13px;
          font-weight:600;
          color:var(--ink);
          margin-top:3px;
      ">
          {count} <span style="font-weight:400;color:var(--ink-soft);font-size:11px;">{T("onions")}</span>
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Stepper
# ---------------------------------------------------------------------------

def stepper(active_step: int) -> str:
    lang = st.session_state.get("lang", "en")
    steps = [
        _TRANSLATIONS.get(lang, _TRANSLATIONS["en"]).get("step_upload",  "Upload"),
        _TRANSLATIONS.get(lang, _TRANSLATIONS["en"]).get("step_perceive","Perceive"),
        _TRANSLATIONS.get(lang, _TRANSLATIONS["en"]).get("step_validate","Validate"),
        _TRANSLATIONS.get(lang, _TRANSLATIONS["en"]).get("step_grade",   "Grade"),
        _TRANSLATIONS.get(lang, _TRANSLATIONS["en"]).get("step_report",  "Report"),
    ]

    items = []

    for index, name in enumerate(steps):

        done = index < active_step
        current = index == active_step

        if done:
            bg = "var(--plum)"
            fg = "#fff"
            border = "var(--plum)"

        elif current:
            bg = "var(--gold)"
            fg = "#fff"
            border = "var(--gold)"

        else:
            bg = "transparent"
            fg = "var(--ink-soft)"
            border = "var(--line)"

        text_color = (
            "var(--ink)"
            if (done or current)
            else "var(--ink-soft)"
        )

        items.append(
            f"""
            <div style="
                display:flex;
                align-items:center;
                gap:8px;
            ">

                <div style="
                    width:26px;
                    height:26px;
                    border-radius:50%;
                    background:{bg};
                    border:1.5px solid {border};
                    color:{fg};
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-size:12px;
                    font-weight:700;
                    flex-shrink:0;
                ">
                    {index + 1}
                </div>

                <span style="
                    font-size:12.5px;
                    font-weight:600;
                    color:{text_color};
                    white-space:nowrap;
                ">
                    {name}
                </span>

            </div>
            """
        )

        if index < len(steps) - 1:

            line_color = (
                "var(--plum)"
                if done
                else "var(--line)"
            )

            items.append(
                f"""
                <div style="
                    flex:1;
                    height:1.5px;
                    background:{line_color};
                    min-width:16px;
                "></div>
                """
            )

    return (
        '<div style="display:flex;align-items:center;gap:10px;'
        'padding:2px 4px;">'
        + "".join(items)
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Flagged record
# ---------------------------------------------------------------------------

def flagged_record_card(record: dict) -> str:

    raw = record.get("raw", {}) or {}

    reason = record.get(
        "reason",
        "Invalid record",
    )

    onion_id = raw.get(
        "onion_id",
        "Unknown onion",
    )

    details = []

    if raw.get("size"):
        details.append(str(raw["size"]))

    if raw.get("color"):
        details.append(str(raw["color"]))

    if raw.get("defect_description"):
        details.append(
            str(raw["defect_description"])
        )

    detail = (
        " &middot; ".join(details)
        if details
        else "No further detail captured"
    )

    return f"""
    <div style="
        display:flex;
        align-items:flex-start;
        gap:12px;
        background:var(--rust-soft);
        border:1px solid var(--rust);
        border-radius:14px;
        padding:13px 17px;
        margin-bottom:8px;
        opacity:0.9;
    ">
        <div style="
            width:30px;height:30px;border-radius:50%;
            background:var(--rust);color:#fff;
            display:flex;align-items:center;justify-content:center;
            font-size:13px;font-weight:700;flex-shrink:0;margin-top:1px;
        ">!</div>
        <div>
            <div style="font-weight:700;font-size:13.5px;color:var(--rust);">
                {escape(str(onion_id))}
            </div>
            <div style="font-size:12.5px;color:var(--ink-soft);margin-top:2px;">
                {escape(detail)}
            </div>
            <div style="font-size:12px;color:var(--ink-soft);margin-top:4px;">
                {T("excluded_label")}: {escape(str(reason))}
            </div>
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Defect analysis
# ---------------------------------------------------------------------------

def render_defect_analysis(data: dict) -> None:

    analysis = data.get(
        "defect_analysis",
        {},
    ) or {}

    categories = (
        analysis.get("categories", {})
        or {}
    )

    total_defective = analysis.get(
        "total_defective",
        0,
    )

    st.markdown(f"### {T('defect_heading')}")

    if total_defective == 0:

        md(f"""
        <div class="og-card">
            <div style="
                font-size:15px;
                font-weight:700;
                color:var(--sage);
            ">
                {T("no_defects")}
            </div>

            <div style="
                margin-top:6px;
                color:var(--ink-soft);
                font-size:13px;
            ">
                {T("no_defects_sub")}
            </div>
        </div>
        """)

        return

    defect_rows = []

    for category, count in categories.items():

        if count:
            defect_rows.append(
                {
                    "Defect Type": category,
                    "Count": int(count),
                }
            )

    if defect_rows:

        cols = st.columns(min(len(defect_rows), 4))

        for index, row in enumerate(defect_rows):
            with cols[index % len(cols)]:
                md(f"""
                <div class="og-card og-slide-up" style="
                    padding:18px 14px;text-align:center;margin-bottom:12px;
                    border-top:3px solid var(--rust);
                ">
                    <div style="
                        font-size:30px;font-weight:700;
                        color:var(--rust);
                        font-family:'Fraunces',serif;
                    ">{row["Count"]}</div>
                    <div style="
                        margin-top:5px;font-size:11px;font-weight:700;
                        color:var(--ink-soft);text-transform:uppercase;letter-spacing:0.05em;
                    ">{escape(str(row["Defect Type"]))}</div>
                </div>
                """)

        # Defect category horizontal bar chart
        defect_fig = go.Figure(
            go.Bar(
                x=[r["Count"] for r in defect_rows],
                y=[r["Defect Type"] for r in defect_rows],
                orientation="h",
                marker_color="#F87171" if st.session_state.get("theme_mode","light") == "dark" else THEME["rust"],
                text=[str(r["Count"]) for r in defect_rows],
                textposition="outside",
                marker_line_width=0,
            )
        )
        _is_dark = st.session_state.get("theme_mode", "light") == "dark"
        _chart_bg = "rgba(0,0,0,0)"
        _chart_font_color = "#F0EDE8" if _is_dark else "#1C1009"
        _chart_grid = "rgba(255,255,255,0.08)" if _is_dark else "#E8DDD0"
        _plot_bg = "rgba(0,0,0,0)"
        defect_fig.update_layout(
            height=max(160, len(defect_rows) * 44),
            margin=dict(l=10, r=30, t=10, b=10),
            xaxis_title="Onion count",
            plot_bgcolor=_plot_bg,
            paper_bgcolor=_chart_bg,
            font=dict(family="Inter, IBM Plex Sans, sans-serif", color=_chart_font_color),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(gridcolor=_chart_grid),
        )
        st.plotly_chart(defect_fig, config={"displayModeBar": False}, use_container_width=True)


# ---------------------------------------------------------------------------
# Overall quality card
# ---------------------------------------------------------------------------

def render_overall_quality(data: dict) -> None:

    quality = get_overall_quality(data)

    status = str(
        quality.get(
            "status",
            "NOT AVAILABLE",
        )
    ).upper()

    description = quality.get(
        "description",
        "",
    )

    recommendation = quality.get(
        "recommendation",
        "",
    )

    if status == "GOOD":
        status_color = "var(--sage)"
        background   = "var(--sage-soft)"
        icon         = "✓"
    elif status == "ACCEPTABLE":
        status_color = "var(--copper-deep)"
        background   = "rgba(200,144,42,0.10)"
        icon         = "!"
    else:
        status_color = "var(--rust)"
        background   = "var(--rust-soft)"
        icon         = "⚠"

    st.markdown(f"### {T('quality_heading')}")

    md(f"""
    <div class="og-card og-fade-in" style="
        background:{background};
        border-left:5px solid {status_color};
        padding:22px 26px;
    ">

        <div style="
            display:flex;
            align-items:center;
            gap:12px;
        ">

            <div style="
                width:42px;
                height:42px;
                border-radius:50%;
                background:{status_color};
                color:white;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:20px;
                font-weight:700;
            ">
                {icon}
            </div>

            <div>

                <div style="
                    font-size:11px;
                    text-transform:uppercase;
                    font-weight:700;
                    letter-spacing:.06em;
                    color:var(--ink-soft);
                ">
                    {T("batch_quality")}
                </div>

                <div style="
                    font-family:'Fraunces',serif;
                    font-size:25px;
                    font-weight:700;
                    color:{status_color};
                ">
                    {escape(status)}
                </div>

            </div>

        </div>

        <div style="
            margin-top:14px;
            font-size:13.5px;
            line-height:1.6;
            color:var(--ink);
        ">
            {escape(str(description))}
        </div>

        <div style="
            margin-top:12px;
            padding-top:12px;
            border-top:1px solid rgba(91,35,51,.12);
        ">

            <div style="
                font-size:11px;
                text-transform:uppercase;
                font-weight:700;
                color:var(--ink-soft);
            ">
                {T("recommendation")}
            </div>

            <div style="
                margin-top:4px;
                font-size:13px;
                line-height:1.55;
                color:var(--ink);
            ">
                {escape(str(recommendation))}
            </div>

        </div>

    </div>
    """)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _display_severity(onion: dict, grade: str) -> str:
    severity = onion.get("severity")
    if severity:
        return str(severity)
    if grade == "Defect":
        return "Major"
    if grade == "URS":
        return "Minor"
    return "None"


def _display_defect_type(onion: dict) -> str:
    defect_type = onion.get("defect_type")
    if defect_type:
        return str(defect_type)

    description = str(onion.get("defect_description") or "").lower()
    if onion.get("sprouting"):
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
    if onion.get("defect_present"):
        return "Other"
    return "None"


# ---------------------------------------------------------------------------
# NIR screening results
# ---------------------------------------------------------------------------

def render_nir_results(data: dict) -> None:
    """Render the NIR internal screening section when nir_results is present."""
    nir = data.get("nir_results")
    if not nir:
        return

    batch_risk = nir.get("batch_risk", "LOW")
    summary    = nir.get("summary", "")
    onions_nir = nir.get("onions", [])
    high_count = nir.get("high_risk_count", 0)
    med_count  = nir.get("medium_risk_count", 0)
    low_count  = nir.get("low_risk_count", 0)

    if batch_risk == "HIGH":
        risk_color = "var(--rust)"
        risk_bg    = "var(--rust-soft)"
        risk_icon  = "⚠️"
    elif batch_risk == "MEDIUM":
        risk_color = "var(--copper-deep)"
        risk_bg    = "rgba(200,144,42,0.10)"
        risk_icon  = "!"
    else:
        risk_color = "var(--sage)"
        risk_bg    = "var(--sage-soft)"
        risk_icon  = "✓"

    st.markdown("### 🔵 Internal NIR Screening Results")

    # Batch-level risk card
    md(f"""
    <div class="og-card" style="
        background:{risk_bg};
        border-left:6px solid {risk_color};
        padding:18px 22px;
        margin-bottom:14px;
    ">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="
                width:38px;height:38px;border-radius:50%;
                background:{risk_color};color:white;
                display:flex;align-items:center;justify-content:center;
                font-size:18px;font-weight:700;flex-shrink:0;
            ">{risk_icon}</div>
            <div>
                <div style="font-size:11px;text-transform:uppercase;font-weight:700;
                            letter-spacing:.06em;color:var(--ink-soft);">
                    Internal Risk Level
                </div>
                <div style="font-family:'Fraunces',serif;font-size:22px;
                            font-weight:700;color:{risk_color};">
                    {escape(batch_risk)}
                </div>
            </div>
        </div>
        <div style="margin-top:10px;font-size:13px;color:var(--ink);line-height:1.6;">
            {escape(summary)}
        </div>
    </div>
    """)

    # Count pills
    md(f"""
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;">
        <div style="background:var(--rust-soft);border:1px solid var(--rust);
                    border-radius:999px;padding:5px 16px;font-size:12px;font-weight:700;
                    color:var(--rust);">
            ⚠ HIGH: {high_count}
        </div>
        <div style="background:rgba(200,144,42,0.10);border:1px solid var(--copper);
                    border-radius:999px;padding:5px 16px;font-size:12px;font-weight:700;
                    color:var(--copper-deep);">
            ! MEDIUM: {med_count}
        </div>
        <div style="background:var(--sage-soft);border:1px solid var(--sage);
                    border-radius:999px;padding:5px 16px;font-size:12px;font-weight:700;
                    color:var(--sage);">
            ✓ LOW: {low_count}
        </div>
    </div>
    """)

    if not onions_nir:
        return

    # Per-onion NIR table
    nir_rows = []
    for o in onions_nir:
        m = o.get("metrics", {})
        nir_rows.append({
            "Onion ID":        o.get("onion_id", "-"),
            "Composite Score": f"{o.get('composite_score', 0):.3f}",
            "Risk":            o.get("risk", "-"),
            "Red Proxy":       f"{m.get('red_proxy', 0):.3f}",
            "Texture Var":     f"{m.get('lap_var_score', 0):.3f}",
            "Entropy":         f"{m.get('entropy_score', 0):.3f}",
            "Ring CoV":        f"{m.get('ring_cov_score', 0):.3f}",
            "Flags":           " | ".join(o.get("internal_flags", [])) or "—",
        })

    import pandas as _pd
    nir_df = _pd.DataFrame(nir_rows)

    def _risk_style(val):
        if val == "HIGH":
            return "color:#F87171;font-weight:700"
        if val == "MEDIUM":
            return "color:#E8A84F;font-weight:700"
        return "color:#4ADE80;font-weight:700"

    styled = nir_df.style.map(_risk_style, subset=["Risk"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Composite score bar chart
    fig = go.Figure(go.Bar(
        x=[o.get("onion_id", "") for o in onions_nir],
        y=[o.get("composite_score", 0) for o in onions_nir],
        marker_color=[
            "#F87171" if o.get("risk") == "HIGH"   else
            "#E8A84F" if o.get("risk") == "MEDIUM" else
            "#4ADE80"
            for o in onions_nir
        ],
        text=[o.get("risk", "") for o in onions_nir],
        textposition="outside",
    ))
    fig.add_hline(y=0.35, line_dash="dash", line_color="#E8A84F",
                  annotation_text="MEDIUM threshold", annotation_position="right")
    fig.add_hline(y=0.60, line_dash="dash", line_color="#F87171",
                  annotation_text="HIGH threshold", annotation_position="right")
    _is_dark_nir = st.session_state.get("theme_mode", "light") == "dark"
    _nir_font = "#F0EDE8" if _is_dark_nir else "#1C1009"
    _nir_grid = "rgba(255,255,255,0.08)" if _is_dark_nir else "#E8DDD0"
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=60, t=16, b=10),
        yaxis=dict(range=[0, 1.0], title="NIR proxy score", gridcolor=_nir_grid),
        xaxis_title="Onion",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, IBM Plex Sans, sans-serif", color=_nir_font),
        showlegend=False,
    )
    st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)

    with st.expander("ℹ️ How NIR-proxy scoring works"):
        st.markdown("""
**Four computer-vision metrics are computed from the RGB image:**

| Metric | What it measures | Weight |
|---|---|---|
| **Red-channel proxy** | Mean red-band intensity deviation from healthy range | 25% |
| **Texture variance** | Laplacian variance in the red channel — high = sub-surface irregularity | 30% |
| **Local entropy** | Shannon entropy over 5×5 patches — high = disordered internal structure | 25% |
| **Ring CoV** | Coefficient of variation across concentric ellipses — high = layer disruption | 20% |

The composite score (0–1) is colour-coded:
🟢 **LOW** (< 0.35) — internal quality appears uniform
🟡 **MEDIUM** (0.35–0.60) — minor irregularity; spot-check recommended
🔴 **HIGH** (≥ 0.60) — significant anomaly; physical inspection before dispatch

> These are CV-based **proxy signals** derived from an RGB image. A real NIR sensor
> provides more accurate readings. Use HIGH results as a triage flag, not a final verdict.
        """)


def build_csv_bytes(data: dict) -> bytes:
    onions = get_onion_records(data)

    grade_result = data.get("grade_result", {}) or {}
    grades = {
        item.get("onion_id"): item.get("grade")
        for item in grade_result.get("per_onion_grades", [])
    }

    output = io.StringIO()

    # Exact CSV schema requested by the project specification.
    fieldnames = [
        "onion_id",
        "size",
        "color",
        "severity",
        "defect_type",
        "defect_description",
        "confidence",
        "final_grade",
    ]

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for onion in onions:
        onion_id = onion.get("onion_id", "")
        confidence = pct(onion.get("confidence", 0))
        grade = grades.get(onion_id, "-")

        writer.writerow({
            "onion_id": onion_id,
            "size": onion.get("size", ""),
            "color": onion.get("color", ""),
            "severity": _display_severity(onion, grade),
            "defect_type": _display_defect_type(onion),
            "defect_description": onion.get("defect_description") or "None",
            "confidence": round(confidence, 3),
            "final_grade": grade,
        })

    return output.getvalue().encode("utf-8-sig")


# ---------------------------------------------------------------------------
# Crop thumbnail helper (Feature 2 — per-onion crop thumbnails)
# ---------------------------------------------------------------------------

def crop_onion_thumbnail(
    image_bytes: bytes,
    bbox: dict,
    thumb_size: int = 120,
) -> bytes | None:
    """Crop a single onion from image_bytes using normalized bbox.

    Returns JPEG thumbnail bytes, or None if cropping is not possible.
    Safe against out-of-range bboxes.
    """
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
        # Clamp to image bounds
        x1 = max(0, int(x * iw))
        y1 = max(0, int(y * ih))
        x2 = min(iw, int((x + bw) * iw))
        y2 = min(ih, int((y + bh) * ih))
        if x2 <= x1 or y2 <= y1:
            return None
        cropped = img.crop((x1, y1, x2, y2))
        # Resize to thumbnail maintaining aspect ratio
        cropped.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=82)
        return buf.getvalue()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# XLSX export builder (frontend-side, for download button)
# ---------------------------------------------------------------------------

def build_xlsx_bytes(data: dict) -> bytes | None:
    """Build XLSX bytes for download in the Streamlit frontend."""
    if not _XLSX_AVAILABLE:
        return None
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill

        onions = get_onion_records(data)
        grade_result = data.get("grade_result", {}) or {}
        batch_id = get_batch_id(data)

        _GREEN  = "FF90EE90"
        _YELLOW = "FFFFF176"
        _RED    = "FFFF9999"
        _GREY   = "FFD3D3D3"

        def _fill(grade: str):
            g = (grade or "").strip()
            if "Grade A" in g or g == "A":
                c = _GREEN
            elif g == "URS":
                c = _YELLOW
            elif g == "Defect":
                c = _RED
            else:
                return None
            return PatternFill(fill_type="solid", fgColor=c)

        wb = Workbook()
        ws1 = wb.active
        ws1.title = "Onion Results"
        headers = ["#","Onion ID","Source Image","Final Grade","Grade Reason",
                   "Size","Colour","Confidence (%)","Sprouting","Defect Present",
                   "Defect Description","Defect Type","Severity","Review Status"]
        hf = PatternFill(fill_type="solid", fgColor=_GREY)
        ws1.append(headers)
        for cell in ws1[1]:
            cell.fill = hf
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center")

        grades_by_id = {
            item.get("onion_id"): item.get("grade")
            for item in grade_result.get("per_onion_grades", [])
        }

        for idx, onion in enumerate(onions, start=1):
            oid   = str(onion.get("onion_id") or "")
            grade = str(grades_by_id.get(oid) or onion.get("final_grade") or "")
            conf  = float(onion.get("confidence") or 0.0)
            row   = [
                idx, oid,
                str(onion.get("source_image") or ""),
                grade,
                str(onion.get("grade_reason") or ""),
                str(onion.get("size") or ""),
                str(onion.get("color") or ""),
                round(conf * 100, 1),
                "Yes" if onion.get("sprouting") else "No",
                "Yes" if onion.get("defect_present") else "No",
                str(onion.get("defect_description") or ""),
                str(onion.get("defect_type") or ""),
                str(onion.get("severity") or ""),
                str(onion.get("review_status") or ""),
            ]
            ws1.append(row)
            f = _fill(grade)
            if f:
                ws1.cell(row=idx+1, column=4).fill = f

        for col in ws1.columns:
            ml = max(len(str(cell.value or "")) for cell in col)
            ws1.column_dimensions[col[0].column_letter].width = min(ml + 4, 50)

        ws2 = wb.create_sheet("Summary")
        summary_rows = [
            ("Batch ID", batch_id),
            ("Generated At", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("Analysis Mode", str(data.get("inspection_mode") or "rgb").upper()),
            ("Demo Mode", "Yes" if data.get("demo_mode") else "No"),
            ("", ""),
            ("Total Validated", grade_result.get("total_validated", 0)),
            ("Grade A Count",   grade_result.get("grade_a_count", 0)),
            ("URS Count",       grade_result.get("urs_count", 0)),
            ("Defect Count",    grade_result.get("defect_count", 0)),
            ("Grade A %",       grade_result.get("grade_a_pct", 0.0)),
            ("URS %",           grade_result.get("urs_pct", 0.0)),
            ("Defect %",        grade_result.get("defect_pct", 0.0)),
        ]
        ws2.column_dimensions["A"].width = 22
        ws2.column_dimensions["B"].width = 30
        gmap = {"Grade A %": _GREEN, "URS %": _YELLOW, "Defect %": _RED}
        for lbl, val in summary_rows:
            ws2.append([lbl, val])
        for row in ws2.iter_rows(min_row=1, max_row=ws2.max_row):
            if row[0].value:
                row[0].font = Font(bold=True)
            colour = gmap.get(str(row[0].value or ""))
            if colour and len(row) > 1:
                row[1].fill = PatternFill(fill_type="solid", fgColor=colour)

        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------

def build_pdf_bytes(
    data: dict,
    image_bytes,
) -> bytes:

    summary = get_summary(data)

    grade_result = data.get(
        "grade_result",
        {},
    ) or {}

    quality = get_overall_quality(data)

    report_text = get_report_text(data)

    batch_id = get_batch_id(data)

    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()

    plum = rl_colors.HexColor(
        THEME["plum_deep"]
    )

    ink = rl_colors.HexColor(
        THEME["ink"]
    )

    ink_soft = rl_colors.HexColor(
        THEME["ink_soft"]
    )

    title_style = ParagraphStyle(
        "Title2",
        parent=styles["Title"],
        textColor=plum,
        fontSize=22,
        spaceAfter=2,
    )

    sub_style = ParagraphStyle(
        "Sub",
        parent=styles["Normal"],
        textColor=ink_soft,
        fontSize=10,
        spaceAfter=14,
    )

    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        textColor=plum,
        fontSize=13,
        spaceBefore=16,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        textColor=ink,
        fontSize=10.2,
        leading=15,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        textColor=ink_soft,
        fontSize=8.5,
    )

    story = []

    story.append(
        Paragraph(
            "OnionGrade AI — Quality Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            f"Team NeuroNex · SIH26031 · "
            f"Batch <b>{escape(batch_id)}</b> · "
            f"Generated "
            f"{datetime.now().strftime('%d %b %Y, %H:%M')}",
            sub_style,
        )
    )

    # ---------------------------------------------------------------
    # Image
    # ---------------------------------------------------------------

    if image_bytes:

        try:

            img = RLImage(
                io.BytesIO(image_bytes)
            )

            max_width = 90 * mm

            ratio = (
                img.imageHeight
                / float(img.imageWidth)
            )

            img.drawWidth = max_width
            img.drawHeight = (
                max_width * ratio
            )

            story.append(img)
            story.append(
                Spacer(1, 10)
            )

        except Exception:
            pass

    # ---------------------------------------------------------------
    # Grade summary
    # ---------------------------------------------------------------

    story.append(
        Paragraph(
            "Grade summary",
            h2_style,
        )
    )

    grade_a = summary.get(
        "grade_a",
        {},
    )

    urs = summary.get(
        "urs",
        {},
    )

    defect = summary.get(
        "defect",
        {},
    )

    total_validated = grade_result.get(
        "total_validated",
        len(get_onion_records(data)),
    )

    summary_rows = [
        [
            "Metric",
            "Percent",
            "Count",
        ],
        [
            "Grade A",
            f"{pct(grade_a.get('percentage', 0)):.1f}%",
            str(grade_a.get("count", 0)),
        ],
        [
            "URS",
            f"{pct(urs.get('percentage', 0)):.1f}%",
            str(urs.get("count", 0)),
        ],
        [
            "Defect",
            f"{pct(defect.get('percentage', 0)):.1f}%",
            str(defect.get("count", 0)),
        ],
        [
            "Total validated",
            "100%",
            str(total_validated),
        ],
    ]

    summary_table = Table(
        summary_rows,
        colWidths=[
            60 * mm,
            40 * mm,
            40 * mm,
        ],
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    rl_colors.HexColor(
                        THEME["plum"]
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    rl_colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9.5,
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        rl_colors.HexColor(
                            "#FBF5EA"
                        ),
                        rl_colors.white,
                    ],
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    rl_colors.HexColor(
                        THEME["line"]
                    ),
                ),
                (
                    "ALIGN",
                    (1, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(
        summary_table
    )

    # ---------------------------------------------------------------
    # Overall assessment
    # ---------------------------------------------------------------

    story.append(
        Paragraph(
            "Overall assessment",
            h2_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>Status:</b> "
            f"{escape(str(quality.get('status', 'NOT AVAILABLE')))}",
            body_style,
        )
    )

    story.append(
        Paragraph(
            escape(
                str(
                    quality.get(
                        "description",
                        "",
                    )
                )
            ),
            body_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>Recommendation:</b> "
            f"{escape(str(quality.get('recommendation', '')))}",
            body_style,
        )
    )

    # ---------------------------------------------------------------
    # Defect analysis
    # ---------------------------------------------------------------

    story.append(
        Paragraph(
            "Defect analysis",
            h2_style,
        )
    )

    defect_analysis = data.get(
        "defect_analysis",
        {},
    ) or {}

    categories = defect_analysis.get(
        "categories",
        {},
    ) or {}

    if categories:

        defect_rows = [
            [
                "Defect type",
                "Count",
            ]
        ]

        for category, count in categories.items():

            defect_rows.append(
                [
                    str(category),
                    str(count),
                ]
            )

        defect_table = Table(
            defect_rows,
            colWidths=[
                100 * mm,
                40 * mm,
            ],
        )

        defect_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        rl_colors.HexColor(
                            THEME["rust"]
                        ),
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        rl_colors.white,
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        rl_colors.HexColor(
                            THEME["line"]
                        ),
                    ),
                    (
                        "ALIGN",
                        (1, 0),
                        (-1, -1),
                        "CENTER",
                    ),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [
                            rl_colors.white,
                            rl_colors.HexColor(
                                "#FBF5EA"
                            ),
                        ],
                    ),
                ]
            )
        )

        story.append(
            defect_table
        )

    else:

        story.append(
            Paragraph(
                "No defect categories were recorded.",
                body_style,
            )
        )

    # ---------------------------------------------------------------
    # Digital report
    # ---------------------------------------------------------------

    story.append(
        Paragraph(
            "Digital report",
            h2_style,
        )
    )

    story.append(
        Paragraph(
            escape(report_text).replace(
                "\n",
                "<br/>",
            ),
            body_style,
        )
    )

    # ---------------------------------------------------------------
    # Per-onion records
    # ---------------------------------------------------------------

    onions = get_onion_records(data)

    if onions:

        story.append(
            Paragraph(
                "Per-onion records",
                h2_style,
            )
        )

        grades_by_id = {
            item.get("onion_id"): item.get("grade")
            for item in grade_result.get(
                "per_onion_grades",
                [],
            )
        }

        header = [
            "ID",
            "Size",
            "Colour",
            "Sprouting",
            "Defect",
            "Grade Reason",
            "Confidence",
            "Grade",
        ]

        rows = [header]
        # Track which row indices are manually overridden (1-based, skip header row 0)
        overridden_row_indices: list[int] = []

        for row_idx, onion in enumerate(onions, start=1):

            onion_id = onion.get(
                "onion_id",
                "",
            )

            confidence = pct(
                onion.get(
                    "confidence",
                    0,
                )
            )

            grade_reason = str(onion.get("grade_reason") or "—")[:40]
            is_override = "manually overridden" in grade_reason.lower()
            if is_override:
                overridden_row_indices.append(row_idx)

            rows.append(
                [
                    str(onion_id),
                    str(
                        onion.get(
                            "size",
                            "",
                        )
                    ),
                    str(
                        onion.get(
                            "color",
                            "",
                        )
                    )[:18],
                    (
                        "Yes"
                        if onion.get(
                            "sprouting"
                        )
                        else "No"
                    ),
                    (
                        "Yes"
                        if onion.get(
                            "defect_present"
                        )
                        else "No"
                    ),
                    grade_reason,
                    f"{confidence * 100:.1f}%",
                    grades_by_id.get(
                        onion_id,
                        onion.get("final_grade", "-"),
                    ),
                ]
            )

        onion_table = Table(
            rows,
            repeatRows=1,
        )

        # Build base styles
        _table_styles = [
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                rl_colors.HexColor(
                    THEME["copper"]
                ),
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                rl_colors.white,
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold",
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                7.5,
            ),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    rl_colors.white,
                    rl_colors.HexColor(
                        "#FBF5EA"
                    ),
                ],
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                rl_colors.HexColor(
                    THEME["line"]
                ),
            ),
            (
                "ALIGN",
                (1, 0),
                (-1, -1),
                "CENTER",
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                4,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                4,
            ),
        ]
        # Mark manually overridden rows with a light gold background + italic font
        _override_bg = rl_colors.HexColor("#FFF3CD")
        for _ri in overridden_row_indices:
            _table_styles.append(
                ("BACKGROUND", (0, _ri), (-1, _ri), _override_bg)
            )
            _table_styles.append(
                ("FONTNAME", (0, _ri), (-1, _ri), "Helvetica-Oblique")
            )

        onion_table.setStyle(TableStyle(_table_styles))

        if overridden_row_indices:
            story.append(
                Paragraph(
                    f"<i>✏️ {len(overridden_row_indices)} grade(s) manually overridden "
                    f"by inspector (highlighted rows).</i>",
                    small_style,
                )
            )

        story.append(
            onion_table
        )

    # ---------------------------------------------------------------
    # Flagged records
    # ---------------------------------------------------------------

    flagged = data.get(
        "flagged",
        [],
    )

    if flagged:

        story.append(
            Paragraph(
                f"Flagged & excluded records "
                f"({len(flagged)})",
                h2_style,
            )
        )

        for record in flagged:

            story.append(
                Paragraph(
                    f"• {escape(str(record.get('reason', 'Invalid record')))}",
                    small_style,
                )
            )

    story.append(
        Spacer(1, 14)
    )

    story.append(
        Paragraph(
            "Generated by OnionGrade AI. "
            "Grades and percentages are computed by a deterministic "
            "Python rule engine, never by the vision model. "
            "Every figure traces back to a validated per-onion record.",
            small_style,
        )
    )

    doc.build(story)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Theme state  (☀️ Light | 🌙 Dark | 🖥️ System)
# ---------------------------------------------------------------------------

st.session_state.setdefault("theme_mode", "light")

# ── Light Mode: Premium warm ivory / cream glass aesthetic ──
_LIGHT_VARS = {
    "bg":               "#F7F3EE",          # warm ivory background
    "surface":          "#FFFFFF",
    "surface-soft":     "#FDF9F4",          # cream-tinted soft surface
    "surface-glass":    "rgba(255,255,255,0.72)",
    "ink":              "#1C1009",          # deep warm charcoal
    "ink-soft":         "#6B5040",          # muted warm brown
    "plum":             "#6B2D4E",          # deep plum/burgundy accent
    "plum-deep":        "#4A1B35",          # darker plum
    "copper":           "#B5651D",          # copper accent
    "copper-deep":      "#8F4D15",
    "gold":             "#C8902A",          # warm gold highlight
    "sage":             "#2D7A4F",          # agricultural green
    "sage-soft":        "#E6F4EC",
    "rust":             "#A23020",          # warm rust/red for errors
    "rust-soft":        "#FBEAE7",
    "line":             "#E8DDD0",          # warm gray border
    "glass-border":     "rgba(107,45,78,0.18)",
    "shadow":           "rgba(28,16,9,0.08)",
    "radial-1":         "rgba(107,45,78,0.10)",
    "radial-2":         "rgba(200,144,42,0.12)",
    "report-bg":        "#FFF8F0",
    "report-border":    "rgba(107,45,78,0.22)",
    "table-header-bg":  "#4A1B35",
    "table-alt-bg":     "#FDF6EE",
    "header-bg":        "rgba(255,255,255,0.82)",
    "header-border":    "rgba(107,45,78,0.15)",
    "sidebar-bg":       "#FBF6F0",
    "sidebar-border":   "rgba(232,221,208,0.8)",
    "card-glow":        "rgba(107,45,78,0.06)",
    "btn-text":         "#4A1B35",
    "nav-active-bg":    "rgba(107,45,78,0.10)",
    "nav-active-border":"#6B2D4E",
    "upload-zone-bg":   "#FDF9F4",
    "ring-track":       "#EFE6D6",
}

# ── Dark Mode: Premium near-black with green/gold accents ──
_DARK_VARS = {
    "bg":               "#0B0E12",          # near-black
    "surface":          "#13181F",          # dark graphite card
    "surface-soft":     "#191F28",          # slightly lighter card
    "surface-glass":    "rgba(255,255,255,0.05)",
    "ink":              "#F0EDE8",          # warm white text
    "ink-soft":         "#9DAAB8",          # muted soft gray
    "plum":             "#C7924A",          # warm gold/copper as primary in dark
    "plum-deep":        "#D9A441",          # bright gold
    "copper":           "#E8A84F",
    "copper-deep":      "#F0C070",
    "gold":             "#D9A441",
    "sage":             "#4ADE80",          # bright green for status
    "sage-soft":        "rgba(74,222,128,0.14)",
    "rust":             "#F87171",
    "rust-soft":        "rgba(248,113,113,0.14)",
    "line":             "rgba(255,255,255,0.10)",
    "glass-border":     "rgba(217,164,65,0.22)",
    "shadow":           "rgba(0,0,0,0.50)",
    "radial-1":         "rgba(74,222,128,0.06)",
    "radial-2":         "rgba(217,164,65,0.07)",
    "report-bg":        "#0F1419",
    "report-border":    "rgba(217,164,65,0.22)",
    "table-header-bg":  "#1A2030",
    "table-alt-bg":     "#141A24",
    "header-bg":        "rgba(19,24,31,0.90)",
    "header-border":    "rgba(217,164,65,0.18)",
    "sidebar-bg":       "#0F1318",
    "sidebar-border":   "rgba(255,255,255,0.08)",
    "card-glow":        "rgba(217,164,65,0.06)",
    "btn-text":         "#F0EDE8",
    "nav-active-bg":    "rgba(217,164,65,0.12)",
    "nav-active-border":"#D9A441",
    "upload-zone-bg":   "#191F28",
    "ring-track":       "rgba(255,255,255,0.12)",
}


def _vars_to_css(vars_dict: dict) -> str:
    return "\n".join(f"  --{k}:{v};" for k, v in vars_dict.items())


_theme_mode = st.session_state["theme_mode"]

if _theme_mode == "dark":
    _root_css = f":root{{\n{_vars_to_css(_DARK_VARS)}\n}}"
elif _theme_mode == "system":
    _root_css = (
        f":root{{\n{_vars_to_css(_LIGHT_VARS)}\n}}\n"
        f"@media (prefers-color-scheme: dark){{\n"
        f"  :root{{\n{_vars_to_css(_DARK_VARS)}\n  }}\n}}"
    )
else:  # light (default)
    _root_css = f":root{{\n{_vars_to_css(_LIGHT_VARS)}\n}}"


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

# Font-size scale — driven by sidebar selector
_FONT_SCALE = st.session_state.get("font_size", "normal")
_FONT_SIZE_MAP = {
    "small":   {"base": "12px",  "h1": "22px", "h2": "17px", "h3": "15px", "caption": "11px"},
    "normal":  {"base": "14px",  "h1": "26px", "h2": "20px", "h3": "16px", "caption": "12px"},
    "medium":  {"base": "15.5px","h1": "29px", "h2": "22px", "h3": "17.5px","caption": "13px"},
    "large":   {"base": "17px",  "h1": "32px", "h2": "25px", "h3": "19px", "caption": "14px"},
    "xlarge":  {"base": "19px",  "h1": "36px", "h2": "28px", "h3": "21px", "caption": "15px"},
}
_FS = _FONT_SIZE_MAP.get(_FONT_SCALE, _FONT_SIZE_MAP["normal"])

md(
    f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>

{_root_css}

/* =====================================================================
   BASE TYPOGRAPHY
   ===================================================================== */
html, body, [class*="css"], .stMarkdown, p, span, label, li {{
  font-family: 'Inter', 'IBM Plex Sans', system-ui, sans-serif;
  color: var(--ink);
  font-size: {_FS["base"]};
  line-height: 1.65;
}}

/* =====================================================================
   APP BACKGROUND — premium subtle gradient
   ===================================================================== */
.stApp {{
  background:
    radial-gradient(ellipse at 10% 0%,   var(--radial-1) 0%, transparent 50%),
    radial-gradient(ellipse at 90% 100%, var(--radial-2) 0%, transparent 50%),
    var(--bg) !important;
  transition: background 0.35s ease;
}}

/* =====================================================================
   HEADINGS — strong visual hierarchy
   ===================================================================== */
h1 {{
  font-family: 'Fraunces', Georgia, serif !important;
  font-size: {_FS["h1"]} !important;
  font-weight: 700 !important;
  letter-spacing: -0.025em !important;
  line-height: 1.12 !important;
  color: var(--plum-deep) !important;
  margin-bottom: 0.25em !important;
}}
h2 {{
  font-family: 'Fraunces', Georgia, serif !important;
  font-size: {_FS["h2"]} !important;
  font-weight: 600 !important;
  letter-spacing: -0.015em !important;
  color: var(--plum-deep) !important;
  border-bottom: 1.5px solid var(--line) !important;
  padding-bottom: 0.28em !important;
  margin-bottom: 0.65em !important;
}}
h3 {{
  font-family: 'Inter', sans-serif !important;
  font-size: {_FS["h3"]} !important;
  font-weight: 700 !important;
  color: var(--ink) !important;
  margin-bottom: 0.4em !important;
  letter-spacing: -0.01em !important;
}}

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ visibility: hidden; }}

div.block-container {{
  padding-top: 1.6rem !important;
  padding-bottom: 3rem !important;
  max-width: 1220px;
}}

/* =====================================================================
   SIDEBAR — premium narrow sidebar
   ===================================================================== */
section[data-testid="stSidebar"] {{
  background: var(--sidebar-bg, var(--surface-soft)) !important;
  border-right: 1px solid var(--sidebar-border, var(--line)) !important;
  backdrop-filter: blur(12px);
}}
section[data-testid="stSidebar"] > div {{
  padding-top: 1.2rem !important;
}}
/* All sidebar text: warm white in dark, deep charcoal in light */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span:not([data-testid]),
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] li {{
  color: var(--ink) !important;
}}
section[data-testid="stSidebar"] h3 {{
  font-size: 10px !important;
  letter-spacing: 0.10em !important;
  text-transform: uppercase !important;
  font-weight: 700 !important;
  color: var(--ink-soft) !important;
  border-bottom: none !important;
  padding-bottom: 0 !important;
  margin-bottom: 6px !important;
}}
/* Sidebar dividers */
section[data-testid="stSidebar"] hr {{
  border-color: var(--line) !important;
  opacity: 0.6 !important;
  margin: 10px 0 !important;
}}
/* Sidebar select / slider */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="slider"] {{
  background: var(--surface) !important;
  border-color: var(--line) !important;
  border-radius: 10px !important;
}}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] input,
section[data-testid="stSidebar"] div[data-baseweb="select"] * {{
  background: var(--surface) !important;
  color: var(--ink) !important;
}}
/* Sidebar segmented control - theme toggle */
section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] {{
  background: var(--surface-soft) !important;
  border: 1.5px solid var(--line) !important;
  border-radius: 999px !important;
  padding: 3px !important;
}}
/* Unselected buttons - transparent bg + readable text in dark mode */
section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] button,
section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] label {{
  background: transparent !important;
  color: var(--ink-soft) !important;
  font-weight: 600 !important;
  border: none !important;
  border-radius: 999px !important;
}}
/* Selected button */
section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] button[aria-selected="true"],
section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] button[data-active="true"],
section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] label[aria-selected="true"],
section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] label[data-active="true"] {{
  background: var(--plum) !important;
  color: #fff !important;
}}
/* Inner text - inherit from parent */
section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] button span,
section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] button p,
section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] label span,
section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] label p {{
  color: inherit !important;
  background: transparent !important;
}}
/* Sidebar radio (theme fallback) */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label {{
  color: var(--ink) !important;
  font-weight: 500 !important;
}}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label span {{
  color: var(--ink) !important;
}}
/* Sidebar select_slider */
section[data-testid="stSidebar"] div[data-testid="stSlider"] span,
section[data-testid="stSidebar"] div[data-testid="stSlider"] p {{
  color: var(--ink) !important;
}}
/* History buttons in sidebar */
section[data-testid="stSidebar"] div.stButton > button {{
  border-radius: 12px !important;
  font-size: 12px !important;
  padding: 0.45em 1em !important;
  text-align: left !important;
  justify-content: flex-start !important;
  color: var(--ink) !important;
  background: var(--surface) !important;
  border-color: var(--line) !important;
}}
section[data-testid="stSidebar"] div.stButton > button:hover {{
  background: var(--nav-active-bg) !important;
  border-color: var(--nav-active-border) !important;
  color: var(--plum-deep) !important;
}}
/* Caption text in sidebar */
section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] div[data-testid="stCaption"],
section[data-testid="stSidebar"] small {{
  color: var(--ink-soft) !important;
}}
/* =====================================================================
   SIDEBAR DARK MODE - explicit text visibility fix
   Covers Streamlit-internal data-testid elements that bypass var(--ink)
   ===================================================================== */
/* Markdown, text, widget labels inside sidebar */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a,
section[data-testid="stSidebar"] [data-testid="stText"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="InputInstructions"],
section[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] span,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] label,
section[data-testid="stSidebar"] [data-testid="stNumberInput"] label,
section[data-testid="stSidebar"] [data-testid="stTextInput"] label,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stCheckbox label {{
  color: var(--ink) !important;
}}
/* Metric / number display in sidebar */
section[data-testid="stSidebar"] [data-testid="stMetricValue"],
section[data-testid="stSidebar"] [data-testid="stMetricLabel"],
section[data-testid="stSidebar"] [data-testid="stMetricDelta"] {{
  color: var(--ink) !important;
}}
/* Select dropdown option text in sidebar */
section[data-testid="stSidebar"] [data-baseweb="menu"] li,
section[data-testid="stSidebar"] [data-baseweb="menu"] [role="option"],
section[data-testid="stSidebar"] [data-baseweb="option"] {{
  background: var(--surface) !important;
  color: var(--ink) !important;
}}
/* Sidebar header/title text */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6 {{
  color: var(--ink) !important;
}}
/* Expander labels in sidebar */
section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary span {{
  color: var(--ink) !important;
}}
/* Tooltip / helper text in sidebar */
section[data-testid="stSidebar"] [data-testid="stTooltipIcon"],
section[data-testid="stSidebar"] [data-testid="stHelp"] {{
  color: var(--ink-soft) !important;
}}


/* =====================================================================
   LIQUID-GLASS BUTTONS — premium iPhone-style
   ===================================================================== */
div.stButton > button,
div.stDownloadButton > button,
div.stFormSubmitButton > button {{
  position: relative;
  overflow: hidden;
  background: linear-gradient(
    145deg,
    var(--surface-glass) 0%,
    rgba(255,255,255,0.03) 100%
  ) !important;
  backdrop-filter: blur(20px) saturate(200%) !important;
  -webkit-backdrop-filter: blur(20px) saturate(200%) !important;
  color: var(--btn-text, var(--ink)) !important;
  border: 1.5px solid var(--glass-border) !important;
  border-radius: 999px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: {_FS["base"]} !important;
  padding: 0.55em 1.65em !important;
  letter-spacing: 0.01em !important;
  box-shadow:
    inset 0 1.5px 0 rgba(255,255,255,0.20),
    0 1px 3px var(--shadow),
    0 4px 16px var(--shadow) !important;
  transition:
    transform 0.18s cubic-bezier(0.34, 1.56, 0.64, 1),
    box-shadow 0.18s ease,
    background 0.18s ease !important;
}}

/* Shine sweep */
div.stButton > button::before,
div.stDownloadButton > button::before {{
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(
    115deg,
    transparent 28%,
    rgba(255,255,255,0.16) 50%,
    transparent 72%
  );
  transform: translateX(-120%);
  transition: transform 0.50s ease;
  pointer-events: none;
  border-radius: inherit;
}}
div.stButton > button:hover::before,
div.stDownloadButton > button:hover::before {{
  transform: translateX(120%);
}}

div.stButton > button:hover,
div.stDownloadButton > button:hover,
div.stFormSubmitButton > button:hover {{
  transform: translateY(-2px) scale(1.018) !important;
  box-shadow:
    inset 0 1.5px 0 rgba(255,255,255,0.24),
    0 6px 12px var(--shadow),
    0 16px 36px var(--shadow) !important;
}}

div.stButton > button:active,
div.stDownloadButton > button:active,
div.stFormSubmitButton > button:active {{
  transform: translateY(0px) scale(0.968) !important;
  box-shadow: 0 1px 4px var(--shadow) !important;
}}

/* PRIMARY — solid plum gradient */
div.stButton > button[kind="primary"],
div.stDownloadButton > button[kind="primary"],
div.stFormSubmitButton > button[kind="primary"] {{
  background: linear-gradient(
    135deg,
    var(--plum) 0%,
    var(--plum-deep) 100%
  ) !important;
  color: #ffffff !important;
  border: 1.5px solid transparent !important;
  box-shadow:
    inset 0 1.5px 0 rgba(255,255,255,0.22),
    0 2px 8px var(--shadow),
    0 10px 28px var(--shadow) !important;
}}
div.stButton > button[kind="primary"]:hover,
div.stDownloadButton > button[kind="primary"]:hover {{
  color: #ffffff !important;
  filter: brightness(1.08);
}}

div.stButton > button:disabled,
div.stDownloadButton > button:disabled {{
  opacity: 0.38 !important;
  transform: none !important;
  box-shadow: none !important;
  cursor: not-allowed !important;
}}

/* Special close-camera button */
.st-key-btn_close_camera button {{
  background: var(--rust-soft) !important;
  color: var(--rust) !important;
  border-color: var(--rust) !important;
}}

/* =====================================================================
   SEGMENTED CONTROL - theme-aware (critical for dark-mode readability)
   ===================================================================== */
div[data-testid="stSegmentedControl"] {{
  background: var(--surface) !important;
  border: 1.5px solid var(--line) !important;
  border-radius: 999px !important;
  padding: 3px !important;
}}
/* Unselected: transparent bg so container color shows through */
div[data-testid="stSegmentedControl"] button,
div[data-testid="stSegmentedControl"] label {{
  background: transparent !important;
  border: none !important;
  border-radius: 999px !important;
  font-size: {_FS["base"]} !important;
  font-weight: 600 !important;
  color: var(--ink-soft) !important;
  transition: color 0.18s ease, background 0.18s ease !important;
}}
/* Selected segment */
div[data-testid="stSegmentedControl"] button[aria-selected="true"],
div[data-testid="stSegmentedControl"] button[data-active="true"],
div[data-testid="stSegmentedControl"] label[aria-selected="true"] {{
  background: var(--plum) !important;
  color: #fff !important;
}}
/* Inner text spans - inherit from parent */
div[data-testid="stSegmentedControl"] button span,
div[data-testid="stSegmentedControl"] button p,
div[data-testid="stSegmentedControl"] label span,
div[data-testid="stSegmentedControl"] label p {{
  color: inherit !important;
  background: transparent !important;
}}
/* Wrapper div - transparent */
div[data-testid="stSegmentedControl"] > div {{
  background: transparent !important;
}}

/* =====================================================================
   RADIO BUTTONS — theme-aware
   ===================================================================== */
div[data-testid="stRadio"] label {{
  color: var(--ink) !important;
  font-size: {_FS["base"]} !important;
  font-weight: 500 !important;
}}
div[data-testid="stRadio"] label span {{
  color: var(--ink) !important;
}}
div[data-testid="stRadio"] label:has(input:checked) span {{
  color: var(--plum-deep) !important;
  font-weight: 700 !important;
}}
/* Radio button circles */
div[data-testid="stRadio"] input[type="radio"] + div {{
  border-color: var(--line) !important;
  background: var(--surface) !important;
}}
div[data-testid="stRadio"] input[type="radio"]:checked + div {{
  border-color: var(--plum) !important;
  background: var(--plum) !important;
}}

/* =====================================================================
   INPUTS & SELECTS
   ===================================================================== */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] {{
  background: var(--surface) !important;
  color: var(--ink) !important;
  border-color: var(--line) !important;
  border-radius: 12px !important;
  font-size: {_FS["base"]} !important;
  font-family: 'Inter', sans-serif !important;
}}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {{
  border-color: var(--plum) !important;
  box-shadow: 0 0 0 3px var(--glass-border) !important;
}}
div[data-baseweb="popover"] li,
ul[role="listbox"] {{
  background: var(--surface) !important;
  color: var(--ink) !important;
  font-size: {_FS["base"]} !important;
}}
div[data-baseweb="popover"] li:hover {{
  background: var(--nav-active-bg) !important;
}}

/* =====================================================================
   TABS
   ===================================================================== */
button[data-baseweb="tab"] {{
  color: var(--ink-soft) !important;
  font-weight: 600 !important;
  font-size: {_FS["base"]} !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
  color: var(--plum) !important;
}}
div[data-baseweb="tab-highlight"] {{ background-color: var(--plum) !important; }}
div[data-baseweb="tab-border"]    {{ background: var(--line) !important; }}

/* =====================================================================
   FILE UPLOADER — premium drag-drop zone
   ===================================================================== */
section[data-testid="stFileUploaderDropzone"] {{
  background: var(--upload-zone-bg, var(--surface-soft)) !important;
  border: 2px dashed var(--line) !important;
  border-radius: 16px !important;
  transition: border-color 0.2s ease, background 0.2s ease !important;
}}
section[data-testid="stFileUploaderDropzone"]:hover {{
  border-color: var(--plum) !important;
  background: var(--surface-glass) !important;
}}
section[data-testid="stFileUploaderDropzone"] button {{
  border-radius: 999px !important;
  border-color: var(--plum) !important;
  color: var(--plum) !important;
  background: transparent !important;
}}
section[data-testid="stFileUploaderDropzone"] *,
section[data-testid="stFileUploaderDropzone"] p,
section[data-testid="stFileUploaderDropzone"] span {{
  color: var(--ink) !important;
}}

/* =====================================================================
   CAMERA
   ===================================================================== */
div[data-testid="stCameraInput"] {{
  background: var(--surface-soft) !important;
  border: 1px solid var(--line) !important;
  border-radius: 18px !important;
  padding: 12px !important;
  overflow: hidden;
}}
div[data-testid="stCameraInput"] video,
div[data-testid="stCameraInput"] img {{ border-radius: 12px !important; }}

/* =====================================================================
   ALERTS & EXPANDERS
   ===================================================================== */
div[data-testid="stAlert"] {{
  border-radius: 14px !important;
  font-size: {_FS["base"]} !important;
  border-width: 1.5px !important;
}}
div[data-testid="stAlert"] * {{ color: inherit !important; }}

div[data-testid="stExpander"] {{
  border-radius: 16px !important;
  border: 1px solid var(--line) !important;
  background: var(--surface-soft) !important;
  overflow: hidden;
}}
div[data-testid="stExpander"] summary {{
  color: var(--ink) !important;
  font-weight: 600 !important;
  font-size: {_FS["base"]} !important;
  padding: 12px 16px !important;
}}
div[data-testid="stExpander"] * {{
  color: var(--ink) !important;
  font-size: {_FS["base"]} !important;
}}

hr, div[data-testid="stDivider"] {{ border-color: var(--line) !important; opacity: 0.7 !important; }}

/* =====================================================================
   DATAFRAME / TABLE — both themes
   ===================================================================== */
div[data-testid="stDataFrame"] {{
  border-radius: 16px !important;
  overflow: hidden;
  border: 1.5px solid var(--line) !important;
  box-shadow: 0 2px 12px var(--shadow) !important;
}}
div[data-testid="stDataFrame"] th,
div[data-testid="stDataFrame"] [role="columnheader"] {{
  background: var(--table-header-bg) !important;
  color: #ffffff !important;
  font-weight: 700 !important;
  font-size: {_FS["caption"]} !important;
  padding: 10px 14px !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
}}
div[data-testid="stDataFrame"] td,
div[data-testid="stDataFrame"] [role="gridcell"] {{
  color: var(--ink) !important;
  font-size: {_FS["base"]} !important;
  padding: 7px 14px !important;
  border-bottom: 1px solid var(--line) !important;
  background: transparent !important;
}}
div[data-testid="stDataFrame"] tr:nth-child(even) td {{
  background: var(--table-alt-bg) !important;
}}

/* =====================================================================
   PREMIUM CARDS
   ===================================================================== */
.og-card {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 24px 28px;
  box-shadow:
    0 1px 3px var(--shadow),
    0 8px 24px var(--shadow),
    inset 0 1px 0 rgba(255,255,255,0.06);
  transition: background 0.28s ease, border-color 0.28s ease, box-shadow 0.28s ease;
  position: relative;
  overflow: hidden;
}}
.og-card:hover {{
  box-shadow:
    0 2px 5px var(--shadow),
    0 14px 38px var(--card-glow),
    inset 0 1px 0 rgba(255,255,255,0.08) !important;
}}

/* =====================================================================
   REPORT BLOCK
   ===================================================================== */
.og-report {{
  background: var(--report-bg, var(--surface-soft));
  border: 1.5px solid var(--report-border, var(--glass-border));
  border-left: 5px solid var(--plum);
  border-radius: 20px;
  padding: 28px 36px;
  box-shadow:
    0 2px 10px var(--shadow),
    0 16px 40px var(--shadow);
  position: relative;
  overflow: hidden;
}}
.og-report::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--plum-deep), var(--copper), var(--gold));
  border-radius: 20px 20px 0 0;
}}
.og-report-heading {{
  font-family: 'Fraunces', serif !important;
  font-size: {_FS["h2"]} !important;
  font-weight: 700 !important;
  color: var(--plum-deep) !important;
  margin-bottom: 6px !important;
  letter-spacing: -0.015em;
}}
.og-report-body {{
  font-size: {_FS["base"]};
  line-height: 1.80;
  color: var(--ink);
  white-space: pre-wrap;
  font-family: 'Inter', 'IBM Plex Sans', sans-serif;
}}

/* =====================================================================
   INLINE TABLES (rule engine / NIR expander)
   ===================================================================== */
.og-table {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
  font-size: {_FS["base"]};
  border-radius: 12px;
  overflow: hidden;
}}
.og-table thead tr {{
  background: var(--plum-deep);
}}
.og-table thead th {{
  color: #fff !important;
  padding: 10px 14px;
  text-align: left;
  font-weight: 700;
  font-size: {_FS["caption"]};
  letter-spacing: 0.05em;
  text-transform: uppercase;
  border: none;
}}
.og-table tbody tr:nth-child(even) {{
  background: var(--table-alt-bg, var(--surface-soft));
}}
.og-table tbody tr:nth-child(odd) {{
  background: var(--surface);
}}
.og-table tbody td {{
  padding: 8px 14px;
  border-bottom: 1px solid var(--line);
  color: var(--ink);
  vertical-align: top;
}}
.og-table tbody tr:hover td {{
  background: var(--nav-active-bg);
}}

/* =====================================================================
   CV SUMMARY CARD
   ===================================================================== */
.og-cv-card {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 26px 30px;
  box-shadow: 0 2px 14px var(--shadow);
}}
.og-cv-section-title {{
  font-size: {_FS["caption"]};
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-soft);
  margin-bottom: 10px;
}}
.og-cv-badge {{
  display: inline-block;
  padding: 3px 12px;
  border-radius: 999px;
  font-size: {_FS["caption"]};
  font-weight: 700;
  margin: 3px 4px 3px 0;
}}

/* =====================================================================
   PROCESSING PROGRESS STEPS
   ===================================================================== */
.og-progress-steps {{
  display: flex;
  gap: 0;
  align-items: stretch;
  background: var(--surface-soft);
  border: 1px solid var(--line);
  border-radius: 16px;
  overflow: hidden;
  margin-bottom: 18px;
}}
.og-progress-step {{
  flex: 1;
  padding: 11px 8px;
  text-align: center;
  font-size: 11px;
  font-weight: 600;
  color: var(--ink-soft);
  border-right: 1px solid var(--line);
  transition: background 0.32s ease, color 0.32s ease;
}}
.og-progress-step:last-child {{ border-right: none; }}
.og-progress-step.active {{
  background: linear-gradient(135deg, var(--plum), var(--plum-deep));
  color: #fff;
}}
.og-progress-step.done {{
  background: var(--sage-soft);
  color: var(--sage);
}}

/* =====================================================================
   INSPECTION MODE CARDS
   ===================================================================== */
.og-mode-card {{
  border-radius: 16px;
  padding: 18px 16px 16px;
  cursor: pointer;
  transition: all 0.18s cubic-bezier(0.34,1.56,0.64,1);
  min-height: 118px;
  position: relative;
  border: 2px solid var(--line);
  background: var(--surface-soft);
}}
.og-mode-card.active-rgb {{
  border-color: var(--copper);
  background: var(--rust-soft);
}}
.og-mode-card.active-nir {{
  border-color: #3b82d4;
  background: rgba(59,130,212,0.08);
}}

/* =====================================================================
   EXPORT CARDS
   ===================================================================== */
.og-export-card {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 20px 24px;
  display: flex;
  align-items: center;
  gap: 18px;
  transition: all 0.22s ease;
  box-shadow: 0 2px 8px var(--shadow);
}}
.og-export-card:hover {{
  border-color: var(--plum);
  box-shadow: 0 4px 16px var(--card-glow), 0 2px 8px var(--shadow);
  transform: translateY(-1px);
}}
.og-export-icon {{
  width: 46px; height: 46px;
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 20px;
  flex-shrink: 0;
}}

/* =====================================================================
   MISC UI ELEMENTS
   ===================================================================== */
.og-input-toggle {{ margin-bottom: 10px; }}

.og-camera-closed {{
  text-align: center;
  padding: 38px 20px;
  border: 2px dashed var(--line);
  border-radius: 18px;
  background: var(--upload-zone-bg, var(--surface-soft));
  margin-bottom: 14px;
}}
.og-camera-closed-icon  {{ font-size: 38px; margin-bottom: 10px; }}
.og-camera-closed-title {{
  font-weight: 700;
  font-size: {_FS["h3"]};
  color: var(--ink);
  margin-bottom: 5px;
}}
.og-camera-closed-sub {{
  font-size: {_FS["caption"]};
  color: var(--ink-soft);
  max-width: 380px;
  margin: 0 auto;
  line-height: 1.55;
}}

/* Pill badge */
.og-pill {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 12px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
}}
.og-pill-green {{
  background: var(--sage-soft);
  color: var(--sage);
  border: 1px solid rgba(74,222,128,0.30);
}}
.og-pill-plum {{
  background: var(--nav-active-bg);
  color: var(--plum-deep);
  border: 1px solid var(--glass-border);
}}

/* Section label */
.og-section-label {{
  font-size: 10.5px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--ink-soft);
  margin-bottom: 8px;
}}

/* =====================================================================
   CAPTION / SMALL TEXT
   ===================================================================== */
small, .og-caption,
div[data-testid="stCaptionContainer"],
div[data-testid="stCaption"] {{
  color: var(--ink-soft) !important;
  font-size: {_FS["caption"]} !important;
}}

div[data-testid="stSpinner"] p {{
  color: var(--ink) !important;
  font-size: {_FS["base"]} !important;
}}

/* =====================================================================
   CHECKBOX & TOGGLE — theme-aware
   ===================================================================== */
div[data-testid="stCheckbox"] label,
div[data-testid="stCheckbox"] label span,
div[data-testid="stCheckbox"] p {{
  color: var(--ink) !important;
  font-size: {_FS["base"]} !important;
}}
div[data-testid="stCheckbox"] input[type="checkbox"] + div {{
  border-color: var(--line) !important;
  background: var(--surface) !important;
}}
div[data-testid="stCheckbox"] input[type="checkbox"]:checked + div {{
  background: var(--plum) !important;
  border-color: var(--plum) !important;
}}
div[data-testid="stToggle"] label,
div[data-testid="stToggle"] label span,
div[data-testid="stToggle"] p {{
  color: var(--ink) !important;
  font-size: {_FS["base"]} !important;
}}

/* =====================================================================
   SELECT SLIDER (font size control) — theme-aware
   ===================================================================== */
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] p,
div[data-testid="stSlider"] span {{
  color: var(--ink) !important;
}}
div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
div[data-testid="stSlider"] div[data-testid="stTickBarMax"] {{
  color: var(--ink-soft) !important;
}}

/* =====================================================================
   NUMBER / METRIC — theme-aware
   ===================================================================== */
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] div[data-testid="stMetricValue"],
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {{
  color: var(--ink) !important;
}}

/* =====================================================================
   MARKDOWN GENERAL — ensure no black-on-dark issues
   ===================================================================== */
.stMarkdown p, .stMarkdown li, .stMarkdown ul, .stMarkdown ol,
.stMarkdown blockquote, .stMarkdown code {{
  color: var(--ink) !important;
}}
/* Inline code */
.stMarkdown code {{
  background: var(--surface-soft) !important;
  color: var(--copper-deep) !important;
  border-radius: 4px !important;
  padding: 1px 5px !important;
  font-size: 0.92em !important;
}}

/* =====================================================================
   ANIMATIONS
   ===================================================================== */
.og-fade-in {{
  animation: ogfade 0.48s cubic-bezier(0.16,1,0.3,1) both;
}}
.og-slide-up {{
  animation: ogslideup 0.42s cubic-bezier(0.16,1,0.3,1) both;
}}
@keyframes ogfade {{
  from {{ opacity: 0; transform: translateY(10px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes ogslideup {{
  from {{ opacity: 0; transform: translateY(18px) scale(0.98); }}
  to   {{ opacity: 1; transform: translateY(0)   scale(1); }}
}}

/* =====================================================================
   RESPONSIVE
   ===================================================================== */
@media (max-width: 768px) {{
  div.block-container {{ padding-top: 1rem !important; }}
  .og-card {{ padding: 18px 16px; border-radius: 14px; }}
  .og-report {{ padding: 20px 18px; }}
  h1 {{ font-size: calc({_FS["h1"]} * 0.80) !important; }}
  h2 {{ font-size: calc({_FS["h2"]} * 0.84) !important; }}
  .og-progress-steps {{ flex-wrap: wrap; }}
  .og-progress-step {{ min-width: 48%; border-right: none; border-bottom: 1px solid var(--line); }}
}}
@media (max-width: 480px) {{
  html, body, p, span, label {{ font-size: max(12px, {_FS["base"]}) !important; }}
  .og-report-body {{ font-size: 13px; }}
}}

</style>
"""
)


# ---------------------------------------------------------------------------
# Header — premium glassmorphism header
# ---------------------------------------------------------------------------

md(
    f"""
<div style="
    position: relative;
    overflow: hidden;
    border-radius: 22px;
    margin-bottom: 20px;
    background: linear-gradient(
        138deg,
        var(--plum-deep) 0%,
        var(--plum) 42%,
        var(--copper) 85%,
        var(--gold) 130%
    );
    padding: 20px 28px;
    box-shadow:
        0 4px 24px var(--shadow),
        0 1px 0 rgba(255,255,255,0.12) inset;
">
  <!-- Decorative glow blobs -->
  <div style="
      position:absolute; right:-30px; top:-50px;
      width:200px; height:200px; border-radius:50%;
      background: radial-gradient(circle, rgba(217,164,65,0.30), transparent 68%);
      pointer-events:none;
  "></div>
  <div style="
      position:absolute; left:30%; bottom:-60px;
      width:140px; height:140px; border-radius:50%;
      background: radial-gradient(circle, rgba(255,255,255,0.08), transparent 68%);
      pointer-events:none;
  "></div>

  <div style="
      position:relative;
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:16px;
      flex-wrap:wrap;
  ">
    <!-- LEFT: Branding -->
    <div style="display:flex; align-items:center; gap:14px;">
      <!-- Onion logo icon -->
      <div style="
          width:48px; height:48px;
          border-radius:14px;
          background: linear-gradient(148deg, #F5E8CC, #D9A441);
          display:flex; align-items:center; justify-content:center;
          flex-shrink:0;
          box-shadow: 0 2px 8px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.30);
          font-size: 26px;
      ">🧅</div>

      <div>
        <div style="
            font-family:'Fraunces',Georgia,serif;
            font-weight:700;
            font-size:26px;
            color:#FFFBF4;
            line-height:1.1;
            letter-spacing:-0.02em;
        ">OnionGrade AI</div>
        <div style="
            font-size:12px;
            color:rgba(255,251,244,0.72);
            margin-top:3px;
            font-weight:400;
            letter-spacing:0.01em;
        ">{T("app_subtitle")}</div>
      </div>
    </div>

    <!-- RIGHT: Status + Team badge -->
    <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap; justify-content:flex-end;">
      <!-- Model Online pill -->
      <div style="
          display:flex; align-items:center; gap:7px;
          background:rgba(255,255,255,0.14);
          backdrop-filter:blur(8px);
          border:1px solid rgba(255,255,255,0.22);
          padding:6px 14px;
          border-radius:999px;
          font-size:12px;
          font-weight:600;
          color:#FFFBF4;
          white-space:nowrap;
      ">
        <span style="
            width:7px; height:7px; border-radius:50%;
            background:#4ADE80;
            box-shadow:0 0 6px #4ADE80;
            flex-shrink:0;
            display:inline-block;
        "></span>
        Model Online
        <span style="
            font-size:10px;
            background:rgba(255,255,255,0.16);
            padding:1px 7px;
            border-radius:999px;
            margin-left:2px;
        ">AI ✦</span>
      </div>

      <!-- Team badge -->
      <div style="
          font-size:11.5px;
          color:rgba(255,251,244,0.90);
          background:rgba(255,255,255,0.13);
          border:1px solid rgba(255,255,255,0.18);
          padding:6px 13px;
          border-radius:999px;
          font-weight:600;
          white-space:nowrap;
          backdrop-filter:blur(8px);
      ">Team NeuroNex · SIH26031</div>
    </div>
  </div>
</div>
"""
)


# ---------------------------------------------------------------------------
# Sidebar — language toggle, batch history, comparison, trend
# ---------------------------------------------------------------------------

with st.sidebar:
    # ── Sidebar branding strip ─────────────────────────────────────────────────
    md(f"""
    <div style="
        display:flex; align-items:center; gap:10px;
        padding:6px 4px 16px;
        border-bottom:1px solid var(--line);
        margin-bottom:14px;
    ">
      <div style="
          width:34px; height:34px; border-radius:10px;
          background:linear-gradient(135deg,var(--plum),var(--plum-deep));
          display:flex; align-items:center; justify-content:center;
          font-size:17px; flex-shrink:0;
          box-shadow:0 2px 8px var(--shadow);
      ">🧅</div>
      <div>
        <div style="font-weight:700;font-size:13.5px;color:var(--ink);font-family:'Fraunces',serif;">OnionGrade AI</div>
        <div style="font-size:10px;color:var(--ink-soft);margin-top:1px;">v1.0 · NeuroNex</div>
      </div>
    </div>

    """)

    # ── System Status block ────────────────────────────────────────────────────
    _backend_ok = "result" in st.session_state or True   # Always show; colour varies with health
    _model_ver   = "Gemini Vision"
    _sb_theme    = st.session_state.get("theme_mode", "light")
    _stat_bg     = "rgba(74,222,128,0.13)" if _sb_theme == "dark" else "rgba(45,122,79,0.09)"
    _stat_color  = "#4ADE80" if _sb_theme == "dark" else "#2D7A4F"
    md(f"""
    <div style="
        margin: 14px 0 6px;
        background:{_stat_bg};
        border:1px solid {_stat_color}40;
        border-radius:14px;
        padding:12px 14px;
    ">
      <div style="
          font-size:9px; font-weight:700; text-transform:uppercase;
          letter-spacing:0.10em; color:var(--ink-soft); margin-bottom:8px;
      ">System Status</div>

      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
        <span style="
            width:8px;height:8px;border-radius:50%;
            background:{_stat_color};
            box-shadow:0 0 5px {_stat_color};
            flex-shrink:0;display:inline-block;
        "></span>
        <span style="font-size:12px;font-weight:600;color:var(--ink);">AI Model Online</span>
      </div>

      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:11px;color:var(--ink-soft);">Model Version</span>
        <span style="
            font-size:11px;font-weight:700;color:var(--ink);
            background:var(--surface);border:1px solid var(--line);
            padding:2px 8px;border-radius:999px;
        ">v1.0</span>
      </div>

      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:5px;">
        <span style="font-size:11px;color:var(--ink-soft);">AI Engine</span>
        <span style="font-size:11px;font-weight:600;color:var(--ink);">Gemini Vision</span>
      </div>
    </div>
    """)

    st.divider()

    # ── Theme toggle ──────────────────────────────────────────────────────────
    st.markdown("### 🎨 Theme")
    _THEME_OPTIONS = {"☀️ Light": "light", "🌙 Dark": "dark", "🖥️ System": "system"}
    _theme_labels = list(_THEME_OPTIONS.keys())
    _current_theme_label = next(
        (label for label, mode in _THEME_OPTIONS.items()
         if mode == st.session_state["theme_mode"]),
        "☀️ Light",
    )
    try:
        theme_choice = st.segmented_control(
            "Theme",
            options=_theme_labels,
            default=_current_theme_label,
            label_visibility="collapsed",
            key="theme_segmented",
        )
    except (AttributeError, TypeError):
        # Older Streamlit without st.segmented_control — fall back to a radio.
        theme_choice = st.radio(
            "Theme",
            options=_theme_labels,
            index=_theme_labels.index(_current_theme_label),
            horizontal=True,
            label_visibility="collapsed",
            key="theme_radio",
        )
    if theme_choice and _THEME_OPTIONS[theme_choice] != st.session_state["theme_mode"]:
        st.session_state["theme_mode"] = _THEME_OPTIONS[theme_choice]
        st.rerun()

    st.divider()

    # ── Language toggle ──────────────────────────────────────────────────────
    st.markdown("### 🌐 Language / भाषा")
    _LANG_OPTIONS = {
        "English":    "en",
        "हिंदी":      "hi",
        "मराठी":      "mr",
        "ગુજરાતી":    "gu",
        "ਪੰਜਾਬੀ":     "pa",
        "தமிழ்":      "ta",
        "తెలుగు":     "te",
        "ಕನ್ನಡ":      "kn",
        "বাংলা":      "bn",
    }
    _current_lang = st.session_state.get("lang", "en")
    _current_label = next(
        (label for label, code in _LANG_OPTIONS.items() if code == _current_lang),
        "English",
    )
    lang_choice = st.selectbox(
        "Language",
        options=list(_LANG_OPTIONS.keys()),
        index=list(_LANG_OPTIONS.keys()).index(_current_label),
        label_visibility="collapsed",
    )
    st.session_state["lang"] = _LANG_OPTIONS[lang_choice]

    st.divider()

    # ── Font Size control ─────────────────────────────────────────────────────
    st.markdown("### 🔡 Text Size")
    _FS_OPTIONS = {
        "Small":       "small",
        "Normal":      "normal",
        "Medium":      "medium",
        "Large":       "large",
        "Extra Large": "xlarge",
    }
    _current_fs_label = next(
        (lbl for lbl, val in _FS_OPTIONS.items()
         if val == st.session_state.get("font_size", "normal")),
        "Normal",
    )
    _fs_choice = st.select_slider(
        "Text size",
        options=list(_FS_OPTIONS.keys()),
        value=_current_fs_label,
        label_visibility="collapsed",
    )
    _fs_val = _FS_OPTIONS[_fs_choice]
    if _fs_val != st.session_state.get("font_size", "normal"):
        st.session_state["font_size"] = _fs_val
        st.rerun()

    st.divider()

    # ── Confidence Threshold Slider (Feature 5) ───────────────────────────────
    st.markdown("### 🎯 Confidence Threshold")
    _conf_threshold = st.slider(
        "Minimum confidence to mark as Accepted",
        min_value=0.50,
        max_value=0.90,
        value=st.session_state.get("confidence_threshold", 0.70),
        step=0.05,
        format="%.2f",
        help=(
            "Detections with confidence ≥ this value are marked **Accepted**. "
            "Detections below it are marked **Low Confidence** (still visible, not removed). "
            "Default 0.70 keeps the current effective behavior."
        ),
    )
    st.session_state["confidence_threshold"] = _conf_threshold
    st.caption(
        f"Accepted ≥ {_conf_threshold:.0%} · Low Confidence < {_conf_threshold:.0%} · "
        "Detections are never silently removed."
    )

    st.divider()

    # ── API Health Status (Feature 10) ────────────────────────────────────────
    st.markdown("### 🔑 API Configuration")
    try:
        _api_health_resp = requests.get(f"{BACKEND_URL}/api-health", timeout=3)
        _api_health = _api_health_resp.json() if _api_health_resp.status_code == 200 else {}
        _api_status = _api_health.get("status", "error")
        if _api_status == "configured":
            st.success("Gemini API: Configured ✓", icon="✅")
        elif _api_status == "not_configured":
            st.info("Gemini API: Not configured\n(demo/offline mode active)", icon="🔵")
        else:
            st.warning("Gemini API: Validation failed", icon="⚠️")
        _api_model = _api_health.get("vision_model", "")
        if _api_model:
            st.caption(f"Model: `{_api_model}`")
    except Exception:
        st.caption("API status unavailable")

    st.divider()

    # ── Batch history (in-session) ────────────────────────────────────────────
    st.markdown(f"### 📋 {T('history_heading')}")

    history: list = st.session_state.get("batch_history", [])

    if not history:
        st.caption(T("history_empty"))
    else:
        for i, entry in enumerate(reversed(history)):
            bid   = entry.get("batch_id", f"Batch {i+1}")
            ga    = entry.get("grade_a_pct", 0)
            total = entry.get("total", 0)
            label = f"**{bid}** — Grade A: {ga}% ({total} {T('onions')})"
            if st.button(label, key=f"hist_{i}", use_container_width=True):
                st.session_state["result"]       = entry["result"]
                st.session_state["image"]        = entry["image"]
                st.session_state["marked_image"] = entry.get("marked_image")
                st.rerun()

        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state["batch_history"] = []
            st.rerun()

    # ── Batch comparison ─────────────────────────────────────────────────────
    if len(history) >= 2:
        st.divider()
        st.markdown(f"### 📊 {T('compare_heading')}")
        batch_labels = [
            f"{e.get('batch_id', f'Batch {i+1}')}"
            for i, e in enumerate(history)
        ]
        sel_a = st.selectbox("Batch A", batch_labels, index=0, key="cmp_a")
        sel_b = st.selectbox("Batch B", batch_labels,
                             index=min(1, len(batch_labels)-1), key="cmp_b")

        if sel_a != sel_b:
            def _entry(label):
                idx = batch_labels.index(label)
                return history[idx]

            ea, eb = _entry(sel_a), _entry(sel_b)
            _is_dark_cmp_pre = st.session_state.get("theme_mode", "light") == "dark"
            _cmp_sage = "#4ADE80" if _is_dark_cmp_pre else THEME["sage"]
            _cmp_copper = "#E8A84F" if _is_dark_cmp_pre else THEME["copper"]
            _cmp_rust = "#F87171" if _is_dark_cmp_pre else THEME["rust"]
            cmp_fig = go.Figure(data=[
                go.Bar(
                    name=sel_a,
                    x=[T("grade_a"), T("urs"), T("defect")],
                    y=[ea.get("grade_a_pct",0), ea.get("urs_pct",0), ea.get("defect_pct",0)],
                    marker_color=[_cmp_sage, _cmp_copper, _cmp_rust],
                ),
                go.Bar(
                    name=sel_b,
                    x=[T("grade_a"), T("urs"), T("defect")],
                    y=[eb.get("grade_a_pct",0), eb.get("urs_pct",0), eb.get("defect_pct",0)],
                    marker_color=["rgba(74,222,128,0.50)", "rgba(232,168,79,0.50)", "rgba(248,113,113,0.50)"],
                ),
            ])
            _is_dark_cmp = st.session_state.get("theme_mode", "light") == "dark"
            _cmp_fc = "#F0EDE8" if _is_dark_cmp else "#1C1009"
            _cmp_gc = "rgba(255,255,255,0.10)" if _is_dark_cmp else "#E8DDD0"
            cmp_fig.update_layout(
                barmode="group", height=220,
                margin=dict(l=4, r=4, t=10, b=10),
                yaxis_title="%",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=10, family="Inter, IBM Plex Sans", color=_cmp_fc),
                legend=dict(orientation="h", y=-0.25, font=dict(color=_cmp_fc)),
                yaxis=dict(gridcolor=_cmp_gc),
            )
            st.plotly_chart(cmp_fig, config={"displayModeBar": False},
                            use_container_width=True)

    # ── Trend chart ──────────────────────────────────────────────────────────
    if len(history) >= 2:
        st.divider()
        st.markdown(f"### 📈 {T('trend_heading')}")
        trend_labels = [e.get("batch_id", f"#{i+1}") for i, e in enumerate(history)]
        trend_vals   = [e.get("grade_a_pct", 0) for e in history]
        _is_dark_trend_pre = st.session_state.get("theme_mode", "light") == "dark"
        _trend_sage = "#4ADE80" if _is_dark_trend_pre else THEME["sage"]
        trend_fig = go.Figure(go.Scatter(
            x=trend_labels, y=trend_vals,
            mode="lines+markers+text",
            text=[f"{v}%" for v in trend_vals],
            textposition="top center",
            textfont=dict(size=9),
            line=dict(color=_trend_sage, width=2.5),
            marker=dict(color=_trend_sage, size=9),
        ))
        _is_dark_trend = st.session_state.get("theme_mode", "light") == "dark"
        _trend_fc = "#F0EDE8" if _is_dark_trend else "#1C1009"
        _trend_gc = "rgba(255,255,255,0.10)" if _is_dark_trend else "#E8DDD0"
        trend_fig.update_layout(
            height=200,
            margin=dict(l=4, r=4, t=10, b=10),
            yaxis=dict(range=[0, 110], title="%", gridcolor=_trend_gc),
            xaxis=dict(tickangle=-35, gridcolor=_trend_gc),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, family="Inter, IBM Plex Sans", color=_trend_fc),
        )
        st.plotly_chart(trend_fig, config={"displayModeBar": False},
                        use_container_width=True)


# ---------------------------------------------------------------------------
# Backend health check
# ---------------------------------------------------------------------------

try:
    health_response = requests.get(f"{BACKEND_URL}/health", timeout=3)
    health_response.raise_for_status()
    health = health_response.json()

    if health.get("demo_mode"):
        st.info(T("health_offline"), icon="🔵")
    else:
        model_name = health.get("vision_model", "Gemini Vision")
        st.success(f"{T('health_online')} (`{model_name}`)", icon="✅")

except requests.exceptions.RequestException:
    st.error(f"{T('health_error')} (`{BACKEND_URL}`)")
    st.stop()


# ---------------------------------------------------------------------------
# Stepper
# ---------------------------------------------------------------------------

current_step = (
    4
    if "result" in st.session_state
    else 0
)

md(
    f"""
    <div class="og-card"
         style="padding:16px 22px;margin-bottom:18px;">
        {stepper(current_step)}
    </div>
    """
)


# ---------------------------------------------------------------------------
# Upload / Camera  +  Batch ID
# ---------------------------------------------------------------------------

md(f"""
<div class="og-card" style="padding:22px 24px;">
  <div style="
      display:flex;align-items:center;gap:8px;
      margin-bottom:14px;
  ">
    <div style="
        width:28px;height:28px;border-radius:8px;
        background:linear-gradient(135deg,var(--plum),var(--plum-deep));
        display:flex;align-items:center;justify-content:center;
        font-size:14px;flex-shrink:0;
    ">🔬</div>
    <div>
      <div style="font-weight:700;font-size:14px;color:var(--ink);">Upload &amp; Analyse</div>
      <div style="font-size:11px;color:var(--ink-soft);">Select inspection mode, then upload or capture a batch photo</div>
    </div>
  </div>
""")

# ── Inspection Mode ───────────────────────────────────────────────────────────
_is_nir = st.session_state.get("inspection_mode", "rgb") == "nir"

md(f"""
<div style="
    font-size:10px;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:0.09em;
    color:var(--ink-soft);
    margin-bottom:10px;
">
    {T("inspection_mode_label")}
</div>
""")

_col_rgb, _col_nir = st.columns(2, gap="small")

# ── RGB card ──
with _col_rgb:
    _rgb_active = not _is_nir
    md(f"""
    <div style="
        border:2px solid {'var(--copper)' if _rgb_active else 'var(--line)'};
        background:{'var(--rust-soft)' if _rgb_active else 'var(--surface-soft)'};
        border-radius:16px;
        padding:18px 16px 14px;
        cursor:pointer;
        transition:all 0.18s cubic-bezier(0.34,1.56,0.64,1);
        min-height:118px;
        position:relative;
        box-shadow:{'0 2px 10px var(--shadow)' if _rgb_active else 'none'};
    ">
        <div style="
            width:36px;height:36px;border-radius:10px;
            background:{'var(--rust-soft)' if _rgb_active else 'var(--surface)'};
            border:1.5px solid {'var(--copper)' if _rgb_active else 'var(--line)'};
            display:flex;align-items:center;justify-content:center;
            font-size:18px;margin-bottom:8px;
        ">🔴</div>
        <div style="
            font-size:13px;font-weight:700;
            color:{'var(--copper-deep)' if _rgb_active else 'var(--ink-soft)'};
            margin-bottom:4px;
        ">External RGB</div>
        <div style="font-size:11px;color:var(--ink-soft);line-height:1.5;">
            Surface quality &amp; visible defects
        </div>
        {'<div style="position:absolute;top:12px;right:13px;width:8px;height:8px;border-radius:50%;background:var(--copper);box-shadow:0 0 5px var(--copper);"></div>' if _rgb_active else ''}
    </div>
    """)
    if st.button(
        T("select_rgb") if _is_nir else T("rgb_selected"),
        key="btn_rgb",
        use_container_width=True,
        type="primary" if not _is_nir else "secondary",
    ):
        st.session_state["inspection_mode"] = "rgb"
        st.rerun()

# ── NIR card ──
with _col_nir:
    _nir_active = _is_nir
    md(f"""
    <div style="
        border:2px solid {'#3b82d4' if _nir_active else 'var(--line)'};
        background:{'rgba(59,130,212,0.10)' if _nir_active else 'var(--surface-soft)'};
        border-radius:16px;
        padding:18px 16px 14px;
        cursor:pointer;
        transition:all 0.18s cubic-bezier(0.34,1.56,0.64,1);
        min-height:118px;
        position:relative;
        box-shadow:{'0 2px 10px var(--shadow)' if _nir_active else 'none'};
    ">
        <div style="
            width:36px;height:36px;border-radius:10px;
            background:{'rgba(59,130,212,0.15)' if _nir_active else 'var(--surface)'};
            border:1.5px solid {'#3b82d4' if _nir_active else 'var(--line)'};
            display:flex;align-items:center;justify-content:center;
            font-size:18px;margin-bottom:8px;
        ">🔵</div>
        <div style="
            font-size:13px;font-weight:700;
            color:{'#3b82d4' if _nir_active else 'var(--ink-soft)'};
            margin-bottom:4px;
        ">Internal NIR</div>
        <div style="font-size:11px;color:var(--ink-soft);line-height:1.5;">
            Hollow heart &amp; internal defects
        </div>
        {'<div style="position:absolute;top:12px;right:13px;width:8px;height:8px;border-radius:50%;background:#3b82d4;box-shadow:0 0 5px #3b82d4;"></div>' if _nir_active else ''}
    </div>
    """)
    if st.button(
        T("select_nir") if not _is_nir else T("nir_selected"),
        key="btn_nir",
        use_container_width=True,
        type="primary" if _is_nir else "secondary",
    ):
        st.session_state["inspection_mode"] = "nir"
        st.rerun()

st.divider()

# Batch ID input
custom_batch_id = st.text_input(
    T("batch_id_label"),
    placeholder="e.g. LOT-MH-001",
    help=T("batch_id_help"),
    max_chars=40,
)

# ── Input mode switch: Upload  vs  Camera ──────────────────────────────────
# IMPORTANT: the camera must never be initialised until the user explicitly
# asks for it — both by switching to "Camera" mode AND by pressing
# "Open Camera". Nothing below calls st.camera_input() unless camera_open
# is True, so no permission prompt / video stream is ever created on load.
st.session_state.setdefault("input_mode", "upload")
st.session_state.setdefault("camera_open", False)

md('<div class="og-input-toggle">')
col_mode_upload, col_mode_camera = st.columns(2, gap="small")
with col_mode_upload:
    if st.button(
        f"📁 {T('upload_heading')}",
        key="btn_mode_upload",
        use_container_width=True,
        type="primary" if st.session_state["input_mode"] == "upload" else "secondary",
    ):
        st.session_state["input_mode"] = "upload"
        st.session_state["camera_open"] = False
        st.rerun()
with col_mode_camera:
    if st.button(
        f"📷 {T('camera_heading')}",
        key="btn_mode_camera",
        use_container_width=True,
        type="primary" if st.session_state["input_mode"] == "camera" else "secondary",
    ):
        st.session_state["input_mode"] = "camera"
        st.rerun()
md("</div>")

uploaded_files = None
camera_photo = None

# Optional NIR file (only shown in upload mode + NIR inspection mode)
_nir_upload_file = None

if st.session_state["input_mode"] == "upload":
    st.caption(T("upload_caption"))
    uploaded_files = st.file_uploader(
        T("upload_heading"),
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Feature 7 — Real NIR/SWIR dual-image upload
    # Only shown when user has selected NIR inspection mode.
    # Provides an optional second image for actual NIR/SWIR data.
    if st.session_state.get("inspection_mode") == "nir":
        st.divider()
        md("""
        <div style="background:rgba(59,130,212,0.08);border:1.5px solid #3b82d4;
                    border-radius:10px;padding:10px 14px;margin-bottom:8px;">
          <div style="font-weight:700;color:#3b82d4;font-size:13px;">
            🔵 Optional: Upload real NIR/SWIR image
          </div>
          <div style="font-size:12px;color:var(--ink-soft);margin-top:3px;">
            If you have a genuine near-infrared or SWIR image from an NIR camera,
            upload it here. The backend will use it for NIR analysis instead of the
            RGB-proxy simulation. Leave empty to use RGB-proxy mode (existing behaviour).
          </div>
        </div>
        """)
        _nir_upload_file = st.file_uploader(
            "NIR/SWIR image (optional)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            key="nir_swir_uploader",
            help=(
                "Upload an actual NIR/SWIR image paired with your RGB image above. "
                "Mode will be shown as 'Real NIR' in the results. "
                "If not uploaded, RGB-proxy NIR mode is used."
            ),
        )
        if _nir_upload_file is not None:
            st.caption(
                f"✓ NIR image loaded: `{_nir_upload_file.name}` "
                f"({_nir_upload_file.size // 1024} KB) — **Real NIR mode**"
            )
        else:
            st.caption("No NIR image — will use **RGB-proxy NIR** mode (simulated).")

else:
    st.caption(T("camera_caption"))

    if not st.session_state["camera_open"]:
        # Camera is CLOSED — no camera widget exists yet, so the browser
        # never asks for camera permission until the button below is clicked.
        md(
            f"""
            <div class="og-camera-closed">
                <div class="og-camera-closed-icon">📷</div>
                <div class="og-camera-closed-title">{T("camera_closed_title")}</div>
                <div class="og-camera-closed-sub">{T("camera_closed_sub")}</div>
            </div>
            """
        )
        _cam_col_a, _cam_col_b, _cam_col_c = st.columns([1, 1.4, 1])
        with _cam_col_b:
            if st.button(
                T("camera_open_btn"),
                key="btn_open_camera",
                use_container_width=True,
                type="primary",
            ):
                st.session_state["camera_open"] = True
                st.rerun()
    else:
        # Camera is OPEN — the widget is created now, this is the only
        # point in the app where st.camera_input() is ever called.
        camera_photo = st.camera_input(
            T("camera_heading"),
            label_visibility="collapsed",
            key="og_camera_widget",
        )
        if st.button(
            T("camera_close_btn"),
            key="btn_close_camera",
            use_container_width=True,
        ):
            st.session_state["camera_open"] = False
            st.rerun()

# Merge: camera photo OR uploaded files
_has_input = bool(uploaded_files) or (camera_photo is not None)

col_btn1, col_btn2 = st.columns([2, 1])
with col_btn1:
    analyse_clicked = st.button(
        T("analyse_btn"),
        disabled=not _has_input,
        type="primary",
        use_container_width=True,
    )
with col_btn2:
    if st.button(
        T("new_analysis_btn"),
        disabled="result" not in st.session_state,
        use_container_width=True,
    ):
        for key in ("result", "image", "marked_image"):
            st.session_state.pop(key, None)
        st.rerun()

md("</div>")


# ---------------------------------------------------------------------------
# Run analysis  (supports multi-image + camera)
# ---------------------------------------------------------------------------

# Build list of (bytes, name, mime) for each source
_inputs: list[tuple[bytes, str, str]] = []
if camera_photo is not None:
    _inputs.append((camera_photo.getvalue(), "camera_capture.jpg", "image/jpeg"))
elif uploaded_files:
    for _f in uploaded_files:
        _inputs.append((_f.getvalue(), _f.name, _f.type or "image/jpeg"))

# Image quality pre-check (warn but don't block analysis)
for _img_bytes, _img_name, _ in _inputs:
    _warn = check_image_quality(_img_bytes)
    if _warn:
        st.warning(f"{_img_name}: {_warn}")

def _progress_steps_html(active: int) -> str:
    """Render the multi-step processing progress bar HTML.

    Feature 9 — Real-Time Analysis Progress (8 stages):
      0. Uploading
      1. Preprocessing
      2. Gemini Analysis
      3. Grading
      4. NIR Analysis  (only meaningful in NIR mode)
      5. Report
      6. Saving History
      7. Complete

    Only stages that have actually started/completed are marked.
    No fake percentage — each step reflects a real completed stage.
    """
    labels = [
        T("prog_uploading"),         # 0
        T("prog_processing"),        # 1 — image preprocessing
        T("prog_analyzing"),         # 2 — Gemini / model
        "Grading",                   # 3 — deterministic Python grading
        "NIR Analysis",              # 4 — NIR screening
        T("prog_preparing"),         # 5 — report assembly
        "Saving History",            # 6 — SQLite save
        T("prog_complete"),          # 7 — done
    ]
    steps_html = ""
    for i, lbl in enumerate(labels):
        if i < active:
            cls = "done"
        elif i == active:
            cls = "active"
        else:
            cls = ""
        steps_html += f'<div class="og-progress-step {cls}">{lbl}</div>'
    return f'<div class="og-progress-steps">{steps_html}</div>'


if _inputs and analyse_clicked:

    # Use custom batch ID or let backend auto-assign
    _base_batch_id = custom_batch_id.strip() if custom_batch_id.strip() else None

    # First image used for the bounding-box display
    uploaded_bytes = _inputs[0][0]

    # Multi-step progress display
    _prog_slot = st.empty()
    _prog_slot.markdown(
        _progress_steps_html(0),
        unsafe_allow_html=True,
    )
    _status_slot = st.empty()
    _status_slot.info(f"📤 {T('prog_uploading')} — {len(_inputs)} image(s) ready…")

    with st.spinner(T("spinner_msg")):
        # ── Multi-image loop: call /analyze for each image, combine ──────────
        all_onions:    list = []
        all_flagged:   list = []
        combined_result: dict | None = None
        _n_images = len(_inputs)

        _insp_mode = st.session_state.get("inspection_mode", "rgb")

        # step 1 — preprocessing (images are preprocessed server-side before Gemini)
        _prog_slot.markdown(_progress_steps_html(1), unsafe_allow_html=True)
        _status_slot.info(f"⚙ {T('prog_processing')} — preprocessing image(s) & connecting to Gemini Vision…")

        _conf_thresh = st.session_state.get("confidence_threshold", 0.70)

        # Determine if user has provided a real NIR image (Feature 7)
        _nir_file_bytes: bytes | None = None
        _nir_file_name: str = ""
        _nir_file_mime: str = "image/jpeg"
        if _insp_mode == "nir" and _nir_upload_file is not None:
            _nir_file_bytes = _nir_upload_file.getvalue()
            _nir_file_name = _nir_upload_file.name
            _nir_file_mime = _nir_upload_file.type or "image/jpeg"

        for _img_idx, (_img_bytes, _img_name, _img_mime) in enumerate(_inputs):
            _response = None
            try:
                # Use /analyze/nir with dual upload when real NIR image present;
                # otherwise use /analyze with inspection_mode param (RGB-proxy NIR).
                if _insp_mode == "nir" and _nir_file_bytes:
                    _files = {
                        "rgb_file": (_img_name, _img_bytes, _img_mime),
                        "nir_file": (_nir_file_name, _nir_file_bytes, _nir_file_mime),
                    }
                    _endpoint = f"{BACKEND_URL}/analyze/nir"
                    _params = {"confidence_threshold": _conf_thresh}
                else:
                    _files = {"file": (_img_name, _img_bytes, _img_mime)}
                    _endpoint = f"{BACKEND_URL}/analyze"
                    _params = {
                        "inspection_mode": _insp_mode,
                        "confidence_threshold": _conf_thresh,
                    }
                _response = requests.post(
                    _endpoint,
                    files=_files,
                    params=_params,
                    timeout=300,
                )
            except requests.exceptions.Timeout:
                _prog_slot.empty()
                _status_slot.empty()
                st.error(f"{_img_name}: Analysis timed out after 300 s.")
                continue
            except requests.exceptions.ConnectionError:
                _prog_slot.empty()
                _status_slot.empty()
                st.error(f"Cannot connect to backend at `{BACKEND_URL}`.")
                break
            except requests.exceptions.RequestException as exc:
                _prog_slot.empty()
                _status_slot.empty()
                st.error(f"Request failed: {exc}")
                continue

            if _response is None or _response.status_code != 200:
                _prog_slot.empty()
                _status_slot.empty()
                try:
                    _detail = _response.json().get("detail", _response.text)
                except Exception:
                    _detail = _response.text if _response else "no response"
                st.error(f"{_img_name}: Analysis failed — {_detail}")
                continue

            try:
                _r = _response.json()
            except ValueError:
                _prog_slot.empty()
                _status_slot.empty()
                st.error(f"{_img_name}: Backend returned invalid JSON.")
                continue

            # step 2 → 3: analyzing (Gemini done) → grading (deterministic)
            _prog_slot.markdown(_progress_steps_html(2), unsafe_allow_html=True)
            _status_slot.info(f"🔍 {T('prog_analyzing')} — Gemini Vision response received…")
            _prog_slot.markdown(_progress_steps_html(3), unsafe_allow_html=True)
            _status_slot.info(f"📐 Grading — deterministic Python rules applied to {len(get_onion_records(_r))} onion(s)…")

            # Collect per-image data
            all_onions.extend(get_onion_records(_r))
            all_flagged.extend(_r.get("flagged", []))
            if combined_result is None:
                combined_result = _r   # use first image as base structure

        # ── Merge multi-image results ────────────────────────────────────────
        if combined_result is not None:
            # step 4 — NIR analysis (mark done whether rgb-proxy or real)
            _prog_slot.markdown(_progress_steps_html(4), unsafe_allow_html=True)
            _nir_note = "real NIR" if _nir_file_bytes else "RGB-proxy"
            _status_slot.info(f"🔵 NIR Analysis — {_nir_note} screening applied…")

            # step 5 — preparing report
            _prog_slot.markdown(_progress_steps_html(5), unsafe_allow_html=True)
            _status_slot.info(f"📊 {T('prog_preparing')} — assembling batch report…")

            # Recompute totals across all images
            _total = len(all_onions)
            _ga = sum(1 for o in all_onions if o.get("final_grade") == "Grade A")
            _urs = sum(1 for o in all_onions if o.get("final_grade") == "URS")
            _def = _total - _ga - _urs
            _ga_pct  = round(_ga / _total * 100, 1) if _total else 0.0
            _urs_pct = round(_urs / _total * 100, 1) if _total else 0.0
            _def_pct = round(100.0 - _ga_pct - _urs_pct, 1) if _total else 0.0

            # Override batch_id with user-supplied value if given
            _effective_bid = (
                _base_batch_id
                or combined_result.get("batch_id", "batch")
            )
            if _n_images > 1:
                _effective_bid = _base_batch_id or f"MULTI-{_n_images}IMG"

            combined_result["batch_id"] = _effective_bid
            combined_result["onions"]   = all_onions
            combined_result["flagged"]  = all_flagged
            combined_result["total_observed"] = _total
            combined_result["summary"] = {
                "grade_a": {"count": _ga,  "percentage": _ga_pct},
                "urs":     {"count": _urs, "percentage": _urs_pct},
                "defect":  {"count": _def, "percentage": _def_pct},
            }
            combined_result["grade_result"].update({
                "total_validated": _total,
                "grade_a_count": _ga, "urs_count": _urs, "defect_count": _def,
                "grade_a_pct": _ga_pct, "urs_pct": _urs_pct, "defect_pct": _def_pct,
            })

            result = combined_result
            st.session_state["result"] = result
            st.session_state["image"]  = uploaded_bytes

            # Bounding-box overlay on first image
            marked_image = draw_onion_markers(uploaded_bytes, get_onion_records(result))
            st.session_state["marked_image"] = marked_image

            # step 6 — saving history
            _prog_slot.markdown(_progress_steps_html(6), unsafe_allow_html=True)
            _status_slot.info("💾 Saving History — persisting to SQLite…")

            # ── Save to in-session batch history (max 10) ────────────────────
            _history = st.session_state.get("batch_history", [])
            _history.append({
                "batch_id":    _effective_bid,
                "grade_a_pct": _ga_pct,
                "urs_pct":     _urs_pct,
                "defect_pct":  _def_pct,
                "total":       _total,
                "result":      result,
                "image":       uploaded_bytes,
                "marked_image": marked_image,
            })
            st.session_state["batch_history"] = _history[-10:]   # keep last 10

            # step 7 — complete
            _prog_slot.markdown(_progress_steps_html(7), unsafe_allow_html=True)
            _status_slot.success(f"✓ {T('prog_complete')} — {_total} onion(s) graded.")

            st.rerun()


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

if "result" not in st.session_state:

    md(
        f"""
        <div class="og-card og-fade-in"
             style="
                 margin-top:18px;
                 text-align:center;
                 padding:52px 30px 44px;
                 background: linear-gradient(
                     160deg,
                     var(--surface) 0%,
                     var(--surface-soft) 100%
                 );
             ">

          <!-- Decorative ring -->
          <div style="
              width:90px;height:90px;border-radius:50%;
              background:linear-gradient(135deg,var(--plum),var(--plum-deep));
              display:flex;align-items:center;justify-content:center;
              margin:0 auto 18px;
              box-shadow:0 8px 28px var(--shadow);
              font-size:40px;
          ">🧅</div>

          <div style="
              font-family:'Fraunces',Georgia,serif;
              font-weight:700;
              font-size:21px;
              color:var(--plum-deep);
              margin-bottom:8px;
              letter-spacing:-0.015em;
          ">
              {T("results_placeholder")}
          </div>

          <div style="
              font-size:13.5px;
              color:var(--ink-soft);
              margin-top:4px;
              max-width:460px;
              margin-left:auto;
              margin-right:auto;
              line-height:1.6;
          ">
              {T("results_sub")}
          </div>

          <!-- Step hints -->
          <div style="
              display:flex;gap:10px;justify-content:center;flex-wrap:wrap;
              margin-top:22px;
          ">
            <div style="
                background:var(--surface-soft);border:1px solid var(--line);
                border-radius:999px;padding:6px 14px;font-size:11.5px;
                font-weight:600;color:var(--ink-soft);
            ">1. Select RGB or NIR mode</div>
            <div style="
                background:var(--surface-soft);border:1px solid var(--line);
                border-radius:999px;padding:6px 14px;font-size:11.5px;
                font-weight:600;color:var(--ink-soft);
            ">2. Upload batch photo</div>
            <div style="
                background:var(--surface-soft);border:1px solid var(--line);
                border-radius:999px;padding:6px 14px;font-size:11.5px;
                font-weight:600;color:var(--ink-soft);
            ">3. Click Analyse batch</div>
          </div>

        </div>
        """
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if "result" in st.session_state:

    data = st.session_state["result"]

    summary = get_summary(data)

    batch_id = get_batch_id(data)

    onions = get_onion_records(data)

    flagged = data.get(
        "flagged",
        [],
    )

    md("<br>")

    # ── Active inspection mode badge ─────────────────────────────────────────
    _res_mode     = data.get("inspection_mode") or st.session_state.get("inspection_mode", "rgb")
    _res_is_nir   = _res_mode == "nir"
    _res_badge_bdr= "#3b82d4" if _res_is_nir else "var(--copper)"
    _res_badge_lbl= T("mode_nir") if _res_is_nir else T("mode_rgb")
    _res_badge_dsc= T("mode_nir_desc") if _res_is_nir else T("mode_rgb_desc")
    # NIR input mode: real_nir or rgb_proxy (Feature 7)
    _nir_input_mode = data.get("nir_input_mode", "")
    _nir_input_badge = ""
    if _res_is_nir and _nir_input_mode:
        if _nir_input_mode == "real_nir":
            _nir_input_badge = (
                '<span style="display:inline-block;margin-top:4px;padding:1px 7px;'
                'background:rgba(59,130,212,0.18);border:1px solid #3b82d4;'
                'border-radius:6px;font-size:10px;font-weight:700;color:#3b82d4;">'
                '📡 Real NIR/SWIR</span>'
            )
        else:
            _nir_input_badge = (
                '<span style="display:inline-block;margin-top:4px;padding:1px 7px;'
                'background:rgba(181,101,29,0.12);border:1px solid var(--copper);'
                'border-radius:6px;font-size:10px;font-weight:600;color:var(--copper);">'
                '🔬 RGB-proxy NIR (simulated)</span>'
            )
    md(f"""
    <div style="
        display:flex;
        align-items:center;
        gap:12px;
        background:var(--surface-soft);
        border:1px solid var(--line);
        border-left:4px solid {_res_badge_bdr};
        border-radius:14px;
        padding:12px 18px;
        margin-bottom:18px;
    ">
        <div style="
            width:36px;height:36px;border-radius:10px;flex-shrink:0;
            background:{'rgba(59,130,212,0.15)' if _res_is_nir else 'var(--rust-soft)'};
            display:flex;align-items:center;justify-content:center;
            font-size:17px;
        ">{'🔵' if _res_is_nir else '🔴'}</div>
        <div>
            <div style="
                font-size:10px;
                text-transform:uppercase;
                font-weight:700;
                letter-spacing:0.07em;
                color:var(--ink-soft);
            ">
                {escape(T('inspection_mode_label'))}
            </div>
            <div style="
                font-size:13px;
                font-weight:700;
                color:var(--ink);
                margin-top:1px;
            ">
                {escape(_res_badge_lbl.lstrip('🔴🔵').strip())}
            </div>
            <div style="
                font-size:11.5px;
                color:var(--ink-soft);
                margin-top:2px;
                line-height:1.5;
            ">
                {escape(_res_badge_dsc)}
            </div>
            {_nir_input_badge}
        </div>
    </div>
    """)

    # ---------------------------------------------------------------
    # Image + grade cards
    # ---------------------------------------------------------------

    image_column, metrics_column = st.columns(
        [1, 2],
        gap="large",
    )

    with image_column:

        md(
            '<div class="og-card og-fade-in" '
            'style="padding:16px;">'
        )

        # ---------------------------------------------------------------
# Numbered onion image
# ---------------------------------------------------------------

        marked_image = st.session_state.get(
            "marked_image"
        )

        original_image = st.session_state.get(
            "image"
        )

        if marked_image:

            st.image(
                marked_image,
                caption=(
                    f"Batch {batch_id} — "
                    "Detected onions — marker number matches the table row"
                ),
                use_column_width=True,
            )

        else:

            st.image(
                original_image,
                caption=f"Batch {batch_id}",
                use_column_width=True,
            )

    md("</div>")

    with metrics_column:

        grade_a = summary.get(
            "grade_a",
            {},
        )

        urs = summary.get(
            "urs",
            {},
        )

        defect = summary.get(
            "defect",
            {},
        )

        _is_dark_rings = st.session_state.get("theme_mode", "light") == "dark"
        _ring_sage   = "#4ADE80" if _is_dark_rings else THEME["sage"]
        _ring_copper = "#E8A84F" if _is_dark_rings else THEME["copper_deep"]
        _ring_rust   = "#F87171" if _is_dark_rings else THEME["rust"]

        md(
            f"""
            <div class="og-fade-in"
                 style="
                     display:flex;
                     gap:14px;
                     margin-bottom:16px;
                 ">

                {metric_ring_card("Grade A", pct(grade_a.get("percentage",0)), int(grade_a.get("count",0)), _ring_sage, "")}
                {metric_ring_card("URS", pct(urs.get("percentage",0)), int(urs.get("count",0)), _ring_copper, "")}
                {metric_ring_card("Defect", pct(defect.get("percentage",0)), int(defect.get("count",0)), _ring_rust, "")}

            </div>
            """
        )

        # -----------------------------------------------------------
        # Grade chart
        # -----------------------------------------------------------

        labels = [
            "Grade A",
            "URS",
            "Defect",
        ]

        values = [
            pct(
                grade_a.get(
                    "percentage",
                    0,
                )
            ),
            pct(
                urs.get(
                    "percentage",
                    0,
                )
            ),
            pct(
                defect.get(
                    "percentage",
                    0,
                )
            ),
        ]

        _is_dark_chart = st.session_state.get("theme_mode", "light") == "dark"
        _chart_fc = "#F0EDE8" if _is_dark_chart else "#1C1009"
        _chart_gc = "rgba(255,255,255,0.08)" if _is_dark_chart else "#E8DDD0"
        # Use theme-aware grade colors
        _sage_col   = "#4ADE80" if _is_dark_chart else THEME["sage"]
        _copper_col = "#E8A84F" if _is_dark_chart else THEME["copper"]
        _rust_col   = "#F87171" if _is_dark_chart else THEME["rust"]

        fig = go.Figure(
            go.Bar(
                x=labels,
                y=values,
                marker_color=[_sage_col, _copper_col, _rust_col],
                text=[f"{value:.1f}%" for value in values],
                textposition="outside",
                textfont=dict(color=_chart_fc, size=12),
                marker_line_width=0,
                width=0.5,
            )
        )

        fig.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="% of validated onions",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, IBM Plex Sans, sans-serif", color=_chart_fc),
            yaxis=dict(
                gridcolor=_chart_gc,
                range=[0, max(100, max(values, default=0) + 10)],
            ),
            xaxis=dict(gridcolor=_chart_gc),
        )

        st.plotly_chart(
            fig,
            config={
                "displayModeBar": False
            },
            use_container_width=True,
        )

    # ---------------------------------------------------------------
    # Overall assessment
    # ---------------------------------------------------------------

    md("<br>")

    render_overall_quality(
        data
    )

    # ---------------------------------------------------------------
    # Defect analysis
    # ---------------------------------------------------------------

    md("<br>")

    render_defect_analysis(
        data
    )

    # ---------------------------------------------------------------
    # NIR screening (only rendered when mode=nir and data is present)
    # ---------------------------------------------------------------

    if data.get("nir_results"):
        md("<br>")
        render_nir_results(data)

    # ---------------------------------------------------------------
    # Per-onion table  +  Manual Override
    # ---------------------------------------------------------------

    md("<br>")
    st.markdown(f"### {T('per_onion_heading')}")
    st.caption(T("per_onion_caption"))

    grade_result = data.get("grade_result", {}) or {}
    grades_by_id = {
        item.get("onion_id"): item.get("grade")
        for item in grade_result.get("per_onion_grades", [])
    }

    _GRADE_OPTIONS = ["(AI decision)", "Grade A", "URS", "Defect"]

    display_rows = []
    for fallback_index, onion in enumerate(onions, start=1):
        onion_id       = onion.get("onion_id", "")
        display_number = get_onion_number(onion_id, fallback_index)
        confidence     = pct(onion.get("confidence", 0))
        grade          = grades_by_id.get(onion_id) or onion.get("final_grade", "-")

        severity = onion.get("severity") or (
            "Major" if grade == "Defect" else ("Minor" if grade == "URS" else "None")
        )
        defect_type = onion.get("defect_type") or "None"

        display_rows.append({
            T("image_num_col"):    display_number,
            T("onion_id_col"):     onion_id,
            T("size_col"):         safe_text(onion.get("size", "")),
            T("colour_col"):       safe_text(onion.get("color", "")),
            T("severity_col"):     severity,
            T("defect_col"):       defect_type,
            T("defect_desc_col"):  safe_text(onion.get("defect_description"), "None"),
            T("confidence_col"):   f"{confidence * 100:.1f}%",
            T("grade_reason_col"): safe_text(onion.get("grade_reason"), "—"),
            T("final_grade_col"):  grade,
            T("override_col"):     "(AI decision)",   # editable column
        })

    if display_rows:
        import pandas as pd
        st.caption(T("override_note"))

        # Editable dataframe — only the Override column is editable
        edited_df = st.data_editor(
            pd.DataFrame(display_rows),
            hide_index=True,
            use_container_width=True,
            column_config={
                T("image_num_col"):    st.column_config.NumberColumn(
                    T("image_num_col"), help="Matches marker on image.", width="small"),
                T("confidence_col"):   st.column_config.TextColumn(
                    T("confidence_col"), width="small"),
                T("grade_reason_col"): st.column_config.TextColumn(
                    T("grade_reason_col"), width="large"),
                T("final_grade_col"):  st.column_config.TextColumn(T("final_grade_col")),
                T("override_col"):     st.column_config.SelectboxColumn(
                    T("override_col"),
                    options=_GRADE_OPTIONS,
                    help="Select a grade to override the AI decision. "
                         "Choose '(AI decision)' to keep the original.",
                    width="medium",
                ),
            },
            disabled=[c for c in list(display_rows[0].keys()) if c != T("override_col")],
        )

        # Recalculate summary if any override was applied
        _override_col = T("override_col")
        _final_col    = T("final_grade_col")
        _overridden_ids = []

        if edited_df is not None and _override_col in edited_df.columns:
            for _, row in edited_df.iterrows():
                override_val = str(row[_override_col]) if row[_override_col] is not None else ""
                if override_val and override_val != "(AI decision)":
                    oid = str(row.get(T("onion_id_col"), ""))
                    _overridden_ids.append(oid)
                    # Patch the onion record in session so PDF/CSV pick it up
                    for o in st.session_state["result"].get("onions", []):
                        if o.get("onion_id") == oid:
                            o["final_grade"] = override_val
                            o["grade_reason"] = f"Manually overridden to {override_val}"

            if _overridden_ids:
                # Recompute summary from patched onions
                _patched = st.session_state["result"].get("onions", [])
                _pt = len(_patched)
                _pga  = sum(1 for o in _patched if o.get("final_grade") == "Grade A")
                _purs = sum(1 for o in _patched if o.get("final_grade") == "URS")
                _pdef = _pt - _pga - _purs
                _pga_pct  = round(_pga / _pt * 100, 1) if _pt else 0.0
                _purs_pct = round(_purs / _pt * 100, 1) if _pt else 0.0
                _pdef_pct = round(100.0 - _pga_pct - _purs_pct, 1) if _pt else 0.0

                st.session_state["result"]["summary"] = {
                    "grade_a": {"count": _pga,  "percentage": _pga_pct},
                    "urs":     {"count": _purs, "percentage": _purs_pct},
                    "defect":  {"count": _pdef, "percentage": _pdef_pct},
                }

                st.info(
                    f"✏️ {len(_overridden_ids)} grade(s) manually overridden: "
                    + ", ".join(_overridden_ids)
                    + f" — Grade A: {_pga_pct}% | URS: {_purs_pct}% | Defect: {_pdef_pct}%"
                )

    else:
        st.info(T("no_validated_records"))

    # ---------------------------------------------------------------
    # Rule engine transparency expander
    # ---------------------------------------------------------------
    with st.expander(T("rule_engine_title")):
        md("""
        <div style="line-height:1.7;color:var(--ink);">
        <p style="color:var(--ink);">All grades are decided by <code>grading_rules.py</code> — a plain Python
        file you can read and modify. Gemini Vision only provides observations.
        Rules are applied in priority order (first match wins):</p>
        <table class="og-table">
          <thead><tr>
            <th>Condition observed</th>
            <th>Grade assigned</th>
            <th>Standard reference</th>
          </tr></thead>
          <tbody>
          <tr>
            <td>Sprouting detected</td>
            <td style="font-weight:700;color:var(--rust);">Defect</td>
            <td style="color:var(--ink-soft);">AGMARK Grade I — sprouted onions excluded</td>
          </tr>
          <tr>
            <td>Defect description contains: <em>rot, decay, mould, mold, major</em></td>
            <td style="font-weight:700;color:var(--rust);">Defect</td>
            <td style="color:var(--ink-soft);">AGMARK Grade I/II — rotten/mouldy onions excluded</td>
          </tr>
          <tr>
            <td>Defect present, minor blemish (no rot/sprouting)</td>
            <td style="font-weight:700;color:var(--copper-deep);">URS</td>
            <td style="color:var(--ink-soft);">AGMARK Grade II equivalent (sub-premium, pending calibration)</td>
          </tr>
          <tr>
            <td>Size = Small</td>
            <td style="font-weight:700;color:var(--copper-deep);">URS</td>
            <td style="color:var(--ink-soft);">AGMARK Grade I: diameter ≥ 45 mm (to be validated in v1.1)</td>
          </tr>
          <tr>
            <td>Medium or Large, no sprouting, no defect</td>
            <td style="font-weight:700;color:var(--sage);">Grade A</td>
            <td style="color:var(--ink-soft);">AGMARK Grade I surface criteria (surface grading only)</td>
          </tr>
          </tbody>
        </table>
        <p style="margin-top:10px;font-size:12px;color:var(--ink-soft);">
          <em>Internal defects (internal rot, double-bulb) are outside the scope of camera-based
          inspection. OnionGrade AI is a digital pre-check — not a replacement for AGMARK
          certification by a licensed inspector.</em>
        </p>
        </div>
        """)
    # ---------------------------------------------------------------
    # Flagged records
    # ---------------------------------------------------------------
    
    flagged = data.get(
        "flagged",
        [],
    )
    if flagged:

        st.caption(T("flagged_caption"))

        with st.expander(
            f"⚠️ {len(flagged)} record(s) {T('flagged_title')}"
        ):

            md(
                "".join(
                    flagged_record_card(record)
                    for record in flagged
                )
            )

    # ---------------------------------------------------------------
    # Per-onion crop thumbnails (Feature 2)
    # ---------------------------------------------------------------

    if onions:
        _orig_img_for_thumbs = st.session_state.get("image")
        if _orig_img_for_thumbs:
            md("<br>")
            with st.expander("🔍 Per-onion crop thumbnails", expanded=False):
                st.caption(
                    "Each tile shows the cropped region for one onion — bounding box from AI. "
                    "Grade colour: 🟢 Grade A · 🟡 URS · 🔴 Defect."
                )
                _thumb_grade_colors = {
                    "Grade A": "rgba(45,122,79,0.15)",
                    "URS":     "rgba(181,101,29,0.15)",
                    "Defect":  "rgba(162,62,46,0.15)",
                }
                _thumb_grade_borders = {
                    "Grade A": "#2D7A4F",
                    "URS":     "#B5651D",
                    "Defect":  "#A23E2E",
                }
                # Render in rows of 6
                _thumb_cols_per_row = 6
                _thumb_rows = [onions[i:i+_thumb_cols_per_row] for i in range(0, len(onions), _thumb_cols_per_row)]
                for _trow in _thumb_rows:
                    _tcols = st.columns(len(_trow))
                    for _tc, _tonion in zip(_tcols, _trow):
                        with _tc:
                            _tbbox = _tonion.get("bbox") if isinstance(_tonion.get("bbox"), dict) else None
                            _tthumb = crop_onion_thumbnail(_orig_img_for_thumbs, _tbbox, thumb_size=120)
                            _tgrade = _tonion.get("final_grade", "-")
                            _tconf  = float(_tonion.get("confidence") or 0.0)
                            _toid   = str(_tonion.get("onion_id") or "")
                            _tsrc   = str(_tonion.get("source_image") or "")
                            _tbg    = _thumb_grade_colors.get(_tgrade, "var(--surface-soft)")
                            _tborder= _thumb_grade_borders.get(_tgrade, "var(--line)")
                            _treview= str(_tonion.get("review_status") or "")
                            if _tthumb:
                                st.image(_tthumb, use_column_width=True)
                            else:
                                md(f'<div style="height:80px;background:var(--surface-soft);border-radius:8px;display:flex;align-items:center;justify-content:center;color:var(--ink-soft);font-size:11px;">No bbox</div>')
                            _badge_color = _thumb_grade_borders.get(_tgrade, "#888")
                            _review_badge = ""
                            if _treview == "Low Confidence":
                                _review_badge = '<span style="font-size:9px;color:orange;">⚠ Low conf</span>'
                            md(f"""
                            <div style="font-size:10px;text-align:center;padding:3px 2px 0;
                                        border-top:2px solid {_tborder};margin-top:2px;">
                              <span style="font-weight:700;color:var(--ink);">{escape(_toid)}</span><br>
                              <span style="font-weight:600;color:{_tborder};">{escape(_tgrade)}</span>
                              &nbsp;<span style="color:var(--ink-soft);">{_tconf*100:.0f}%</span>
                              {_review_badge}
                            </div>
                            """)

    # ---------------------------------------------------------------
    # Per-onion confidence scatter chart
    # ---------------------------------------------------------------

    if onions:
        md("<br>")
        st.markdown(f"### {T('confidence_heading')}")
        st.caption(T("confidence_caption"))

        conf_ids = [o.get("onion_id", f"#{i+1}") for i, o in enumerate(onions)]
        conf_vals = [pct(o.get("confidence", 0)) * 100 for o in onions]
        conf_grades = [o.get("final_grade", "-") for o in onions]

        _is_dark_conf = st.session_state.get("theme_mode", "light") == "dark"
        _conf_fc   = "#F0EDE8" if _is_dark_conf else "#1C1009"
        _conf_gc   = "rgba(255,255,255,0.08)" if _is_dark_conf else "#E8DDD0"
        _conf_sage   = "#4ADE80" if _is_dark_conf else THEME["sage"]
        _conf_copper = "#E8A84F" if _is_dark_conf else THEME["copper"]
        _conf_rust   = "#F87171" if _is_dark_conf else THEME["rust"]
        _grade_colors = {
            "Grade A": _conf_sage,
            "URS":     _conf_copper,
            "Defect":  _conf_rust,
        }
        conf_marker_colors = [_grade_colors.get(g, "#9DAAB8") for g in conf_grades]

        conf_fig = go.Figure()
        conf_fig.add_trace(go.Scatter(
            x=conf_ids,
            y=conf_vals,
            mode="markers+text",
            marker=dict(
                color=conf_marker_colors,
                size=16,
                line=dict(width=2, color="rgba(255,255,255,0.70)"),
            ),
            text=[f"{v:.0f}%" for v in conf_vals],
            textposition="top center",
            textfont=dict(size=10, family="Inter, IBM Plex Sans", color=_conf_fc),
            hovertemplate="%{x}: %{y:.1f}% confidence<extra></extra>",
        ))
        conf_fig.add_hline(
            y=50,
            line_dash="dash",
            line_color=_conf_rust,
            annotation_text="Validation floor 50%",
            annotation_position="bottom right",
            annotation_font_size=10,
            annotation_font_color=_conf_rust,
        )
        conf_fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=30),
            yaxis=dict(title="Confidence (%)", range=[0, 110], gridcolor=_conf_gc),
            xaxis=dict(gridcolor=_conf_gc, tickangle=-35),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, IBM Plex Sans, sans-serif", color=_conf_fc),
            showlegend=False,
        )
        st.plotly_chart(conf_fig, config={"displayModeBar": False}, use_container_width=True)

    # ---------------------------------------------------------------
    # Digital report
    # ---------------------------------------------------------------

    md("<br>")

    report_text = get_report_text(
        data
    )

    md(
        f"""
        <div class="og-report og-fade-in">

          <div style="
              display:flex;
              align-items:center;
              gap:12px;
              margin-bottom:18px;
          ">
            <div style="
                width:40px; height:40px; border-radius:10px;
                background:linear-gradient(135deg,var(--plum),var(--copper));
                color:#fff; display:flex; align-items:center;
                justify-content:center; font-size:18px; flex-shrink:0;
            ">📄</div>
            <div style="flex:1; display:flex; align-items:center;
                        justify-content:space-between; flex-wrap:wrap; gap:8px;">
              <div class="og-report-heading">{T("report_heading")}</div>
              <span style="
                  font-size:10px; font-weight:700; letter-spacing:.05em;
                  color:var(--copper-deep); background:var(--rust-soft);
                  padding:3px 10px; border-radius:999px; text-transform:uppercase;
              ">LangChain · AI Generated</span>
            </div>
          </div>

          <div style="
              height:2px;
              background:linear-gradient(90deg,var(--plum),var(--copper),transparent);
              border-radius:2px;
              margin-bottom:20px;
              opacity:0.4;
          "></div>

          <div class="og-report-body">{escape(report_text)}</div>

        </div>
        """
    )

    # ---------------------------------------------------------------
    # Export section — premium cards + download buttons
    # ---------------------------------------------------------------

    md("<br>")

    md(f"""
    <div style="margin-bottom:6px;">
      <div style="font-family:'Fraunces',Georgia,serif;font-size:{_FS['h2']};font-weight:700;
                  color:var(--plum-deep);margin-bottom:4px;">{T('export_heading')}</div>
      <div style="font-size:12px;color:var(--ink-soft);">Download your grading results in your preferred format.</div>
    </div>
    """)

    csv_bytes = build_csv_bytes(data)
    json_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")

    # PDF uses the numbered image so the report matches the dashboard.
    marked_image_for_pdf = st.session_state.get("marked_image")
    if not marked_image_for_pdf:
        marked_image_for_pdf = st.session_state.get("image")
    pdf_bytes = build_pdf_bytes(data, marked_image_for_pdf)
    xlsx_bytes = build_xlsx_bytes(data)

    # Premium export cards with description + download button beneath
    export_col1, export_col2, export_col3, export_col4 = st.columns(4, gap="medium")

    with export_col1:
        md(f"""
        <div class="og-export-card og-fade-in">
          <div class="og-export-icon" style="background:var(--sage-soft);">📊</div>
          <div style="flex:1;">
            <div style="font-weight:700;font-size:14px;color:var(--ink);">Export CSV</div>
            <div style="font-size:11.5px;color:var(--ink-soft);margin-top:2px;">Comma Separated Values · Per-onion records</div>
          </div>
        </div>
        """)
        st.download_button(
            T("export_csv"),
            data=csv_bytes,
            file_name=f"{batch_id}_onions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with export_col2:
        md(f"""
        <div class="og-export-card og-fade-in">
          <div class="og-export-icon" style="background:var(--nav-active-bg);">📋</div>
          <div style="flex:1;">
            <div style="font-weight:700;font-size:14px;color:var(--ink);">Export JSON</div>
            <div style="font-size:11.5px;color:var(--ink-soft);margin-top:2px;">Full structured report · All fields included</div>
          </div>
        </div>
        """)
        st.download_button(
            T("export_json"),
            data=json_bytes,
            file_name=f"{batch_id}_report.json",
            mime="application/json",
            use_container_width=True,
        )

    with export_col3:
        md(f"""
        <div class="og-export-card og-fade-in">
          <div class="og-export-icon" style="background:var(--rust-soft);">📄</div>
          <div style="flex:1;">
            <div style="font-weight:700;font-size:14px;color:var(--ink);">Export PDF</div>
            <div style="font-size:11.5px;color:var(--ink-soft);margin-top:2px;">Printable quality report · Batch certified</div>
          </div>
        </div>
        """)
        st.download_button(
            T("export_pdf"),
            data=pdf_bytes,
            file_name=f"{batch_id}_quality_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with export_col4:
        md(f"""
        <div class="og-export-card og-fade-in">
          <div class="og-export-icon" style="background:rgba(0,180,100,0.12);">📗</div>
          <div style="flex:1;">
            <div style="font-weight:700;font-size:14px;color:var(--ink);">Export XLSX</div>
            <div style="font-size:11.5px;color:var(--ink-soft);margin-top:2px;">Excel workbook · Color-coded grades</div>
          </div>
        </div>
        """)
        if xlsx_bytes:
            st.download_button(
                "⬇️ Export XLSX",
                data=xlsx_bytes,
                file_name=f"{batch_id}_onions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.button(
                "XLSX unavailable",
                disabled=True,
                use_container_width=True,
                help="Install openpyxl to enable XLSX export.",
            )

    # ---------------------------------------------------------------
    # CV Summary — structured analysis panel
    # ---------------------------------------------------------------

    md("<br>")
    st.markdown(f"### 🗂 {T('cv_mode_heading')}")

    _cv_onions   = onions  # validated onion list
    _cv_summary  = get_summary(data)
    _cv_quality  = get_overall_quality(data)
    _cv_flagged  = data.get("flagged", [])
    _cv_batch_id = get_batch_id(data)
    _cv_total    = int(_cv_summary.get("grade_a", {}).get("count", 0)) \
                 + int(_cv_summary.get("urs", {}).get("count", 0)) \
                 + int(_cv_summary.get("defect", {}).get("count", 0))

    # AI confidence stats from onion records
    _conf_vals = [pct(o.get("confidence", 0)) * 100 for o in _cv_onions]
    _avg_conf  = round(sum(_conf_vals) / len(_conf_vals), 1) if _conf_vals else 0.0
    _max_conf_onion = max(_cv_onions, key=lambda o: pct(o.get("confidence", 0)), default={})
    _min_conf_onion = min(_cv_onions, key=lambda o: pct(o.get("confidence", 0)), default={})

    # Top defect types
    from collections import Counter as _Counter
    _defect_types = [
        o.get("defect_type") or "None"
        for o in _cv_onions
        if o.get("defect_present") or o.get("final_grade") in ("URS", "Defect")
    ]
    _top_defects = _Counter(_defect_types).most_common(4)

    # Grade badge helper
    def _grade_badge(grade: str) -> str:
        if grade == "Grade A":
            return f'<span class="og-cv-badge" style="background:var(--sage-soft);color:var(--sage);">Grade A</span>'
        if grade == "URS":
            return f'<span class="og-cv-badge" style="background:rgba(200,144,42,0.12);color:var(--copper-deep);">URS</span>'
        if grade == "Defect":
            return f'<span class="og-cv-badge" style="background:var(--rust-soft);color:var(--rust);">Defect</span>'
        return f'<span class="og-cv-badge" style="background:var(--surface-soft);color:var(--ink-soft);">{escape(str(grade))}</span>'

    _status_val  = str(_cv_quality.get("status", "—")).upper()
    _status_desc = _cv_quality.get("description", "—")
    _recommendation = _cv_quality.get("recommendation", "—")

    _ga_count  = _cv_summary.get("grade_a", {}).get("count", 0)
    _urs_count = _cv_summary.get("urs", {}).get("count", 0)
    _def_count = _cv_summary.get("defect", {}).get("count", 0)
    _ga_pct    = pct(_cv_summary.get("grade_a", {}).get("percentage", 0))
    _urs_pct   = pct(_cv_summary.get("urs", {}).get("percentage", 0))
    _def_pct   = pct(_cv_summary.get("defect", {}).get("percentage", 0))

    _cv_col1, _cv_col2 = st.columns(2, gap="large")

    with _cv_col1:
        md(f"""
        <div class="og-cv-card og-fade-in">
          <div class="og-cv-section-title">📦 {T('cv_batch_info')}</div>
          <table class="og-table">
            <tbody>
              <tr>
                <td style="font-weight:600;">Batch ID</td>
                <td><code>{escape(_cv_batch_id)}</code></td>
              </tr>
              <tr>
                <td style="font-weight:600;">{T('cv_total_onions')}</td>
                <td><strong>{_cv_total}</strong></td>
              </tr>
              <tr>
                <td style="font-weight:600;">{T('cv_flagged_count')}</td>
                <td><strong>{len(_cv_flagged)}</strong></td>
              </tr>
              <tr>
                <td style="font-weight:600;">{T('cv_batch_status')}</td>
                <td><strong style="color:var(--{'sage' if _status_val=='GOOD' else 'rust' if _status_val=='POOR' else 'copper'});">{escape(_status_val)}</strong></td>
              </tr>
            </tbody>
          </table>
          <div style="margin-top:14px;">
            <div class="og-cv-section-title">💬 {T('recommendation')}</div>
            <div style="color:var(--ink);line-height:1.6;">{escape(_recommendation)}</div>
          </div>
        </div>
        """)

    with _cv_col2:
        # Top defects list HTML
        _defect_html = "".join(
            f'<tr><td style="font-weight:600;">{escape(str(d))}</td><td style="text-align:right;">{c}</td></tr>'
            for d, c in _top_defects
        ) if _top_defects else "<tr><td colspan='2' style='color:var(--sage);'>✓ No defects</td></tr>"

        md(f"""
        <div class="og-cv-card og-fade-in">
          <div class="og-cv-section-title">📊 {T('cv_grade_breakdown')}</div>
          <table class="og-table">
            <thead><tr><th>Grade</th><th>Count</th><th>%</th></tr></thead>
            <tbody>
              <tr>
                <td>{_grade_badge("Grade A")}</td>
                <td style="text-align:center;font-weight:700;">{_ga_count}</td>
                <td style="text-align:center;font-weight:700;color:var(--sage);">{_ga_pct:.1f}%</td>
              </tr>
              <tr>
                <td>{_grade_badge("URS")}</td>
                <td style="text-align:center;font-weight:700;">{_urs_count}</td>
                <td style="text-align:center;font-weight:700;color:var(--copper-deep);">{_urs_pct:.1f}%</td>
              </tr>
              <tr>
                <td>{_grade_badge("Defect")}</td>
                <td style="text-align:center;font-weight:700;">{_def_count}</td>
                <td style="text-align:center;font-weight:700;color:var(--rust);">{_def_pct:.1f}%</td>
              </tr>
            </tbody>
          </table>

          <div style="margin-top:16px;">
            <div class="og-cv-section-title">🔬 {T('cv_top_defects')}</div>
            <table class="og-table">
              <tbody>{_defect_html}</tbody>
            </table>
          </div>
        </div>
        """)

    # AI confidence row
    md(f"""
    <div class="og-cv-card og-fade-in" style="margin-top:14px;">
      <div class="og-cv-section-title">🤖 {T('cv_ai_confidence')}</div>
      <div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;">
        <div style="flex:1;min-width:160px;">
          <div style="font-size:32px;font-weight:700;color:var(--plum-deep);font-family:'Fraunces',serif;">{_avg_conf:.1f}%</div>
          <div style="color:var(--ink-soft);font-size:12px;margin-top:2px;">{T('cv_avg_confidence')}</div>
        </div>
        <div style="flex:2;min-width:200px;">
          <table class="og-table">
            <tbody>
              <tr>
                <td style="color:var(--ink-soft);">{T('cv_highest_conf')}</td>
                <td style="font-weight:700;">
                  <code>{escape(str(_max_conf_onion.get('onion_id','-')))}</code>
                  &nbsp;{pct(_max_conf_onion.get('confidence',0))*100:.1f}%
                </td>
              </tr>
              <tr>
                <td style="color:var(--ink-soft);">{T('cv_lowest_conf')}</td>
                <td style="font-weight:700;">
                  <code>{escape(str(_min_conf_onion.get('onion_id','-')))}</code>
                  &nbsp;{pct(_min_conf_onion.get('confidence',0))*100:.1f}%
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      <div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--line);
                  font-size:12px;color:var(--ink-soft);">
        💡 {T('cv_export_hint')}
      </div>
    </div>
    """)


# ---------------------------------------------------------------------------
# Previous Scans — SQLite history (Feature 1)
# Loaded outside the "result" block so it always shows at the bottom.
# ---------------------------------------------------------------------------

md("<br>")
st.markdown("---")
st.markdown("## 🗄️ Previous Scans (Persistent History)")
st.caption(
    "Scans are stored in a local SQLite database. "
    "Reloading a scan does **not** call Gemini — it replays the saved result."
)

try:
    _history_resp = requests.get(f"{BACKEND_URL}/history?limit=20", timeout=5)
    _db_scans = _history_resp.json().get("scans", []) if _history_resp.status_code == 200 else []
except Exception:
    _db_scans = []

if not _db_scans:
    st.info("No previous scans found. Run an analysis to save results.", icon="📂")
else:
    # Build a summary table
    import pandas as _pd_hist
    _scan_rows = []
    for _sc in _db_scans:
        _scan_rows.append({
            "ID":           _sc.get("id", ""),
            "Batch ID":     _sc.get("batch_id", ""),
            "Date/Time":    str(_sc.get("created_at", ""))[:19].replace("T", " "),
            "Mode":         str(_sc.get("analysis_mode", "")).upper(),
            "Onions":       _sc.get("total_validated", 0),
            "Grade A %":    f"{_sc.get('grade_a_pct', 0):.1f}%",
            "URS %":        f"{_sc.get('urs_pct', 0):.1f}%",
            "Defect %":     f"{_sc.get('defect_pct', 0):.1f}%",
            "Status":       _sc.get("overall_status", ""),
            "Demo":         "Yes" if _sc.get("demo_mode") else "No",
        })
    _hist_df = _pd_hist.DataFrame(_scan_rows)
    st.dataframe(_hist_df, use_container_width=True, hide_index=True)

    _selected_scan_id = st.selectbox(
        "Select a scan to reload (no Gemini call)",
        options=[s.get("id") for s in _db_scans],
        format_func=lambda sid: next(
            (f"#{s['id']} — {s['batch_id']} ({str(s.get('created_at',''))[:10]})"
             for s in _db_scans if s["id"] == sid), str(sid)
        ),
        key="history_scan_selector",
    )

    _hist_col_load, _hist_col_del = st.columns([3, 1])
    with _hist_col_load:
        if st.button("📂 Reload selected scan (no Gemini call)", use_container_width=True, type="primary"):
            try:
                _reload_resp = requests.get(f"{BACKEND_URL}/history/{_selected_scan_id}", timeout=5)
                if _reload_resp.status_code == 200:
                    _reloaded = _reload_resp.json().get("result", {})
                    if _reloaded:
                        st.session_state["result"] = _reloaded
                        st.session_state.pop("image", None)
                        st.session_state.pop("marked_image", None)
                        st.success(f"Scan #{_selected_scan_id} reloaded from history (no Gemini call).", icon="✅")
                        st.rerun()
                    else:
                        st.error("Reloaded scan is empty — record may be corrupted.")
                else:
                    st.error(f"Could not load scan #{_selected_scan_id}: {_reload_resp.text}")
            except Exception as _ex:
                st.error(f"Error reloading scan: {_ex}")
    with _hist_col_del:
        if st.button("🗑️ Delete scan", use_container_width=True):
            try:
                _del_resp = requests.delete(f"{BACKEND_URL}/history/{_selected_scan_id}", timeout=5)
                if _del_resp.status_code == 200:
                    st.success(f"Scan #{_selected_scan_id} deleted.", icon="🗑️")
                    st.rerun()
                else:
                    st.error(f"Delete failed: {_del_resp.text}")
            except Exception as _ex:
                st.error(f"Error deleting scan: {_ex}")
