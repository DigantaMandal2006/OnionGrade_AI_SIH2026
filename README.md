# OnionGrade AI — real-stack build

Team NeuroNex · SIH26031 · Quality assessment and grading of onion

This is the actual technology stack from the idea slide, not a mockup:

- **Frontend:** Streamlit
- **Backend/API:** FastAPI
- **AI & LLM:** Google Gemini Vision API (perception) + LangChain (report generation)
- **Data layer:** Pydantic (validation) + Python rule engine (grading logic)
- **Visualization:** Plotly
- **Deployment:** Uvicorn + Docker

## How the pipeline works (matches the System Flow slide)

```
Streamlit upload
   -> Gemini Vision (analyser) perceives size / colour / sprouting / defects
   -> Pydantic validates the structured per-onion JSON, flags incomplete records
   -> Deterministic Python rules (core/grading_rules.py) classify each onion
   -> core/calculations.py computes Grade A % / URS % / Defect % — never the AI
   -> LangChain turns those fixed numbers into a readable report
   -> Result shown on the Streamlit dashboard
```

Gemini only ever returns *observations*. Grades and percentages are always
computed by plain Python, so every number on the dashboard can be traced
back to a per-onion record — this is the "AI perceives, Python decides"
principle from the slide.

## Demo mode (no API key needed)

If `GOOGLE_API_KEY` isn't set, the backend automatically runs in **DEMO_MODE**:
`vision_service.py` returns a fixed, realistic mock perception (same JSON shape
Gemini would return, including one deliberately low-confidence/occluded onion
that gets flagged by validation) and `report_service.py` falls back to a
template report instead of calling an LLM. This lets you run and demo the
entire pipeline — upload, validate, grade, report — without any key.

Add a real `GOOGLE_API_KEY` to switch to live Gemini Vision + LangChain calls.

## Run locally

**1. Backend**
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # optionally add GOOGLE_API_KEY
uvicorn app.main:app --reload
```
Backend runs at `http://localhost:8000` (docs at `/docs`).

**2. Frontend** (separate terminal)
```bash
cd frontend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Opens the dashboard at `http://localhost:8501`.

## Run with Docker (backend)

```bash
cd backend
docker build -t oniongrade-backend .
docker run -p 8000:8000 --env-file .env oniongrade-backend
```

## Project structure

```
oniongrade-ai/
  backend/
    app/
      main.py                 FastAPI app, /analyze endpoint, wires the pipeline
      core/
        schemas.py             Pydantic models — what Gemini is allowed to perceive
        vision_service.py      Gemini Vision call (+ DEMO_MODE mock)
        grading_rules.py       Deterministic per-onion classification
        calculations.py        Grade A % / URS % / Defect % computation
        report_service.py      LangChain report generation from fixed numbers
    requirements.txt
    Dockerfile
    .env.example
  frontend/
    streamlit_app.py           Upload, dashboard, charts, report, export
    requirements.txt
```
