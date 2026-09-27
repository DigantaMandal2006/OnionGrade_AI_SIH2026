@echo off
cd /d %~dp0

where py >nul 2>nul
if errorlevel 1 (
    echo [Error] Python launcher "py" not found. Install Python 3.12 from:
    echo https://www.python.org/downloads/release/python-3120/
    pause
    exit /b 1
)

py -3.12 -c "exit()" >nul 2>nul
if errorlevel 1 (
    echo [Error] Python 3.12 is not installed on this PC.
    echo Install it from: https://www.python.org/downloads/release/python-3120/
    echo ^(tick "Add python.exe to PATH" during install^)
    pause
    exit /b 1
)

if not exist venv (
    echo [Setup] Creating virtual environment with Python 3.12...
    py -3.12 -m venv venv
)

call venv\Scripts\activate.bat

if not exist venv\installed.flag (
    echo [Setup] Installing backend dependencies, please wait...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [Error] Dependency install failed. See the error above.
        pause
        exit /b 1
    )
    echo done > venv\installed.flag
)

if not exist .env (
    copy .env.example .env >nul
)

echo Starting backend on http://localhost:8000
uvicorn app.main:app
pause
