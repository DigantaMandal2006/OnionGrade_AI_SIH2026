@echo off
echo Starting OnionGrade AI (backend + frontend)...
start "OnionGrade Backend"  cmd /k "%~dp0backend\start_backend.bat"
timeout /t 3 /nobreak >nul
start "OnionGrade Frontend" cmd /k "%~dp0frontend\start_frontend.bat"
echo Dono servers alag windows me start ho rahe hain.
echo Backend:  http://localhost:8000/docs
echo Frontend: http://localhost:8501
