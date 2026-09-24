@echo off
cd /d "%~dp0"
echo ===================================================
echo   PRISM-Phish: Starting Web Dashboard
echo   Open your browser at: http://localhost:8000/
echo ===================================================
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
) else (
    python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
)
pause
