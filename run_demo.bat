@echo off
cd /d "%~dp0"
echo ===================================================
echo   PRISM-Phish: Starting Interactive Demo
echo ===================================================
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" demo.py
) else (
    python demo.py
)
pause
