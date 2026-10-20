@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo CET6 RAG - Step 3: Ask DeepSeek
echo ============================================================
echo.

set "PY=%~dp0.venv\Scripts\python.exe"

"%PY%" step3_ask.py

echo.
if errorlevel 1 (
    echo [FAILED] Please take a screenshot of this window.
) else (
    echo [DONE]
)
echo.
pause
