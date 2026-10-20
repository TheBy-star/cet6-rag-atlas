@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo CET6 RAG - Step 4: Streamlit UI
echo ============================================================
echo.

set "PY=%~dp0.venv\Scripts\python.exe"
set "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"

"%PY%" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo Installing Streamlit. This may take a few minutes...
    "%PY%" -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple >nul 2>nul
    "%PY%" -m pip install --no-cache-dir --timeout 30 --retries 1 --use-deprecated=legacy-resolver streamlit==1.66.0
    if errorlevel 1 (
        echo.
        echo [ERROR] Streamlit installation failed.
        echo Please take a screenshot of this window.
        pause
        exit /b 1
    )
)

echo Starting the web app...
echo.

start "CET6 RAG Server" /min "%PY%" -m streamlit run app.py --server.port 8501
timeout /t 6 /nobreak >nul
start "" http://localhost:8501

echo Browser should open automatically.
echo If it does not, open: http://localhost:8501
echo.
echo To stop the app, close the minimized "CET6 RAG Server" window.
echo.
pause
