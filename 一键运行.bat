@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo CET6 RAG - Step 1
echo ============================================================
echo.

set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo Creating Python virtual environment...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo.
        echo [ERROR] Could not create .venv with Python 3.12.
        echo Please install Python 3.12 first.
        pause
        exit /b 1
    )
)

"%PY%" -c "import langchain_core, langchain_text_splitters" >nul 2>nul
if errorlevel 1 (
    echo Fixing the pip mirror address...
    set "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"
    "%PY%" -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple >nul 2>nul

    echo Installing step-1 packages. This may take a few minutes...
    "%PY%" -m pip install --no-cache-dir --timeout 20 --retries 1 --use-deprecated=legacy-resolver langchain-core==1.6.12 langchain-text-splitters==1.1.3
    if errorlevel 1 (
        echo.
        echo [ERROR] Package installation failed.
        echo Please check the network, then run this file again.
        pause
        exit /b 1
    )
)

echo.
echo Running step1_chunk.py ...
echo.
"%PY%" step1_chunk.py

echo.
if errorlevel 1 (
    echo [FAILED] Please take a screenshot of this window.
) else (
    echo [SUCCESS] Your chunks are in:
    echo %~dp0data\chunks\chunks.jsonl
)
echo.
pause
