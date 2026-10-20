@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo CET6 RAG - Step 2: BGE + ChromaDB
echo ============================================================
echo.

set "PY=%~dp0.venv\Scripts\python.exe"
set "HF_ENDPOINT=https://hf-mirror.com"
set "PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple"

"%PY%" -c "import chromadb, fastembed" >nul 2>nul
if errorlevel 1 (
    echo Installing step-2 packages. This may take a few minutes...
    "%PY%" -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple >nul 2>nul
    "%PY%" -m pip install --no-cache-dir --timeout 30 --retries 1 --use-deprecated=legacy-resolver chromadb==1.5.9 fastembed==0.9.0
    if errorlevel 1 (
        echo.
        echo [ERROR] Package installation failed.
        echo Please take a screenshot of this window.
        pause
        exit /b 1
    )
)

echo.
echo Building the vector index...
echo.
"%PY%" step2_build_index.py

echo.
if errorlevel 1 (
    echo [FAILED] Please take a screenshot of this window.
) else (
    echo [SUCCESS] ChromaDB is ready:
    echo %~dp0data\chroma_db
)
echo.
pause
