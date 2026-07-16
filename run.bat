@echo off
REM Nara Images - double-click this to start, then the browser opens by itself.
cd /d "%~dp0backend"

python -c "import fastapi, uvicorn, cv2, numpy" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    python -m pip install -r "%~dp0requirements.txt"
)

echo.
echo   Nara Images  -  http://127.0.0.1:8000
echo   Press Ctrl+C to stop.
echo.

start "" http://127.0.0.1:8000
python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
