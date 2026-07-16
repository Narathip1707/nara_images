@echo off
REM Nara Images - double-click this to start, then the browser opens by itself.
REM
REM NOTE: keep this file pure ASCII. cmd.exe cannot reliably parse a .bat
REM containing Thai text, and `chcp 65001` does not fix it.

setlocal

REM ---------------------------------------------------------------- find Python
REM Do NOT trust the first `python` on PATH. On machines with msys2/MinGW or the
REM Microsoft Store stub it resolves to an interpreter with no pip and no cv2:
REM   C:\msys64\ucrt64\bin\python.exe: No module named pip
REM So: probe candidates and take the first one that actually has pip.

set "PY="

REM 1) the Windows Python Launcher - always finds a real python.org install
py -3 -m pip --version >nul 2>&1
if not errorlevel 1 set "PY=py -3"

REM 2) plain `python`, but only if pip really works
if not defined PY (
    python -m pip --version >nul 2>&1
    if not errorlevel 1 set "PY=python"
)

REM 3) python3
if not defined PY (
    python3 -m pip --version >nul 2>&1
    if not errorlevel 1 set "PY=python3"
)

if not defined PY goto nopython

for /f "delims=" %%v in ('%PY% -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%v"
echo Using Python: %PYEXE%

REM ------------------------------------------------------------ install deps
%PY% -c "import fastapi, uvicorn, cv2, numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies, this may take a minute...
    %PY% -m pip install -q -r "%~dp0requirements.txt"
    if errorlevel 1 goto pipfailed
)

%PY% -c "import fastapi, uvicorn, cv2, numpy" >nul 2>&1
if errorlevel 1 goto pipfailed

REM ------------------------------------------------------------------- images
if not exist "%~dp0images\*.png" if not exist "%~dp0images\*.jpg" if not exist "%~dp0images\*.tif" (
    echo.
    echo   Note: no test images in images\ yet.
    echo   Run get_images.bat, or just drag your own image onto the web page.
)

REM ------------------------------------------------------------------- serve
echo.
echo   Nara Images  -  http://127.0.0.1:8000
echo   Press Ctrl+C to stop.
echo.

cd /d "%~dp0backend"
start "" http://127.0.0.1:8000
%PY% -m uvicorn app:app --host 127.0.0.1 --port 8000
goto end

REM ------------------------------------------------------------------ errors
:nopython
echo.
echo   ERROR: no usable Python found.
echo.
echo   A Python was maybe on your PATH, but it has no pip - that happens with
echo   the msys2/MinGW python and with the Microsoft Store stub.
echo.
echo   Install Python 3.10+ from https://www.python.org/downloads/
echo   and TICK "Add python.exe to PATH" in the installer.
echo   Then close this window, open a new one, and run run.bat again.
echo.
pause
exit /b 1

:pipfailed
echo.
echo   ERROR: could not install the dependencies.
echo.
echo   Try running this by hand to see the real error:
echo     %PY% -m pip install -r "%~dp0requirements.txt"
echo.
echo   If you are behind a proxy or offline, that is usually the cause.
echo.
pause
exit /b 1

:end
pause
