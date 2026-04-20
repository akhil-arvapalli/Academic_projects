@echo off
setlocal

set "ROOT=%~dp0"
pushd "%ROOT%" >nul

echo Starting SENTINEL_v3 backend...

if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" app.py
) else (
    echo [WARN] venv Python not found. Falling back to system Python.
    python app.py
)

set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo Backend exited with code %EXITCODE%.
    pause
)

popd >nul
endlocal
