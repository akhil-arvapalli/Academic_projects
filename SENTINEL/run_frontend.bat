@echo off
setlocal

set "ROOT=%~dp0"
pushd "%ROOT%frontend" >nul

echo Starting SENTINEL_v3 frontend...
npm run dev

set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo Frontend exited with code %EXITCODE%.
    pause
)

popd >nul
endlocal
