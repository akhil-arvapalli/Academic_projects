@echo off
setlocal

set "ROOT=%~dp0"
set "APP_NAME=SENTINEL_v3"

echo Launching %APP_NAME% backend and frontend in separate terminals...
start "%APP_NAME% Backend"  cmd /k ""%ROOT%run_backend.bat""
start "%APP_NAME% Frontend" cmd /k ""%ROOT%run_frontend.bat""

endlocal
