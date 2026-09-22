@echo off
setlocal
call :main %*
set "EXIT_CODE=%errorlevel%"
echo.
echo Script finished with exit code %EXIT_CODE% (0 = success).
pause
exit /b %EXIT_CODE%

:main
cd /d "%~dp0.."
if not defined GSF827_BACKEND set "GSF827_BACKEND=cu128"
if "%GSF827_BACKEND%"=="cpu" goto backend_ok
if "%GSF827_BACKEND%"=="cu128" goto backend_ok
echo ERROR: GSF827_BACKEND must be cpu or cu128.
exit /b 2
:backend_ok
where uv >nul 2>&1
if errorlevel 1 exit /b 1
uv run --locked --extra %GSF827_BACKEND% python -m surgenet_dino.temporal %*
exit /b %errorlevel%
