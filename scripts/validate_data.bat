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
where uv >nul 2>&1
if errorlevel 1 exit /b 1
if not exist uv.lock exit /b 1
set "DATASET=%~1"
if "%DATASET%"=="" set "DATASET=H:\dataset\JSCAS_preprocessed\labeled_1fps_v1"
if not exist "%DATASET%\dataset.json" exit /b 1
set "REPORT=%~2"
if not "%REPORT%"=="" goto report_ready
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss_fff"') do set "RUN_ID=%%I"
if not defined RUN_ID exit /b 1
set "REPORT=%DATASET%\validation\%RUN_ID%"
:report_ready
echo Full image audit. Report: "%REPORT%"
uv run --locked --extra cu128 python -m surgenet_dino validate-data --dataset "%DATASET%" --output-dir "%REPORT%"
exit /b %errorlevel%
