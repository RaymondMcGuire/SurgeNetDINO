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
if errorlevel 1 (
    echo ERROR: uv was not found. Run scripts\init.bat cu128 first.
    exit /b 1
)
if not exist uv.lock exit /b 1
set "FEATURES=%~1"
if "%FEATURES%"=="" set "FEATURES=H:\dataset\JSCAS_experiments\frozen_dinov2\20260920_122914_522\cache\features.pt"
set "PROFILE=%~2"
if "%PROFILE%"=="" set "PROFILE=low"
if "%PROFILE%"=="low" goto profile_ok
if "%PROFILE%"=="medium" goto profile_ok
if "%PROFILE%"=="high" goto profile_ok
echo ERROR: PROFILE must be low, medium or high.
exit /b 1

:profile_ok
set "RUN_DIR=%~3"
if not "%RUN_DIR%"=="" goto run_dir_ok
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss_fff"') do set "RUN_ID=%%I"
if not defined RUN_ID exit /b 1
set "RUN_DIR=H:\dataset\JSCAS_experiments\frozen_dinov2\head_regularized_%RUN_ID%"

:run_dir_ok
if not exist "%FEATURES%" (
    echo ERROR: Feature cache not found: "%FEATURES%"
    echo Pass an existing features.pt as the first argument, or run scripts\train.bat to create one.
    exit /b 1
)
if exist "%RUN_DIR%" (
    echo ERROR: Output already exists. Choose a NEW run directory.
    exit /b 1
)
echo Reusing frozen features: "%FEATURES%"
echo Output: "%RUN_DIR%"
echo Original images and manifests will be verified. DINO features will NOT be re-extracted.
uv run --locked --extra cu128 python -m surgenet_dino train-head --features "%FEATURES%" --output-dir "%RUN_DIR%" --gpu-profile "%PROFILE%"
exit /b %errorlevel%
