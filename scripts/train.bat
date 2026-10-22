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
set "PROFILE=%~1"
if "%PROFILE%"=="" set "PROFILE=low"
if "%PROFILE%"=="low" goto profile_ok
if "%PROFILE%"=="medium" goto profile_ok
if "%PROFILE%"=="high" goto profile_ok
echo Usage: train.bat [low^|medium^|high] [NEW_RUN_DIR] [SPLIT_JSON] [PROFILES_JSON]
exit /b 1

:profile_ok
set "RUN_DIR=%~2"
if not "%RUN_DIR%"=="" goto run_dir_ok
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss_fff"') do set "RUN_ID=%%I"
if not defined RUN_ID exit /b 1
set "RUN_DIR=H:\dataset\JSCAS_experiments\frozen_dinov2\%RUN_ID%"

:run_dir_ok
set "SPLIT_JSON=H:\dataset\JSCAS_experiments\splits\video_holdout_9_3_v1\split.json"
if not "%~3"=="" set "SPLIT_JSON=%~3"
if not exist "%SPLIT_JSON%" (
    echo ERROR: Split file not found: "%SPLIT_JSON%"
    exit /b 1
)
set "PROFILE_ARGS="
set "PROFILES_JSON=%~4"
if not defined PROFILES_JSON goto profiles_ready
if not exist "%PROFILES_JSON%" (
    echo ERROR: Calibration file not found: "%PROFILES_JSON%"
    exit /b 1
)
set PROFILE_ARGS=--profiles "%PROFILES_JSON%"
:profiles_ready
if exist "%RUN_DIR%" (
    echo ERROR: Output directory exists. Choose a new run directory.
    exit /b 1
)
echo GPU profile: %PROFILE%
echo Output: "%RUN_DIR%"
echo [1/2] Validate the split and cache frozen features for training and validation.
uv run --locked --extra cu128 python -m surgenet_dino cache-features --split "%SPLIT_JSON%" --output-dir "%RUN_DIR%\cache" --gpu-profile "%PROFILE%" %PROFILE_ARGS%
if errorlevel 1 exit /b %errorlevel%
echo [2/2] Train a regularized linear head with sub-epoch validation and early stopping.
uv run --locked --extra cu128 python -m surgenet_dino train-head --features "%RUN_DIR%\cache\features.pt" --output-dir "%RUN_DIR%\train" --gpu-profile "%PROFILE%" %PROFILE_ARGS%
exit /b %errorlevel%
