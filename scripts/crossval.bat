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
set "RUN_DIR=H:\dataset\JSCAS_experiments\frozen_dinov2\crossval_regularized_%RUN_ID%"
:run_dir_ok
for %%I in ("%RUN_DIR%") do set "RUN_NAME=%%~nxI"
set "SPLIT_DIR=%~4"
if "%SPLIT_DIR%"=="" set "SPLIT_DIR=H:\dataset\JSCAS_experiments\splits\%RUN_NAME%"
set "PROFILES=%~5"
if not exist "%FEATURES%" exit /b 1
set "PROFILE_ARGS="
if not defined PROFILES goto profiles_ready
if not exist "%PROFILES%" exit /b 1
set PROFILE_ARGS=--profiles "%PROFILES%"
:profiles_ready
if exist "%RUN_DIR%" exit /b 1
if exist "%SPLIT_DIR%" exit /b 1
echo [1/2] Validate cached sources once, then train four video-disjoint folds.
echo Output: "%RUN_DIR%"
uv run --locked --extra cu128 python -m surgenet_dino crossval-head --features "%FEATURES%" --output-dir "%RUN_DIR%" --split-output-dir "%SPLIT_DIR%" --folds 4 --gpu-profile "%PROFILE%" %PROFILE_ARGS%
if errorlevel 1 exit /b %errorlevel%
echo [2/2] Plot completed cross-validation results in an isolated plotting environment.
uv run --isolated --no-project --with matplotlib==3.11.2 python -m scripts.plot_crossval "%RUN_DIR%"
exit /b %errorlevel%
