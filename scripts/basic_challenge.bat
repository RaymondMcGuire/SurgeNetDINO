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
set "SOURCE_ROOT=%~1"
if "%SOURCE_ROOT%"=="" set "SOURCE_ROOT=H:\dataset\JSCAS AI Challenge"
set "RUN_DIR=%~2"
if not "%RUN_DIR%"=="" goto run_dir_ok
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss_fff"') do set "RUN_ID=%%I"
if not defined RUN_ID exit /b 1
set "RUN_DIR=outputs\basic_challenge\%RUN_ID%"

:run_dir_ok
set "MODEL_PYTHON=.venv\Scripts\python.exe"
set "VIDEO_PYTHON=.cache\uv\archive-v0\SruTPdM1WCWkIu0S\Scripts\python.exe"
set "CHECKPOINT=outputs\challenge_final_20261004\checkpoint.pt"
set "TEMPLATE=%SOURCE_ROOT%\jscas_ai_challenge_2026_data1\jscas_ai_challenge_2026_data1\submission_format\basic_challenge\submission.csv"
for %%F in ("%MODEL_PYTHON%" "%VIDEO_PYTHON%" "%CHECKPOINT%" "%TEMPLATE%") do (
    if not exist "%%~F" (
        echo ERROR: Required file not found: "%%~F"
        exit /b 1
    )
)
if exist "%RUN_DIR%" (
    echo ERROR: Output exists. Choose a NEW run directory: "%RUN_DIR%"
    exit /b 1
)
echo Template: "%TEMPLATE%"
echo Output: "%RUN_DIR%"
echo Checking the existing CUDA environment. No training or package installation.
"%MODEL_PYTHON%" -m surgenet_dino doctor --require-cuda
if errorlevel 1 exit /b %errorlevel%

echo [1/3] Extract only the template's 1 fps images from videos 02, 07, 11, 15.
"%VIDEO_PYTHON%" -m tools.basic_extract --source "%SOURCE_ROOT%" --template "%TEMPLATE%" --output-dir "%RUN_DIR%\data"
if errorlevel 1 exit /b %errorlevel%

echo [2/3] Extract frozen DINO features. GPU low, CPU 2 threads.
"%MODEL_PYTHON%" -m surgenet_dino.temporal.submission cache-images --checkpoint "%CHECKPOINT%" --template "%TEMPLATE%" --image-root "%RUN_DIR%\data" --output-dir "%RUN_DIR%\features"
if errorlevel 1 exit /b %errorlevel%

echo [3/3] Predict Basic labels and export the completed CSV in template order.
"%MODEL_PYTHON%" -m surgenet_dino.temporal.submission predict --checkpoint "%CHECKPOINT%" --template "%TEMPLATE%" --images "%RUN_DIR%\features\images.pt" --output-dir "%RUN_DIR%\predictions"
if errorlevel 1 exit /b %errorlevel%
echo.
echo Basic submission created: "%RUN_DIR%\predictions\submission.csv"
echo The original template and videos were not modified. No submission was sent.
exit /b 0
