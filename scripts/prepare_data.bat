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
    echo ERROR: Install uv first. See README.md.
    exit /b 1
)
set "SOURCE_ROOT=%~1"
if "%SOURCE_ROOT%"=="" set "SOURCE_ROOT=H:\dataset\JSCAS AI Challenge"
set "DATA_ROOT=%~2"
if "%DATA_ROOT%"=="" set "DATA_ROOT=%SOURCE_ROOT%\..\JSCAS_preprocessed"
set "EXPERIMENTS_ROOT=%~3"
if "%EXPERIMENTS_ROOT%"=="" set "EXPERIMENTS_ROOT=%SOURCE_ROOT%\..\JSCAS_experiments"
echo Source: "%SOURCE_ROOT%"
echo Dataset: "%DATA_ROOT%\labeled_1fps_v1"
echo Split: "%EXPERIMENTS_ROOT%\splits\video_holdout_9_3_v1\split.json"
echo Extract and audit all 12 labeled videos, including 01. No GPU or model training.
echo uv prepares isolated OpenCV/pandas/Pillow dependencies. Allow at least 120 GiB free space.
uv run --no-project --python 3.12 --script tools\prepare_labeled.py prepare --source "%SOURCE_ROOT%" --output "%DATA_ROOT%\labeled_1fps_v1" --split-output "%EXPERIMENTS_ROOT%\splits\video_holdout_9_3_v1"
exit /b %errorlevel%
