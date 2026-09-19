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
set "BACKEND=%~1"
if "%BACKEND%"=="" set "BACKEND=cu128"
set "MODEL=%~2"
if "%MODEL%"=="" set "MODEL=dinov2_vitb"
if /i "%BACKEND%"=="cu128" (
    set "BACKEND=cu128"
    set "DEVICE=cuda"
) else if /i "%BACKEND%"=="cpu" (
    set "BACKEND=cpu"
    set "DEVICE=cpu"
) else (
    echo ERROR: backend must be cu128 or cpu.
    exit /b 1
)
if not exist uv.lock (
    echo ERROR: run scripts\init.bat first.
    exit /b 1
)
echo Official surgical weights: compatibility/forward check, NOT classification accuracy.
uv run --locked --extra "%BACKEND%" python -m surgenet_dino smoke --model "%MODEL%" --weights surgenet --device "%DEVICE%" --report "outputs/%MODEL%/pretrained_smoke.json" --overwrite
exit /b %errorlevel%
