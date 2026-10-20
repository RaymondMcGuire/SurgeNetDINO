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
    echo ERROR: uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)
set "BACKEND=%~1"
if "%BACKEND%"=="" set "BACKEND=cu128"
if /i "%BACKEND%"=="cu128" (
    set "BACKEND=cu128"
) else if /i "%BACKEND%"=="cpu" (
    set "BACKEND=cpu"
) else (
    echo ERROR: backend must be cu128 or cpu.
    exit /b 1
)
if not exist uv.lock (
    echo ERROR: restore the committed uv.lock before initialization.
    exit /b 1
)
uv python install 3.12
if errorlevel 1 exit /b 1
uv sync --locked --extra "%BACKEND%" --group dev
if errorlevel 1 exit /b 1
if /i "%BACKEND%"=="cu128" (
    uv run --locked --extra cu128 python -m surgenet_dino doctor --require-cuda
) else (
    uv run --locked --extra cpu python -m surgenet_dino doctor
)
exit /b %errorlevel%
