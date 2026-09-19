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
    echo ERROR: run scripts\init.bat first.
    exit /b 1
)
uv run --locked --extra "%BACKEND%" python -m surgenet_dino doctor
if errorlevel 1 exit /b 1
uv run --locked --extra "%BACKEND%" ruff check surgenet_dino tests examples load_models.py
if errorlevel 1 exit /b 1
uv run --locked --extra "%BACKEND%" python -m pytest -q
exit /b %errorlevel%
