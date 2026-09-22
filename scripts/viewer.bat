@echo off
setlocal
cd /d "%~dp0.."
rem The viewer uses only the standard library; never sync or change the training environment.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m tools.viewer %*
) else (
    python -m tools.viewer %*
)
if errorlevel 1 pause
