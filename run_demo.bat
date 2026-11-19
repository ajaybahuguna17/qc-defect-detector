@echo off
cd /d "%~dp0"
set "PY=%QC_PYTHON%"
if not defined PY (where python >nul 2>nul && set "PY=python")
if not defined PY (where py >nul 2>nul && set "PY=py")
if not defined PY (
  echo Python not found.
  echo Install Python, or run once:  setx QC_PYTHON "C:\path\to\python.exe"
  pause
  exit /b 1
)
echo Using Python: %PY%
"%PY%" phase1-vision\demo\app.py
pause
