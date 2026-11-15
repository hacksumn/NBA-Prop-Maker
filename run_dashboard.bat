@echo off
REM Rebuilds the read-only Command Center dashboard from the existing project files and opens it.
REM Safe to run any time (including during the morning run): it only READS data/models/output/logs
REM and writes one file, output\dashboard_pro.html.
cd /d "%~dp0"
python scripts\build_dashboard.py
if errorlevel 1 (
  echo.
  echo Dashboard build failed. See the message above.
  pause
  exit /b 1
)
start "" "%~dp0output\dashboard_pro.html"
