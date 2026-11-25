@echo off
setlocal
cd /d "%~dp0"
set "DOBOT_PANEL_PYTHONW=%~dp0..\.venv\Scripts\pythonw.exe"
if not exist "%DOBOT_PANEL_PYTHONW%" (
    echo Python environment not found: %DOBOT_PANEL_PYTHONW%
    echo Run tools\gripper_panel.py with the project's Python 3.11 environment.
    pause
    exit /b 1
)
start "" "%DOBOT_PANEL_PYTHONW%" -X utf8 -B "%~dp0tools\gripper_panel.py"
