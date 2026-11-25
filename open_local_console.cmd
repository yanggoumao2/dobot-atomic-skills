@echo off
setlocal
cd /d "%~dp0"
set "DOBOT_CONSOLE_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not exist "%DOBOT_CONSOLE_PYTHON%" (
    echo Python environment not found: %DOBOT_CONSOLE_PYTHON%
    pause
    exit /b 1
)
"%DOBOT_CONSOLE_PYTHON%" -X utf8 -B "%~dp0tools\local_console.py"
pause
