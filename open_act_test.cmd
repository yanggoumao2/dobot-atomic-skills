@echo off
cd /d "%~dp0"
"%~dp0..\.venv\Scripts\python.exe" -X utf8 tools\act_real_test.py %*
pause
