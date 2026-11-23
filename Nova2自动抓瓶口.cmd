@echo off
setlocal
cd /d "%~dp0"
if not defined NOVA2_PYTHON set "NOVA2_PYTHON=C:\software\anaconda3\envs\dobot-xtrainer\python.exe"
if not exist "%NOVA2_PYTHON%" (
  echo Set NOVA2_PYTHON to the Dobot Python executable.
  pause
  exit /b 1
)
if "%~1"=="" (
  "%NOVA2_PYTHON%" -X utf8 -u -m chem_demo.run_neck_pick --live
) else (
  "%NOVA2_PYTHON%" -X utf8 -u -m chem_demo.run_neck_pick %*
)
set "pick_exit=%errorlevel%"
if not "%pick_exit%"=="0" pause
exit /b %pick_exit%
