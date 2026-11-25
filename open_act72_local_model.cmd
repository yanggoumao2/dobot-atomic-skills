@echo off
cd /d "%~dp0"
set PYTHONNOUSERSITE=1
set CUDA_VISIBLE_DEVICES=0
set PYTHONPATH=
set "ACT_CHECKPOINT=%~1"
if "%ACT_CHECKPOINT%"=="" set "ACT_CHECKPOINT=best.pt"
"%~dp0..\.venv-act-inference\Scripts\python.exe" -X utf8 tools\act_inference_server.py --run "%~dp0.local\models\act72_15k" --checkpoint "%ACT_CHECKPOINT%"
pause
