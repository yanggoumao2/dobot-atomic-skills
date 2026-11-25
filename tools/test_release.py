"""Offline release gate: all first-party tests, then full synthetic workflow."""
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ['DOBOT_CONFIG'] = str(ROOT / 'scripts/dobot_config/dobot_settings.ini')
for command in ([sys.executable, '-m', 'pytest', 'tests', '-v', '-o', 'faulthandler_timeout=45'],
                [sys.executable, 'tests/offline_workflow.py']):
    print('RUN', ' '.join(command), flush=True)
    subprocess.run(command, check=True, timeout=240)
print('RELEASE OFFLINE FLOW PASSED', flush=True)
