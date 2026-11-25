"""Bounded virtual display lifecycle for container tests."""
import os
from pathlib import Path
import subprocess
import sys
import time

def main():
    display = subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1280x1024x24', '-nolisten', 'tcp', '-ac'])
    try:
        deadline = time.monotonic() + 15
        while not Path('/tmp/.X11-unix/X99').exists():
            if display.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Virtual display failed to start within 15 seconds')
            time.sleep(.1)
        print('Virtual display ready', flush=True)
        env = dict(os.environ, DISPLAY=':99')
        return subprocess.run([sys.executable, 'tools/test_release.py'], env=env, timeout=300).returncode
    finally:
        display.terminate()
        try: display.wait(timeout=5)
        except subprocess.TimeoutExpired:
            display.kill(); display.wait()

if __name__ == '__main__': raise SystemExit(main())
