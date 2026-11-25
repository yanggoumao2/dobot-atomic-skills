"""Desktop launcher: reuse a running console or start it in the background."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
URL = 'http://127.0.0.1:8765'
HTTP = build_opener(ProxyHandler({}))


def ready():
    try:
        with HTTP.open(URL + '/api/state', timeout=1) as response:
            state = json.load(response)
        return Path(state['root']).resolve() == ROOT and bool(state['token'])
    except (OSError, ValueError, KeyError):
        return False


def main():
    if not ready():
        log = ROOT / '.local/console-launcher.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a', encoding='utf-8') as output:
            process = subprocess.Popen(
                [sys.executable, '-X', 'utf8', '-B', str(ROOT / 'tools/local_console.py'), '--no-browser'],
                cwd=ROOT, stdout=output, stderr=output,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        deadline = time.monotonic() + 15
        while not ready():
            if process.poll() is not None or time.monotonic() >= deadline:
                import tkinter.messagebox
                tkinter.messagebox.showerror('Nova2控制台', f'本地网页服务启动失败，请查看 {log}')
                return 1
            time.sleep(0.25)
    webbrowser.open(URL)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
