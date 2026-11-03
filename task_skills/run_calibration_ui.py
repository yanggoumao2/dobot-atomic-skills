"""Launch the local Dobot camera-to-base calibration wizard.

Default bind is loopback only. The process does not open a camera or robot at
startup; both require explicit operator actions in the browser.
"""

from __future__ import annotations

import argparse
import ipaddress
import sys
import threading
import webbrowser
from pathlib import Path

from task_skills.calibration.http_server import create_server


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "calibration-data"
WEB_ROOT = Path(__file__).resolve().parent / "calibration" / "web"


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1", help="监听地址；默认仅回环")
    p.add_argument("--port", type=int, default=8765, help="HTTP 端口，0 表示系统分配")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA,
                   help="会话目录；必须由本应用管理，默认 calibration-data/")
    p.add_argument("--open-browser", action="store_true", help="启动后打开本机浏览器")
    p.add_argument("--allow-remote-bind", action="store_true",
                   help="显式允许非回环监听；仍无公网认证，不建议使用")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 0 <= args.port <= 65535:
        print("ERROR: port must be 0..65535", file=sys.stderr)
        return 1
    if not _is_loopback(args.host) and not args.allow_remote_bind:
        print("ERROR: non-loopback bind requires --allow-remote-bind", file=sys.stderr)
        return 2
    if not WEB_ROOT.is_dir():
        print(f"ERROR: web assets missing: {WEB_ROOT}", file=sys.stderr)
        return 1
    try:
        server = create_server(args.host, args.port, args.data_dir, WEB_ROOT,
                               allow_remote_bind=args.allow_remote_bind)
    except OSError as error:
        print(f"ERROR: cannot bind {args.host}:{args.port}: {error}", file=sys.stderr)
        return 1
    host, port = server.server_address[:2]
    if host in ("0.0.0.0", "::"):
        display_host = "127.0.0.1"
    else:
        display_host = host
    url = f"http://{display_host}:{port}/"
    print("Dobot 相机到基座标定向导", flush=True)
    print(f"URL: {url}", flush=True)
    print(f"数据目录: {args.data_dir.resolve()}", flush=True)
    print("启动不会连接相机、机械臂或串口；请在页面内显式操作。", flush=True)
    if not _is_loopback(args.host):
        print("WARNING: 已显式允许非回环监听；该界面没有公网身份认证。", flush=True)
    if args.open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n正在关闭标定服务…", flush=True)
    finally:
        server.camera.close()
        server.service.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
