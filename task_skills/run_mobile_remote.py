"""LAN pendant for one Nova2 operator. Hardware I/O runs on one worker at a time."""
import argparse
import collections
import hmac
import ipaddress
import json
import math
import secrets
import signal
import socket
import ssl
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .mobile_arm import MobileArm, MobileCleanupError
from .run_cartesian_jog import JogStopped

DRIVE_TIMEOUT = 0.42
SPEED_RANGE_MM_S = (5.0, 150.0)
VIEWS = {"front": 0, "left": 90, "back": 180, "right": -90}
# Robot front in each arm's base frame: -Y on both. Both home TCPs (+-117.6,
# -344.8) and every taught work pose sit at Y -345..-385, i.e. in front. (The
# photo poses, left +X / right -X, swing the arms outward to clear the top
# camera; reading them as "forward" was a 90 deg error, fixed 2026-11-21.)
DEFAULT_FRONT_DEG = {"left": 270, "right": 270}
ACTIONS = ("/api/open", "/api/recover", "/api/step", "/api/home", "/api/camera", "/api/gripper")
POSTS = ("/api/pair", "/api/drive", "/api/release", "/api/stop", "/api/settings",
         "/api/front", *ACTIONS)


def operator_basis(view, front_deg):
    """Base-frame unit vectors for the operator's right and forward.

    The operator stands on side ``view`` of the robot and faces it, so forward
    (thumb up) points from the operator towards the robot and right is 90 deg
    clockwise of that, seen from above (base Z up). Angles are multiples of 90,
    so the result is exactly axis aligned.
    """
    facing = math.radians(front_deg + VIEWS[view] + 180)
    forward = (round(math.cos(facing)), round(math.sin(facing)))
    right = (round(math.sin(facing)), round(-math.cos(facing)))
    return right, forward


def to_base(view, front_deg, x, y, z):
    right, forward = operator_basis(view, front_deg)
    return (x * right[0] + y * forward[0], x * right[1] + y * forward[1], z)


class Settings:
    """Per-arm front direction and saved camera joints, kept in one JSON file."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.data = {"left": {}, "right": {}}
        if self.path and self.path.exists():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            for arm in ("left", "right"):
                if isinstance(loaded.get(arm), dict):
                    self.data[arm] = loaded[arm]

    def front(self, arm):
        value = self.data[arm].get("front_deg", DEFAULT_FRONT_DEG[arm])
        return value if value in (0, 90, 180, 270) else DEFAULT_FRONT_DEG[arm]

    def update(self, arm, **values):
        self.data[arm].update(values)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(self.path)


class RequestError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Remote:
    def __init__(self, live, *, factory=MobileArm, clock=time.monotonic, settings=None):
        self.live, self.factory, self.clock = live, factory, clock
        self.settings = settings or Settings()
        self.speed, self.view = 30.0, "front"
        self.gripper = None
        self.after = None
        self.lock = threading.RLock()
        self.code = f"{secrets.randbelow(10**8):08d}"
        self.token = None
        self.attempts = collections.deque()
        self.last_seen = 0
        self.phase, self.arm, self.revision = "idle", None, 0
        self.pose, self.mode, self.message = None, None, "请选择机械臂"
        self.device = self.worker = None
        self.cancel = threading.Event()
        self.stop_reason = ""
        self.vector = None
        self.seq = -1
        self.last_drive = 0
        self.cleanup_failed = False
        self.closing = False

    def state(self):
        return dict(live=self.live, phase=self.phase, arm=self.arm, revision=self.revision,
                    pose=self.pose, mode=self.mode, message=self.message, serverTime=self.clock(),
                    speed=self.speed, view=self.view, gripper=self.gripper,
                    front={arm: self.settings.front(arm) for arm in ("left", "right")})

    def transition(self, phase, message):
        self.phase, self.message = phase, message
        self.revision += 1
        # service.log is the only record of what happened on site.
        try:
            print(f"{time.strftime('%H:%M:%S')} [{self.arm or '-'}] {phase}: {message}", flush=True)
        except (OSError, ValueError):
            pass

    def pair(self, code):
        now = self.clock()
        while self.attempts and now - self.attempts[0] >= 60:
            self.attempts.popleft()
        if len(self.attempts) >= 8:
            raise RequestError(429, "尝试过多，请稍后再试")
        if not isinstance(code, str) or not code.isascii() or not hmac.compare_digest(code, self.code):
            self.attempts.append(now)
            raise RequestError(403, "配对码不正确，请查看电脑上的配对卡")
        if self.token and (now - self.last_seen < 10 or self.phase not in ("idle", "fault")):
            raise RequestError(409, "已有手机连接。请先停止控制并关闭原页面，等待 10 秒后重试")
        self.token, self.last_seen = secrets.token_urlsafe(32), now
        self.seq = -1
        return {"token": self.token}

    def release_drag(self):
        self.vector = None
        if self.phase == "driving":
            self.transition("settling", "摇杆已回中，正在确认停止")

    def sample(self):
        with self.lock:
            if self.clock() - self.last_drive >= DRIVE_TIMEOUT:
                self.release_drag()
            return self.vector if self.phase == "driving" else None

    def drive(self, path, body):
        seq = body.get("seq")
        if type(seq) is not int or not self.seq < seq < 2**53:
            raise RequestError(409, "摇杆输入已过期，请松手后重试")
        if path == "/api/release":
            self.seq = seq
            self.release_drag()
            return
        values = [body.get(k) for k in "xyz"]
        if any(type(v) not in (float, int) or not -1 <= v <= 1 for v in values):
            raise RequestError(400, "摇杆方向无效")
        magnitude = math.hypot(*values)
        if magnitude > 1.000001:
            raise RequestError(400, "摇杆幅度无效")
        if (self.phase not in ("ready", "driving") or type(body.get("revision")) is not int
                or body["revision"] != self.revision or self.clock() - self.last_seen >= 5):
            raise RequestError(409, "状态已变化，请松手并重新拖动")
        vector = to_base(self.view, self.settings.front(self.arm), *values)
        self.seq, self.last_drive, self.vector = seq, self.clock(), vector
        if self.phase == "ready":
            self.transition("driving", "拇指控制中，松手停止")
            self.start(self.device.drive, self.sample, self.cancel)

    def api(self, path, body, token):
        with self.lock:
            if self.closing:
                raise RequestError(503, "遥控服务正在退出")
            if path == "/api/pair":
                return self.pair(body.get("code"))
            if not self.token or not token.isascii() or not hmac.compare_digest(token, self.token):
                raise RequestError(401, "请在本机配对后操作")
            self.expire()
            if path in ("/api/drive", *ACTIONS):
                issued = body.get("issued")
                if type(issued) not in (int, float) or not -.15 <= self.clock() - issued <= .4:
                    raise RequestError(409, "指令延迟过高或已过期，请松手后重试")
            if path == "/api/state":
                self.last_seen = self.clock()
            elif path == "/api/stop":
                self.stop("已停止控制")
            elif path in ("/api/drive", "/api/release"):
                self.drive(path, body)
            elif path == "/api/settings":
                speed, view = body.get("speed"), body.get("view")
                if speed is not None:
                    if type(speed) not in (int, float) or not math.isfinite(speed) or \
                            not SPEED_RANGE_MM_S[0] <= speed <= SPEED_RANGE_MM_S[1]:
                        raise RequestError(400, "速度需在 5-150 mm/s")
                    self.speed = float(speed)
                    if self.device is not None:
                        self.device.set_speed(speed)
                if view is not None:
                    if view not in VIEWS:
                        raise RequestError(400, "站位无效")
                    if self.phase in ("driving", "settling"):
                        raise RequestError(409, "请松开摇杆后再切换站位")
                    self.view = view
            elif path == "/api/front":
                arm, deg = body.get("arm"), body.get("deg")
                if arm not in ("left", "right") or deg not in (0, 90, 180, 270):
                    raise RequestError(400, "方向校准值无效")
                if self.phase in ("driving", "settling") and self.arm == arm:
                    raise RequestError(409, "请松开摇杆后再校准方向")
                self.settings.update(arm, front_deg=deg)
            elif path in ACTIONS:
                if self.clock() - self.last_seen >= 5:
                    raise RequestError(409, "连接已超时，请刷新状态")
                revision = body.get("revision")
                if type(revision) is not int or revision != self.revision:
                    raise RequestError(409, "状态已变化，请确认面板后再操作")
                clear = path == "/api/recover"
                if clear and self.phase == "ready":
                    self.transition("moving", "正在清除错误并重新使能")
                    self.start(self.device.recover, self.cancel)
                elif path == "/api/open" or clear:
                    arm = body.get("arm")
                    if arm not in ("left", "right") or self.phase not in (
                            ("idle", "fault") if clear else ("idle",)):
                        raise RequestError(409, "请先停止当前控制，再选择机械臂")
                    self.arm = arm
                    self.device = self.factory(arm, self.live)
                    if hasattr(self.device, "set_speed"):
                        self.device.set_speed(self.speed)
                    self.gripper = None
                    self.cancel = threading.Event()
                    if clear:
                        # A fresh open re-reads the controller and refuses unsafe
                        # modes, so it is also the way out of a failed release.
                        self.cleanup_failed = False
                        self.transition("opening", "正在上电（如需）、清除错误、重新使能所选机械臂")
                        self.start(self.device.open, self.cancel, True)
                    else:
                        self.transition("opening", "正在连接并检查所选机械臂")
                        self.start(self.device.open, self.cancel)
                elif path != "/api/step":
                    if self.phase != "ready":
                        raise RequestError(409, "机械臂尚未就绪")
                    self.action(path, body)
                else:
                    axis, direction, mm = (body.get(key) for key in ("axis", "direction", "mm"))
                    if (axis not in ("X", "Y", "Z") or type(direction) is not int
                            or direction not in (-1, 1) or type(mm) not in (int, float)
                            or mm not in (0.5, 1, 5)):
                        raise RequestError(400, "无效的轴、方向或步长")
                    if self.phase != "ready":
                        raise RequestError(409, "机械臂尚未就绪")
                    # Axis and direction arrive in the operator's frame:
                    # X = operator right, Y = operator forward.
                    if axis != "Z":
                        local = (direction, 0) if axis == "X" else (0, direction)
                        base = to_base(self.view, self.settings.front(self.arm), *local, 0)
                        axis = "X" if base[0] else "Y"
                        direction = int(base[0] or base[1])
                    self.transition("moving", f"正在沿 {axis} 轴移动 {direction * mm:+g} mm")
                    self.start(self.device.step, axis, direction, mm, self.cancel)
            else:
                raise RequestError(404, "接口不存在")
            return self.state()

    def action(self, path, body):
        device = self.device
        if path == "/api/home":
            self.transition("moving", "正在复位到初始关节位")
            self.start(device.go_home, self.cancel)
        elif path == "/api/camera":
            self.transition("moving", "正在移动到拍摄位")
            self.start(device.go_photo, self.cancel)
        else:
            act = body.get("action")
            if act not in ("open", "close"):
                raise RequestError(400, "夹爪指令无效")
            from .mobile_arm import GRIPPER_CLOSED, GRIPPER_OPEN
            self.transition("moving", "正在张开夹爪" if act == "open" else "正在闭合夹爪")
            self.after = "gripper"
            self.start(device.set_gripper, GRIPPER_OPEN if act == "open" else GRIPPER_CLOSED,
                       self.cancel)

    def start(self, operation, *args):
        self.worker = threading.Thread(target=self.run, args=(operation, args), name="nova2-arm")
        self.worker.start()

    def run(self, operation, args):
        error = ""
        after, self.after = self.after, None
        try:
            if operation and not self.cancel.is_set():
                try:
                    snapshot = operation(*args)
                except Exception as exc:
                    # A gripper fault (serial timeout, rejected move) does not move
                    # the arm, so keep the session and report it instead of
                    # disabling the robot.
                    if after != "gripper" or isinstance(exc, (JogStopped, MobileCleanupError)):
                        raise
                    with self.lock:
                        if not self.cancel.is_set():
                            self.transition("ready", f"夹爪未完成：{exc}")
                            return
                    raise
                with self.lock:
                    if not self.cancel.is_set():
                        self.pose, self.mode = snapshot["pose"], snapshot["mode"]
                        self.vector = None
                        message = "可以拖动摇杆或单步微调"
                        if after == "gripper":
                            self.gripper = getattr(self.device, "gripper_position", None)
                            message = "夹爪已到位（位置反馈，不代表已夹稳）"
                        self.transition("ready", message)
                        return
        except JogStopped as exc:
            if not self.cancel.is_set():
                error = str(exc) or "机械臂已停止"
        except MobileCleanupError as exc:
            with self.lock:
                self.cleanup_failed = True
                error = f"{exc}。请到现场核对机械臂，确认安全后可点“上电 / 清错并使能”重试。"
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        with self.lock:
            if self.phase != "stopping":
                self.transition("stopping", "正在结束控制")
        try:
            self.device.close()
        except Exception as exc:
            with self.lock:
                self.cleanup_failed = True
                error = f"{error} 释放控制失败：{exc}。请到现场核对机械臂，确认安全后可点“上电 / 清错并使能”重试。".strip()
        with self.lock:
            self.device, self.arm, self.pose, self.mode = None, None, None, None
            self.transition("fault" if error else "idle", error or self.stop_reason)

    def stop(self, reason):
        if self.phase == "stopping":
            return
        if self.phase in ("fault", "idle"):
            self.transition("idle", reason)
            return
        needs_worker = self.phase == "ready"
        self.cancel.set()
        self.vector = None
        self.stop_reason = reason
        self.transition("stopping", "正在停止并释放机械臂")
        if needs_worker:
            self.start(None)

    def expire(self):
        with self.lock:
            if self.phase == "driving" and self.clock() - self.last_drive >= DRIVE_TIMEOUT:
                self.release_drag()
            if self.phase in ("opening", "ready", "moving", "driving", "settling") and self.clock() - self.last_seen >= 5:
                self.stop("手机连接超时，已结束控制")

    def shutdown(self):
        with self.lock:
            self.closing = True
            self.stop("电脑端已结束控制")
            worker = self.worker
        if worker:
            worker.join()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        self.request.settimeout(4)
        self.request = self.server.tls.wrap_socket(self.request, server_side=True)
        super().setup()

    def log_message(self, *args):
        pass

    def send_data(self, status, data, kind="application/json; charset=utf-8"):
        if isinstance(data, dict):
            data = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(data)

    def handle_api(self, post):
        try:
            if self.headers.get("Host") != self.server.authority:
                raise RequestError(403, "请使用电脑配对卡上的地址")
            if self.path == "/" and not post:
                self.send_data(200, self.server.panel, "text/html; charset=utf-8")
                return
            if post and self.headers.get("Origin") not in (None, "https://" + self.server.authority):
                raise RequestError(403, "来源不符")
            body = {}
            if post:
                if self.path not in POSTS:
                    raise RequestError(404, "接口不存在")
                if self.headers.get_content_type() != "application/json":
                    raise RequestError(415, "需要 JSON 请求")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024:
                    raise RequestError(400, "请求大小不正确")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise RequestError(400, "请求格式不正确")
            elif self.path != "/api/state":
                raise RequestError(404, "接口不存在")
            bearer = self.headers.get("Authorization", "")
            token = bearer[7:] if bearer.startswith("Bearer ") else ""
            result = self.server.remote.api(self.path, body, token)
            status = 202 if self.path in (*ACTIONS, "/api/stop") else 200
            self.send_data(status, result)
        except RequestError as exc:
            self.close_connection = True
            self.send_data(exc.status, {"error": str(exc)})
        except (ValueError, UnicodeError):
            self.close_connection = True
            self.send_data(HTTPStatus.BAD_REQUEST, {"error": "请求格式不正确"})

    def do_GET(self):
        self.handle_api(False)

    def do_POST(self):
        self.handle_api(True)


class Server(ThreadingHTTPServer):
    def service_actions(self):
        if self.stop_file.exists():
            self.stop_file.unlink()
            raise KeyboardInterrupt
        self.remote.expire()

    def handle_error(self, request, client_address):
        import sys
        if not isinstance(sys.exc_info()[1], (ConnectionError, socket.timeout, ssl.SSLError)):
            super().handle_error(request, client_address)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--port", type=int, default=9443)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--card-dir", type=Path, required=True)
    parser.add_argument("--settings", type=Path,
                        default=Path(__file__).resolve().parent.parent / "calibration-data"
                        / "mobile-remote" / "settings.json",
                        help="front direction and saved camera pose per arm")
    args = parser.parse_args()
    address = ipaddress.IPv4Address(args.bind)
    if not address.is_private or address.is_unspecified or address.is_link_local or address.is_multicast:
        parser.error("请选择电脑的局域网 IPv4 地址")
    from .mobile_pairing import prepare_tls, write_card
    cert, key = prepare_tls(args.card_dir, str(address))
    remote = Remote(args.live, settings=Settings(args.settings))
    with Server((str(address), args.port), Handler) as server:
        server.tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server.tls.minimum_version = ssl.TLSVersion.TLSv1_2
        server.tls.load_cert_chain(cert, key)
        server.panel = Path(__file__).with_name("mobile_remote.html").read_bytes()
        server.authority = f"{address}:{server.server_port}"
        server.remote = remote
        server.stop_file = args.card_dir / "stop.request"
        server.stop_file.unlink(missing_ok=True)
        write_card(args.card_dir, "https://" + server.authority, remote.code, cert, args.live)
        print(f"手机打开 https://{server.authority} ，配对码见电脑本地配对卡。", flush=True)
        def terminate(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, terminate)
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            pass
        finally:
            remote.shutdown()


if __name__ == "__main__":
    main()
