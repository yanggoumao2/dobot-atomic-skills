"""Serve one local camera as an MJPEG stream for a remote frontend.

Read-only with respect to the robot: it opens a camera and nothing else. One
capture thread owns the device; every HTTP client gets the latest encoded
frame, so N viewers do not mean N camera reads.

Endpoints:
  GET /stream.mjpg   multipart/x-mixed-replace MJPEG (use as <img src=...>)
  GET /snapshot.jpg  one JPEG
  GET /health        JSON status (no token needed, carries no image)
  GET /              minimal viewer page

Auth: if a token is configured, image endpoints require either
``Authorization: Bearer <token>`` or ``?token=<token>`` (an <img> tag cannot
send headers, so the query form exists for browsers).

Run with the calibration venv (OpenCV 4.8.1 reads this camera; 5.0 does not):
  .venv-calibration/Scripts/python.exe -B -m task_skills.run_camera_stream --device 1 --host 100.79.172.21
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# MSMF hardware transforms make this top camera stall: it opens, then every
# grab times out (~10 s, error 0x8000000A). Off by default (2026-11-13).
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2  # noqa: E402  (must follow the environment setting above)

BOUNDARY = "frame"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class FrameSource:
    """Owns the camera; keeps only the newest JPEG."""

    def __init__(self, device: int, width: int, height: int, rotate: int, fps: float, quality: int):
        self.device, self.width, self.height = device, width, height
        self.rotation = ROTATIONS[rotate]
        self.period = 1.0 / fps
        self.quality = quality
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._stop = threading.Event()
        self.status: dict = {"open": False, "device": device}
        self._capture = None
        self._thread: threading.Thread | None = None

    def _open(self, reads: int = 5):
        """Open the device and return (capture, first frame), or (None, reason)."""
        capture = cv2.VideoCapture(self.device)
        if not capture.isOpened():
            capture.release()
            return None, "cannot open (is another program using it?)"
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        for _ in range(reads):  # the first grabs after opening can fail on this camera
            ok, frame = capture.read()
            if ok and frame is not None:
                return capture, frame
        capture.release()
        return None, "opened but returned no frame"

    def start(self, attempts: int = 3) -> None:
        reason = ""
        for attempt in range(1, attempts + 1):
            capture, result = self._open()
            if capture is not None:
                break
            reason = result
            print(f"camera {self.device}: attempt {attempt}/{attempts} {reason}", flush=True)
            time.sleep(1.0)
        else:
            raise RuntimeError(f"camera {self.device} {reason}; try replugging its USB cable")
        frame = result
        self._capture = capture
        self.status.update(open=True, actual_wh=[frame.shape[1], frame.shape[0]],
                           backend=capture.getBackendName(), reconnects=0)
        self._publish(frame)
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _reopen(self) -> None:
        """Release the stalled device and reopen it until it answers or we stop.

        ``open`` stays True meanwhile so viewers keep their connection and
        resume when frames return, instead of the stream ending for good.
        """
        self.status["reconnecting"] = True
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        delay = 1.0
        while not self._stop.is_set():
            capture, result = self._open(reads=3)
            if capture is not None:
                self._capture = capture
                self.status["reconnects"] = self.status.get("reconnects", 0) + 1
                self.status.pop("reconnecting", None)
                self.status.pop("error", None)
                print(f"camera {self.device}: reopened (reconnect #{self.status['reconnects']})", flush=True)
                self._publish(result)
                return
            self.status["error"] = result
            print(f"camera {self.device}: reopen failed ({result}); retry in {delay:.0f} s", flush=True)
            self._stop.wait(delay)
            delay = min(delay * 2, 10.0)

    def _publish(self, frame) -> None:
        if self.rotation is not None:
            frame = cv2.rotate(frame, self.rotation)
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return
        with self._cond:
            self._jpeg = encoded.tobytes()
            self._seq += 1
            self.status["frames"] = self._seq
            self.status["last_frame_unix"] = time.time()
            self._cond.notify_all()

    def _loop(self) -> None:
        failures = 0
        next_at = time.monotonic()
        while not self._stop.is_set():
            if self._capture is None:
                self._reopen()
                continue
            ok, frame = self._capture.read()
            if not ok or frame is None:
                # A failed grab on MSMF already blocked ~10 s, so two in a row
                # means the device is stalled: reopen it rather than keep waiting.
                failures += 1
                if failures >= 2:
                    print(f"camera {self.device}: no frames, reopening", flush=True)
                    failures = 0
                    self._reopen()
                else:
                    time.sleep(0.03)
                continue
            failures = 0
            now = time.monotonic()
            if now >= next_at:  # read every frame to keep the driver buffer fresh, encode at the cap
                self._publish(frame)
                next_at = now + self.period

    def wait_newer(self, seq: int, timeout: float = 5.0) -> tuple[int, bytes | None]:
        with self._cond:
            self._cond.wait_for(lambda: self._seq != seq or self._stop.is_set() or not self.status["open"],
                                timeout=timeout)
            return self._seq, self._jpeg

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=12)  # a stalled MSMF grab can block ~10 s
        if self._capture is not None:
            self._capture.release()
        with self._cond:
            self._cond.notify_all()


def make_handler(source: FrameSource, token: str | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "DobotCameraStream/1"

        def log_message(self, fmt, *args):  # keep the token out of the console log
            path = urlparse(self.path).path
            print(f"{self.client_address[0]} {self.command} {path} {args[1] if len(args) > 1 else ''}", flush=True)

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Authorization")
            self.send_header("Cache-Control", "no-store")

        def _authorized(self, query) -> bool:
            if not token:
                return True
            header = self.headers.get("Authorization", "")
            given = header[7:] if header.startswith("Bearer ") else (query.get("token") or [""])[0]
            return hmac.compare_digest(given.encode(), token.encode())

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path == "/health":
                body = json.dumps({**source.status, "auth_required": bool(token)}).encode()
                return self._send(200, body, "application/json")
            if url.path not in {"/", "/stream.mjpg", "/snapshot.jpg"}:
                return self._send(404, b'{"error":"not found"}', "application/json")
            if not self._authorized(query):
                return self._send(401, b'{"error":"missing or wrong token"}', "application/json")
            if url.path == "/":
                suffix = f"?token={query['token'][0]}" if "token" in query else ""
                page = (f'<!doctype html><meta charset="utf-8"><title>camera</title>'
                        f'<body style="margin:0;background:#111">'
                        f'<img src="/stream.mjpg{suffix}" style="width:100%;height:auto"></body>').encode()
                return self._send(200, page, "text/html; charset=utf-8")
            if url.path == "/snapshot.jpg":
                _, jpeg = source.wait_newer(-1, timeout=0)
                if jpeg is None:
                    return self._send(503, b'{"error":"no frame"}', "application/json")
                return self._send(200, jpeg, "image/jpeg")
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Connection", "close")
            self.end_headers()
            seq = -1
            try:
                while source.status.get("open"):
                    seq, jpeg = source.wait_newer(seq)
                    if jpeg is None:
                        continue
                    self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                                     f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            self.close_connection = True

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--device", type=int, required=True,
                        help="OpenCV camera index; indices move when USB cameras are replugged "
                             "(2026-11-12: 0 = RealSense wrist, 1 = top camera, 2 = laptop webcam)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--rotate", type=int, choices=sorted(ROTATIONS), default=180,
                        help="clockwise rotation; the top camera is mounted upside down")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--token-env", default="DOBOT_STREAM_TOKEN",
                        help="environment variable holding the access token")
    parser.add_argument("--no-token", action="store_true", help="allow access without a token")
    args = parser.parse_args(argv)
    if not 1 <= args.fps <= 60:
        parser.error("--fps must be 1-60")
    if not 10 <= args.quality <= 100:
        parser.error("--quality must be 10-100")
    token = None if args.no_token else (os.environ.get(args.token_env) or None)
    if token is None and not args.no_token and args.host not in LOOPBACK:
        parser.error(f"non-loopback binding needs a token in ${args.token_env} (or pass --no-token)")

    source = FrameSource(args.device, args.width, args.height, args.rotate, args.fps, args.quality)
    source.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(source, token))
    server.daemon_threads = True
    print(f"camera stream: http://{args.host}:{server.server_port}/stream.mjpg "
          f"device={args.device} size={source.status['actual_wh']} rotate={args.rotate} "
          f"fps<={args.fps:g} auth={'token' if token else 'none'}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        source.stop()
        print("camera stream stopped; camera released", flush=True)


if __name__ == "__main__":
    main()
