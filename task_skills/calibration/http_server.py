"""Loopback HTTP server and local web UI for calibration."""

from __future__ import annotations

import json
import mimetypes
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .camera import CameraManager
from .service import CalibrationService, IMAGE_CONTENT_TYPES
from .state import CalibrationError, SessionStore, validate_settings

MAX_JSON_BODY = 2 * 1024 * 1024
MAX_IMAGE_BODY = 32 * 1024 * 1024
STATIC_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


class CalibrationHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, service: CalibrationService, camera: CameraManager,
                 web_root: Path, *, allow_remote_bind: bool = False):
        self.service = service
        self.camera = camera
        self.web_root = web_root.resolve()
        self.csrf_token = secrets.token_urlsafe(32)
        self.allow_remote_bind = allow_remote_bind
        self._write_slots = threading.BoundedSemaphore(12)
        super().__init__(address, CalibrationRequestHandler)

    @property
    def allowed_hosts(self) -> set[str]:
        host = self.server_address[0]
        allowed = {"127.0.0.1", "localhost", "::1"}
        allowed.add(str(host))
        if self.allow_remote_bind:
            allowed.add(str(host))
        return allowed

    def shutdown(self):
        super().shutdown()


class CalibrationRequestHandler(BaseHTTPRequestHandler):
    server: CalibrationHTTPServer
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def do_PUT(self):
        self._handle("PUT")

    def do_PATCH(self):
        self._handle("PATCH")

    def do_DELETE(self):
        self._handle("DELETE")

    def do_OPTIONS(self):
        self._send_json(405, {"ok": False, "code": "METHOD_NOT_ALLOWED",
                              "message": "不支持跨源预检"})

    def log_message(self, fmt, *args):
        print(f"[calibration-ui] {self.address_string()} {fmt % args}", flush=True)

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; "
                         "script-src 'self'; connect-src 'self'; object-src 'none'; "
                         "base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")

    def _send_json(self, status: int, value: Any) -> None:
        data = json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self._security_headers()
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def _send_bytes(self, status: int, content_type: str, data: bytes,
                    extra_headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self._security_headers()
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def _send_error(self, error: CalibrationError) -> None:
        self._send_json(error.status, {
            "ok": False,
            "code": error.code,
            "message": error.message,
        })

    def _check_origin_and_host(self) -> None:
        host_header = self.headers.get("Host", "")
        host_name = host_header.rsplit(":", 1)[0].strip("[]").lower() if host_header else ""
        if not host_name or host_name not in {item.lower() for item in self.server.allowed_hosts}:
            raise CalibrationError("HOST_REJECTED", "Host 头不在允许的本地地址中", status=403)
        origin = self.headers.get("Origin")
        if origin:
            port = self.server.server_address[1]
            allowed = {
                f"http://127.0.0.1:{port}", f"http://localhost:{port}",
                f"http://[::1]:{port}",
            }
            if origin.rstrip("/") not in {item.rstrip("/") for item in allowed}:
                raise CalibrationError("CROSS_ORIGIN_REJECTED", "拒绝跨源请求", status=403)
        fetch_site = self.headers.get("Sec-Fetch-Site")
        if fetch_site and fetch_site not in ("same-origin", "none"):
            raise CalibrationError("CROSS_ORIGIN_REJECTED", "拒绝跨站请求", status=403)

    def _check_write_token(self) -> None:
        provided = self.headers.get("X-Calibration-Token", "")
        if not provided or not secrets.compare_digest(provided, self.server.csrf_token):
            raise CalibrationError("CSRF_TOKEN_INVALID", "缺少或错误的本机写操作令牌", status=403)

    def _content_length(self, maximum: int) -> int:
        if self.headers.get("Transfer-Encoding"):
            raise CalibrationError("CHUNKED_NOT_SUPPORTED", "不支持分块请求体", status=400)
        values = self.headers.get_all("Content-Length", [])
        if len(values) != 1:
            raise CalibrationError("CONTENT_LENGTH_REQUIRED", "必须提供唯一 Content-Length", status=400)
        try:
            length = int(values[0])
        except ValueError as error:
            raise CalibrationError("INVALID_CONTENT_LENGTH", "Content-Length 不合法") from error
        if length < 1 or length > maximum:
            raise CalibrationError("REQUEST_TOO_LARGE", f"请求体必须在 1–{maximum} 字节", status=413)
        return length

    def _raw_body(self, maximum: int) -> bytes:
        length = self._content_length(maximum)
        data = self.rfile.read(length)
        if len(data) != length:
            raise CalibrationError("INCOMPLETE_BODY", "请求体不完整")
        return data

    def _json_body(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise CalibrationError("INVALID_CONTENT_TYPE", "此接口需要 application/json", status=415)
        raw = self._raw_body(MAX_JSON_BODY)
        try:
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                               parse_constant=lambda token: (_ for _ in ()).throw(
                                   ValueError(f"nonfinite JSON: {token}")))
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            raise CalibrationError("INVALID_JSON", "JSON 不合法或包含非有限数") from error
        if not isinstance(value, dict):
            raise CalibrationError("INVALID_JSON", "请求体必须是 JSON 对象")
        return value

    def _image_body(self) -> tuple[bytes, str]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type not in IMAGE_CONTENT_TYPES:
            raise CalibrationError("UNSUPPORTED_IMAGE_TYPE", "上传必须使用 image/png 或 image/jpeg",
                                   status=415)
        return self._raw_body(MAX_IMAGE_BODY), content_type

    def _handle(self, method: str) -> None:
        try:
            self._check_origin_and_host()
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            query = parse_qs(parsed.query)
            if method not in ("GET", "HEAD") and path.startswith("/api/"):
                self._check_write_token()
                if not self.server._write_slots.acquire(blocking=False):
                    raise CalibrationError("SERVER_BUSY", "并发写请求过多", status=429)
                try:
                    self._route(method, path, query)
                finally:
                    self.server._write_slots.release()
            else:
                self._route(method, path, query)
        except CalibrationError as error:
            self._send_error(error)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            import traceback
            traceback.print_exc()
            try:
                self._send_json(500, {"ok": False, "code": "INTERNAL_ERROR",
                                      "message": "服务器内部错误；详情仅在本地终端日志"})
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _route(self, method: str, path: str, query: dict[str, list[str]]) -> None:
        if method == "GET" and path in ("/", "/index.html"):
            return self._serve_static("index.html")
        if method == "GET" and path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):])
        if method == "GET" and path == "/api/bootstrap":
            return self._send_json(200, {
                "ok": True,
                "csrf_token": self.server.csrf_token,
                "dependencies": self.server.service.dependencies(),
                "sessions": self.server.service.list_sessions(),
                "limits": {
                    "json_bytes": MAX_JSON_BODY,
                    "image_bytes": MAX_IMAGE_BODY,
                    "max_workers": 2,
                },
                "camera": self.server.camera.status(),
            })
        if path == "/api/camera/status" and method == "GET":
            return self._send_json(200, {"ok": True, "camera": self.server.camera.status()})
        if path == "/api/camera/frame" and method == "GET":
            data = self.server.camera.frame_jpeg()
            return self._send_bytes(200, "image/jpeg", data)
        if method == "GET" and path == "/api/sessions":
            return self._send_json(200, {"ok": True, "sessions": self.server.service.list_sessions()})
        if method == "POST" and path == "/api/sessions":
            body = self._json_body()
            return self._send_json(201, {"ok": True, "session": self.server.service.create_session(body)})
        if method == "POST" and path == "/api/camera/start":
            body = self._json_body()
            session_id = body.get("session_id")
            state = self.server.service.store.read(session_id)
            width, height = state["settings"]["image_size_wh"]
            requested_width = body.get("width", width)
            requested_height = body.get("height", height)
            if [requested_width, requested_height] != [width, height]:
                raise CalibrationError("CAMERA_GEOMETRY_MISMATCH",
                                       f"相机预览必须使用任务设置的 {width}×{height}", status=409)
            status = self.server.camera.start(body.get("device", 0), width, height)
            return self._send_json(200, {"ok": True, "camera": status})
        if method == "POST" and path == "/api/camera/stop":
            return self._send_json(200, {"ok": True, "camera": self.server.camera.stop()})
        if method == "POST" and path == "/api/camera/capture":
            body = self._json_body()
            session_id = body.get("session_id")
            target = body.get("target")
            try:
                data, _size = self.server.camera.capture_png()
                if target == "intrinsics":
                    split = body.get("split")
                    record = self.server.service.add_intrinsics_image(
                        session_id, data, "image/png", split,
                        body.get("filename", "camera_capture.png"))
                    return self._send_json(201, {"ok": True, "image": record})
                if target == "board":
                    result = self.server.service.upload_board_image(
                        session_id, data, "image/png", body.get("filename", "fixed_board.png"))
                    return self._send_json(201, {"ok": True, **result})
                raise CalibrationError("INVALID_CAPTURE_TARGET", "采集目标只能是 intrinsics 或 board")
            except Exception:
                self.server.camera.stop()
                raise

        prefix = self._session_prefix(path)
        if prefix is None:
            raise CalibrationError("NOT_FOUND", "未知接口", status=404)
        session_id, tail = prefix
        if method == "GET" and tail == "":
            return self._send_json(200, {"ok": True, "session": self.server.service.session_view(session_id)})
        if method == "PATCH" and tail == "/settings":
            return self._send_json(200, {"ok": True,
                                         "session": self.server.service.update_settings(session_id, self._json_body())})
        if method == "POST" and tail == "/intrinsics/images":
            split = query.get("split", [""])[0]
            filename = query.get("filename", ["camera.png"])[0]
            data, content_type = self._image_body()
            record = self.server.service.add_intrinsics_image(session_id, data, content_type,
                                                              split, filename)
            return self._send_json(201, {"ok": True, "image": record})
        if method == "POST" and tail == "/intrinsics/images/batch":
            return self._send_json(200, {"ok": True, **self.server.service.batch_intrinsics_images(
                session_id, self._json_body())})
        if tail.startswith("/intrinsics/images/") and method in ("PATCH", "DELETE"):
            image_id = tail[len("/intrinsics/images/"):]
            if "/" in image_id or not image_id:
                raise CalibrationError("NOT_FOUND", "图像接口不存在", status=404)
            if method == "DELETE":
                self.server.service.delete_intrinsics_image(session_id, image_id)
                return self._send_json(200, {"ok": True})
            body = self._json_body()
            return self._send_json(200, {"ok": True, "image": self.server.service.update_intrinsics_split(
                session_id, image_id, body.get("split"))})
        if method == "POST" and tail == "/intrinsics/run":
            self._json_body()
            return self._send_json(202, {"ok": True, "job_id": self.server.service.run_intrinsics(session_id)})
        if method == "POST" and tail == "/board/image":
            filename = query.get("filename", ["fixed_board.png"])[0]
            data, content_type = self._image_body()
            return self._send_json(201, {"ok": True, **self.server.service.upload_board_image(
                session_id, data, content_type, filename)})
        if method == "POST" and tail == "/board/detect":
            self._json_body()
            return self._send_json(200, {"ok": True, **self.server.service.detect_board(session_id)})
        if method == "POST" and tail == "/board/confirm":
            return self._send_json(200, {"ok": True, **self.server.service.confirm_board(
                session_id, self._json_body())})
        if tail.startswith("/arms/"):
            parts = [part for part in tail[len("/arms/"):].split("/") if part]
            if len(parts) < 2:
                raise CalibrationError("NOT_FOUND", "机械臂接口不存在", status=404)
            arm, action = parts[0], "/".join(parts[1:])
            if action == "metadata" and method == "PUT":
                return self._send_json(200, {"ok": True, "arm": self.server.service.update_arm_metadata(
                    session_id, arm, self._json_body())})
            if action == "points" and method == "PUT":
                return self._send_json(200, {"ok": True, "arm": self.server.service.update_teach_points(
                    session_id, arm, self._json_body())})
            if action == "teach/import" and method == "POST":
                return self._send_json(200, {"ok": True, "arm": self.server.service.import_teach_csv(
                    session_id, arm, self._json_body())})
            if action == "teach/export" and method == "GET":
                text = self.server.service.export_teach_csv(session_id, arm)
                return self._send_bytes(200, "text/csv; charset=utf-8", text.encode("utf-8-sig"), {
                    "Content-Disposition": f'attachment; filename="{session_id}-{arm}-teach.csv"'
                })
            if action == "get-pose" and method == "POST":
                return self._send_json(200, {"ok": True, **self.server.service.get_pose(
                    session_id, arm, self._json_body())})
            if action == "fit" and method == "POST":
                self._json_body()
                return self._send_json(202, {"ok": True, "job_id": self.server.service.run_fit(
                    session_id, arm)})
            if action == "verification" and method == "PUT":
                return self._send_json(200, {"ok": True, "verification": self.server.service.update_verification_draft(
                    session_id, arm, self._json_body())})
            if action == "verification/import" and method == "POST":
                return self._send_json(200, {"ok": True, "verification": self.server.service.import_verify_csv(
                    session_id, arm, self._json_body())})
            if action == "verification/export" and method == "GET":
                text = self.server.service.export_verify_csv(session_id, arm)
                return self._send_bytes(200, "text/csv; charset=utf-8", text.encode("utf-8-sig"), {
                    "Content-Disposition": f'attachment; filename="{session_id}-{arm}-verification.csv"'
                })
            if action == "verify" and method == "POST":
                self._json_body()
                return self._send_json(202, {"ok": True, "job_id": self.server.service.run_verification(
                    session_id, arm)})
        if method == "GET" and tail.startswith("/jobs/"):
            suffix = tail[len("/jobs/"):].split("/")
            job_id = suffix[0]
            state = self.server.service.session_view(session_id)
            job = next((item for item in state["jobs"] if item["id"] == job_id), None)
            if not job:
                raise CalibrationError("JOB_NOT_FOUND", "后台任务不存在", status=404)
            if len(suffix) == 2 and suffix[1] == "log":
                offset = int(query.get("offset", ["0"])[0])
                text, end = self.server.service.job_log(session_id, job_id, offset)
                return self._send_json(200, {"ok": True, "job": job, "log": text, "next_offset": end})
            if len(suffix) == 1:
                return self._send_json(200, {"ok": True, "job": job})
        if method == "POST" and tail == "/query":
            return self._send_json(200, {"ok": True, **self.server.service.query(
                session_id, self._json_body())})
        if method == "POST" and tail == "/depth-confirmation":
            return self._send_json(200, {"ok": True, "depth_registration": self.server.service.confirm_depth(
                session_id, self._json_body())})
        if method == "POST" and tail == "/exports":
            return self._send_json(201, {"ok": True, "export": self.server.service.create_export(
                session_id, self._json_body())})
        if method == "GET" and tail.startswith("/assets/"):
            relative = tail[len("/assets/"):]
            path = self.server.service.asset(session_id, relative)
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix.lower() == ".zip":
                content_type = "application/zip"
            disposition = "attachment" if path.suffix.lower() in (".zip", ".csv") else "inline"
            return self._send_bytes(200, content_type, path.read_bytes(), {
                "Content-Disposition": f'{disposition}; filename="{path.name}"'
            })
        raise CalibrationError("NOT_FOUND", "未知接口", status=404)

    def _session_prefix(self, path: str) -> tuple[str, str] | None:
        marker = "/api/sessions/"
        if not path.startswith(marker):
            return None
        remainder = path[len(marker):]
        parts = remainder.split("/", 1)
        if not parts[0]:
            return None
        return parts[0], ("/" + parts[1] if len(parts) > 1 else "")

    def _serve_static(self, relative: str) -> None:
        if not relative or relative.startswith(".") or "\\" in relative:
            raise CalibrationError("STATIC_NOT_FOUND", "静态文件不存在", status=404)
        path = (self.server.web_root / relative).resolve()
        try:
            path.relative_to(self.server.web_root)
        except ValueError as error:
            raise CalibrationError("STATIC_NOT_FOUND", "静态文件不存在", status=404) from error
        if not path.is_file():
            raise CalibrationError("STATIC_NOT_FOUND", "静态文件不存在", status=404)
        suffix = path.suffix.lower()
        if suffix not in STATIC_CONTENT_TYPES:
            raise CalibrationError("STATIC_NOT_FOUND", "静态文件类型不允许", status=404)
        return self._send_bytes(200, STATIC_CONTENT_TYPES[suffix], path.read_bytes())


def create_server(host: str, port: int, data_dir: Path, web_root: Path, *,
                  allow_remote_bind: bool = False, fit_hook=None) -> CalibrationHTTPServer:
    store = SessionStore(data_dir)
    service = CalibrationService(store, fit_hook=fit_hook)
    camera = CameraManager()
    return CalibrationHTTPServer((host, port), service, camera, web_root,
                                 allow_remote_bind=allow_remote_bind)
