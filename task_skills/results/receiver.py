"""Standalone loopback receiver for the explicitly provisional 3.3 mock API."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import threading
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .common import ENDPOINT_PATH, MAX_REQUEST_BYTES, canonical_json, content_digest, safe_bundle_id, validate_bundle


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class ReceiverServer:
    """Local mock HTTP receiver with durable receipts and idempotent POSTs.

    The listener defaults to loopback. Stored receipt filenames are hashes of
    validated bundle IDs, so incoming IDs are never used as filesystem paths.
    """

    def __init__(
        self,
        data_dir: str | os.PathLike[str],
        host: str = "127.0.0.1",
        port: int = 0,
        token: str | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.receipts_dir = self.data_dir / "receipts"
        self.host = host
        self.port = int(port)
        self.token = token
        self._lock = threading.RLock()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._handler_class = self._make_handler()

    def _receipt_path(self, bundle_id: str) -> Path:
        import hashlib

        safe_bundle_id(bundle_id)
        return self.receipts_dir / (hashlib.sha256(bundle_id.encode("utf-8")).hexdigest() + ".json")

    def _authorized(self, handler: BaseHTTPRequestHandler) -> bool:
        if self.token is None:
            return True
        expected = f"Bearer {self.token}"
        received = handler.headers.get("Authorization", "")
        return hmac.compare_digest(received, expected)

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "ProvisionalResultsReceiver/1"
            sys_version = ""

            def log_message(self, format: str, *args: Any) -> None:
                # The token and submitted data are never written to stdout.
                return

            def _send_json(self, status: int, value: Any) -> None:
                raw = canonical_json(value)
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                path = urllib.parse.urlsplit(self.path).path
                if path in {"/health", "/mock-3.3/v1/health"}:
                    self._send_json(200, {"ok": True, "contract": "provisional-3.3-mock/v1"})
                    return
                if not owner._authorized(self):
                    self._send_json(401, {"error": "unauthorized"})
                    return
                if path.rstrip("/") == ENDPOINT_PATH:
                    items = []
                    for receipt_path in sorted(owner.receipts_dir.glob("*.json")) if owner.receipts_dir.exists() else []:
                        try:
                            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            continue
                        items.append(
                            {
                                "bundle_id": receipt.get("bundle_id"),
                                "content_digest": receipt.get("content_digest"),
                                "accepted_at": receipt.get("accepted_at"),
                            }
                        )
                    self._send_json(200, {"items": items})
                    return
                prefix = ENDPOINT_PATH + "/"
                if path.startswith(prefix):
                    raw_id = urllib.parse.unquote(path[len(prefix):])
                    try:
                        bundle_id = safe_bundle_id(raw_id)
                    except ValueError:
                        self._send_json(400, {"error": "invalid bundle_id"})
                        return
                    receipt_path = owner._receipt_path(bundle_id)
                    try:
                        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    except FileNotFoundError:
                        self._send_json(404, {"error": "receipt not found"})
                        return
                    except (OSError, json.JSONDecodeError):
                        self._send_json(500, {"error": "receipt unavailable"})
                        return
                    self._send_json(200, receipt)
                    return
                self._send_json(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                path = urllib.parse.urlsplit(self.path).path
                if path.rstrip("/") != ENDPOINT_PATH:
                    self._send_json(404, {"error": "not found"})
                    return
                if not owner._authorized(self):
                    self._send_json(401, {"error": "unauthorized"})
                    return
                if self.headers.get_content_type() != "application/json":
                    self._send_json(415, {"error": "content type must be application/json"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", ""))
                except ValueError:
                    self._send_json(400, {"error": "invalid content length"})
                    return
                if length < 0:
                    self._send_json(400, {"error": "invalid content length"})
                    return
                if length > MAX_REQUEST_BYTES:
                    self._send_json(413, {"error": "request exceeds the 24 MiB limit"})
                    return
                raw = self.rfile.read(length)
                if len(raw) != length:
                    self._send_json(400, {"error": "incomplete request body"})
                    return
                try:
                    envelope = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._send_json(400, {"error": "invalid JSON"})
                    return
                if not isinstance(envelope, dict) or not isinstance(envelope.get("bundle"), dict):
                    self._send_json(400, {"error": "request must contain a bundle object"})
                    return
                bundle = envelope["bundle"]
                try:
                    validate_bundle(bundle)
                except (ValueError, TypeError) as exc:
                    self._send_json(400, {"error": f"invalid bundle: {exc}"})
                    return
                expected_digest = content_digest(bundle)
                if envelope.get("content_digest") != expected_digest:
                    self._send_json(400, {"error": "content_digest does not match bundle"})
                    return
                bundle_id = bundle["bundle_id"]
                receipt_path = owner._receipt_path(bundle_id)
                with owner._lock:
                    if receipt_path.exists():
                        try:
                            previous = json.loads(receipt_path.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            self._send_json(500, {"error": "existing receipt unavailable"})
                            return
                        if previous.get("content_digest") != expected_digest:
                            self._send_json(409, {"error": "bundle_id already exists with different content"})
                            return
                        receipt = previous
                    else:
                        accepted_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
                        receipt = {
                            "accepted": True,
                            "bundle_id": bundle_id,
                            "content_digest": expected_digest,
                            "accepted_at": accepted_at,
                            "receipt_url": ENDPOINT_PATH + "/" + urllib.parse.quote(bundle_id, safe=""),
                            "bundle": bundle,
                        }
                        try:
                            _atomic_write_json(receipt_path, receipt)
                        except OSError:
                            self._send_json(500, {"error": "could not persist receipt"})
                            return
                # The ACK fields are repeated on every exact duplicate POST.
                self._send_json(
                    200,
                    {
                        "accepted": True,
                        "bundle_id": receipt["bundle_id"],
                        "content_digest": receipt["content_digest"],
                        "receipt_url": receipt["receipt_url"],
                    },
                )

        return Handler

    def start(self) -> "ReceiverServer":
        if self._httpd is not None:
            return self
        self._httpd = self._create_httpd()
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="results-mock-receiver", daemon=True)
        self._thread.start()
        return self

    def _create_httpd(self) -> ThreadingHTTPServer:
        httpd = ThreadingHTTPServer((self.host, self.port), self._handler_class)
        httpd.daemon_threads = True
        httpd.owner = self  # type: ignore[attr-defined]
        return httpd

    @property
    def address(self) -> tuple[str, int]:
        if self._httpd is None:
            return self.host, self.port
        host, port = self._httpd.server_address[:2]
        return str(host), int(port)

    @property
    def url(self) -> str:
        host, port = self.address
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{port}"

    def serve_forever(self) -> None:
        # Run one listener loop in the background and block the caller on its
        # thread. Ctrl+C interrupts the caller, which can then stop the loop
        # from a different thread without deadlocking BaseServer.shutdown().
        self.start()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def stop(self) -> None:
        httpd = self._httpd
        if httpd is None:
            return
        httpd.shutdown()
        httpd.server_close()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._httpd = None
        self._thread = None

    def __enter__(self) -> "ReceiverServer":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the provisional local result receiver (not an official 3.3 API).")
    parser.add_argument("--host", default="127.0.0.1", help="listener address; defaults to loopback")
    parser.add_argument("--port", type=int, default=8333)
    parser.add_argument("--data-dir", required=True, help="directory for durable mock receipts")
    parser.add_argument("--token", help="optional bearer token")
    args = parser.parse_args(argv)
    server = ReceiverServer(args.data_dir, host=args.host, port=args.port, token=args.token)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
