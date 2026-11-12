"""Bounded standard-library HTTP control plane for the partner's draft contract."""

import json
import logging
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

from .contracts import ContractError, canonical


MAX_BODY = 2 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


class GatewayServer(ThreadingHTTPServer):
    allow_reuse_address = True
    request_queue_size = 16

    def __init__(self, address, gateway, token=None):
        self.gateway, self.token = gateway, token
        self.slots = threading.BoundedSemaphore(16)
        super().__init__(address, GatewayHandler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def service_actions(self):
        self.gateway.maintain()


class GatewayHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *_):
        # Do not log authorization headers, packages or caller-chosen IDs.
        pass

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def _handle(self, method):
        try:
            if self.server.token is not None:
                provided = self.headers.get("Authorization", "").encode("utf-8")
                expected = ("Bearer " + self.server.token).encode("utf-8")
                if not secrets.compare_digest(provided, expected):
                    raise ContractError("UNAUTHORIZED", "A valid bearer token is required", status=401)
            url = urlsplit(self.path)
            path = unquote(url.path)
            body = self._body() if method == "POST" else None
            status, value = self._route(method, path, parse_qs(url.query), body)
        except ContractError as error:
            status, value = error.status, error.body
        except (ValueError, TypeError, KeyError, RecursionError) as error:
            status, value = 400, ContractError("INVALID_REQUEST", "Invalid JSON or field shape", status=400).body
        except Exception:
            LOGGER.exception("Gateway request failed")
            status, value = 500, ContractError("INTERNAL_ERROR", "Gateway request failed", status=500).body
        try:
            data = canonical(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            # A disconnected client can retry the persisted idempotency key.
            pass
        self.close_connection = True

    def _body(self):
        if self.headers.get("Transfer-Encoding"):
            raise ContractError("INVALID_REQUEST", "Chunked bodies are not supported", status=400)
        if len(self.headers.get_all("Content-Length", [])) != 1:
            raise ContractError("INVALID_REQUEST", "One Content-Length header is required", status=400)
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            raise ContractError("INVALID_CONTENT_TYPE", "Use application/json", status=415)
        try:
            size = int(self.headers["Content-Length"])
        except ValueError as error:
            raise ContractError("INVALID_REQUEST", "Invalid Content-Length", status=400) from error
        if size < 1 or size > MAX_BODY:
            raise ContractError("REQUEST_TOO_LARGE", "Expected body size 1..2 MiB", status=413)
        raw = self.rfile.read(size)
        if len(raw) != size:
            raise ContractError("INVALID_REQUEST", "Incomplete request body", status=400)
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))
        if not isinstance(value, dict):
            raise ContractError("INVALID_REQUEST", "Request must be an object", status=400)
        return value

    def _route(self, method, path, query, body):
        gateway = self.server.gateway
        if method == "GET" and path == "/health":
            return 200, {"status": "ok", "mode": gateway.mode, "catalog_revision": gateway.catalog.revision,
                         "active_execution_id": gateway.active, "quarantined": bool(gateway.store.quarantine_reason())}
        if method == "GET" and path in {"/v1/catalog", "/v1/capabilities"}:
            return 200, gateway.catalog.snapshot()
        if method == "GET" and path == "/v1/delivery-status":
            return 200, {**gateway.store.delivery_status(), "upstream_configured": gateway.upstream is not None}
        if method == "GET" and path == "/v1/results-delivery-status":
            return 200, gateway.result_delivery_status()
        key = self.headers.get("Idempotency-Key")
        if method == "POST" and path == "/device-gateway/v1/execution-plans/validate":
            return 200, gateway.preflight(body, key)
        if method == "POST" and path == "/v1/execute-graph":
            return 202, gateway.submit_new_execution(body)
        if method == "POST" and path in {"/v1/execute-graph/idempotent", "/device-gateway/v1/executions"}:
            return 202, gateway.submit(body, key, two_phase=path.startswith("/device-gateway/"))
        for prefix in ("/v1/executions/", "/device-gateway/v1/executions/"):
            if path.startswith(prefix):
                suffix = path[len(prefix):].split("/")
                execution_id = suffix[0]
                if len(suffix) == 1 and method == "GET":
                    return 200, gateway.status(execution_id)
                if len(suffix) == 2 and suffix[1] == "events" and method == "GET":
                    after, limit = int(query.get("after", ["0"])[0]), int(query.get("limit", ["100"])[0])
                    if not 0 <= after <= 2 ** 63 - 1 or not 1 <= limit <= 1000:
                        raise ContractError("INVALID_REQUEST", "Invalid event pagination", status=400)
                    return 200, {"events": gateway.store.events(execution_id, after, limit)}
                if len(suffix) == 2 and suffix[1] == "cancel" and method == "POST":
                    return 200, gateway.cancel(execution_id)
        raise ContractError("NOT_FOUND", "Unknown endpoint", status=404)
