"""Configured compiler transport; request bodies cannot choose callback destinations."""

import json
import threading
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ..graph import utc_now
from .contracts import ContractError, canonical
from .partner_wire import to_partner


SNAPSHOT_PATH = "/v1/integrations/catalog/snapshots"


def _is_event_path(path):
    return path == "/v1/integrations/device-state-events" or (
        path.startswith("/v1/executions/") and path.endswith("/events"))


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Upstream:
    def __init__(self, base_url, token=None, timeout=3.0, allow_http=False):
        parts = urlsplit(base_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("Expected a compiler base URL without credentials, query or fragment")
        if parts.scheme == "http" and parts.hostname not in {"127.0.0.1", "localhost", "::1"} and not allow_http:
            raise ValueError("Remote compiler transport requires HTTPS")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.opener = build_opener(NoRedirects())

    def request(self, path, body=None):
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = canonical(body).encode("utf-8")
            if body.get("event_id"):
                headers["Idempotency-Key"] = body["event_id"]
        request = Request(self.base_url + path, data=data, headers=headers, method="POST" if data is not None else "GET")
        with self.opener.open(request, timeout=self.timeout) as response:
            payload = response.read(2 * 1024 * 1024 + 1)
            if len(payload) > 2 * 1024 * 1024:
                raise ValueError("Compiler response exceeds size limit")
            return json.loads(payload) if payload else {}

    def fetch_plan(self, run_id):
        result = self.request("/v1/integrations/execution-plans/" + quote(run_id, safe=""))
        if not isinstance(result, dict):
            raise ContractError("UPSTREAM_INVALID_RESPONSE", "Compiler returned no package", status=502)
        return result.get("package", result)


class DeliveryWorker:
    def __init__(self, store, upstream, interval=0.5, publish_snapshots=False):
        self.store, self.upstream, self.interval = store, upstream, interval
        self.publish_snapshots = publish_snapshots
        self.demo_events = False  # set by DemoOverrideGateway
        self.stop = threading.Event()
        self.thread = None
        self.lock = threading.Lock()

    def publish_catalog(self, catalog):
        snapshot = catalog.snapshot()
        snapshot["generated_at"] = utc_now()
        revision = snapshot["catalog_revision"]
        stages = [("validate", "/v1/integrations/catalog/validate")]
        if self.publish_snapshots:
            stages.append(("snapshots", SNAPSHOT_PATH))
        for stage, path in stages:
            self.store.enqueue("catalog:" + revision, path, snapshot, revision + ":" + stage)

    def pump(self):
        if self.upstream is None:
            return 0
        delivered = 0
        with self.lock:
            for item in self.store.pending_deliveries():
                if self.stop.is_set():
                    break
                if item["path"] == SNAPSHOT_PATH and not self.publish_snapshots:
                    # Queued by an older run; never replace the partner's catalog unasked.
                    continue
                try:
                    result = self.upstream.request(item["path"], to_partner(item["path"], json.loads(item["body"]),
                                                                            demo=self.demo_events))
                    if not isinstance(result, dict) or ("accepted" in result and result["accepted"] is not True) or "code" in result:
                        self.store.delivery_result(item["id"], False, "UPSTREAM_REJECTED")
                        continue
                    diagnostics = result.get("diagnostics", [])
                    if not isinstance(diagnostics, list) or any(not isinstance(d, dict) or d.get("severity") == "blocking" for d in diagnostics):
                        self.store.delivery_result(item["id"], False, "UPSTREAM_BLOCKING_DIAGNOSTIC")
                        continue
                except HTTPError as error:
                    if error.code == 409 and _is_event_path(item["path"]):
                        # event_id is the spec's dedupe key: 409 on an event means the
                        # receiver already has it (a retry after a lost acknowledgement).
                        self.store.delivery_result(item["id"], True)
                        delivered += 1
                        continue
                    # 404 can occur until dispatch registers the returned execution_id.
                    self.store.delivery_result(item["id"], False, "HTTP_" + str(error.code))
                except (URLError, TimeoutError, ValueError, OSError, RecursionError, HTTPException) as error:
                    self.store.delivery_result(item["id"], False, type(error).__name__)
                else:
                    self.store.delivery_result(item["id"], True)
                    delivered += 1
        return delivered

    def start(self):
        if self.upstream is not None:
            self.thread = threading.Thread(target=self._run, name="dobot-event-delivery")
            self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            self.pump()
            self.stop.wait(self.interval)

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join()
