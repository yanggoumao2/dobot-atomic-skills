"""Demo mode: every execution request runs one fixed package, whatever was posted.

For a live demonstration where the counterparty's planner is not ready yet: the
partner posts any JSON object to an execution route and this gateway executes the
fixed package instead (by default the 20-node beaker pour). Everything else --
token check, preflight, device lease, isolation, final homing -- is unchanged,
because only the package inside the request is swapped before the normal path.

What is swapped, per route:
- ``POST /v1/execute-graph``: same rule as the real direct push -- with an
  ``Idempotency-Key``/``request_id`` a retry returns the original run; without one
  every call is a new run.
- ``POST /v1/execute-graph/idempotent`` and the two-phase routes: node keys are
  derived from the caller's request key, so a retry with the same key replays the
  same execution and a new key starts a new one. Without any key, the idempotent
  route falls back to a fresh execution; the two-phase routes keep requiring one.

Every received body is written to ``<data-dir>/demo-incoming/`` so the partner's
real graphs are kept for later. Responses are exactly what the normal gateway
returns (no extra field: the partner's client rejected one); the substitution is
logged to the console instead.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from pathlib import Path
from uuid import uuid4

from .contracts import canonical
from .service import Gateway

# Copied from the posted graph onto the executed demo package (see _demo_for).
LINEAGE_FIELDS = ("package_id", "experiment_id", "run_id")


class DemoOverrideGateway(Gateway):
    def __init__(self, *args, demo_package_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_package_path = Path(demo_package_path)
        try:
            with open(self.demo_package_path, encoding="utf-8") as handle:
                package = json.load(handle)
            # Fail at startup, not at the partner's first request.
            self.demo_package = self.catalog.validate(package, check_state=False).package
        except Exception:
            self.close()
            raise
        self.demo_incoming = Path(self.directory) / "demo-incoming"
        self.demo_incoming.mkdir(parents=True, exist_ok=True)
        self._demo_local = threading.local()
        # Our pour nodes/devices do not exist in the partner's graph: pushed events
        # report them as telemetry and keep the node identity in the payload.
        self.delivery.demo_events = True

    # ------------------------------------------------------------------ helpers

    def _record_incoming(self, route, body):
        name = time.strftime("%Y%m%d-%H%M%S") + "-" + route + "-" + uuid4().hex[:8] + ".json"
        path = self.demo_incoming / name
        try:
            text = canonical(body)
        except Exception:
            text = json.dumps(body, ensure_ascii=False, default=repr)
        path.write_text(text, encoding="utf-8")
        return path

    @staticmethod
    def _incoming_package(body):
        if not isinstance(body, dict):
            return {}
        for field in ("graph", "package"):
            if isinstance(body.get(field), dict):
                return body[field]
        return body

    def _demo_for(self, body, request_key=None):
        """The demo package, wearing the posted graph's lineage IDs.

        The partner checks that the receipt's ``graph_id`` equals the ``package_id``
        it sent (and events carry ``graph_id`` too), so the executed copy takes over
        ``package_id``/``experiment_id``/``run_id`` from the posted graph.
        """
        package = copy.deepcopy(self.demo_package)
        incoming = self._incoming_package(body)
        for field in LINEAGE_FIELDS:
            value = incoming.get(field)
            if isinstance(value, str) and value.strip() and len(value) <= 200:
                package[field] = value
        if request_key is not None:
            for node in package["nodes"]:
                if node["node_type"] == "action":
                    seed = (request_key + "\0" + node["node_id"]).encode("utf-8")
                    node["skill_invocation"]["idempotency_key"] = "demo-" + hashlib.sha256(seed).hexdigest()[:40]
        return package

    @staticmethod
    def _request_key(body, header_key):
        request_key = body.get("request_id") if isinstance(body, dict) else None
        key = header_key or request_key
        return key if isinstance(key, str) and key.strip() else None

    def _annotate(self, response, route, recorded):
        # The partner's client rejected an extra response field, so responses stay
        # exactly as the normal gateway returns them. The substitution is only
        # visible here, in the console, and in demo-incoming/.
        print(f"[demo] {route}: received body saved as {recorded.name}; "
              f"executed {self.demo_package_path.name}", flush=True)
        return response

    # ------------------------------------------------------------------ routes

    def submit_new_execution(self, body, key=None):
        if getattr(self._demo_local, "inner", False):
            return super().submit_new_execution(body, key)
        recorded = self._record_incoming("execute-graph", body)
        # Same idempotency rule as the real direct push: a key (header first, else
        # request_id) makes a retry return the original run instead of pouring twice.
        request_key = self._request_key(body, key)
        self._demo_local.inner = True
        try:
            response = super().submit_new_execution({"graph": self._demo_for(body)}, request_key)
        finally:
            self._demo_local.inner = False
        return self._annotate(response, "execute-graph", recorded)

    def submit(self, body, key, two_phase=False):
        # Gateway.submit_new_execution calls submit() itself; that inner call
        # already carries the demo package and must pass through untouched.
        if getattr(self._demo_local, "inner", False):
            return super().submit(body, key, two_phase)
        route = "executions" if two_phase else "idempotent"
        recorded = self._record_incoming(route, body)
        request_key = self._request_key(body, key)
        if request_key is None and not two_phase:
            self._demo_local.inner = True
            try:
                response = super().submit_new_execution({"graph": self._demo_for(body)})
            finally:
                self._demo_local.inner = False
            return self._annotate(response, route, recorded)
        swapped = {"graph": self._demo_for(body, request_key or "")}
        if request_key is not None:
            swapped["request_id"] = request_key
        if isinstance(body, dict) and "validation_token" in body:
            swapped["validation_token"] = body["validation_token"]
        self._demo_local.inner = True
        try:
            response = super().submit(swapped, request_key, two_phase)
        finally:
            self._demo_local.inner = False
        return self._annotate(response, route, recorded)

    def preflight(self, body, key):
        recorded = self._record_incoming("validate", body)
        request_key = self._request_key(body, key)
        swapped = {"graph": self._demo_for(body, request_key or "")}
        if request_key is not None:
            swapped["request_id"] = request_key
        if isinstance(body, dict) and "reserve_resources" in body:
            swapped["reserve_resources"] = body["reserve_resources"]
        response = super().preflight(swapped, request_key)
        return self._annotate(response, "validate", recorded)
