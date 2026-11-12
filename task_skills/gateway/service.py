"""One execution owner behind both single-push and two-phase protocols."""

import copy
import json
import os
import threading
import time
from pathlib import Path
from uuid import uuid4

from ..device_lease import DeviceLease
from ..run_graph import MockAdapter, run_graph
from .contracts import Catalog, ContractError, identifier
from .store import Store, TERMINAL
from .upstream import DeliveryWorker


ENVELOPE_KEYS = {"graph", "package", "request_id"}


def bare_package(body):
    """Return ``body`` itself when the body *is* an execution package.

    The counterparty serializes the graph document and posts that document. Both
    shapes are therefore accepted: the ``{"graph": ...}`` envelope this gateway
    documents, and the package as the whole body. A body carrying both envelope
    and package keys is ambiguous and stays rejected.
    """
    if not isinstance(body, dict) or not ENVELOPE_KEYS.isdisjoint(body):
        return None
    if "nodes" in body and ("schema_version" in body or "package_id" in body):
        return body
    return None


class Gateway:
    def __init__(self, scene, directory, mode="mock", allow_live=False, upstream=None,
                 adapter_factory=None, lease_path=None, reservation_seconds=60,
                 results_delivery=None):
        if mode not in {"mock", "live"}:
            raise ValueError("Mode must be mock or live")
        if mode == "live" and (not allow_live or os.name != "nt"):
            raise ValueError("Live execution requires explicit --allow-live and Windows hardware environment")
        self.scene, self.directory, self.mode = scene, Path(directory), mode
        self.catalog = Catalog(scene, mode)
        self.store = Store(directory)
        try:
            self.store.bind_mode(mode)
        except Exception:
            self.store.close()
            raise
        self.lock = threading.RLock()
        self.active = None
        self.cancel_event = None
        self.worker = None
        self._execution_workers = []
        self.reservation_token = None
        self.reservation_until = 0
        self.reservation_seconds = reservation_seconds
        self.lease_path, self.lease = lease_path, None
        self.upstream, self.adapter_factory = upstream, adapter_factory
        self.results_delivery = results_delivery
        self._result_capture_lock = threading.Lock()
        self._result_capture_errors = {}
        self.closed = False
        self.store.recover(mode == "live")
        self.delivery = DeliveryWorker(self.store, upstream)
        self.delivery.publish_catalog(self.catalog)
        for device in self.catalog.snapshot()["devices"]:
            self.store.device_event(device["device_id"], "idle" if mode == "mock" else "offline", mode=mode)
        self.delivery.start()
        if self.results_delivery is not None:
            self._recover_result_captures()
            self.results_delivery.start()

    def _capture_result(self, execution_id):
        """Copy committed execution evidence to the independent result queue."""
        with self._result_capture_lock:
            self._capture_result_locked(execution_id)

    def _capture_result_locked(self, execution_id):
        record = self.store.get(execution_id)
        if record["status"] not in TERMINAL:
            return
        events, after = [], 0
        while True:
            page = self.store.events(execution_id, after=after, limit=1000)
            if not page:
                break
            events.extend(page)
            after = page[-1]["sequence"]
        terminal = next((e for e in reversed(events)
                         if e["event_type"] == "execution_" + record["status"]), None)
        result = dict(terminal["payload"]) if terminal else {}
        result["status"] = record["status"]
        if record["error"] is not None:
            result["error"] = record["error"]
        directory = self.directory / "executions" / execution_id
        self.results_delivery.capture(execution_id, record["package"], result,
                                      events, directory, self.mode)
        with self.store.transaction() as db:
            db.execute("DELETE FROM metadata WHERE name=?", ("results-capture:" + execution_id,))
        self._result_capture_errors.pop(execution_id, None)

    def _pending_result_captures(self):
        with self.store.lock:
            rows = self.store.db.execute(
                "SELECT name FROM metadata WHERE name LIKE 'results-capture:%'").fetchall()
        return sorted(set(row[0].split(":", 1)[1] for row in rows)
                      | set(self._result_capture_errors))

    def _recover_result_captures(self):
        # Only executions explicitly enrolled in this channel are recovered.
        # Enabling it must not silently publish historical experiments.
        for execution_id in self._pending_result_captures():
            try:
                self._capture_result(execution_id)
            except Exception as error:
                self._result_capture_errors[execution_id] = type(error).__name__

    def result_delivery_status(self):
        if self.results_delivery is None:
            return {"configured": False}
        return {"configured": True, "capture_pending": self._pending_result_captures(),
                "capture_errors": dict(self._result_capture_errors),
                "delivery": self.results_delivery.status()}

    def _plan(self, body):
        if not isinstance(body, dict):
            raise ContractError("INVALID_REQUEST", "Request must be a JSON object", status=400)
        if bare_package(body) is not None:
            # Posted as the graph document itself; wrap it here so the rest of the
            # pipeline only ever sees one shape.
            body = {"package": body}
        if body.get("resume_from_node_id") is not None or body.get("execution_id") is not None:
            raise ContractError("UNSUPPORTED_RESUME", "This profile accepts fresh graphs only")
        if "mode" in body:
            raise ContractError("INVALID_REQUEST", "Execution mode is configured on the server")
        if "graph" in body and "package" in body:
            raise ContractError("INVALID_REQUEST", "Supply graph or package, not both")
        package = body.get("graph", body.get("package"))
        if package is None and body.get("run_id") is not None:
            run_id = identifier(body["run_id"], "run_id")
            if self.upstream is None:
                raise ContractError("MISSING_EXECUTION_PACKAGE", "Supply complete graph/package, or configure the compiler URL")
            try:
                package = self.upstream.fetch_plan(run_id)
            except ContractError:
                raise
            except Exception as error:
                raise ContractError("UPSTREAM_UNAVAILABLE", "Unable to fetch execution package", status=502) from error
        if package is None:
            raise ContractError("MISSING_EXECUTION_PACKAGE", "A complete execution package is required")
        if body.get("run_id") is not None:
            identifier(body["run_id"], "run_id")
            if isinstance(package, dict) and package.get("run_id", body["run_id"]) != body["run_id"]:
                raise ContractError("LINEAGE_MISMATCH", "Package run_id differs from request run_id")
        return self.catalog.validate(package, check_state=False)

    @staticmethod
    def _key(body, header_key):
        request_key = body.get("request_id") if isinstance(body, dict) else None
        if header_key and request_key and header_key != request_key:
            raise ContractError("IDEMPOTENCY_CONFLICT", "Header and request_id disagree", status=409)
        return identifier(header_key or request_key, "Idempotency-Key")

    def _available(self):
        if self.closed:
            raise ContractError("GATEWAY_CLOSED", "Gateway is shutting down", status=503)
        reason = self.store.quarantine_reason()
        if reason:
            raise ContractError("STOP_UNCONFIRMED", "Rig is quarantined; onsite verification required", status=409)
        if self.active is not None:
            raise ContractError("RESOURCE_CONFLICT", "Another execution owns the rig", status=409)
        if self.reservation_token and self.reservation_until <= time.time():
            self.reservation_token = None
            self._release_lease()

    def _acquire_lease(self, family="embodied"):
        if family == "chemistry":
            # The dobot.* atoms run through the chemistry driver, which takes the
            # same rig lease (same DOBOT_DEVICE_LOCK_PATH) in the session its first
            # atom opens. Taking the lease here as well would make every chemistry
            # execution collide with itself. Ownership is still exclusive; it just
            # belongs to the driver for this execution. A legacy CLI or ZMQ process
            # holding the directory is reported here, before anything runs.
            # Mock stays filesystem-independent: a stale live lease must not decide
            # whether a simulated run is allowed.
            if self.mode == "live" and DeviceLease(self.lease_path).path.exists():
                raise ContractError("RESOURCE_CONFLICT", "Legacy control or quarantine owns the rig", status=409)
            return
        if self.mode == "live" and self.lease is None:
            lease = DeviceLease(self.lease_path)
            try:
                lease.acquire()
            except RuntimeError as error:
                raise ContractError("RESOURCE_CONFLICT", "Legacy control or quarantine owns the rig", status=409) from error
            try:
                shared_scene = json.loads(lease.scene_stamp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                shared_scene = None
            if (lease.scene_marker.exists() or self.store.verified_scene_revision() != self.catalog.revision
                    or shared_scene != self.store.scene_identity()):
                lease.release()
                raise ContractError("SCENE_RESET_REQUIRED", "Confirm onsite initial object/gripper state with the operator reset tool", status=409)
            self.lease = lease

    def _release_lease(self):
        if self.lease:
            self.lease.release()
            self.lease = None

    def preflight(self, body, key):
        plan = self._plan(body)
        key = self._key(body, key)
        reserve = body.get("reserve_resources", True)
        if type(reserve) is not bool:
            raise ContractError("INVALID_REQUEST", "reserve_resources must be boolean")
        with self.lock:
            old = self.store.reservation_by_key(key)
            if old:
                old_response = json.loads(old["response"])
                if old["digest"] != plan.fingerprint:
                    raise ContractError("IDEMPOTENCY_CONFLICT", "Preflight key used for another package", status=409)
                if old["expires"] <= time.time() and not old["execution_id"]:
                    raise ContractError("VALIDATION_TOKEN_EXPIRED", "Use a new preflight key", status=409)
                if not old["execution_id"] and old["token"] != self.reservation_token:
                    raise ContractError("RESERVATION_EXPIRED", "Preflight reservation was lost; use a new key", status=409)
                if old_response.get("reserve_resources", True) != reserve:
                    raise ContractError("IDEMPOTENCY_CONFLICT", "reserve_resources differs from original preflight", status=409)
                return old_response
            self._available()
            if self.reservation_token:
                raise ContractError("RESOURCE_CONFLICT", "Another preflight reserved the rig", status=409)
            plan = self.catalog.validate(plan.package, self.store.object_state())
            if not reserve:
                return {"accepted": True, "validation_token": None, "catalog_revision": self.catalog.revision,
                        "reservations": [], "diagnostics": [], "package": plan.package}
            self._acquire_lease(plan.family)
            token = "validation-" + uuid4().hex
            expiry = time.time() + self.reservation_seconds
            response = {"accepted": True, "validation_token": token, "catalog_revision": self.catalog.revision,
                        "reserve_resources": True,
                        "expires_at_unix": expiry, "diagnostics": [], "package": plan.package,
                        "reservations": [{"node_id": n["node_id"], "device_id": n["skill_invocation"]["device_id"]}
                                         for n in plan.package["nodes"] if n["node_type"] == "action"]}
            try:
                self.store.reserve(key, plan, expiry, response)
            except Exception:
                self._release_lease()
                raise
            self.reservation_token, self.reservation_until = token, expiry
            return response

    def submit(self, body, key, two_phase=False):
        plan = self._plan(body)
        key = self._key(body, key)
        token = body.get("validation_token")
        with self.lock:
            old = self.store.by_key(key)
            if old:
                if old["digest"] != plan.fingerprint:
                    raise ContractError("IDEMPOTENCY_CONFLICT", "Dispatch key used for another package", status=409)
                return self._response(old)
            self._available()
            if two_phase:
                identifier(token, "validation_token")
                reservation = self.store.reservation(token)
                if reservation is None or reservation["expires"] <= time.time():
                    raise ContractError("VALIDATION_TOKEN_EXPIRED", "Validation token is unknown or expired", status=409)
                if reservation["digest"] != plan.fingerprint:
                    raise ContractError("VALIDATION_TOKEN_MISMATCH", "Token is bound to a different package", status=409)
                if reservation["execution_id"]:
                    raise ContractError("VALIDATION_TOKEN_CONSUMED", "Use the original dispatch key", status=409)
                if self.reservation_token != token:
                    raise ContractError("RESERVATION_EXPIRED", "Resource reservation was lost", status=409)
            elif self.reservation_token:
                raise ContractError("RESOURCE_CONFLICT", "Use the two-phase route for the reserved package", status=409)
            plan = self.catalog.validate(plan.package, self.store.object_state())
            self._acquire_lease(plan.family)
            try:
                record = self.store.create(key, plan, token if two_phase else None)
            except Exception:
                if not self.reservation_token:
                    self._release_lease()
                raise
            self.reservation_token = None
            self.active = record["id"]
            self.cancel_event = threading.Event()
            self.worker = threading.Thread(target=self._execute, args=(record["id"], plan, self.cancel_event), name="dobot-execution")
            self._execution_workers = [worker for worker in self._execution_workers if worker.is_alive()]
            self._execution_workers.append(self.worker)
            self.worker.start()
            return self._response(record)

    def submit_new_execution(self, body):
        """Submit a fresh execution of a direct package with server-generated keys."""
        if not isinstance(body, dict):
            raise ContractError("INVALID_REQUEST", "Request must be a JSON object", status=400)
        if bare_package(body) is not None:
            return self.submit_new_execution({"package": body})
        if set(body) - ENVELOPE_KEYS:
            raise ContractError("INVALID_REQUEST", "Repeat accepts only graph/package and optional request_id", status=400)
        if "graph" in body and "package" in body:
            raise ContractError("INVALID_REQUEST", "Supply graph or package, not both", status=400)
        package = body.get("graph", body.get("package"))
        if package is None:
            raise ContractError("MISSING_EXECUTION_PACKAGE", "A complete execution package is required")
        if "request_id" in body:
            identifier(body["request_id"], "request_id")

        # Validate the caller's package before changing any idempotency data. Catalog.validate
        # also returns a deep copy, so caller-owned request data remains untouched.
        plan = self.catalog.validate(package, check_state=False)
        repeated_package = copy.deepcopy(plan.package)
        dispatch_key = "repeat-" + uuid4().hex
        for node in repeated_package["nodes"]:
            if node["node_type"] == "action":
                node["skill_invocation"]["idempotency_key"] = "repeat-" + uuid4().hex
        return self.submit({"graph": repeated_package, "request_id": dispatch_key}, dispatch_key)

    @staticmethod
    def _response(record):
        return {"execution_id": record["id"], "status": record["status"], "graph_id": record["package"]["package_id"],
                "accepted_at": record["accepted_at"]}

    def _execute(self, execution_id, plan, cancel):
        adapter = None
        result = {"status": "interrupted", "error": {"code": "EXECUTION_INTERRUPTED", "retryable": False}, "stop": {"confirmed": False}}
        def emit(event_type, **kwargs):
            with self.lock:
                if cancel.is_set() and kwargs.get("status") in {"running", "finalizing"}:
                    kwargs["status"] = "cancelling"
                self.store.event(execution_id, event_type, **kwargs)
        if self.results_delivery is not None:
            try:
                with self.store.transaction() as db:
                    db.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)",
                               ("results-capture:" + execution_id, "pending"))
            except Exception as error:
                self._result_capture_errors[execution_id] = type(error).__name__
        try:
            if self.adapter_factory:
                adapter = self.adapter_factory(cancel)
            elif self.mode == "mock":
                adapter = MockAdapter(cancel)
            elif plan.family == "chemistry":
                from .chemistry import ChemistryAdapter

                adapter = ChemistryAdapter(cancel)
            else:
                from ..dobot_adapter import DobotHardwareAdapter
                adapter = DobotHardwareAdapter.from_live(("left", "right"))
                adapter.set_cancel_event(cancel)
            for device in self.catalog.snapshot()["devices"]:
                self.store.device_event(device["device_id"], "running", execution_id, self.mode)
            result = run_graph(plan, adapter, self.scene, self.directory / "executions" / execution_id,
                               cancel, emit, live=self.mode == "live")
        except Exception as error:
            result["exception_type"] = type(error).__name__
            if self.mode == "mock":
                result["stop"] = {"confirmed": True}
        finally:
            if adapter is not None:
                try:
                    adapter.close()
                except Exception:
                    result["status"] = "interrupted"
                    result["stop"] = {"confirmed": False}
            with self.lock:
                if result["status"] == "succeeded" and cancel.is_set():
                    result["status"] = "cancelled"
                if result.get("stop", {}).get("confirmed") is False:
                    result["status"] = "interrupted"
                    result["error"] = {"code": "STOP_UNCONFIRMED", "retryable": False}
                    self.store.quarantine("Stop feedback was not confirmed")
                elif self.mode == "live" and result["status"] in {"failed", "cancelled", "interrupted"}:
                    self.store.quarantine("Object and gripper state requires onsite reset after incomplete live execution")
                else:
                    self._release_lease()
                emit("execution_" + result["status"], status=result["status"], payload=result,
                     error=result.get("error"), block_unfinished=True)
                for device in self.catalog.snapshot()["devices"]:
                    state = "fault" if result["status"] == "interrupted" else "idle" if self.mode == "mock" else "offline"
                    self.store.device_event(device["device_id"], state, mode=self.mode)
                self.active = None
            if self.results_delivery is not None:
                try:
                    self._capture_result(execution_id)
                except Exception as error:
                    # Upload/capture failure must not rewrite the robot outcome.
                    # The registry enrollment permits a later recovery attempt.
                    self._result_capture_errors[execution_id] = type(error).__name__

    def status(self, execution_id):
        with self.lock:
            record = self.store.get(execution_id)
            return {**self._response(record), "node_statuses": record["node_statuses"], "error": record["error"],
                    "catalog_revision": record["package"].get("catalog_revision", self.catalog.revision), "mode": self.mode,
                    "modeled_object_state": self.store.object_state(),
                    "verification": {"level": "command_trace", "physical_success_verified": False}}

    def cancel(self, execution_id):
        with self.lock:
            record = self.store.get(execution_id)
            if record["status"] in TERMINAL:
                return self.status(execution_id)
            if self.active != execution_id:
                raise ContractError("EXECUTION_INTERRUPTED", "No active worker owns this execution", status=409)
            self.cancel_event.set()
            self.store.event(execution_id, "execution_cancelling", status="cancelling")
            return self.status(execution_id)

    def maintain(self):
        if self.results_delivery is not None:
            self._recover_result_captures()
        with self.lock:
            if not self.active and self.reservation_token and time.time() >= self.reservation_until:
                self.reservation_token = None
                self._release_lease()

    def close(self):
        with self.lock:
            self.closed = True
            if self.cancel_event and self.active:
                self.cancel_event.set()
        shutdown_deadline = time.monotonic() + 15
        for worker in self._execution_workers:
            worker.join(timeout=max(0, shutdown_deadline - time.monotonic()))
            if worker.is_alive():
                self.store.quarantine("Worker shutdown was not confirmed")
                raise RuntimeError("Execution worker has not stopped; retain ownership and registry")
        self.delivery.close()
        if self.results_delivery is not None:
            self.results_delivery.close()
        if not self.store.quarantine_reason():
            self._release_lease()
        self.store.close()
