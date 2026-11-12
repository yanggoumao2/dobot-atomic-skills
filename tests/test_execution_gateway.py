import copy
import json
import os
import threading
import time
import unittest
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import BadStatusLine, IncompleteRead
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from task_skills.device_lease import DeviceLease
from task_skills.dobot_adapter import DobotHardwareAdapter
from task_skills.atomic_graph import AtomicNode
from task_skills.gateway.contracts import (CHEMISTRY_SKILLS, Catalog, ContractError,
                                            digest)
from task_skills.gateway.server import GatewayServer
from task_skills.gateway.service import Gateway
from task_skills.gateway.store import Store
from task_skills.gateway.upstream import DeliveryWorker, Upstream
from task_skills.gateway.operator import reset_scene
from task_skills.models import load_scene_config
from task_skills.run_graph import MockAdapter, run_graph


SCENE = Path(__file__).resolve().parents[1] / "task_skills/scene_config.json"


def package(revision=None, suffix="1", hold=0):
    def action(node_id, skill, arm, obj, ref):
        return {"node_id": node_id, "node_type": "action", "status": "pending", "timeout_s": 30,
                "skill_invocation": {"skill_id": "embodied." + skill, "skill_version": "1.0.0",
                    "device_id": "dobot-nova-" + arm,
                    "parameters": {"object_id": obj, "source_ref" if skill == "pick" else "destination_ref": ref},
                    "idempotency_key": suffix + ":" + node_id},
                "on_fail": {"strategy": "decision_agent", "max_retries": 1, "allowed_actions": ["safe_stop", "human_confirmation"]}}
    nodes = [action("pick-A", "pick", "left", "A", "object:A"),
             action("pick-B", "pick", "right", "B", "object:B"),
             {"node_id": "hold-A", "node_type": "wait", "status": "pending", "duration_s": hold, "timeout_s": 5},
             action("place-A", "place", "left", "A", "target:target_A"),
             action("place-B", "place", "right", "B", "target:target_B"),
             action("repick-B", "pick", "right", "B", "target:target_B"),
             action("return-B", "place", "right", "B", "object:B")]
    edges = [("pick-A", "hold-A"), ("hold-A", "place-A"), ("pick-B", "place-B"),
             ("place-A", "repick-B"), ("place-B", "repick-B"), ("repick-B", "return-B")]
    result = {"schema_version": "2.1", "graph_version": "0.1", "package_id": "runtime-" + suffix,
              "recipe_id": "pick-place", "version": "1.0", "experiment_id": "experiment-" + suffix,
              "entry_node_ids": ["pick-A", "pick-B"], "nodes": nodes,
              "edges": [{"source": a, "target": b, "edge_type": "success"} for a, b in edges]}
    if revision is not None:
        result["catalog_revision"] = revision
    return result


CHEMISTRY_DAG = Path(__file__).resolve().parents[1] / "examples/chemistry/right_beaker_pour_into_left_tube.json"
CHEMISTRY_PACKAGE = Path(__file__).resolve().parents[1] / "docs/execution-api/examples/runtime-chemistry-beaker-pour.json"


def chemistry_package():
    with open(CHEMISTRY_PACKAGE, encoding="utf-8") as handle:
        return json.load(handle)


def wait_package(suffix, revision=None):
    result = {"schema_version": "2.1", "graph_version": "0.1", "package_id": "wait-" + suffix,
              "experiment_id": "experiment-" + suffix, "entry_node_ids": ["wait"],
              "nodes": [{"node_id": "wait", "node_type": "wait", "status": "pending", "duration_s": 0}],
              "edges": []}
    if revision is not None:
        result["catalog_revision"] = revision
    return result


class FakeLiveAdapter(MockAdapter):
    def snapshot_all(self):
        from task_skills.layout import physical_nova_home_degrees
        return {
            arm: {
                "mode": 5,
                "errors": "0,{0},GetErrorID();",
                "joints": "0,{%s},GetAngle();" % ",".join(
                    str(value) for value in physical_nova_home_degrees(arm)),
            }
            for arm in ("left", "right")
        }

    def configure_robot(self, arm, speed):
        pass

    def stop_and_verify(self):
        return {"confirmed": True, "hardware_connected": False}


class CompilerHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        data = json.dumps({"package": self.server.package}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock:
            self.server.received.append((self.path, body))
            status = 200
            if "/executions/" in self.path and self.server.reject_first_event:
                self.server.reject_first_event = False
                status = 404
        self.send_response(status)
        self.end_headers()
        self.wfile.write(b'{"accepted": true}')


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.gateway = Gateway(load_scene_config(SCENE), Path(self.temp.name) / "data")
        self.server = GatewayServer(("127.0.0.1", 0), self.gateway, token="test-token")
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02})
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.gateway.close()
        self.temp.cleanup()

    def request(self, path, body=None, key=None, authorized=True):
        headers = {"Authorization": "Bearer test-token"} if authorized else {}
        if key:
            headers["Idempotency-Key"] = key
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        try:
            response = urlopen(Request(self.url + path, data=data, headers=headers), timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.code, json.loads(response.read())

    def submit(self, graph=None, key="dispatch-1"):
        return self.request("/v1/execute-graph/idempotent", {"graph": graph or package(self.gateway.catalog.revision)}, key)

    def finish(self, execution_id):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            result = self.gateway.status(execution_id)
            if result["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
                return result
            time.sleep(0.01)
        self.fail("Execution did not terminate")

    def reset_mock_scene(self):
        data_dir = Path(self.temp.name) / "data"
        scene = self.gateway.scene
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.gateway.close()
        reset_scene(data_dir, scene, "mock", data_dir / "mock-reset-owner")
        self.gateway = Gateway(scene, data_dir)
        self.server = GatewayServer(("127.0.0.1", 0), self.gateway, token="test-token")
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02})
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.server.server_port)

    def test_http_pipeline_bypasses_natural_language_and_preserves_join(self):
        with patch("task_skills.parser.parse_instruction", side_effect=AssertionError("NL parser must not run")):
            code, result = self.submit()
            self.assertEqual(code, 202)
            done = self.finish(result["execution_id"])
        self.assertEqual(done["status"], "succeeded")
        self.assertEqual(set(done["node_statuses"].values()), {"succeeded"})
        events = self.gateway.store.events(result["execution_id"])
        completed = {e["node_id"]: e["sequence"] for e in events if e["event_type"] == "action_completed"}
        repick = next(e["sequence"] for e in events if e.get("node_id") == "repick-B" and e["event_type"] == "action_started")
        self.assertGreater(repick, max(completed["place-A"], completed["place-B"]))
        self.assertFalse(done["verification"]["physical_success_verified"])

    def test_duplicate_dispatch_returns_original_record_without_more_actions(self):
        adapters = []
        self.gateway.adapter_factory = lambda cancel: adapters.append(MockAdapter(cancel)) or adapters[-1]
        first = self.submit()[1]
        self.finish(first["execution_id"])
        second = self.submit()[1]
        self.assertEqual(first["execution_id"], second["execution_id"])
        self.assertEqual(len(adapters), 1)
        altered = package(self.gateway.catalog.revision)
        altered["nodes"][2]["duration_s"] = 0.1
        code, response = self.submit(altered)
        self.assertEqual((code, response["code"]), (409, "IDEMPOTENCY_CONFLICT"))

    def test_execute_graph_creates_fresh_execution_with_server_generated_node_keys(self):
        graph = package(self.gateway.catalog.revision, "fresh-static")
        graph["run_id"] = "run-fresh-static"
        original = copy.deepcopy(graph)
        # Partner spec 7.2 envelope: resume fields are always present, null here.
        body = {"graph": graph, "request_id": "dispatch-1", "resume_from_node_id": None, "execution_id": None}

        code, first = self.request("/v1/execute-graph", body, "dispatch-1")
        self.assertEqual(code, 202)
        self.assertEqual(self.finish(first["execution_id"])["status"], "succeeded")
        first_package = self.gateway.store.get(first["execution_id"])["package"]
        first_keys = [node["skill_invocation"]["idempotency_key"] for node in first_package["nodes"]
                      if node["node_type"] == "action"]

        # Spec 9.2: a retry with the same key is the same execution, not a second motion.
        code, retried = self.request("/v1/execute-graph", body, "dispatch-1")
        self.assertEqual(code, 202)
        self.assertEqual(retried["execution_id"], first["execution_id"])

        # A new dispatch of the same compiled run (same node keys, new request key) runs.
        self.reset_mock_scene()
        body = dict(body, request_id="dispatch-2")
        code, second = self.request("/v1/execute-graph", body, "dispatch-2")
        self.assertEqual(code, 202)
        self.assertEqual(self.finish(second["execution_id"])["status"], "succeeded")
        second_package = self.gateway.store.get(second["execution_id"])["package"]
        second_keys = [node["skill_invocation"]["idempotency_key"] for node in second_package["nodes"]
                       if node["node_type"] == "action"]

        self.assertNotEqual(first["execution_id"], second["execution_id"])
        self.assertEqual(graph, original, "Fresh submission must not mutate the caller-owned package")
        self.assertEqual(first_package["package_id"], graph["package_id"])
        self.assertEqual(second_package["package_id"], graph["package_id"])
        self.assertEqual(second_package["run_id"], graph["run_id"])
        self.assertEqual([node["node_id"] for node in second_package["nodes"]],
                         [node["node_id"] for node in graph["nodes"]])
        self.assertEqual(len(second_keys), len(set(second_keys)))
        self.assertTrue(set(first_keys).isdisjoint(second_keys))
        self.assertNotEqual(second_keys, [node["skill_invocation"]["idempotency_key"]
                                         for node in graph["nodes"] if node["node_type"] == "action"])

    def test_execute_graph_rejects_a_second_request_while_execution_is_active(self):
        entered, release = threading.Event(), threading.Event()

        class HeldAdapter(MockAdapter):
            def execute_atomic(self, node, scene):
                if node.node_id == "pick-A:0":
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError("Test did not release held action")
                return super().execute_atomic(node, scene)

        self.gateway.adapter_factory = lambda cancel: HeldAdapter(cancel)
        body = {"graph": package(self.gateway.catalog.revision, "held-repeat"),
                "request_id": "same-request-id"}
        code, first = self.request("/v1/execute-graph", body, "same-header-key")
        self.assertEqual(code, 202)
        try:
            self.assertTrue(entered.wait(2), "Execution did not reach the held action")
            # Same key while running: the original execution, no second worker.
            code, again = self.request("/v1/execute-graph", body, "same-header-key")
            self.assertEqual((code, again["execution_id"]), (202, first["execution_id"]))
            # A different key while running is a real second request: refused.
            code, blocked = self.request("/v1/execute-graph", dict(body, request_id="other"), "other")
            self.assertEqual((code, blocked["code"]), (409, "RESOURCE_CONFLICT"))
            self.assertEqual(self.gateway.active, first["execution_id"])
        finally:
            release.set()
        self.assertEqual(self.finish(first["execution_id"])["status"], "succeeded")

    def test_execute_graph_rejects_resume_run_id_and_two_phase_shapes(self):
        graph = package(self.gateway.catalog.revision)
        unsupported = [
            {"run_id": "upstream-only"},
            {"graph": graph, "run_id": "upstream-only"},
            {"graph": graph, "validation_token": "validation-token"},
            {"graph": graph, "execution_id": "execution-old"},
            {"graph": graph, "resume_from_node_id": "pick-A"},
            {"graph": graph, "mode": "mock"},
            {"graph": graph, "unexpected": True},
            {"graph": graph, "package": graph},
        ]
        for body in unsupported:
            with self.subTest(body=list(body)):
                code, response = self.request("/v1/execute-graph", body, "fixed-key")
                self.assertEqual((code, response["code"]), (400, "INVALID_REQUEST"))
        malformed_graph = copy.deepcopy(graph)
        malformed_graph["nodes"] = "not-a-node-list"
        code, response = self.request("/v1/execute-graph", {"graph": malformed_graph}, "fixed-key")
        self.assertEqual((code, response["code"]), (422, "INVALID_GRAPH"))
        code, response = self.request("/v1/execute-graph", {"graph": graph}, authorized=False)
        self.assertEqual((code, response["code"]), (401, "UNAUTHORIZED"))

    def test_new_execution_service_does_not_mutate_the_supplied_package(self):
        graph = package(self.gateway.catalog.revision, "direct-repeat")
        original = copy.deepcopy(graph)
        result = self.gateway.submit_new_execution({"graph": graph})
        self.assertEqual(graph, original)
        self.assertEqual(self.finish(result["execution_id"])["status"], "succeeded")

    def test_execute_graph_does_not_require_an_idempotency_header(self):
        graph = package(self.gateway.catalog.revision, "no-idempotency-key")
        code, result = self.request("/v1/execute-graph", {"graph": graph})
        self.assertEqual(code, 202)
        self.assertEqual(self.finish(result["execution_id"])["status"], "succeeded")

    def test_execute_graph_obeys_reservation_gate(self):
        graph = package(self.gateway.catalog.revision, "gated-submission")
        self.gateway.preflight({"package": graph}, "submission-reservation")
        code, response = self.request("/v1/execute-graph", {"graph": graph}, "submission-key")
        self.assertEqual((code, response["code"]), (409, "RESOURCE_CONFLICT"))

    def test_execute_graph_obeys_quarantine_gate(self):
        graph = package(self.gateway.catalog.revision, "quarantined-submission")
        self.gateway.store.quarantine("test quarantine")
        code, response = self.request("/v1/execute-graph", {"graph": graph}, "submission-key")
        self.assertEqual((code, response["code"]), (409, "STOP_UNCONFIRMED"))

    def test_concurrent_duplicate_requests_start_one_worker(self):
        graph = package(self.gateway.catalog.revision, hold=0.1)
        adapters = []
        self.gateway.adapter_factory = lambda cancel: adapters.append(MockAdapter(cancel)) or adapters[-1]
        with ThreadPoolExecutor(max_workers=6) as pool:
            responses = list(pool.map(lambda _: self.submit(graph), range(6)))
        self.assertEqual(len(responses), 6)
        self.assertTrue(all(code == 202 for code, _ in responses))
        results = [result for _, result in responses]
        self.assertEqual(len({r["execution_id"] for r in results}), 1)
        self.assertEqual(self.finish(results[0]["execution_id"])["status"], "succeeded")
        self.assertEqual(len(adapters), 1)

    def test_two_phase_reservation_is_package_bound_and_consumed_once(self):
        graph = package(self.gateway.catalog.revision)
        code, validation = self.request("/device-gateway/v1/execution-plans/validate", {"package": graph}, "pre-1")
        self.assertEqual(code, 200)
        code, blocked = self.submit()
        self.assertEqual((code, blocked["code"]), (409, "RESOURCE_CONFLICT"))
        changed = copy.deepcopy(graph)
        changed["nodes"][2]["duration_s"] = 0.1
        code, mismatch = self.request("/device-gateway/v1/executions", {
            "package": changed, "validation_token": validation["validation_token"]}, "dispatch-1")
        self.assertEqual(mismatch["code"], "VALIDATION_TOKEN_MISMATCH")
        body = {"package": graph, "validation_token": validation["validation_token"]}
        code, submitted = self.request("/device-gateway/v1/executions", body, "dispatch-1")
        self.assertEqual(code, 202)
        self.finish(submitted["execution_id"])
        self.assertEqual(self.request("/device-gateway/v1/executions", body, "dispatch-1")[1]["execution_id"], submitted["execution_id"])
        self.assertEqual(self.request("/device-gateway/v1/executions", body, "dispatch-2")[1]["code"], "VALIDATION_TOKEN_CONSUMED")

    def test_expired_reservation_refuses_formal_submission(self):
        self.gateway.reservation_seconds = 0.01
        graph = package(self.gateway.catalog.revision)
        validation = self.gateway.preflight({"package": graph}, "pre-1")
        time.sleep(0.03)
        with self.assertRaises(ContractError) as error:
            self.gateway.submit({"package": graph, "validation_token": validation["validation_token"]}, "d", True)
        self.assertEqual(error.exception.body["code"], "VALIDATION_TOKEN_EXPIRED")

    def test_cancel_wait_stops_downstream_and_returns_only_after_worker_finishes(self):
        graph = package(self.gateway.catalog.revision, hold=3)
        result = self.submit(graph)[1]
        execution_id = result["execution_id"]
        deadline = time.monotonic() + 2
        while self.gateway.status(execution_id)["node_statuses"]["hold-A"] != "running" and time.monotonic() < deadline:
            time.sleep(0.01)
        code, response = self.request("/v1/executions/" + execution_id + "/cancel", {})
        self.assertEqual(code, 200)
        done = self.finish(execution_id)
        self.assertEqual(done["status"], "cancelled")
        self.assertEqual(done["node_statuses"]["place-A"], "blocked")
        self.assertIsNone(self.gateway.active)

    def test_failure_stops_other_branches_and_retains_actual_failed_node(self):
        class Failing(MockAdapter):
            def execute_atomic(self, node, scene):
                if node.arm == "left" and node.operation == "close_gripper":
                    raise RuntimeError("Injected controller failure")
                return super().execute_atomic(node, scene)
        self.gateway.adapter_factory = lambda cancel: Failing(cancel, 0.01)
        result = self.submit()[1]
        done = self.finish(result["execution_id"])
        self.assertEqual(done["status"], "failed")
        self.assertEqual(done["node_statuses"]["pick-A"], "failed")
        self.assertEqual(done["node_statuses"]["repick-B"], "blocked")

    def test_skill_timeout_is_total_across_internal_actions(self):
        self.gateway.adapter_factory = lambda cancel: MockAdapter(cancel, 0.02)
        graph = package(self.gateway.catalog.revision)
        graph["nodes"][0]["timeout_s"] = 0.025
        result = self.submit(graph)[1]
        done = self.finish(result["execution_id"])
        self.assertEqual(done["status"], "failed")
        self.assertEqual(done["error"]["code"], "MOTION_TIMEOUT")

    def test_unsupported_semantics_are_rejected_before_any_adapter(self):
        self.gateway.adapter_factory = lambda _: self.fail("Adapter constructed for invalid graph")
        mutations = [
            ("UNSUPPORTED_NODE_TYPE", lambda g: g["nodes"][0].update(node_type="observe")),
            ("UNSUPPORTED_EDGE_TYPE", lambda g: g["edges"][0].update(edge_type="failure")),
            ("UNSUPPORTED_SKILL", lambda g: g["nodes"][0]["skill_invocation"].update(skill_id="embodied.load")),
            ("UNSUPPORTED_CONDITION", lambda g: g["nodes"][0].update(preconditions=["object_detected"])),
            ("UNSUPPORTED_CONDITION", lambda g: g["nodes"][0].update(completion_condition="physical_grasp_verified")),
            ("UNSUPPORTED_RESUME", lambda g: g["nodes"][0].update(status="succeeded")),
            ("INVALID_GRAPH", lambda g: g["edges"].append({"source": "return-B", "target": "pick-B", "edge_type": "success"})),
            ("INVALID_GRAPH", lambda g: g["nodes"].append(copy.deepcopy(g["nodes"][0]))),
            ("DEVICE_INCOMPATIBLE", lambda g: g["nodes"][0]["skill_invocation"]["parameters"].update(object_id="B")),
        ]
        for expected, change in mutations:
            with self.subTest(expected=expected):
                graph = package(self.gateway.catalog.revision)
                change(graph)
                code, response = self.submit(graph)
                self.assertGreaterEqual(code, 400)
                self.assertEqual(response["code"], expected)

    def test_node_array_order_does_not_change_dependencies(self):
        graph = package(self.gateway.catalog.revision)
        graph["nodes"].reverse()
        result = self.submit(graph)[1]
        self.assertEqual(self.finish(result["execution_id"])["status"], "succeeded")

    def test_same_arm_unordered_actions_and_invalid_object_lifecycle_rejected(self):
        graph = package(self.gateway.catalog.revision)
        graph["edges"] = [e for e in graph["edges"] if e["target"] != "place-A"]
        graph["entry_node_ids"].append("place-A")
        self.assertEqual(self.submit(graph)[1]["code"], "RESOURCE_CONFLICT")
        graph = package(self.gateway.catalog.revision)
        graph["nodes"][0]["skill_invocation"].update(skill_id="embodied.place")
        graph["nodes"][0]["skill_invocation"]["parameters"] = {"object_id": "A", "destination_ref": "target:target_A"}
        self.assertEqual(self.submit(graph)[1]["code"], "INVALID_OBJECT_STATE")

    def test_new_dispatch_key_cannot_replay_previously_used_node_keys(self):
        first = self.submit()[1]
        self.finish(first["execution_id"])
        graph = package(self.gateway.catalog.revision)
        graph["nodes"][0]["skill_invocation"]["parameters"]["source_ref"] = "target:target_A"
        self.assertEqual(self.submit(graph, key="different")[1]["code"], "IDEMPOTENCY_CONFLICT")

    def test_successful_object_state_is_preserved_for_the_next_graph(self):
        first = self.submit()[1]
        self.finish(first["execution_id"])
        graph = package(self.gateway.catalog.revision, "next")
        self.assertEqual(self.submit(graph, "next")[1]["code"], "INVALID_OBJECT_STATE")
        graph["nodes"][0]["skill_invocation"]["parameters"]["source_ref"] = "target:target_A"
        result = self.submit(graph, "next")[1]
        self.assertEqual(self.finish(result["execution_id"])["status"], "succeeded")

    def test_large_integer_parameter_returns_validation_error(self):
        graph = package(self.gateway.catalog.revision)
        graph["nodes"][0]["timeout_s"] = 10 ** 400
        code, response = self.submit(graph)
        self.assertEqual((code, response["code"]), (422, "DEVICE_PARAMETER_OUT_OF_RANGE"))
        result = self.submit()[1]
        code, response = self.request("/v1/executions/" + result["execution_id"] + "/events?after=" + str(10 ** 100))
        self.assertEqual((code, response["code"]), (400, "INVALID_REQUEST"))

    def test_unconfirmed_stop_quarantines_and_blocks_new_work(self):
        class Unconfirmed(MockAdapter):
            def execute_atomic(self, node, scene):
                raise RuntimeError("Injected motion fault")
            def stop_and_verify(self):
                return {"confirmed": False}
        self.gateway.adapter_factory = lambda cancel: Unconfirmed(cancel)
        result = self.submit()[1]
        done = self.finish(result["execution_id"])
        self.assertEqual((done["status"], done["error"]["code"]), ("interrupted", "STOP_UNCONFIRMED"))
        self.assertEqual(self.submit(package(self.gateway.catalog.revision, "2"), "2")[1]["code"], "STOP_UNCONFIRMED")

    def test_mock_independent_arm_skills_execute_concurrently(self):
        barrier = threading.Barrier(2)
        class Parallel(MockAdapter):
            def execute_atomic(self, node, scene):
                if node.node_id in {"pick-A:0", "pick-B:0"}:
                    barrier.wait(timeout=2)
                return super().execute_atomic(node, scene)
        self.gateway.adapter_factory = lambda cancel: Parallel(cancel)
        result = self.submit()[1]
        self.assertEqual(self.finish(result["execution_id"])["status"], "succeeded")

    def test_live_independent_arm_skills_enter_concurrently_and_failure_cancels_sibling(self):
        live_catalog = Catalog(self.gateway.scene, "live")
        plan = live_catalog.validate(package(live_catalog.revision))
        cancel = threading.Event()
        entry_barrier = threading.Barrier(2)
        entry_lock = threading.Lock()
        entered_arms, calls = set(), []

        class FakeLiveAdapter(MockAdapter):
            def snapshot_all(self):
                from task_skills.layout import physical_nova_home_degrees
                return {
                    arm: {
                        "mode": 5,
                        "errors": "0,{0},GetErrorID();",
                        "joints": "0,{%s},GetAngle();" % ",".join(
                            str(value) for value in physical_nova_home_degrees(arm)),
                    }
                    for arm in ("left", "right")
                }

            def configure_robot(self, arm, speed):
                pass

            def execute_atomic(self, node, scene):
                with entry_lock:
                    calls.append(node.node_id)
                if node.node_id in {"pick-A:0", "pick-B:0"}:
                    with entry_lock:
                        entered_arms.add(node.arm)
                    entry_barrier.wait(timeout=2)
                    if node.arm == "left":
                        raise RuntimeError("Injected fake-live controller failure")
                    if self.cancel.wait(2):
                        raise InterruptedError("Execution cancelled")
                    raise AssertionError("Sibling was not cancelled after the other arm failed")
                if self.cancel.is_set():
                    raise InterruptedError("Execution cancelled")
                return {"fake_live": True}

            def stop_and_verify(self):
                return {"confirmed": True, "hardware_connected": False}

        adapter = FakeLiveAdapter(cancel)
        events = []
        try:
            with TemporaryDirectory() as temp:
                result = run_graph(plan, adapter, self.gateway.scene, temp, cancel,
                                   lambda event_type, **kwargs: events.append((event_type, kwargs)), live=True)
        finally:
            adapter.close()

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["node_id"], "pick-A")
        self.assertTrue(result["stop"]["confirmed"])
        self.assertEqual(entered_arms, {"left", "right"})
        self.assertEqual(set(calls), {"pick-A:0", "pick-B:0"})
        node_results = {
            kwargs.get("node_id"): event_type
            for event_type, kwargs in events if kwargs.get("node_id")
        }
        self.assertEqual(node_results["pick-A"], "action_failed")
        self.assertEqual(node_results["pick-B"], "action_cancelled")

    def test_live_success_preserves_wait_and_join_readiness(self):
        live_catalog = Catalog(self.gateway.scene, "live")
        plan = live_catalog.validate(package(live_catalog.revision, hold=0.02))
        cancel = threading.Event()
        pick_barrier = threading.Barrier(2)
        entry_lock = threading.Lock()
        entered_arms, calls = set(), []

        class FakeLiveAdapter(MockAdapter):
            def snapshot_all(self):
                from task_skills.layout import physical_nova_home_degrees
                return {
                    arm: {
                        "mode": 5,
                        "errors": "0,{0},GetErrorID();",
                        "joints": "0,{%s},GetAngle();" % ",".join(
                            str(value) for value in physical_nova_home_degrees(arm)),
                    }
                    for arm in ("left", "right")
                }

            def configure_robot(self, arm, speed):
                pass

            def execute_atomic(self, node, scene):
                with entry_lock:
                    calls.append(node.node_id)
                if node.node_id in {"pick-A:0", "pick-B:0"}:
                    with entry_lock:
                        entered_arms.add(node.arm)
                    pick_barrier.wait(timeout=2)
                if self.cancel.is_set():
                    raise InterruptedError("Execution cancelled")
                return {"fake_live": True, "node_id": node.node_id}

            def stop_and_verify(self):
                return {"confirmed": True, "hardware_connected": False}

        adapter = FakeLiveAdapter(cancel)
        events = []
        try:
            with TemporaryDirectory() as temp:
                result = run_graph(plan, adapter, self.gateway.scene, temp, cancel,
                                   lambda event_type, **kwargs: events.append((event_type, kwargs)), live=True)
        finally:
            adapter.close()

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(entered_arms, {"left", "right"})
        self.assertIn("final:left", calls)
        self.assertIn("final:right", calls)
        wait_completed = next(
            index for index, (event_type, kwargs) in enumerate(events)
            if event_type == "wait_completed" and kwargs.get("node_id") == "hold-A"
        )
        started = {
            kwargs["node_id"]: index
            for index, (event_type, kwargs) in enumerate(events)
            if event_type == "action_started"
        }
        completed = {
            kwargs["node_id"]: index
            for index, (event_type, kwargs) in enumerate(events)
            if event_type == "action_completed"
        }
        self.assertGreater(started["place-A"], wait_completed)
        self.assertGreater(started["repick-B"], max(completed["place-A"], completed["place-B"]))
        self.assertGreater(started["return-B"], completed["repick-B"])
        self.assertIn("action_completed", [event_type for event_type, kwargs in events
                                           if kwargs.get("node_id") == "return-B"])

    def test_live_final_homes_run_together_and_success_waits_for_both(self):
        live_catalog = Catalog(self.gateway.scene, "live")
        plan = live_catalog.validate(package(live_catalog.revision))
        cancel = threading.Event()
        homes_entered = threading.Event()
        release_homes = threading.Event()
        home_barrier = threading.Barrier(2)
        home_lock = threading.Lock()
        entered_arms, completed_arms = set(), set()

        class ParallelHome(FakeLiveAdapter):
            def execute_atomic(self, node, scene):
                if node.operation == "move_safe_pose":
                    with home_lock:
                        entered_arms.add(node.arm)
                        if len(entered_arms) == 2:
                            homes_entered.set()
                    home_barrier.wait(timeout=2)
                    if not release_homes.wait(4):
                        raise TimeoutError("Test did not release final homes")
                    with home_lock:
                        completed_arms.add(node.arm)
                    return {"homed": node.arm}
                return super().execute_atomic(node, scene)

        adapter = ParallelHome(cancel)
        events = []
        with ThreadPoolExecutor(max_workers=1) as runner:
            with TemporaryDirectory() as temp:
                result_future = runner.submit(
                    run_graph, plan, adapter, self.gateway.scene, temp,
                    cancel, lambda event_type, **kwargs: events.append((event_type, kwargs)), True)
                try:
                    self.assertTrue(homes_entered.wait(2), "Both arms did not enter final homing")
                    self.assertFalse(result_future.done(), "Execution returned before either home was released")
                finally:
                    release_homes.set()
                result = result_future.result(timeout=5)
        adapter.close()

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(entered_arms, {"left", "right"})
        self.assertEqual(completed_arms, {"left", "right"})
        finalizing = {kwargs["payload"]["arm"] for event_type, kwargs in events
                      if event_type == "arm_finalizing"}
        finalized = {kwargs["payload"]["arm"] for event_type, kwargs in events
                     if event_type == "arm_finalized"}
        self.assertEqual(finalizing, {"left", "right"})
        self.assertEqual(finalized, {"left", "right"})

    def test_live_final_home_failure_stops_sibling_and_rechecks_stop(self):
        live_catalog = Catalog(self.gateway.scene, "live")
        plan = live_catalog.validate(package(live_catalog.revision))

        for confirmed in (True, False):
            with self.subTest(stop_confirmed=confirmed):
                cancel = threading.Event()
                home_barrier = threading.Barrier(2)
                sibling_active, stop_called, sibling_done = (
                    threading.Event(), threading.Event(), threading.Event())
                stop_states, stop_calls = [], []

                class FailingHome(FakeLiveAdapter):
                    def execute_atomic(self, node, scene):
                        if node.operation == "move_safe_pose":
                            home_barrier.wait(timeout=2)
                            if node.arm == "left":
                                raise RuntimeError("Injected left home failure")
                            sibling_active.set()
                            if not stop_called.wait(2):
                                raise TimeoutError("Stop was not requested for home sibling")
                            sibling_active.clear()
                            sibling_done.set()
                            raise InterruptedError("Sibling stopped during final homing")
                        return super().execute_atomic(node, scene)

                    def stop_and_verify(self):
                        if not stop_calls:
                            sibling_active.wait(2)
                        stop_states.append(sibling_active.is_set())
                        stop_calls.append(True)
                        stop_called.set()
                        return {"confirmed": confirmed, "hardware_connected": False}

                adapter = FailingHome(cancel)
                events = []
                with TemporaryDirectory() as temp:
                    result = run_graph(plan, adapter, self.gateway.scene, temp, cancel,
                                       lambda event_type, **kwargs: events.append((event_type, kwargs)), live=True)
                adapter.close()

                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error"]["arm"], "left")
                self.assertEqual(result["error"]["phase"], "final_home")
                self.assertEqual(result["stop"]["confirmed"], confirmed)
                self.assertTrue(sibling_done.is_set())
                self.assertEqual(len(stop_calls), 2)
                self.assertTrue(stop_states[0], "Initial stop check must run while sibling home is active")
                self.assertFalse(stop_states[-1], "Final stop check must run after home workers drain")
                self.assertTrue(any(event_type == "arm_finalization_failed"
                                    and kwargs["payload"]["arm"] == "left"
                                    for event_type, kwargs in events))

    def test_live_finalization_event_failure_stops_and_drains_started_home(self):
        live_catalog = Catalog(self.gateway.scene, "live")
        plan = live_catalog.validate(package(live_catalog.revision))
        cancel = threading.Event()
        home_started, stop_called, home_done = threading.Event(), threading.Event(), threading.Event()
        active_lock = threading.Lock()
        active_arms, stop_states, started_events = set(), [], []

        class ActiveHome(FakeLiveAdapter):
            def execute_atomic(self, node, scene):
                if node.operation == "move_safe_pose":
                    with active_lock:
                        active_arms.add(node.arm)
                    home_started.set()
                    if not stop_called.wait(2):
                        raise TimeoutError("Finalization failure did not trigger stop")
                    with active_lock:
                        active_arms.discard(node.arm)
                    home_done.set()
                    raise InterruptedError("Home cancelled after event failure")
                return super().execute_atomic(node, scene)

            def stop_and_verify(self):
                with active_lock:
                    stop_states.append(set(active_arms))
                stop_called.set()
                return {"confirmed": True, "hardware_connected": False}

        def emit(event_type, **kwargs):
            if event_type == "arm_finalizing":
                arm = kwargs["payload"]["arm"]
                started_events.append(arm)
                if arm == "right":
                    self.assertTrue(home_started.wait(1), "First arm home did not start before the second event")
                    raise OSError("Injected event store failure")

        adapter = ActiveHome(cancel)
        with TemporaryDirectory() as temp:
            result = run_graph(plan, adapter, self.gateway.scene, temp, cancel, emit, live=True)
        adapter.close()

        self.assertEqual(started_events, ["left", "right"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "FINALIZATION_EVENT_FAILED")
        self.assertEqual(result["error"]["phase"], "final_home")
        self.assertEqual(result["error"]["arm"], "right")
        self.assertTrue(result["stop"]["confirmed"])
        self.assertTrue(home_done.is_set())
        self.assertEqual(len(stop_states), 2)
        self.assertEqual(stop_states[0], {"left"}, "Prompt stop must happen while the submitted home is active")
        self.assertEqual(stop_states[-1], set(), "Final stop verification must follow worker drain")

    def test_live_cancel_during_final_homing_cannot_return_success(self):
        live_catalog = Catalog(self.gateway.scene, "live")
        plan = live_catalog.validate(package(live_catalog.revision))
        cancel = threading.Event()
        homes_entered = threading.Event()
        home_barrier = threading.Barrier(2)
        home_lock = threading.Lock()
        entered_arms, stop_states = set(), []

        class CancellableHome(FakeLiveAdapter):
            def execute_atomic(self, node, scene):
                if node.operation == "move_safe_pose":
                    home_barrier.wait(timeout=2)
                    with home_lock:
                        entered_arms.add(node.arm)
                        if len(entered_arms) == 2:
                            homes_entered.set()
                    if self.cancel.wait(2):
                        raise InterruptedError("Cancelled during final homing")
                    raise TimeoutError("Test did not cancel final homing")
                return super().execute_atomic(node, scene)

            def stop_and_verify(self):
                stop_states.append(set(entered_arms))
                return {"confirmed": True, "hardware_connected": False}

        adapter = CancellableHome(cancel)
        events = []
        with ThreadPoolExecutor(max_workers=1) as runner:
            with TemporaryDirectory() as temp:
                result_future = runner.submit(
                    run_graph, plan, adapter, self.gateway.scene, temp, cancel,
                    lambda event_type, **kwargs: events.append((event_type, kwargs)), True)
                self.assertTrue(homes_entered.wait(2), "Both arms did not enter final homing")
                self.assertFalse(result_future.done())
                cancel.set()
                result = result_future.result(timeout=4)
        adapter.close()

        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["error"]["phase"], "final_home")
        self.assertIn(result["error"]["arm"], {"left", "right"})
        self.assertEqual(result["stop"]["confirmed"], True)
        self.assertEqual(len(stop_states), 2)
        self.assertTrue(all(arms == {"left", "right"} for arms in stop_states))
        self.assertEqual(sum(event_type == "arm_finalization_cancelled" for event_type, _ in events), 2)

    def test_catalog_revision_is_optional_opaque_and_fingerprinted(self):
        live_catalog = Catalog(self.gateway.scene, "live")
        mock_catalog = Catalog(self.gateway.scene, "mock")
        self.assertEqual(live_catalog.snapshot()["execution_profile"]["cross_arm_motion_concurrency"], 2)
        self.assertEqual(mock_catalog.snapshot()["execution_profile"]["cross_arm_motion_concurrency"], 2)
        self.assertNotEqual(live_catalog.revision, mock_catalog.revision)
        self.assertEqual(live_catalog.snapshot()["catalog_revision"], live_catalog.revision)

        base = wait_package("revision-fingerprint")
        fingerprints = []
        for revision in ("random-upstream-value", 847263, None,
                         {"source": "caller"}, [7, "snapshot"]):
            graph = copy.deepcopy(base)
            graph["catalog_revision"] = revision
            caller_copy = copy.deepcopy(graph)
            plan = live_catalog.validate(graph)
            self.assertEqual(graph, caller_copy, "Validation must preserve caller-owned data")
            self.assertEqual(plan.package, caller_copy)
            self.assertEqual(plan.fingerprint, digest(caller_copy))
            self.assertEqual(plan.package["catalog_revision"], revision)
            fingerprints.append(plan.fingerprint)
        self.assertEqual(len(fingerprints), len(set(fingerprints)),
                         "Different catalog_revision metadata must produce different fingerprints")

        without_metadata = wait_package("revision-fingerprint-optional")
        plan = live_catalog.validate(without_metadata)
        self.assertNotIn("catalog_revision", plan.package)
        self.assertEqual(plan.fingerprint, digest(without_metadata))

    def test_execution_and_two_phase_routes_preserve_optional_catalog_metadata(self):
        for index, revision in enumerate((None, "random-http-metadata", 912345.625)):
            suffix = "metadata-%d" % index
            graph = wait_package(suffix + "-fresh", revision)
            original = copy.deepcopy(graph)
            code, response = self.request("/v1/execute-graph", {"graph": graph})
            self.assertEqual(code, 202)
            self.assertEqual(self.finish(response["execution_id"])["status"], "succeeded")
            stored = self.gateway.store.get(response["execution_id"])
            self.assertEqual(stored["package"], original)

            graph = wait_package(suffix + "-idempotent", revision)
            original = copy.deepcopy(graph)
            code, response = self.request(
                "/v1/execute-graph/idempotent", {"graph": graph}, "dispatch-" + suffix)
            self.assertEqual(code, 202)
            self.assertEqual(self.finish(response["execution_id"])["status"], "succeeded")
            stored = self.gateway.store.get(response["execution_id"])
            self.assertEqual(stored["package"], original)

            graph = wait_package(suffix + "-two-phase", revision)
            original = copy.deepcopy(graph)
            code, validation = self.request(
                "/device-gateway/v1/execution-plans/validate", {"package": graph}, "validate-" + suffix)
            self.assertEqual(code, 200)
            self.assertEqual(validation["package"], original)
            code, response = self.request(
                "/device-gateway/v1/executions",
                {"package": graph, "validation_token": validation["validation_token"]}, "two-phase-" + suffix)
            self.assertEqual(code, 202)
            self.assertEqual(self.finish(response["execution_id"])["status"], "succeeded")
            stored = self.gateway.store.get(response["execution_id"])
            self.assertEqual(stored["package"], original)

    def test_authentication_request_shape_and_resume_rejected(self):
        self.assertEqual(self.request("/v1/catalog", authorized=False)[0], 401)
        graph = package(self.gateway.catalog.revision)
        idempotent_path = "/v1/execute-graph/idempotent"
        self.assertEqual(self.request(idempotent_path, {"graph": graph})[0], 422)
        self.assertEqual(self.request(idempotent_path, {"graph": graph, "resume_from_node_id": "pick-A"}, "x")[1]["code"], "UNSUPPORTED_RESUME")
        self.assertEqual(self.request(idempotent_path, {"graph": graph, "mode": "live"}, "x")[1]["code"], "INVALID_REQUEST")

    def test_removed_repeat_route_returns_not_found(self):
        graph = package(self.gateway.catalog.revision)
        code, response = self.request("/v1/execute-graph/repeat", {"graph": graph}, "fixed-key")
        self.assertEqual((code, response["code"]), (404, "NOT_FOUND"))

    def test_compiler_catalog_fetch_and_event_delivery_survive_registration_race(self):
        compiler = ThreadingHTTPServer(("127.0.0.1", 0), CompilerHandler)
        compiler.received, compiler.lock, compiler.reject_first_event = [], threading.Lock(), True
        compiler.package = package(self.gateway.catalog.revision)
        thread = threading.Thread(target=compiler.serve_forever, kwargs={"poll_interval": 0.02})
        thread.start()
        upstream = Upstream("http://127.0.0.1:" + str(compiler.server_port))
        self.gateway.upstream = upstream
        self.gateway.delivery.upstream = upstream
        try:
            validation = self.gateway.preflight({"run_id": "run-1"}, "pre")
            result = self.gateway.submit({"package": validation["package"], "validation_token": validation["validation_token"]}, "d", True)
            self.finish(result["execution_id"])
            for _ in range(30):
                self.gateway.delivery.pump()
                if not self.gateway.store.delivery_status()["pending"]:
                    break
            self.assertEqual(self.gateway.store.delivery_status()["pending"], 0)
            paths = [p for p, _ in compiler.received]
            self.assertIn("/v1/integrations/catalog/validate", paths)
            # Publishing a snapshot replaces the partner's catalog; it is opt-in.
            self.assertNotIn("/v1/integrations/catalog/snapshots", paths)
            events = [b for p, b in compiler.received if "/executions/" in p]
            self.assertEqual(events[0]["event_id"], events[1]["event_id"])
            unique = {e["event_id"]: e for e in events}
            self.assertEqual([e["sequence"] for e in unique.values()], list(range(1, len(unique) + 1)))
            self.assertTrue(all(e["execution_id"] == result["execution_id"] for e in events))
        finally:
            compiler.shutdown()
            compiler.server_close()
            thread.join()


class RegistryAndDriverTests(unittest.TestCase):
    def test_restart_does_not_claim_a_lost_preflight_reservation(self):
        with TemporaryDirectory() as temp:
            scene = load_scene_config(SCENE)
            gateway = Gateway(scene, temp)
            graph = package(gateway.catalog.revision)
            gateway.preflight({"package": graph}, "pre")
            gateway.close()
            restarted = Gateway(scene, temp)
            try:
                with self.assertRaises(ContractError) as caught:
                    restarted.preflight({"package": graph}, "pre")
                self.assertEqual(caught.exception.body["code"], "RESERVATION_EXPIRED")
                self.assertTrue(restarted.preflight({"package": graph}, "new-pre")["accepted"])
            finally:
                restarted.close()

    def test_invalid_ack_and_recursive_response_remain_retryable_in_outbox(self):
        class FakeUpstream:
            response = []
            def request(self, *_):
                if isinstance(self.response, Exception):
                    raise self.response
                return self.response
        with TemporaryDirectory() as temp:
            store = Store(temp)
            upstream = FakeUpstream()
            delivery = DeliveryWorker(store, upstream)
            store.enqueue("test", "/events", {"event_id": "same-id"})
            try:
                for response in ([], {"accepted": "true"}, {"accepted": False}, {"diagnostics": [{"severity": "blocking"}]}, RecursionError(), BadStatusLine("bad"), IncompleteRead(b"")):
                    upstream.response = response
                    self.assertEqual(delivery.pump(), 0)
                    self.assertEqual(store.delivery_status()["pending"], 1)
                    self.assertEqual(store.pending_deliveries()[0]["id"], "same-id")
                upstream.response = {"accepted": True}
                self.assertEqual(delivery.pump(), 1)
                self.assertEqual(store.delivery_status()["pending"], 0)
            finally:
                store.close()

    def test_onsite_reset_requires_free_owners_and_retains_history(self):
        with TemporaryDirectory() as temp:
            scene = load_scene_config(SCENE)
            directory, owner = Path(temp) / "db", Path(temp) / "owner"
            gateway = Gateway(scene, directory)
            graph = package(gateway.catalog.revision)
            result = gateway.submit({"graph": graph}, "key")
            gateway.worker.join()
            with self.assertRaises(RuntimeError):
                reset_scene(directory, scene, "mock", owner)
            gateway.close()
            lease = DeviceLease(owner, "legacy-control").acquire()
            with self.assertRaises(RuntimeError):
                reset_scene(directory, scene, "mock", owner)
            lease.release()
            self.assertTrue(lease.scene_marker.exists())
            reset_scene(directory, scene, "mock", owner)
            self.assertTrue(lease.scene_marker.exists())
            restarted = Gateway(scene, directory)
            try:
                self.assertEqual(restarted.store.object_state()["left"]["location"], "object:A")
                self.assertEqual(restarted.submit({"graph": graph}, "key")["execution_id"], result["execution_id"])
            finally:
                restarted.close()

    def test_registry_mode_is_fixed_and_scene_stamp_invalidates_other_directories(self):
        with TemporaryDirectory() as temp:
            scene = load_scene_config(SCENE)
            owner = Path(temp) / "owner"
            first, second = Path(temp) / "live-1", Path(temp) / "live-2"
            reset_scene(first, scene, "live", owner)
            with self.assertRaises(ValueError):
                Gateway(scene, first, mode="mock")
            def check_gate(directory):
                gateway = object.__new__(Gateway)
                gateway.mode, gateway.lease_path, gateway.lease = "live", owner, None
                gateway.store, gateway.catalog = Store(directory), Catalog(scene, "live")
                try:
                    gateway._acquire_lease()
                finally:
                    gateway._release_lease()
                    gateway.store.close()
            check_gate(first)
            reset_scene(second, scene, "live", owner)
            check_gate(second)
            with self.assertRaises(ContractError) as caught:
                check_gate(first)
            self.assertEqual(caught.exception.body["code"], "SCENE_RESET_REQUIRED")
            lease = DeviceLease(owner, "legacy-control").acquire()
            lease.release()
            with self.assertRaises(ContractError):
                check_gate(second)
            reset_scene(second, scene, "live", owner)
            check_gate(second)

    def test_cancel_gate_prevents_late_gripper_command_after_stop(self):
        entered, resume, calls, errors = threading.Event(), threading.Event(), [], []
        class Dashboard:
            def StopScript(self):
                return "0,{},StopScript();"
            def RobotMode(self):
                return "0,{5},RobotMode();"
        class Gripper:
            def get_current_position(self):
                return 127
            def move(self, target, speed, force):
                calls.append(target)
                return 0
        class Adapter(DobotHardwareAdapter):
            def _gripper_position(self, arm):
                # Pause before the I/O gate, reproducing a stale pre-command check.
                if threading.current_thread().name == "blocked-skill":
                    entered.set()
                    if not resume.wait(2):
                        raise TimeoutError("Test did not resume skill")
                return super()._gripper_position(arm)
        adapter = Adapter.from_components({"left": Dashboard()}, {}, {"left": Gripper()})
        cancel = threading.Event()
        adapter.set_cancel_event(cancel)
        def skill():
            try:
                adapter.close_gripper("left")
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=skill, name="blocked-skill")
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            cancel.set()
            self.assertTrue(adapter.stop_and_verify()["confirmed"])
        finally:
            resume.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(calls, [127])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], InterruptedError)

    def test_device_lease_is_shared_and_owner_checked(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "owner"
            first = DeviceLease(path).acquire()
            with self.assertRaises(RuntimeError):
                DeviceLease(path).acquire()
            first.release()
            with DeviceLease(path):
                self.assertTrue(path.exists())

    def test_process_guard_prevents_a_second_registry_owner(self):
        with TemporaryDirectory() as temp:
            store = Store(temp)
            try:
                with self.assertRaises(RuntimeError):
                    Store(temp)
            finally:
                store.close()

    def test_restart_retains_completed_idempotency_and_interrupts_unfinished_intent(self):
        with TemporaryDirectory() as temp:
            scene = load_scene_config(SCENE)
            gateway = Gateway(scene, temp)
            graph = package(gateway.catalog.revision)
            completed = gateway.submit({"graph": graph}, "key")
            gateway.worker.join()
            graph2 = package(gateway.catalog.revision, "2")
            interrupted = gateway.store.create("crash-key", gateway.catalog.validate(graph2))
            gateway.close()
            restarted = Gateway(scene, temp)
            try:
                self.assertEqual(restarted.submit({"graph": graph}, "key")["execution_id"], completed["execution_id"])
                self.assertEqual(restarted.status(interrupted["id"])["status"], "interrupted")
                self.assertEqual(restarted.submit({"graph": graph2}, "crash-key")["status"], "interrupted")
                self.assertIsNone(restarted.worker)
            finally:
                restarted.close()

    def test_driver_wait_is_cancellable_without_hardware(self):
        adapter = DobotHardwareAdapter.from_components({}, {}, {})
        cancel = threading.Event()
        adapter.set_cancel_event(cancel)
        timer = threading.Timer(0.02, cancel.set)
        timer.start()
        started = time.monotonic()
        try:
            with self.assertRaises(InterruptedError):
                adapter.execute_atomic(AtomicNode("wait", 0, "wait", "left", parameters={"seconds": 10}), None)
        finally:
            timer.join()
        self.assertLess(time.monotonic() - started, 1)

    def test_stop_confirmation_is_independent_of_cancel_flag(self):
        class Dashboard:
            def StopScript(self):
                return "0,{},StopScript();"
            def RobotMode(self):
                return "0,{5},RobotMode();"
        adapter = DobotHardwareAdapter.from_components({"left": Dashboard()}, {}, {})
        cancel = threading.Event()
        cancel.set()
        adapter.set_cancel_event(cancel)
        self.assertTrue(adapter.stop_and_verify()["confirmed"])

    def test_stop_attempts_other_arm_when_one_fails_and_holds_grippers(self):
        stopped, held = [], []
        class Dashboard:
            def __init__(self, arm):
                self.arm = arm
            def StopScript(self):
                stopped.append(self.arm)
                if self.arm == "left":
                    raise OSError("Disconnected")
                return "0,{},StopScript();"
            def RobotMode(self):
                return "0,{5},RobotMode();"
        class Gripper:
            def get_current_position(self):
                return 127
            def move(self, target, speed, force):
                held.append(target)
                return 0
        adapter = DobotHardwareAdapter.from_components(
            {arm: Dashboard(arm) for arm in ("left", "right")}, {},
            {arm: Gripper() for arm in ("left", "right")})
        result = adapter.stop_and_verify()
        self.assertFalse(result["confirmed"])
        self.assertEqual(set(stopped), {"left", "right"})
        self.assertEqual(held, [127, 127])
        self.assertTrue(result["arms"]["right"]["stop_confirmed"])

    def test_live_mode_cannot_be_enabled_by_default(self):
        with TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                Gateway(load_scene_config(SCENE), temp, mode="live")


class ChemistryGatewayTests(unittest.TestCase):
    """The dobot.* atom group rides the embodied envelope; these pin that seam."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.gateway = Gateway(load_scene_config(SCENE), Path(self.temp.name) / "data")
        self.server = GatewayServer(("127.0.0.1", 0), self.gateway, token="test-token")
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02})
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.gateway.close()
        self.temp.cleanup()

    def request(self, path, body=None, key=None):
        headers = {"Authorization": "Bearer test-token"}
        if key:
            headers["Idempotency-Key"] = key
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        try:
            response = urlopen(Request(self.url + path, data=data, headers=headers), timeout=10)
        except HTTPError as error:
            response = error
        with response:
            return response.code, json.loads(response.read())

    def finish(self, execution_id):
        # The 20-node graph really sleeps through its 8 s and 5 s stagger waits.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            result = self.gateway.status(execution_id)
            if result["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
                return result
            time.sleep(0.01)
        self.fail("Execution did not terminate")

    def test_shipped_chemistry_example_is_the_bridge_output(self):
        from task_skills.gateway.chemistry import to_runtime_package

        shipped = chemistry_package()
        with open(CHEMISTRY_DAG, encoding="utf-8") as handle:
            authored = json.load(handle)
        rebuilt = to_runtime_package(authored, run_id=shipped["run_id"],
                                     catalog_revision=shipped["catalog_revision"])
        self.assertEqual(rebuilt, shipped, "The delivered JSON must be the bridge's own output")
        self.assertEqual(len(shipped["nodes"]), 20)
        self.assertEqual(len(shipped["edges"]), 24)
        self.assertEqual(shipped["entry_node_ids"],
                         ["left_tube_approach", "right_beaker_approach"])

    def test_chemistry_package_validates_as_its_own_family(self):
        plan = self.gateway.catalog.validate(chemistry_package())
        self.assertEqual(plan.family, "chemistry")
        self.assertEqual(len(plan.order), 20)
        operation = plan.operations["left_tube_approach"][0]
        self.assertEqual((operation.operation, operation.arm), ("chemistry_atom", "left"))
        self.assertEqual(operation.parameters["skill"], "arm.move_joint")
        self.assertEqual(operation.parameters["parameters"]["target_ref"], "left.tube.approach")
        pour = plan.operations["pour_into_tube"][0]
        self.assertEqual(pour.parameters["skill"], "arm.rotate_about_pivot")
        self.assertEqual(pour.parameters["parameters"]["speed"], 15)
        for node_id in ("work_stagger_8s", "return_stagger_5s"):
            self.assertEqual(plan.operations[node_id][0].operation, "wait")
            self.assertEqual(plan.arms[node_id], "system")

    def test_chemistry_defaults_come_from_the_chemistry_catalog(self):
        graph = chemistry_package()
        node = next(item for item in graph["nodes"] if item["node_id"] == "left_tube_approach")
        self.assertEqual(node["skill_invocation"]["parameters"], {"target_ref": "left.tube.approach",
                                                                 "grip_requirement": "open",
                                                                 "speed": 20, "tilt_limit_deg": 1})
        node["skill_invocation"]["parameters"] = {"target_ref": "left.tube.approach"}
        normalized = self.gateway.catalog.validate(graph).operations["left_tube_approach"][0]
        self.assertEqual(normalized.parameters["parameters"],
                         {"target_ref": "left.tube.approach", "speed": 20,
                          "grip_requirement": "any", "tilt_limit_deg": 1})

    def test_pose_reference_outside_the_chemistry_profile_is_rejected(self):
        graph = chemistry_package()
        next(item for item in graph["nodes"]
             if item["node_id"] == "left_tube_approach")["skill_invocation"]["parameters"]["target_ref"] = "left.tube.nowhere"
        with self.assertRaises(ContractError) as caught:
            self.gateway.catalog.validate(graph)
        self.assertEqual(caught.exception.status, 422)
        self.assertIn("missing from profile.poses", caught.exception.body["message"])
        code, body = self.request("/v1/execute-graph", {"graph": graph}, "chem-bad-pose")
        self.assertEqual((code, body["code"]), (422, "INVALID_GRAPH"))

    def test_unknown_dobot_skill_is_rejected(self):
        graph = chemistry_package()
        next(item for item in graph["nodes"]
             if item["node_id"] == "pour_into_tube")["skill_invocation"]["skill_id"] = "dobot.arm.stir"
        code, body = self.request("/v1/execute-graph", {"graph": graph}, "chem-bad-skill")
        self.assertEqual((code, body["code"]), (422, "UNSUPPORTED_SKILL"))

    def test_mixing_chemistry_and_embodied_nodes_is_rejected(self):
        graph = chemistry_package()
        graph["nodes"].append({
            "node_id": "pick-B", "node_type": "action", "status": "pending", "timeout_s": 30,
            "skill_invocation": {"skill_id": "embodied.pick", "skill_version": "1.0.0",
                                 "device_id": "dobot-nova-right",
                                 "parameters": {"object_id": "B", "source_ref": "object:B"},
                                 "idempotency_key": "chem-mixed:pick-B"},
            "on_fail": {"strategy": "decision_agent", "max_retries": 0, "allowed_actions": ["safe_stop"]}})
        graph["edges"].append({"source": "right_arm_home", "target": "pick-B", "edge_type": "success"})
        code, body = self.request("/v1/execute-graph", {"graph": graph}, "chem-mixed")
        self.assertEqual((code, body["code"]), (422, "INVALID_GRAPH"))
        self.assertIn("one skill family", body["message"])

    def test_catalog_publishes_the_dobot_atom_group(self):
        code, body = self.request("/v1/catalog")
        self.assertEqual(code, 200)
        published = {skill["skill_id"]: skill for skill in body["atomic_skills"]}
        self.assertEqual(set(CHEMISTRY_SKILLS) - set(published), set())
        self.assertEqual(published["dobot.arm.move_joint"]["parameters"]["speed"]["default"], 20)
        self.assertEqual(published["dobot.gripper.set_position"]["parameters"]["speed"]["default"], 10)
        self.assertEqual(published["dobot.arm.rotate_about_pivot"]["parameters"]["speed"]["default"], 15)
        self.assertEqual(published["dobot.arm.rotate_about_pivot"]["parameters"]["source_ref"]["const"],
                         "right.beaker.work")
        self.assertFalse(published["dobot.gripper.set_position"]["verification"]["physical_success_verified"])
        self.assertIn("dobot.arm.move_joint", body["devices"][0]["supported_skill_ids"])

    def test_chemistry_package_executes_in_mock_mode_without_the_object_state_machine(self):
        before = self.gateway.store.object_state()
        code, body = self.request("/v1/execute-graph", {"graph": chemistry_package()}, "chem-e2e")
        self.assertEqual(code, 202)
        done = self.finish(body["execution_id"])
        self.assertEqual(done["status"], "succeeded")
        self.assertEqual(len(done["node_statuses"]), 20)
        self.assertEqual(set(done["node_statuses"].values()), {"succeeded"})
        self.assertEqual(self.gateway.store.object_state(), before,
                         "dobot.* atoms move a tool, not registered objects A/B")
        events = self.gateway.store.events(body["execution_id"])
        completed = {event["node_id"] for event in events if event["event_type"] == "action_completed"}
        self.assertEqual(completed, {"left_tube_approach", "right_beaker_approach", "left_tube_insert",
                                     "right_beaker_insert", "left_tube_close", "right_beaker_close",
                                     "right_beaker_to_work", "left_tube_to_work", "pour_into_tube",
                                     "right_beaker_untip", "left_tube_return", "left_tube_release",
                                     "left_tube_retreat", "right_beaker_return", "right_beaker_release",
                                     "right_beaker_retreat", "left_arm_home", "right_arm_home"})

    def test_the_package_itself_is_accepted_as_the_whole_body(self):
        # The counterparty posts the serialized graph document, so the gateway must
        # accept it unwrapped as well as under {"graph": ...}.
        code, body = self.request("/v1/execute-graph", chemistry_package())
        self.assertEqual(code, 202)
        self.assertEqual(body["graph_id"], "runtime-right_beaker_pour_into_left_tube")
        self.assertEqual(self.finish(body["execution_id"])["status"], "succeeded")

    def test_unwrapped_body_still_needs_a_key_on_the_idempotent_route(self):
        code, body = self.request("/v1/execute-graph/idempotent", chemistry_package())
        self.assertEqual((code, body["code"], body["path"]), (422, "INVALID_GRAPH", "Idempotency-Key"))

    def test_live_chemistry_start_gate_is_the_driver_preflight_not_the_home_pose(self):
        catalog = Catalog(self.gateway.scene, "live")
        plan = catalog.validate(chemistry_package())
        calls = []

        class FakeChemistryAdapter:
            def preflight(self, package_plan):
                calls.append("preflight")
                return {"success": True, "note": "fake chemistry preflight"}

            def snapshot_all(self):
                calls.append("snapshot_all")
                return {"hardware_connected": True, "mode": "chemistry"}

            def configure_robot(self, arm, speed):
                calls.append("configure_robot")

            def execute_atomic(self, node, scene):
                calls.append((node.arm, node.operation))
                return {"fake": True}

            def stop_and_verify(self):
                return {"confirmed": True}

            def close(self):
                pass

        with TemporaryDirectory() as temp:
            result = run_graph(plan, FakeChemistryAdapter(), self.gateway.scene, temp,
                               threading.Event(), lambda *args, **kwargs: None, live=True)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["family"], "chemistry")
        self.assertEqual(result["preflight"], {"success": True, "note": "fake chemistry preflight"})
        self.assertEqual(calls[0], "preflight")
        self.assertNotIn("configure_robot", calls, "The embodied home gate must not run for chemistry")
        atoms = [call for call in calls if isinstance(call, tuple)]
        self.assertEqual(sorted(arm for arm, operation in atoms if operation == "move_safe_pose"),
                         ["left", "right"], "A finished graph still parks both arms at home")

    def test_chemistry_live_preflight_failure_stops_before_any_atom(self):
        catalog = Catalog(self.gateway.scene, "live")
        plan = catalog.validate(chemistry_package())
        calls = []

        class RefusingAdapter:
            def preflight(self, package_plan):
                calls.append("preflight")
                return {"success": False, "error": "left.tube.work is not reachable from the current state"}

            def snapshot_all(self):
                return {}

            def configure_robot(self, arm, speed):
                calls.append("configure_robot")

            def execute_atomic(self, node, scene):
                calls.append(("atom", node.node_id))
                return {}

            def stop_and_verify(self):
                return {"confirmed": True}

            def close(self):
                pass

        with TemporaryDirectory() as temp:
            result = run_graph(plan, RefusingAdapter(), self.gateway.scene, temp,
                               threading.Event(), lambda *args, **kwargs: None, live=True)
        self.assertEqual(result["status"], "failed")
        self.assertIn("not reachable", result["error"]["message"])
        self.assertEqual([call for call in calls if isinstance(call, tuple)], [])

    def test_chemistry_adapter_preflight_skips_a_package_without_chemistry_atoms(self):
        from task_skills.gateway.chemistry import ChemistryAdapter

        plan = self.gateway.catalog.validate(package(self.gateway.catalog.revision))
        evidence = ChemistryAdapter(threading.Event()).preflight(plan)
        self.assertEqual(evidence, {"success": True, "note": "no chemistry atoms in this package"})

    def test_the_gateway_answers_the_chemistry_drivers_consent_gate(self):
        """A live POST is the on-site consent ``LiveDriver.home`` asks for.

        Without it the final park of a finished chemistry run raised, which turned
        a successful package into ``FINAL_HOME_FAILED`` and quarantined the rig.
        """
        from task_skills.gateway.chemistry import ChemistryAdapter

        adapter = ChemistryAdapter(threading.Event())
        self.assertTrue(adapter.driver().confirm(),
                        "a live gateway run must not fail its final park for want of a prompt")

        injected = ChemistryAdapter(threading.Event(), confirmation=lambda: False)
        self.assertFalse(injected.driver().confirm(),
                         "a confirmation injected by the caller still decides")

    def test_the_gateways_final_park_does_not_raise_for_a_parked_arm(self):
        """The exact call that failed on 2026-10-15 17:23 CST, minus the hardware.

        The run had finished all twenty nodes and then the final park blew up in
        ``LiveDriver.home``. Only the two hardware reads are replaced here; the
        gateway's own path - adapter -> driver -> consent gate -> parked probe -
        is the code that failed.
        """
        from task_skills.layout import physical_nova_home_degrees
        from task_skills.gateway.chemistry import ChemistryAdapter

        adapter = ChemistryAdapter(threading.Event())
        driver = adapter.driver()
        driver._rig = lambda arm, profile, with_motion=True: SimpleNamespace(arm=arm)
        driver._read_state = lambda rig: {"source": "offline", "mode": 5,
                                          "joints": list(physical_nova_home_degrees(rig.arm)),
                                          "pose": [0.0] * 6}
        evidence = adapter.execute_atomic(
            AtomicNode("final:left", 0, "move_safe_pose", "left", timeout_seconds=180),
            self.gateway.scene)
        self.assertTrue(evidence["success"], evidence)
        self.assertTrue(evidence["arms"]["left"]["skipped"], evidence)



if __name__ == "__main__":
    unittest.main()
