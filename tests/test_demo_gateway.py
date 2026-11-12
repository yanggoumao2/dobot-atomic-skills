"""Demo override: whatever is posted, the fixed pour package is what runs (mock only)."""

import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from task_skills.gateway.contracts import ContractError
from task_skills.gateway.demo_override import DemoOverrideGateway
from task_skills.gateway.server import GatewayServer
from task_skills.models import load_scene_config

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "task_skills/scene_config.json"
POUR = ROOT / "docs/execution-api/examples/runtime-chemistry-beaker-pour.json"
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


class DemoOverrideGatewayTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.data = Path(self.temp.name) / "data"
        self.gateway = DemoOverrideGateway(load_scene_config(SCENE), self.data, demo_package_path=POUR)
        self.server = GatewayServer(("127.0.0.1", 0), self.gateway, token="t")
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02})
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.server.server_port)
        with open(POUR, encoding="utf-8") as handle:
            self.pour_node_ids = {node["node_id"] for node in json.load(handle)["nodes"]}

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.gateway.close()
        self.temp.cleanup()

    def post(self, path, body, key=None):
        headers = {"Authorization": "Bearer t", "Content-Type": "application/json"}
        if key:
            headers["Idempotency-Key"] = key
        try:
            response = urlopen(Request(self.url + path, data=json.dumps(body).encode(), headers=headers), timeout=10)
        except HTTPError as error:
            response = error
        with response:
            return response.code, json.loads(response.read())

    def wait(self, execution_id, cancel=False):
        if cancel:
            self.gateway.cancel(execution_id)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status = self.gateway.status(execution_id)
            if status["status"] in TERMINAL:
                return status
            time.sleep(0.02)
        self.fail("execution did not terminate")

    def test_any_body_runs_the_pour_and_is_repeatable(self):
        junk = {"hello": "this is not a graph", "nodes": "nope"}
        code, body = self.post("/v1/execute-graph", junk)
        self.assertEqual(code, 202, body)
        # The response must be exactly the normal gateway's: no extra fields.
        self.assertEqual(set(body), {"execution_id", "status", "graph_id", "accepted_at"})
        done = self.wait(body["execution_id"])
        self.assertEqual(done["status"], "succeeded")
        self.assertEqual(set(done["node_statuses"]), self.pour_node_ids)
        self.assertEqual(set(done["node_statuses"].values()), {"succeeded"})
        saved = list((self.data / "demo-incoming").glob("*-execute-graph-*.json"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(json.loads(saved[0].read_text(encoding="utf-8")), junk)

        # A second, different body must start a new run, not collide on node keys.
        other = {"schema_version": "x", "package_id": "partner-1", "nodes": [{"node_id": "pick"}]}
        code, second = self.post("/v1/execute-graph", other)
        self.assertEqual(code, 202, second)
        self.assertNotEqual(second["execution_id"], body["execution_id"])
        self.assertEqual(self.wait(second["execution_id"], cancel=True)["status"] in TERMINAL, True)

    def test_receipt_and_events_carry_the_posted_package_id(self):
        # The partner rejects a receipt whose graph_id differs from the package_id it sent.
        body = {"graph": {"package_id": "runtime_c012853b402343439f1e95c294a32200",
                          "experiment_id": "e3f73afb51e8420c8d1ac3107afc844c", "nodes": []},
                "request_id": "new:runtime_c012853b402343439f1e95c294a32200:entry"}
        code, receipt = self.post("/v1/execute-graph", body, key=body["request_id"])
        self.assertEqual(code, 202, receipt)
        self.assertEqual(receipt["graph_id"], "runtime_c012853b402343439f1e95c294a32200")
        status = self.wait(receipt["execution_id"], cancel=True)
        self.assertEqual(set(status["node_statuses"]), self.pour_node_ids)
        events = self.gateway.store.events(receipt["execution_id"], 0, 1000)
        self.assertEqual({event["graph_id"] for event in events}, {"runtime_c012853b402343439f1e95c294a32200"})
        self.assertTrue(self.gateway.delivery.demo_events)

    def test_demo_events_hide_pour_nodes_from_the_partner(self):
        from task_skills.gateway.partner_wire import to_partner
        path = "/v1/executions/e-1/events"
        node_event = {"event_id": "x", "event_type": "action_completed", "node_id": "left_tube_approach",
                      "device_id": "dobot-nova-left", "skill_id": "dobot.arm.move_joint",
                      "skill_version": "1.0.0", "graph_id": "runtime_c0", "payload": {}}
        sent = to_partner(path, node_event, demo=True)
        self.assertEqual(sent["event_type"], "telemetry_received")
        for field in ("node_id", "device_id", "skill_id", "skill_version"):
            self.assertNotIn(field, sent)
        self.assertEqual(sent["payload"]["demo_node"]["node_id"], "left_tube_approach")
        failed = to_partner(path, dict(node_event, event_type="action_failed", error_code="SKILL_FAILED"), demo=True)
        self.assertEqual((failed["event_type"], failed["error_code"]), ("action_failed", "SKILL_FAILED"))
        done = to_partner(path, {"event_id": "y", "event_type": "execution_succeeded", "payload": {}}, demo=True)
        self.assertEqual(done["event_type"], "experiment_completed")

    def test_partner_push_envelope_retry_does_not_pour_twice(self):
        body = {"graph": {"package_id": "partner-run"}, "resume_from_node_id": None,
                "execution_id": None, "request_id": "req-1"}
        code, first = self.post("/v1/execute-graph", body, key="req-1")
        self.assertEqual(code, 202, first)
        code, retry = self.post("/v1/execute-graph", body, key="req-1")
        self.assertEqual((code, retry["execution_id"]), (202, first["execution_id"]))
        self.wait(first["execution_id"], cancel=True)

    def test_idempotent_route_replays_by_key_and_needs_no_key(self):
        code, first = self.post("/v1/execute-graph/idempotent", {"anything": 1}, key="k-1")
        self.assertEqual(code, 202, first)
        code, again = self.post("/v1/execute-graph/idempotent", {"something else": 2}, key="k-1")
        self.assertEqual(code, 202, again)
        self.assertEqual(again["execution_id"], first["execution_id"])
        self.wait(first["execution_id"], cancel=True)

        code, keyless = self.post("/v1/execute-graph/idempotent", {"anything": 3})
        self.assertEqual(code, 202, keyless)
        self.assertNotEqual(keyless["execution_id"], first["execution_id"])
        status = self.wait(keyless["execution_id"], cancel=True)
        self.assertEqual(set(status["node_statuses"]), self.pour_node_ids)

    def test_two_phase_route_runs_the_pour(self):
        code, checked = self.post("/device-gateway/v1/execution-plans/validate", {"x": 1}, key="tp-1")
        self.assertEqual(code, 200, checked)
        self.assertTrue(checked["validation_token"])
        code, run = self.post("/device-gateway/v1/executions",
                              {"y": 2, "validation_token": checked["validation_token"]}, key="tp-1")
        self.assertEqual(code, 202, run)
        status = self.wait(run["execution_id"], cancel=True)
        self.assertEqual(set(status["node_statuses"]), self.pour_node_ids)

    def test_invalid_demo_package_fails_at_startup(self):
        bad = Path(self.temp.name) / "bad.json"
        bad.write_text(json.dumps({"schema_version": "1", "nodes": []}), encoding="utf-8")
        with self.assertRaises(ContractError):
            DemoOverrideGateway(load_scene_config(SCENE), Path(self.temp.name) / "d2", demo_package_path=bad)


if __name__ == "__main__":
    unittest.main()
