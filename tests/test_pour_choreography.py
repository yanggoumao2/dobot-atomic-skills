import copy
import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from task_skills.chemistry.choreography import ChoreographyError, choreograph
from task_skills.gateway.contracts import Catalog, ContractError
from task_skills.gateway.server import GatewayServer
from task_skills.gateway.service import Gateway
from task_skills.models import load_scene_config

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "task_skills" / "scene_config.json"
REFERENCE = ROOT / "docs" / "execution-api" / "examples" / "runtime-chemistry-beaker-pour.json"


def guidance():
    return {
        "schema_version": "2.1", "graph_version": "0.1",
        "package_id": "runtime-right_beaker_pour_into_left_tube",
        "run_id": "beaker-pour-into-tube-001",
        "experiment_id": "experiment-right_beaker_pour_into_left_tube",
        "recipe_id": "dobot-chemistry-task", "version": "1.0",
        "catalog_revision": "dobot-43e65f5027ab5e8246e0",
        "entry_node_ids": ["pour"],
        "nodes": [pour_node("pour")], "edges": [],
    }


def pour_node(node_id):
    return {"node_id": node_id, "node_type": "action", "status": "pending",
            "skill_invocation": {
                "skill_id": "embodied.pour", "skill_version": "1.0.0",
                "parameters": {"source_object": "beaker", "target_object": "tube",
                               "tilt_deg": 80, "motion_speed": 20}}}


def without_prefix(value):
    return json.loads(json.dumps(value).replace("pour.", ""))


class ChoreographyTests(unittest.TestCase):
    def test_expansion_equals_the_published_live_package(self):
        package, lineage = choreograph(guidance())
        reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        ours = {node["node_id"]: node for node in without_prefix(package["nodes"])}
        self.assertEqual(ours, {node["node_id"]: node for node in reference["nodes"]})
        edge = lambda item: (item["source"], item["target"], item["edge_type"])
        self.assertEqual(sorted(map(edge, without_prefix(package["edges"]))),
                         sorted(map(edge, reference["edges"])))
        self.assertEqual(without_prefix(package["entry_node_ids"]), sorted(reference["entry_node_ids"]))
        self.assertEqual(sorted(lineage["pour"]), sorted(n["node_id"] for n in package["nodes"]))

    def test_the_arms_only_tilt_after_both_reach_their_work_poses(self):
        package, _ = choreograph(guidance())
        sources = {e["source"] for e in package["edges"] if e["target"] == "pour.pour_into_tube"}
        self.assertEqual(sources, {"pour.left_tube_to_work", "pour.right_beaker_to_work"})

    def test_ordered_pours_join_every_exit_to_every_entry(self):
        document = guidance()
        document["nodes"] = [pour_node("first"), pour_node("second")]
        document["edges"] = [{"source": "first", "target": "second", "edge_type": "success"}]
        package, _ = choreograph(document)
        self.assertEqual(len(package["nodes"]), 40)
        joins = {(e["source"], e["target"]) for e in package["edges"]
                 if e["source"].startswith("first.") and e["target"].startswith("second.")}
        self.assertEqual(joins, {("first.%s_arm_home" % arm, "second." + entry)
                                 for arm in ("left", "right")
                                 for entry in ("left_tube_approach", "right_beaker_approach")})

    def test_unsafe_guidance_is_rejected(self):
        def changed(**parameters):
            document = guidance()
            document["nodes"][0]["skill_invocation"]["parameters"].update(parameters)
            return document

        unordered = guidance()
        unordered["nodes"] = [pour_node("first"), pour_node("second")]
        cycle = guidance()
        cycle["nodes"] = [pour_node("first"), pour_node("second")]
        cycle["edges"] = [{"source": "first", "target": "second", "edge_type": "success"},
                          {"source": "second", "target": "first", "edge_type": "success"}]
        for document in (changed(source_object="tube"), changed(target_ref="left.tube.grasp"),
                         changed(tilt_deg=120), changed(motion_speed=36), unordered, cycle):
            with self.subTest(document=document), self.assertRaises(ChoreographyError):
                choreograph(document)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog(load_scene_config(SCENE))

    def test_validation_returns_an_executable_chemistry_plan(self):
        plan = self.catalog.validate(guidance(), check_state=False)
        self.assertEqual(plan.family, "chemistry")
        self.assertEqual(len(plan.order), 20)
        self.assertEqual(self.catalog.validate(plan.package, check_state=False).fingerprint, plan.fingerprint)

    def test_guidance_errors_use_the_contract_error_shape(self):
        document = guidance()
        document["nodes"][0]["skill_invocation"]["parameters"]["tilt_deg"] = 120
        with self.assertRaises(ContractError) as caught:
            self.catalog.validate(document, check_state=False)
        self.assertEqual(caught.exception.status, 422)
        self.assertEqual(caught.exception.body["code"], "INVALID_GRAPH")

    def test_a_pour_cannot_overlap_another_node_on_the_same_arm(self):
        document = guidance()
        document["nodes"].append({
            "node_id": "extra", "node_type": "action", "status": "pending",
            "skill_invocation": {"skill_id": "dobot.arm.move_to_home", "skill_version": "1.0.0",
                                 "device_id": "dobot-nova-left", "idempotency_key": "run:extra",
                                 "parameters": {}}})
        document["entry_node_ids"].append("extra")
        with self.assertRaises(ContractError) as caught:
            self.catalog.validate(document, check_state=False)
        self.assertEqual(caught.exception.body["code"], "RESOURCE_CONFLICT")

    def test_the_catalog_publishes_the_guidance_skill(self):
        snapshot = self.catalog.snapshot()
        self.assertIn("embodied.pour", [skill["skill_id"] for skill in snapshot["atomic_skills"]])
        self.assertIn("embodied.pour", snapshot["devices"][0]["supported_skill_ids"])

    def test_atomic_packages_are_not_rewritten(self):
        reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
        plan = self.catalog.validate(copy.deepcopy(reference), check_state=False)
        self.assertEqual(plan.package, reference)


class GatewayHttpTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.gateway = Gateway(load_scene_config(SCENE), Path(self.temp.name) / "data")
        self.server = GatewayServer(("127.0.0.1", 0), self.gateway, token="test-token")
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.01,))
        self.thread.start()
        self.addCleanup(self._stop)

    def _stop(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.gateway.close()

    def request(self, path, body=None):
        request = Request("http://127.0.0.1:%d%s" % (self.server.server_port, path),
                          data=None if body is None else json.dumps(body).encode("utf-8"),
                          headers={"Authorization": "Bearer test-token",
                                   "Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())

    def test_a_posted_guidance_graph_runs_every_atom(self):
        status, accepted = self.request("/v1/execute-graph", guidance())
        self.assertEqual(status, 202)
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            _, state = self.request("/v1/executions/" + accepted["execution_id"])
            if state["status"] not in {"accepted", "running", "finalizing"}:
                break
            time.sleep(0.1)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(len(state["node_statuses"]), 20)
        self.assertEqual(set(state["node_statuses"].values()), {"succeeded"})

    def test_a_rejected_guidance_graph_starts_nothing(self):
        document = guidance()
        document["nodes"][0]["skill_invocation"]["parameters"]["source_object"] = "tube"
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/execute-graph", document)
        self.assertEqual(caught.exception.code, 422)
        self.assertIsNone(self.gateway.active)


if __name__ == "__main__":
    unittest.main()
