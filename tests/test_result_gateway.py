"""Result upload failures must stay separate from execution outcomes."""

import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from task_skills.gateway.service import Gateway
from task_skills.models import load_scene_config
from task_skills.run_graph import MockAdapter


ROOT = Path(__file__).resolve().parents[1]


def wait_package():
    return {"schema_version": "2.1", "graph_version": "0.1",
            "package_id": "result-test", "experiment_id": "experiment-result-test",
            "entry_node_ids": ["wait"], "edges": [],
            "nodes": [{"node_id": "wait", "node_type": "wait",
                       "status": "pending", "duration_s": 0}]}


class RecordingDelivery:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def capture(self, *args):
        if self.fail:
            raise OSError("simulated disk failure")
        self.calls.append(args)

    def start(self):
        pass

    def close(self):
        pass

    def status(self):
        return {"captured": len(self.calls)}


class ResultGatewayTests(unittest.TestCase):
    def scene(self):
        return load_scene_config(ROOT / "task_skills/scene_config.json")

    def run_execution(self, gateway, package=None):
        response = gateway.submit_new_execution(package or wait_package())
        gateway.worker.join(10)
        self.assertFalse(gateway.worker.is_alive())
        return response["execution_id"]

    def test_capture_has_committed_terminal_and_original_evidence(self):
        with TemporaryDirectory() as temp:
            delivery = RecordingDelivery()
            gateway = Gateway(self.scene(), temp, results_delivery=delivery)
            try:
                execution_id = self.run_execution(gateway)
                self.assertEqual(len(delivery.calls), 1)
                args = delivery.calls[0]
                self.assertEqual(args[0], execution_id)
                self.assertEqual(args[2]["status"], "succeeded")
                self.assertEqual(args[3][-1]["event_type"], "execution_succeeded")
                self.assertTrue((args[4] / "submitted_package.json").is_file())
                self.assertEqual(args[5], "mock")
                self.assertEqual(gateway.result_delivery_status()["capture_pending"], [])
            finally:
                gateway.close()

    def test_capture_failure_preserves_outcome_and_recovers_on_restart(self):
        with TemporaryDirectory() as temp:
            first = Gateway(self.scene(), temp, results_delivery=RecordingDelivery(fail=True))
            try:
                execution_id = self.run_execution(first)
                self.assertEqual(first.status(execution_id)["status"], "succeeded")
                self.assertEqual(first.result_delivery_status()["capture_pending"], [execution_id])
            finally:
                first.close()
            delivery = RecordingDelivery()
            recovered = Gateway(self.scene(), temp, results_delivery=delivery)
            try:
                self.assertEqual(len(delivery.calls), 1)
                self.assertEqual(delivery.calls[0][0], execution_id)
                recovered.maintain()
                self.assertEqual(len(delivery.calls), 1)
            finally:
                recovered.close()

    def test_enabling_channel_does_not_publish_unenrolled_history(self):
        with TemporaryDirectory() as temp:
            old = Gateway(self.scene(), temp)
            try:
                self.run_execution(old)
            finally:
                old.close()
            delivery = RecordingDelivery()
            new = Gateway(self.scene(), temp, results_delivery=delivery)
            try:
                self.assertEqual(delivery.calls, [])
            finally:
                new.close()

    def test_action_failure_preserves_message_and_traceback(self):
        class FailingAdapter(MockAdapter):
            def execute_atomic(self, node, scene):
                raise ValueError("synthetic malformed gripper reply")

        with TemporaryDirectory() as temp:
            package = json.loads((ROOT / "docs/execution-api/examples/runtime-chemistry-beaker-pour.json")
                                 .read_text(encoding="utf-8"))
            delivery = RecordingDelivery()
            gateway = Gateway(self.scene(), temp, adapter_factory=FailingAdapter,
                              results_delivery=delivery)
            try:
                self.run_execution(gateway, package)
                self.assertEqual(delivery.calls[0][2]["status"], "failed")
                failures = [e for e in delivery.calls[0][3]
                            if e["event_type"] == "action_failed"]
                self.assertTrue(failures)
                self.assertEqual(failures[0]["payload"]["message"], "synthetic malformed gripper reply")
                self.assertIn("ValueError", failures[0]["payload"]["traceback"])
            finally:
                gateway.close()

    def test_shutdown_waits_for_previous_execution_still_capturing(self):
        with TemporaryDirectory() as temp:
            delivery = RecordingDelivery()
            gateway = Gateway(self.scene(), temp, results_delivery=delivery)
            entered, release, closed = threading.Event(), threading.Event(), threading.Event()
            original = gateway._capture_result
            first_id = []

            def paused_capture(execution_id):
                if not first_id:
                    first_id.append(execution_id)
                    entered.set()
                    release.wait(5)
                original(execution_id)

            gateway._capture_result = paused_capture
            close_thread = None
            try:
                gateway.submit_new_execution(wait_package())
                self.assertTrue(entered.wait(5))
                older_worker = gateway.worker
                self.run_execution(gateway)

                def close_gateway():
                    gateway.close()
                    closed.set()

                close_thread = threading.Thread(target=close_gateway)
                close_thread.start()
                self.assertFalse(closed.wait(0.1))
                release.set()
                close_thread.join(5)
                self.assertTrue(closed.is_set())
                self.assertFalse(older_worker.is_alive())
                self.assertEqual(len(delivery.calls), 2)
            finally:
                release.set()
                if close_thread is not None:
                    close_thread.join(5)
                if not closed.is_set():
                    gateway.close()

    def test_crash_recovery_capture_includes_registry_interrupted_status(self):
        with TemporaryDirectory() as temp:
            first = Gateway(self.scene(), temp)
            try:
                plan = first.catalog.validate(wait_package(), check_state=False)
                record = first.store.create("crash-result", plan)
                execution_id = record["id"]
                with first.store.transaction() as db:
                    db.execute("INSERT INTO metadata VALUES(?,?)",
                               ("results-capture:" + execution_id, "pending"))
            finally:
                first.close()
            delivery = RecordingDelivery()
            recovered = Gateway(self.scene(), temp, results_delivery=delivery)
            try:
                self.assertEqual(delivery.calls[0][2]["status"], "interrupted")
                self.assertEqual(delivery.calls[0][2]["error"]["code"], "EXECUTION_INTERRUPTED")
            finally:
                recovered.close()


if __name__ == "__main__":
    unittest.main()
