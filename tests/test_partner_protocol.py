"""Pins the gateway to the partner's 设备能力原子技能与执行图接口规范 1.0-draft."""

import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError

from task_skills.gateway.contracts import Catalog
from task_skills.gateway.store import Store
from task_skills.gateway.upstream import DeliveryWorker
from task_skills.models import load_scene_config

SCENE = Path(__file__).resolve().parents[1] / "task_skills/scene_config.json"


class StubUpstream:
    def __init__(self, outcome):
        self.outcome, self.calls = outcome, []

    def request(self, path, body=None):
        self.calls.append(path)
        if type(self.outcome) is int:
            raise HTTPError(path, self.outcome, "stub", {}, io.BytesIO(b""))
        return self.outcome


class DeliveryAcknowledgementTests(unittest.TestCase):
    def deliver(self, path, outcome):
        with TemporaryDirectory() as temp:
            store = Store(Path(temp) / "data")
            try:
                store.enqueue("stream-1", path, {"event_id": "event-1"})
                delivered = DeliveryWorker(store, StubUpstream(outcome)).pump()
                return delivered, store.pending_deliveries()
            finally:
                store.db.close()
                store.guard.close()

    def test_a_2xx_json_object_acknowledges_an_event(self):
        for body in ({}, {"status": "stored"}, {"accepted": True}):
            with self.subTest(body=body):
                delivered, pending = self.deliver("/v1/executions/execution-1/events", body)
                self.assertEqual((delivered, pending), (1, []))

    def test_explicit_rejection_is_not_an_acknowledgement(self):
        delivered, pending = self.deliver("/v1/executions/execution-1/events", {"accepted": False})
        self.assertEqual(delivered, 0)
        self.assertEqual(len(pending), 1)

    def test_409_on_an_event_means_already_received(self):
        for path in ("/v1/executions/execution-1/events", "/v1/integrations/device-state-events"):
            with self.subTest(path=path):
                delivered, pending = self.deliver(path, 409)
                self.assertEqual((delivered, pending), (1, []))

    def test_409_on_a_catalog_publish_stays_pending(self):
        delivered, pending = self.deliver("/v1/integrations/catalog/snapshots", 409)
        self.assertEqual(delivered, 0)
        self.assertEqual(len(pending), 1)


class PartnerWireTests(unittest.TestCase):
    """Shapes the partner's receiver accepted or rejected on 2026-11-12 (422 details)."""

    def test_every_gateway_event_type_maps_into_the_partner_enum(self):
        from task_skills.gateway.partner_wire import PARTNER_EVENT_TYPES, to_partner
        gateway_types = ["execution_started", "action_started", "action_completed", "action_failed",
                         "action_cancelled", "action_interrupted", "wait_started", "wait_completed",
                         "execution_finalizing", "arm_finalizing", "arm_finalized",
                         "arm_finalization_failed", "arm_finalization_cancelled", "execution_cancelling",
                         "execution_succeeded", "execution_failed", "execution_cancelled",
                         "execution_interrupted"]
        for event_type in gateway_types:
            with self.subTest(event_type=event_type):
                sent = to_partner("/v1/executions/e-1/events",
                                  {"event_id": "x", "event_type": event_type, "payload": {}})
                self.assertIn(sent["event_type"], PARTNER_EVENT_TYPES)
                self.assertEqual(sent["payload"]["gateway_event_type"], event_type)

    def test_terminal_and_failure_mapping(self):
        from task_skills.gateway.partner_wire import to_partner
        path = "/v1/executions/e-1/events"
        done = to_partner(path, {"event_type": "execution_succeeded", "payload": {"status": "succeeded"}})
        self.assertEqual(done["event_type"], "experiment_completed")
        timeout = to_partner(path, {"event_type": "action_failed", "error_code": "MOTION_TIMEOUT", "retryable": False})
        self.assertEqual(timeout["event_type"], "timeout")
        failed = to_partner(path, {"event_type": "execution_failed",
                                   "payload": {"error": {"code": "SKILL_FAILED", "retryable": False}}})
        self.assertEqual((failed["event_type"], failed["error_code"], failed["retryable"]),
                         ("action_failed", "SKILL_FAILED", False))

    def test_device_event_drops_fields_the_partner_forbids(self):
        from task_skills.gateway.partner_wire import to_partner
        sent = to_partner("/v1/integrations/device-state-events",
                          {"event_id": "d", "device_id": "dobot-nova-left", "capability_version": "1.0.0",
                           "sequence": 1, "occurred_at": "t", "state": "offline", "health": "degraded",
                           "mode": "live", "hardware_connected": False})
        self.assertNotIn("mode", sent)
        self.assertNotIn("hardware_connected", sent)
        self.assertEqual(sent["state"], "offline")

    def test_catalog_is_reduced_to_spec_fields(self):
        from task_skills.gateway.partner_wire import (DEVICE_FIELDS, PARAMETER_FIELDS, SKILL_FIELDS,
                                                      SKILL_KINDS, SNAPSHOT_FIELDS, to_partner)
        snapshot = Catalog(load_scene_config(SCENE), "live").snapshot()
        snapshot["generated_at"] = "2026-11-12T00:00:00Z"
        sent = to_partner("/v1/integrations/catalog/validate", snapshot)
        self.assertLessEqual(set(sent), SNAPSHOT_FIELDS)
        for device in sent["devices"]:
            self.assertLessEqual(set(device), DEVICE_FIELDS)
        for skill in sent["atomic_skills"]:
            with self.subTest(skill=skill["skill_id"]):
                self.assertLessEqual(set(skill), SKILL_FIELDS)
                self.assertIn(skill["skill_kind"], SKILL_KINDS)
                for spec in skill["parameters"].values():
                    self.assertLessEqual(set(spec), PARAMETER_FIELDS)
                for item in skill.get("preconditions", []):
                    self.assertIsInstance(item, dict)

    def test_queued_snapshot_is_never_sent_unless_enabled(self):
        with TemporaryDirectory() as temp:
            store = Store(Path(temp) / "data")
            try:
                store.enqueue("catalog:r", "/v1/integrations/catalog/snapshots", {"devices": []}, "r:snapshots")
                upstream = StubUpstream({"accepted": True})
                self.assertEqual(DeliveryWorker(store, upstream).pump(), 0)
                self.assertEqual(upstream.calls, [])
                self.assertEqual(DeliveryWorker(store, upstream, publish_snapshots=True).pump(), 1)
            finally:
                store.db.close()
                store.guard.close()


class CatalogSnapshotTests(unittest.TestCase):
    def test_devices_offer_every_capability_their_skills_require(self):
        snapshot = Catalog(load_scene_config(SCENE), "mock").snapshot()
        skills = {skill["skill_id"]: skill for skill in snapshot["atomic_skills"]}
        for device in snapshot["devices"]:
            for skill_id in device["supported_skill_ids"]:
                with self.subTest(device=device["device_id"], skill=skill_id):
                    self.assertIn(skill_id, skills)
                    self.assertLessEqual(set(skills[skill_id]["required_capabilities"]),
                                         set(device["capabilities"]))

    def test_skills_carry_the_spec_required_fields(self):
        required = {"skill_id", "version", "name_zh", "lifecycle", "implements", "method",
                    "required_capabilities", "parameters", "completion_conditions",
                    "abort_conditions", "execution_semantics"}
        snapshot = Catalog(load_scene_config(SCENE), "mock").snapshot()
        for skill in snapshot["atomic_skills"]:
            with self.subTest(skill=skill["skill_id"]):
                self.assertLessEqual(required, set(skill))


if __name__ == "__main__":
    unittest.main()
