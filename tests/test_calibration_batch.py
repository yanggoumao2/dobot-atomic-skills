"""Batch delete / re-split of intrinsics images, through the service and the HTTP route."""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

from task_skills.calibration.http_server import create_server
from task_skills.calibration.service import CalibrationService
from task_skills.calibration.state import CalibrationError, SessionStore


def png(seed: int) -> bytes:
    """Distinct blank frames: duplicates are rejected, so each image differs by one pixel."""
    frame = np.zeros((960, 1280, 3), dtype=np.uint8)
    frame[0, seed] = 255
    ok, encoded = cv2.imencode(".png", frame)
    assert ok
    return encoded.tobytes()


class BatchIntrinsicsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="calibration-batch-")
        self.addCleanup(temp.cleanup)
        self.service = CalibrationService(SessionStore(Path(temp.name) / "sessions"))
        self.addCleanup(self.service.close)
        self.session_id = self.service.create_session({"display_name": "batch", "mode": "real"})["session_id"]
        self.ids = [self.service.add_intrinsics_image(self.session_id, png(i), "image/png", "train",
                                                      f"board_{i:03d}.png")["id"] for i in range(6)]

    def images(self):
        return self.service.session_view(self.session_id)["intrinsics"]["images"]

    def test_batch_split_changes_only_the_named_images(self):
        result = self.service.batch_intrinsics_images(
            self.session_id, {"action": "split", "split": "val", "image_ids": self.ids[1:3]})
        self.assertEqual(result["count"], 2)
        self.assertEqual([image["split"] for image in self.images()],
                         ["train", "val", "val", "train", "train", "train"])
        again = self.service.batch_intrinsics_images(
            self.session_id, {"action": "split", "split": "val", "image_ids": self.ids[1:3]})
        self.assertEqual(again["count"], 0)  # already val: nothing changed, nothing invalidated

    def test_batch_delete_and_clear_all(self):
        result = self.service.batch_intrinsics_images(
            self.session_id, {"action": "delete", "image_ids": [self.ids[0], self.ids[5]]})
        self.assertEqual((result["count"], result["remaining"]), (2, 4))
        self.assertEqual([image["id"] for image in self.images()], self.ids[1:5])
        cleared = self.service.batch_intrinsics_images(self.session_id, {"action": "delete", "all": True})
        self.assertEqual((cleared["count"], cleared["remaining"]), (4, 0))
        # The gallery accepts the same frames again after a reset.
        self.service.add_intrinsics_image(self.session_id, png(0), "image/png", "train", "again.png")
        self.assertEqual(len(self.images()), 1)

    def test_an_unknown_id_rejects_the_whole_batch(self):
        with self.assertRaises(CalibrationError) as error:
            self.service.batch_intrinsics_images(
                self.session_id, {"action": "delete", "image_ids": [self.ids[0], "img-missing"]})
        self.assertEqual(error.exception.code, "IMAGE_NOT_FOUND")
        self.assertEqual(len(self.images()), 6)

    def test_invalid_requests_are_refused(self):
        for body, code in (({"action": "rename", "all": True}, "INVALID_BATCH_ACTION"),
                           ({"action": "split", "split": "test", "all": True}, "INVALID_SPLIT"),
                           ({"action": "delete"}, "INVALID_IMAGE_IDS"),
                           ({"action": "delete", "image_ids": []}, "INVALID_IMAGE_IDS"),
                           ({"action": "delete", "all": "yes"}, "INVALID_IMAGE_IDS")):
            with self.subTest(body=body), self.assertRaises(CalibrationError) as error:
                self.service.batch_intrinsics_images(self.session_id, body)
            self.assertEqual(error.exception.code, code)
        self.assertEqual(len(self.images()), 6)

    def test_http_route_runs_the_batch(self):
        temp = tempfile.TemporaryDirectory(prefix="calibration-batch-http-")
        self.addCleanup(temp.cleanup)
        web_root = Path(__file__).resolve().parents[1] / "task_skills" / "calibration" / "web"
        server = create_server("127.0.0.1", 0, Path(temp.name) / "sessions", web_root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        host, port = server.server_address
        conn = http.client.HTTPConnection(host, port, timeout=10)
        conn.request("GET", "/api/bootstrap")
        token = json.loads(conn.getresponse().read())["csrf_token"]
        headers = {"Content-Type": "application/json", "X-Calibration-Token": token,
                   "Origin": f"http://{host}:{port}", "Host": f"{host}:{port}"}
        conn.request("POST", "/api/sessions", json.dumps({"display_name": "http", "mode": "real"}), headers)
        session_id = json.loads(conn.getresponse().read())["session"]["session_id"]
        ids = []
        for i in range(3):
            conn.request("POST", f"/api/sessions/{session_id}/intrinsics/images?split=train&filename=b{i}.png",
                         png(i), {**headers, "Content-Type": "image/png"})
            ids.append(json.loads(conn.getresponse().read())["image"]["id"])
        conn.request("POST", f"/api/sessions/{session_id}/intrinsics/images/batch",
                     json.dumps({"action": "delete", "image_ids": ids[:2]}), headers)
        response = conn.getresponse()
        body = json.loads(response.read())
        self.assertEqual(response.status, 200, body)
        self.assertEqual((body["count"], body["remaining"]), (2, 1))


if __name__ == "__main__":
    unittest.main()
