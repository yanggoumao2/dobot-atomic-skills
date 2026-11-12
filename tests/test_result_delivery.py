import base64
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from task_skills.results import DeliveryError, ResultsDelivery, deliver_bundle, make_bundle
from task_skills.results.delivery import _execution_files
from task_skills.results.receiver import ReceiverServer


def sample_bundle(*, bundle_id="test-bundle", result=None, execution_dir=None):
    return make_bundle(
        "exec-17",
        {
            "task_id": "task-3",
            "package_id": "package-9",
            "target_result": {"placed": True},
            "task_constraints": {"timeout_s": 30},
            "actions": [{"command": "move", "target": [1, 2, 3]}],
            "equipment_config": {"arm": "left"},
        },
        result or {
            "source": "synthetic_fixture",
            "execution_status": "completed",
            "task_outcome": "unknown",
            "actual_commands": [{"command": "move", "sent": True}],
            "actual_state_feedback": {"joint_state": [0, 0, 0]},
        },
        [{"event": "step_finished", "timestamp": "2026-10-23T10:00:00Z"}],
        execution_dir,
        "real",
        bundle_id=bundle_id,
    )


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.025)
    return predicate()


class ResultDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_bundle_preserves_source_hashes_and_marks_synthetic_provenance(self):
        execution_dir = self.tmp_path / "execution"
        execution_dir.mkdir()
        (execution_dir / "run.log").write_text("command returned 0\n", encoding="utf-8")
        (execution_dir / "frame.png").write_bytes(b"frame-pixels")
        (execution_dir / ".result_capture_pending").write_text("pending", encoding="utf-8")
        (execution_dir / ".hidden").mkdir()
        (execution_dir / ".hidden" / "private.log").write_text("ignore", encoding="utf-8")

        bundle = sample_bundle(execution_dir=execution_dir)

        self.assertEqual(bundle["provenance"]["kind"], "synthetic")
        self.assertEqual(bundle["result"]["execution_status"], "completed")
        self.assertEqual(bundle["result"]["task_outcome"], "unknown")
        self.assertEqual(len(bundle["data_categories"]), 10)
        self.assertEqual(bundle["data_categories"]["actual_state_feedback"]["status"], "available")
        self.assertEqual(bundle["data_categories"]["data_quality_collection"]["status"], "unknown")
        self.assertEqual([item["name"] for item in bundle["media"]["items"]], ["frame.png"])
        names = {attachment["name"] for attachment in bundle["attachments"]}
        self.assertIn("run.log", names)
        self.assertNotIn(".result_capture_pending", names)
        self.assertNotIn(".hidden/private.log", names)

        attachments = {item["attachment_id"]: item for item in bundle["attachments"]}
        for source_name in ("package", "result", "events"):
            source = bundle["source_documents"][source_name]
            raw = base64.b64decode(attachments[source["attachment_id"]]["data_base64"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), source["sha256"])
            self.assertIsNotNone(json.loads(raw.decode("utf-8")))
        media = bundle["media"]["items"][0]
        media_attachment = attachments[media["attachment_id"]]
        self.assertEqual(base64.b64decode(media_attachment["data_base64"]), b"frame-pixels")
        self.assertEqual(media["sha256"], hashlib.sha256(b"frame-pixels").hexdigest())

    def test_receiver_accepts_idempotent_duplicate_and_rejects_conflicting_content(self):
        with ReceiverServer(self.tmp_path / "receiver") as receiver:
            bundle = sample_bundle()
            first = deliver_bundle(receiver.url, bundle)
            duplicate = deliver_bundle(receiver.url, bundle)
            self.assertEqual(first, duplicate)
            self.assertIs(first["accepted"], True)
            self.assertEqual(first["bundle_id"], bundle["bundle_id"])

            detail_url = receiver.url + first["receipt_url"]
            import urllib.request

            with urllib.request.urlopen(detail_url, timeout=2) as response:
                detail = json.loads(response.read().decode("utf-8"))
            self.assertEqual(detail["content_digest"], first["content_digest"])
            self.assertEqual(detail["bundle"]["bundle_id"], bundle["bundle_id"])

            conflicting = sample_bundle(result={"source": "synthetic", "task_outcome": "success"})
            with self.assertRaisesRegex(DeliveryError, "HTTP 409"):
                deliver_bundle(receiver.url, conflicting)

    def test_receiver_requires_token_when_configured(self):
        with ReceiverServer(self.tmp_path / "receiver", token="local-secret") as receiver:
            with self.assertRaisesRegex(DeliveryError, "HTTP 401"):
                deliver_bundle(receiver.url, sample_bundle())
            ack = deliver_bundle(receiver.url, sample_bundle(), token="local-secret")
            self.assertIs(ack["accepted"], True)

    def test_blocking_server_wrapper_stops_cleanly(self):
        receiver = ReceiverServer(self.tmp_path / "receiver")
        caller = threading.Thread(target=receiver.serve_forever, daemon=True)
        caller.start()
        try:
            self.assertTrue(
                wait_for(lambda: self._health_is_ready(receiver.url), timeout=3.0)
            )
        finally:
            receiver.stop()
            caller.join(timeout=2.0)
        self.assertFalse(caller.is_alive())

    @staticmethod
    def _health_is_ready(url):
        import urllib.request

        try:
            with urllib.request.urlopen(url + "/health", timeout=0.5) as response:
                return response.status == 200
        except OSError:
            return False

    def test_delivery_rejects_mismatched_acknowledgement(self):
        class BadAckHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                envelope = json.loads(self.rfile.read(length).decode("utf-8"))
                raw = json.dumps(
                    {
                        "accepted": True,
                        "bundle_id": envelope["bundle"]["bundle_id"],
                        "content_digest": "0" * 64,
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        server = ThreadingHTTPServer(("127.0.0.1", 0), BadAckHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            with self.assertRaisesRegex(DeliveryError, "content_digest"):
                deliver_bundle(url, sample_bundle())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_background_retry_survives_restart_and_recapture_is_idempotent(self):
        # Reserve an ephemeral port so the first attempt fails, then bind the
        # mock receiver to that same port and recreate the sender from its queue.
        import socket

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        receiver_url = f"http://127.0.0.1:{port}"
        queue = self.tmp_path / "sender"
        execution_dir = self.tmp_path / "execution"
        execution_dir.mkdir()
        (execution_dir / "result.json").write_text("first capture", encoding="utf-8")

        first_sender = ResultsDelivery(queue, receiver_url)
        bundle_id = first_sender.capture(
            "exec-restart",
            {"task_id": "task", "package_id": "pkg"},
            {"source": "synthetic", "task_outcome": "unknown"},
            [],
            execution_dir,
            "synthetic",
        )
        first_sender.start()
        try:
            self.assertTrue(wait_for(lambda: first_sender.status()["items"][0]["attempts"] >= 1))
            self.assertEqual(first_sender.status()["items"][0]["state"], "pending")
        finally:
            first_sender.close()

        (execution_dir / ".result_capture_pending").write_text("transient", encoding="utf-8")
        with ReceiverServer(self.tmp_path / "receiver", port=port) as receiver:
            restarted = ResultsDelivery(queue, receiver_url)
            recaptured_id = restarted.capture(
                "exec-restart",
                {"task_id": "task", "package_id": "pkg"},
                {"source": "synthetic", "task_outcome": "success"},
                [{"different": True}],
                execution_dir,
                "synthetic",
            )
            self.assertEqual(recaptured_id, bundle_id)
            restarted.start()
            try:
                self.assertTrue(wait_for(lambda: restarted.status()["delivered"] == 1, timeout=6.0))
                item = restarted.status()["items"][0]
                self.assertGreaterEqual(item["attempts"], 2)
                self.assertIsNone(item["last_error"])
                import urllib.request

                with urllib.request.urlopen(receiver.url + item["ack"]["receipt_url"], timeout=2) as response:
                    receipt = json.loads(response.read().decode("utf-8"))
                self.assertEqual(receipt["bundle"]["result"]["task_outcome"], "unknown")
            finally:
                restarted.close()

    def test_remote_plain_http_requires_explicit_opt_in(self):
        bundle = sample_bundle()
        with self.assertRaisesRegex(ValueError, "allow_http=True"):
            deliver_bundle("http://example.com", bundle)

    def test_gateway_status_fallback_and_runtime_mode_protect_provenance(self):
        bundle = make_bundle(
            "exec-status",
            {"task_id": "task", "package_id": "pkg"},
            {"status": "completed", "source": "real"},
            [],
            None,
            "simulation",
            bundle_id="status-fallback",
        )
        self.assertEqual(bundle["execution"]["execution_status"], "completed")
        self.assertEqual(bundle["result"]["task_outcome"], "unknown")
        self.assertEqual(bundle["provenance"]["kind"], "mock")
        synthetic = make_bundle(
            "exec-synthetic-mode",
            {"task_id": "task", "package_id": "pkg", "source": "real"},
            {"source": "real"},
            [],
            None,
            "synthetic",
            bundle_id="synthetic-mode",
        )
        self.assertEqual(synthetic["provenance"]["kind"], "synthetic")

    def test_oversized_media_is_hashed_and_reported_without_inline_read(self):
        execution_dir = self.tmp_path / "large-execution"
        execution_dir.mkdir()
        media_path = execution_dir / "long-run.mp4"
        media_path.write_bytes(b"0" * (16 * 1024 * 1024 + 1))

        bundle = sample_bundle(execution_dir=execution_dir)

        self.assertEqual(bundle["media"]["items"], [])
        self.assertEqual(len(bundle["media"]["missing"]), 1)
        missing = bundle["media"]["missing"][0]
        self.assertEqual(missing["name"], "long-run.mp4")
        self.assertEqual(missing["status"], "export_restricted")
        self.assertEqual(missing["size_bytes"], 16 * 1024 * 1024 + 1)
        self.assertEqual(missing["sha256"], hashlib.sha256(media_path.read_bytes()).hexdigest())
        self.assertEqual(bundle["data_categories"]["video_images"]["status"], "export_restricted")

    def test_category_status_strings_keep_missing_semantics(self):
        bundle = make_bundle(
            "exec-categories",
            {
                "task_id": "task",
                "package_id": "pkg",
                "data_categories": {
                    "identifiers": "可提供",
                    "task_targets_constraints": {
                        "status": "本次未记录",
                        "missing_reason": "No target field was collected.",
                    },
                },
            },
            {"source": "synthetic"},
            [],
            None,
            "synthetic",
            bundle_id="category-statuses",
        )
        self.assertEqual(bundle["data_categories"]["identifiers"]["status"], "available")
        missing = bundle["data_categories"]["task_targets_constraints"]
        self.assertEqual(missing["status"], "not_recorded")
        self.assertEqual(missing["missing_reason"], "No target field was collected.")

    def test_execution_directory_symlinks_are_not_attached(self):
        execution_dir = self.tmp_path / "safe-execution"
        outside_dir = self.tmp_path / "outside"
        execution_dir.mkdir()
        outside_dir.mkdir()
        (outside_dir / "private.log").write_text("outside", encoding="utf-8")
        try:
            (execution_dir / "external").symlink_to(outside_dir, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"directory symlinks are unavailable: {exc}")

        bundle = sample_bundle(execution_dir=execution_dir)

        self.assertNotIn("external/private.log", {item["name"] for item in bundle["attachments"]})

    def test_execution_file_resolving_outside_root_is_excluded(self):
        execution_dir = self.tmp_path / "safe-execution"
        outside_dir = self.tmp_path / "outside"
        execution_dir.mkdir()
        outside_dir.mkdir()
        candidate = execution_dir / "junction-target.log"
        candidate.write_text("must not escape", encoding="utf-8")
        outside_target = outside_dir / "junction-target.log"
        original_resolve = Path.resolve

        def resolve_with_external_target(path, strict=False):
            if path == candidate:
                return outside_target
            return original_resolve(path, strict=strict)

        with patch.object(Path, "resolve", resolve_with_external_target):
            files = _execution_files(execution_dir)

        self.assertNotIn("junction-target.log", {name for name, _ in files})

    def test_worker_survives_attempt_record_write_failure_and_delivers_other_records(self):
        import task_skills.results.delivery as delivery_module

        queue = self.tmp_path / "sender"
        with ReceiverServer(self.tmp_path / "receiver") as receiver:
            sender = ResultsDelivery(queue, receiver.url)
            for execution_id in ("first", "second"):
                sender.capture(
                    execution_id,
                    {"task_id": "task", "package_id": execution_id},
                    {"source": "synthetic", "task_outcome": "unknown"},
                    [],
                    None,
                    "synthetic",
                )

            original_write = delivery_module._atomic_write_json
            failed_once = {"value": False}

            def fail_first_write(path, value):
                if not failed_once["value"]:
                    failed_once["value"] = True
                    raise OSError("simulated disk busy")
                return original_write(path, value)

            with patch("task_skills.results.delivery._atomic_write_json", side_effect=fail_first_write):
                sender.start()
                try:
                    self.assertTrue(wait_for(lambda: sender.status()["delivered"] == 2))
                    status = sender.status()
                finally:
                    sender.close()

            self.assertIn("simulated disk busy", status["last_worker_error"])

    def test_worker_reports_and_recovers_from_malformed_retry_time(self):
        queue = self.tmp_path / "sender"
        with ReceiverServer(self.tmp_path / "receiver") as receiver:
            sender = ResultsDelivery(queue, receiver.url)
            sender.capture(
                "bad-retry-time",
                {"task_id": "task", "package_id": "pkg"},
                {"source": "synthetic"},
                [],
                None,
                "synthetic",
            )
            record_path = next((queue / "outbox").glob("*.json"))
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["next_attempt_at"] = "invalid-time"
            record_path.write_text(json.dumps(record), encoding="utf-8")
            sender.start()
            try:
                self.assertTrue(
                    wait_for(lambda: "ValueError" in (sender.status()["last_worker_error"] or ""))
                )
                record["next_attempt_at"] = 0.0
                record_path.write_text(json.dumps(record), encoding="utf-8")
                self.assertTrue(wait_for(lambda: sender.status()["delivered"] == 1, timeout=4.0))
            finally:
                sender.close()


if __name__ == "__main__":
    unittest.main()
