import base64
import json
import tempfile
import unittest
from pathlib import Path

from task_skills.results.common import validate_bundle
from task_skills.results_demo import SCENARIOS, _CHEMISTRY_PACKAGE, run_demo


class ResultsDemoTests(unittest.TestCase):
    def test_loopback_demo_delivers_gateway_run_and_synthetic_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "demo"
            report = run_demo(output)

            self.assertEqual(report["demo_status"], "passed")
            self.assertFalse(report["formal_partner_3_3_receiver"])
            self.assertFalse(report["hardware_connected"])
            self.assertTrue(report["gateway"]["bundle_delivered"])
            self.assertEqual(report["gateway"]["terminal_status"], "succeeded")
            self.assertEqual(len(report["synthetic_scenarios"]), 9)
            self.assertTrue(report["protocol_checks"]["exact_duplicate_reuses_receipt"])
            self.assertTrue(report["protocol_checks"]["mismatched_duplicate_returns_http_409"])

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["synthetic_fixture_count"], len(SCENARIOS))
            self.assertFalse(manifest["hardware_connected"])
            self.assertTrue((output / "gateway-package.json").is_file())
            self.assertTrue((output / "gateway-receipt.json").is_file())
            gateway_receipt = json.loads((output / "gateway-receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(gateway_receipt["bundle"]["provenance"]["kind"], "mock")
            self.assertEqual(gateway_receipt["bundle"]["execution"]["execution_status"], "succeeded")

            expected_outcomes = {scenario["name"]: (scenario["terminal_status"], scenario["task_outcome"])
                                 for scenario in SCENARIOS}
            for scenario in SCENARIOS:
                case_dir = output / "cases" / scenario["name"]
                bundle = json.loads((case_dir / "bundle.json").read_text(encoding="utf-8"))
                receipt = json.loads((case_dir / "receipt.json").read_text(encoding="utf-8"))
                validate_bundle(bundle)
                self.assertEqual(bundle["provenance"]["kind"], "synthetic")
                self.assertTrue(bundle["identifiers"]["execution_id"].startswith(
                    "synthetic-execution-" + scenario["name"] + "-"))
                self.assertTrue(bundle["bundle_id"].startswith("synthetic-bundle-"))
                self.assertEqual(receipt["bundle_id"], bundle["bundle_id"])
                self.assertEqual(receipt["bundle"], bundle)
                self.assertEqual(
                    (bundle["execution"]["execution_status"], bundle["execution"]["task_outcome"]),
                    expected_outcomes[scenario["name"]],
                )

                attachments = {item["attachment_id"]: json.loads(
                    base64.b64decode(item["data_base64"], validate=True).decode("utf-8"))
                    for item in bundle["attachments"]}
                fixture_events = attachments["source-events"]
                fixture_package = attachments["source-package"]
                fixture_result = attachments["source-result"]
                self.assertTrue(fixture_package["package_id"].startswith("synthetic-package-"))
                self.assertFalse(fixture_package["hardware_connected"])
                self.assertEqual(fixture_package["source_metadata"]["kind"], "synthetic")
                self.assertGreaterEqual(len(fixture_result["stages"]), 2)
                self.assertTrue(fixture_result["commands_sent"])
                self.assertTrue(fixture_result["feedback"])
                for command in fixture_result["commands_sent"]:
                    self.assertTrue(command["command_id"].startswith("synthetic-command-"))
                    self.assertFalse(command["hardware_connected"])
                    self.assertEqual(command["source_metadata"]["kind"], "synthetic")
                for feedback in fixture_result["feedback"]:
                    self.assertTrue(feedback["feedback_id"].startswith("synthetic-feedback-"))
                    self.assertFalse(feedback["hardware_connected"])
                    self.assertEqual(feedback["provenance"]["kind"], "synthetic")
                for event in fixture_events:
                    self.assertTrue(event["event_id"].startswith("synthetic-event-"))
                    self.assertFalse(event["hardware_connected"])
                    self.assertEqual(event["source_metadata"]["kind"], "synthetic")
                if fixture_result.get("measurement"):
                    measurement = fixture_result["measurement"]
                    self.assertTrue(measurement["measurement_id"].startswith("synthetic-measurement-"))
                    self.assertFalse(measurement["hardware_connected"])
                    self.assertEqual(measurement["provenance"]["kind"], "synthetic")

            missing = json.loads((output / "cases" / "missing-evidence" / "bundle.json").read_text(encoding="utf-8"))
            self.assertEqual(missing["data_categories"]["actual_state_feedback"]["status"], "not_recorded")
            self.assertIn("No media was generated", missing["data_categories"]["video_images"]["missing_reason"])

            first_receipts = {item["bundle_id"] for item in report["synthetic_scenarios"]}
            second_report = run_demo(output)
            second_receipts = {item["bundle_id"] for item in second_report["synthetic_scenarios"]}
            self.assertTrue(first_receipts.isdisjoint(second_receipts))
            self.assertEqual(report["receiver_receipt_count"], 10)
            self.assertEqual(second_report["receiver_receipt_count"], 10)
            self.assertGreater(second_report["receiver_total_receipt_count"],
                               report["receiver_total_receipt_count"])

    def test_example_source_is_present(self):
        self.assertTrue(_CHEMISTRY_PACKAGE.is_file())


if __name__ == "__main__":
    unittest.main()
