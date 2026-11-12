"""Run a synthetic result-bundle demo against the local mock endpoints.

This module intentionally uses only Python's standard library.  It starts a
loopback mock receiver and a mock-mode execution gateway in-process, then sends
one shortened copy of the shipped chemistry package through the gateway's real
HTTP endpoint.  Additional failure and measurement bundles are explicitly
synthetic fixtures; none of this code connects to a robot or claims to exercise
a formal partner 3.3 receiver.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .gateway.server import GatewayServer
from .gateway.service import Gateway
from .gateway.contracts import canonical
from .models import load_scene_config
from .results import ResultsDelivery, deliver_bundle, make_bundle
from .results.common import ENDPOINT_PATH, content_digest
from .results.receiver import ReceiverServer


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHEMISTRY_PACKAGE = _REPO_ROOT / "docs" / "execution-api" / "examples" / "runtime-chemistry-beaker-pour.json"
_SCENE = Path(__file__).with_name("scene_config.json")
_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "interrupted"}
_SOURCE_METADATA = {
    "kind": "synthetic",
    "source": "task_skills.results_demo fixture generator",
    "hardware_connected": False,
}


SCENARIOS: tuple[dict[str, Any], ...] = (
    {"name": "motion-timeout", "terminal_status": "failed", "task_outcome": "failure",
     "failure_code": "MOTION_TIMEOUT", "feedback_status": "timeout"},
    {"name": "controller-alarm", "terminal_status": "failed", "task_outcome": "failure",
     "failure_code": "CONTROLLER_ALARM", "feedback_status": "alarm"},
    {"name": "controller-disconnect", "terminal_status": "interrupted", "task_outcome": "failure",
     "failure_code": "CONTROLLER_DISCONNECTED", "feedback_status": "disconnected"},
    {"name": "gripper-failure", "terminal_status": "failed", "task_outcome": "failure",
     "failure_code": "GRIPPER_NOT_CLOSED", "feedback_status": "gripper_failure"},
    {"name": "insufficient-pour", "terminal_status": "succeeded", "task_outcome": "failure",
     "failure_code": None, "feedback_status": "command_completed",
     "measurement": {"metric": "transferred_volume", "value": 13.2, "target_value": 20.0,
                     "unit": "mL", "judgment": "below_target"}},
    {"name": "excess-pour", "terminal_status": "succeeded", "task_outcome": "failure",
     "failure_code": None, "feedback_status": "command_completed",
     "measurement": {"metric": "transferred_volume", "value": 26.4, "target_value": 20.0,
                     "unit": "mL", "judgment": "above_target"}},
    {"name": "spill", "terminal_status": "succeeded", "task_outcome": "failure",
     "failure_code": None, "feedback_status": "command_completed",
     "measurement": {"metric": "spilled_volume", "value": 1.5, "target_value": 0.0,
                     "unit": "mL", "judgment": "spill_detected"}},
    {"name": "human-cancel", "terminal_status": "cancelled", "task_outcome": "cancelled",
     "failure_code": "EXECUTION_CANCELLED", "feedback_status": "cancelled"},
    {"name": "missing-evidence", "terminal_status": "succeeded", "task_outcome": "indeterminate",
     "failure_code": None, "feedback_status": "not_recorded", "measurement": None},
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _request_json(url: str, *, method: str = "GET", value: Any = None,
                  timeout: float = 5.0) -> tuple[int, Any]:
    data = None if value is None else canonical(value).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw.decode("utf-8")) if raw else None
    except HTTPError as error:
        raw = error.read()
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = None
        return error.code, body


def _synthetic_package(template: dict[str, Any], name: str) -> dict[str, Any]:
    """Copy the chemistry template and assign fixture-only identifiers."""
    package = copy.deepcopy(template)
    package_id = "synthetic-package-" + name
    run_id = "synthetic-run-" + name
    package["package_id"] = package_id
    package["run_id"] = run_id
    package["experiment_id"] = "synthetic-experiment-" + name
    package["catalog_revision"] = "synthetic-catalog-revision"
    package["hardware_connected"] = False
    package["provenance"] = copy.deepcopy(_SOURCE_METADATA)
    package["source_metadata"] = {**_SOURCE_METADATA, "template_package_id": template.get("package_id")}

    node_ids = {}
    for node in package.get("nodes", []):
        old_id = node.get("node_id", "node")
        new_id = "synthetic-stage-" + name + "-" + str(old_id).replace("_", "-")
        node_ids[old_id] = new_id
        node["node_id"] = new_id
        node["hardware_connected"] = False
        node["source_metadata"] = copy.deepcopy(_SOURCE_METADATA)
        invocation = node.get("skill_invocation")
        if isinstance(invocation, dict):
            invocation["idempotency_key"] = "synthetic-key-" + name + "-" + str(old_id).replace("_", "-")
    for edge in package.get("edges", []):
        edge["source"] = node_ids.get(edge.get("source"), edge.get("source"))
        edge["target"] = node_ids.get(edge.get("target"), edge.get("target"))
    package["entry_node_ids"] = [node_ids.get(node_id, node_id) for node_id in package.get("entry_node_ids", [])]
    return package


def _command_node(package: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    actions = [node for node in package.get("nodes", []) if node.get("node_type") == "action"]
    preferred = None
    if scenario["name"] == "gripper-failure":
        preferred = next((node for node in actions
                         if "gripper" in node.get("skill_invocation", {}).get("skill_id", "")), None)
    elif scenario["name"] in {"insufficient-pour", "excess-pour", "spill", "missing-evidence"}:
        preferred = next((node for node in actions
                         if "rotate_about_pivot" in node.get("skill_invocation", {}).get("skill_id", "")), None)
    return preferred or (actions[0] if actions else {})


def _synthetic_case(template: dict[str, Any], scenario: dict[str, Any], run_tag: str) -> dict[str, Any]:
    name = scenario["name"]
    fixture_id = name + "-" + run_tag
    package = _synthetic_package(template, fixture_id)
    node = _command_node(package, scenario)
    invocation = node.get("skill_invocation", {})
    command_id = "synthetic-command-" + fixture_id
    stage_id = "synthetic-stage-" + fixture_id + "-active"
    command = {
        "command_id": command_id,
        "stage_id": stage_id,
        "node_id": node.get("node_id", "synthetic-node-" + name),
        "skill_id": invocation.get("skill_id", "synthetic.unknown"),
        "device_id": invocation.get("device_id", "synthetic-device-unknown"),
        "parameters": copy.deepcopy(invocation.get("parameters", {})),
        "status": "issued",
        "hardware_connected": False,
        "source_metadata": copy.deepcopy(_SOURCE_METADATA),
    }
    failure_code = scenario.get("failure_code")
    failure = None if failure_code is None else {
        "code": failure_code,
        "retryable": False,
        "message": "Synthetic fixture outcome; no controller or robot was contacted.",
    }
    measurement = scenario.get("measurement")
    synthetic_measurement = None if measurement is None else {
        "measurement_id": "synthetic-measurement-" + fixture_id,
        **copy.deepcopy(measurement),
        "provenance": copy.deepcopy(_SOURCE_METADATA),
        "hardware_connected": False,
        "source_metadata": copy.deepcopy(_SOURCE_METADATA),
    }
    stage_status = "failed" if failure_code else "completed"
    if scenario["terminal_status"] == "cancelled":
        stage_status = "cancelled"
    if scenario["name"] == "missing-evidence":
        stage_status = "completed_unverified"
    feedback = {
        "feedback_id": "synthetic-feedback-" + fixture_id,
        "command_id": command_id,
        "status": scenario["feedback_status"],
        "failure": failure,
        "controller_state": "synthetic_unavailable" if scenario["feedback_status"] == "disconnected" else "synthetic_mock",
        "gripper_state": "synthetic_not_confirmed" if scenario["name"] == "gripper-failure" else "synthetic_unmeasured",
        "measurement": synthetic_measurement,
        "hardware_connected": False,
        "provenance": copy.deepcopy(_SOURCE_METADATA),
        "source_metadata": copy.deepcopy(_SOURCE_METADATA),
    }
    stages = [
        {"stage_id": "synthetic-stage-" + fixture_id + "-approach", "name": "approach",
         "status": "completed", "hardware_connected": False,
         "source_metadata": copy.deepcopy(_SOURCE_METADATA)},
        {"stage_id": stage_id, "name": "transfer_or_failure_point", "status": stage_status,
         "command_id": command_id, "hardware_connected": False,
         "source_metadata": copy.deepcopy(_SOURCE_METADATA)},
    ]
    events = [
        {"event_id": "synthetic-event-" + fixture_id + "-001", "sequence": 1,
         "event_type": "execution_started", "status": "running",
         "hardware_connected": False, "provenance": copy.deepcopy(_SOURCE_METADATA),
         "source_metadata": copy.deepcopy(_SOURCE_METADATA)},
        {"event_id": "synthetic-event-" + fixture_id + "-002", "sequence": 2,
         "event_type": "stage_completed", "status": "completed", "stage": stages[0],
         "hardware_connected": False, "provenance": copy.deepcopy(_SOURCE_METADATA),
         "source_metadata": copy.deepcopy(_SOURCE_METADATA)},
        {"event_id": "synthetic-event-" + fixture_id + "-003", "sequence": 3,
         "event_type": "stage_started", "status": "running", "stage": stages[1],
         "hardware_connected": False, "provenance": copy.deepcopy(_SOURCE_METADATA),
         "source_metadata": copy.deepcopy(_SOURCE_METADATA)},
        {"event_id": "synthetic-event-" + fixture_id + "-004", "sequence": 4,
         "event_type": "command_issued", "status": "running", "command": command,
         "hardware_connected": False, "provenance": copy.deepcopy(_SOURCE_METADATA),
         "source_metadata": copy.deepcopy(_SOURCE_METADATA)},
        {"event_id": "synthetic-event-" + fixture_id + "-005", "sequence": 5,
         "event_type": "feedback_received", "status": feedback["status"], "feedback": feedback,
         "hardware_connected": False, "provenance": copy.deepcopy(_SOURCE_METADATA),
         "source_metadata": copy.deepcopy(_SOURCE_METADATA)},
        {"event_id": "synthetic-event-" + fixture_id + "-006", "sequence": 6,
         "event_type": "stage_" + stage_status, "status": stage_status, "stage": stages[1],
         "hardware_connected": False, "provenance": copy.deepcopy(_SOURCE_METADATA),
         "source_metadata": copy.deepcopy(_SOURCE_METADATA)},
        {"event_id": "synthetic-event-" + fixture_id + "-007", "sequence": 7,
         "event_type": "execution_" + scenario["terminal_status"],
         "status": scenario["terminal_status"],
         "hardware_connected": False, "provenance": copy.deepcopy(_SOURCE_METADATA),
         "source_metadata": copy.deepcopy(_SOURCE_METADATA)},
    ]
    task_outcome = scenario["task_outcome"]
    result = {
        "status": scenario["terminal_status"],
        "execution_status": scenario["terminal_status"],
        "error": failure,
        "task_outcome": task_outcome,
        "task_outcome_source": "synthetic_fixture_judgment",
        "judgment": {"outcome": task_outcome, "failure_code": failure_code,
                     "rule": "synthetic demo fixture", "physical_success_verified": False},
        "stages": stages,
        "commands_sent": [command],
        "feedback": [feedback],
        "measurement": synthetic_measurement,
        "evidence": {"kind": "synthetic", "available": scenario["name"] != "missing-evidence",
                     "note": "Fabricated demo data; not a physical observation."},
        "provenance": copy.deepcopy(_SOURCE_METADATA),
        "source_metadata": copy.deepcopy(_SOURCE_METADATA),
        "hardware_connected": False,
    }
    if scenario["name"] == "missing-evidence":
        result["category_statuses"] = {
            "actual_state_feedback": {
                "status": "not_recorded", "data": None,
                "missing_reason": "Synthetic case intentionally has no state measurement.",
            },
            "video_images": {
                "status": "not_recorded", "data": None,
                "missing_reason": "No media was generated or referenced in this demo.",
            },
        }
    return {"package": package, "result": result, "events": events,
            "execution_id": "synthetic-execution-" + fixture_id}


def synthetic_bundles(template: dict[str, Any], output_dir: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Create and persist the nine synthetic outcome bundles."""
    bundles = []
    run_tag = uuid.uuid4().hex[:12]
    for scenario in SCENARIOS:
        case = _synthetic_case(template, scenario, run_tag)
        case_dir = output_dir / "cases" / scenario["name"]
        bundle = make_bundle(
            case["execution_id"], case["package"], case["result"], case["events"],
            None, "synthetic", bundle_id="synthetic-bundle-" + case["execution_id"].removeprefix("synthetic-execution-"),
            source_kind="synthetic",
        )
        _write_json(case_dir / "bundle.json", bundle)
        bundles.append((scenario, bundle))
    return bundles


def _short_timed_package() -> dict[str, Any]:
    package = _read_json(_CHEMISTRY_PACKAGE)
    for node in package.get("nodes", []):
        if node.get("node_type") == "wait":
            node["duration_s"] = min(float(node.get("duration_s", 0)), 0.02)
    package["demo_metadata"] = {
        "kind": "mock",
        "source_example": "docs/execution-api/examples/runtime-chemistry-beaker-pour.json",
        "timing_note": "Only wait durations were shortened for this loopback demo.",
        "hardware_connected": False,
    }
    return package


def _poll_execution(base_url: str, execution_id: str, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, result = _request_json(base_url + "/v1/executions/" + quote(execution_id, safe=""))
        if status != 200:
            raise RuntimeError(f"Gateway status request returned HTTP {status}: {result}")
        if result.get("status") in _TERMINAL_STATUSES:
            return result
        time.sleep(0.05)
    raise TimeoutError(f"Gateway execution {execution_id} did not finish within {timeout:g}s")


def _poll_result_delivery(gateway: Gateway, execution_id: str, timeout: float = 15.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        delivery = gateway.result_delivery_status().get("delivery", {})
        match = next((item for item in delivery.get("items", [])
                      if item.get("execution_id") == execution_id), None)
        if match and match.get("state") == "delivered":
            return match
        time.sleep(0.05)
    raise TimeoutError(f"ResultsDelivery did not deliver gateway execution {execution_id}: "
                       f"{gateway.result_delivery_status()}")


def run_demo(output_dir: str | Path) -> dict[str, Any]:
    """Run the complete loopback flow and write its bundles, manifest and report."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gateway_dir = output_dir / "gateway-data"
    receiver_dir = output_dir / "receiver-data"
    template = _read_json(_CHEMISTRY_PACKAGE)

    receiver = ReceiverServer(receiver_dir, host="127.0.0.1", port=0)
    gateway: Gateway | None = None
    server: GatewayServer | None = None
    server_thread = None
    try:
        receiver.start()
        results_delivery = ResultsDelivery(gateway_dir / "results-outbox", receiver.url, allow_http=True)
        gateway = Gateway(load_scene_config(_SCENE), gateway_dir, "mock", results_delivery=results_delivery)
        server = GatewayServer(("127.0.0.1", 0), gateway)
        import threading

        server_thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                                          name="results-demo-gateway", daemon=True)
        server_thread.start()
        gateway_url = f"http://127.0.0.1:{server.server_port}"

        submitted_package = _short_timed_package()
        _write_json(output_dir / "gateway-package.json", submitted_package)
        status, submission = _request_json(gateway_url + "/v1/execute-graph", method="POST",
                                           value=submitted_package)
        if status != 202:
            raise RuntimeError(f"Mock gateway submission returned HTTP {status}: {submission}")
        execution_id = submission["execution_id"]
        execution = _poll_execution(gateway_url, execution_id)
        if execution.get("status") != "succeeded":
            raise RuntimeError(f"Mock chemistry package ended with status {execution.get('status')}")
        delivery_record = _poll_result_delivery(gateway, execution_id)

        scenario_bundles = synthetic_bundles(template, output_dir)
        endpoint = receiver.url + ENDPOINT_PATH
        scenario_reports = []
        for scenario, bundle in scenario_bundles:
            ack = deliver_bundle(endpoint, bundle, allow_http=True)
            duplicate_ack = deliver_bundle(endpoint, bundle, allow_http=True)
            duplicate_idempotent = duplicate_ack == ack
            if not duplicate_idempotent:
                raise RuntimeError(f"Duplicate POST changed the ACK for {bundle['bundle_id']}")
            receipt_status, receipt = _request_json(
                receiver.url + ENDPOINT_PATH + "/" + quote(bundle["bundle_id"], safe=""))
            if receipt_status != 200 or receipt.get("content_digest") != ack.get("content_digest"):
                raise RuntimeError(f"Could not verify receipt for {bundle['bundle_id']}")
            scenario_reports.append({
                "name": scenario["name"],
                "bundle_id": bundle["bundle_id"],
                "terminal_status": scenario["terminal_status"],
                "task_outcome": scenario["task_outcome"],
                "provenance": bundle["provenance"]["kind"],
                "duplicate_idempotent": duplicate_idempotent,
                "receipt_verified": True,
            })
            _write_json(output_dir / "cases" / scenario["name"] / "receipt.json", receipt)

        # Verify a same-ID payload mutation conflicts without altering its stored receipt.
        conflict_bundle = copy.deepcopy(scenario_bundles[0][1])
        conflict_bundle["demo_conflict_probe"] = "intentional payload change"
        conflict_status, conflict_body = _request_json(endpoint, method="POST",
                                                        value={"bundle": conflict_bundle,
                                                               "content_digest": content_digest(conflict_bundle)})
        if conflict_status != 409:
            raise RuntimeError(f"Mismatched duplicate should conflict; got HTTP {conflict_status}: {conflict_body}")

        # The gateway sender captured the real mock execution. Read and persist its exact receipt.
        gateway_bundle_id = delivery_record.get("bundle_id")
        gateway_receipt = None
        if gateway_bundle_id:
            receipt_status, gateway_receipt = _request_json(
                receiver.url + ENDPOINT_PATH + "/" + quote(gateway_bundle_id, safe=""))
            if receipt_status != 200:
                raise RuntimeError(f"Gateway result bundle receipt returned HTTP {receipt_status}")
        if not gateway_receipt:
            raise RuntimeError("Gateway ResultsDelivery ACK had no bundle ID or receipt")
        _write_json(output_dir / "gateway-receipt.json", gateway_receipt)

        # Ensure all nine fixture receipts and the gateway's mock receipt are visible.
        list_status, receipt_list = _request_json(endpoint)
        if list_status != 200:
            raise RuntimeError(f"Receiver list returned HTTP {list_status}")
        receipt_ids = {item.get("bundle_id") for item in receipt_list.get("items", [])}
        expected_ids = {bundle["bundle_id"] for _, bundle in scenario_bundles}
        if not expected_ids.issubset(receipt_ids) or gateway_bundle_id not in receipt_ids:
            raise RuntimeError("Receiver listing is missing one or more expected receipts")

        manifest = {
            "manifest_version": 1,
            "contract_label": "provisional-3.3-mock/v1",
            "formal_partner_3_3_receiver": False,
            "hardware_connected": False,
            "synthetic_fixture_count": len(scenario_bundles),
            "gateway_execution": {
                "execution_id": execution_id,
                "bundle_id": gateway_bundle_id,
                "status": execution["status"],
                "mode": "mock",
                "source_example": "docs/execution-api/examples/runtime-chemistry-beaker-pour.json",
                "waits_shortened_in_submitted_copy": True,
            },
            "cases": [
                {"name": scenario["name"], "bundle_id": bundle["bundle_id"],
                 "terminal_status": scenario["terminal_status"], "task_outcome": scenario["task_outcome"],
                 "provenance": bundle["provenance"]["kind"], "bundle_path": f"cases/{scenario['name']}/bundle.json"}
                for scenario, bundle in scenario_bundles
            ],
        }
        _write_json(output_dir / "manifest.json", manifest)

        report = {
            "demo_status": "passed",
            "contract_label": "provisional-3.3-mock/v1",
            "formal_partner_3_3_receiver": False,
            "receiver_kind": "temporary loopback mock receiver",
            "hardware_connected": False,
            "gateway": {
                "mode": "mock",
                "execution_id": execution_id,
                "terminal_status": execution["status"],
                "bundle_id": gateway_bundle_id,
                "bundle_delivered": True,
                "bundle_provenance": gateway_receipt["bundle"].get("provenance", {}).get("kind"),
                "package_source": "existing runtime-chemistry-beaker-pour example; wait durations shortened in demo copy",
            },
            "synthetic_scenarios": scenario_reports,
            "protocol_checks": {
                "exact_duplicate_reuses_receipt": all(item["duplicate_idempotent"] for item in scenario_reports),
                "mismatched_duplicate_returns_http_409": True,
                "all_expected_receipts_listed": True,
            },
            "receiver_receipt_count": len(expected_ids) + 1,
            "receiver_total_receipt_count": len(receipt_ids),
            "output_directory": str(output_dir),
        }
        _write_json(output_dir / "demo_report.json", report)
        return report
    finally:
        try:
            try:
                if server is not None:
                    try:
                        server.shutdown()
                    finally:
                        server.server_close()
            finally:
                if server_thread is not None:
                    server_thread.join(timeout=2.0)
        finally:
            try:
                if gateway is not None:
                    gateway.close()
            finally:
                receiver.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the provisional results contract demo using local mock HTTP services.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="directory for generated synthetic bundles, manifest, report and mock receipts")
    args = parser.parse_args(argv)
    try:
        report = run_demo(args.output_dir)
    except (OSError, ValueError, RuntimeError, TimeoutError, URLError) as error:
        parser.exit(1, f"results demo failed: {type(error).__name__}: {error}\n")
    print(json.dumps({"demo_status": report["demo_status"], "output_directory": report["output_directory"],
                      "receiver_receipt_count": report["receiver_receipt_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
