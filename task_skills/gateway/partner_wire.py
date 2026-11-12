"""Translate outgoing pushes to the partner's strict schema (extra fields forbidden).

Measured on 2026-11-12 against the partner's FastAPI receiver: every model rejects
unknown keys (``extra_forbidden``), ``event_type`` and ``skill_kind`` are closed
enums, and ``preconditions`` must be objects. Local records keep the richer
gateway form; only what leaves the machine is reduced here, at delivery time, so
items already queued in the outbox are fixed too.

Event type mapping (gateway -> partner):
  action_started / action_completed     -> unchanged
  action_failed with MOTION_TIMEOUT      -> timeout
  action_failed / _cancelled / _interrupted, execution_failed / _cancelled /
  _interrupted, arm_finalization_failed / _cancelled -> action_failed
  execution_succeeded                    -> experiment_completed
  wait_completed                         -> condition_satisfied
  everything else (execution_started, wait_started, arm_finalizing, ...) -> telemetry_received
The original type is kept in ``payload.gateway_event_type``.
"""

from __future__ import annotations

import copy

PARTNER_EVENT_TYPES = {
    "action_started", "telemetry_received", "condition_satisfied", "action_completed",
    "action_failed", "timeout", "human_intervention", "experiment_completed", "effect_verified",
}
EXECUTION_EVENT_FIELDS = {
    "event_id", "execution_id", "graph_id", "node_id", "device_id", "skill_id", "skill_version",
    "sequence", "event_type", "error_code", "retryable", "payload", "occurred_at",
}
DEVICE_EVENT_FIELDS = {
    "event_id", "device_id", "capability_version", "sequence", "occurred_at", "state", "health",
    "active_execution_id", "node_id", "alarms",
}
SNAPSHOT_FIELDS = {"schema_version", "catalog_revision", "provider", "generated_at", "devices", "atomic_skills"}
DEVICE_FIELDS = {
    "device_id", "name", "device_type", "manufacturer", "model", "serial_number", "capability_version",
    "capabilities", "supported_skill_ids", "parameter_ranges", "parameter_envelope", "precision",
    "capacity", "sensors", "sensor_details", "resource_pool", "concurrency_limit",
    "supported_container_materials", "interlocks", "safety", "control_interface", "profile", "status",
}
SKILL_FIELDS = {
    "skill_id", "version", "name_zh", "lifecycle", "skill_kind", "implements", "method",
    "required_capabilities", "required_sensors", "optional_sensors", "parameters",
    "applicable_device_types", "supported_container_materials", "material_constraints",
    "preconditions", "completion_conditions", "abort_conditions", "execution_semantics", "failure_codes",
}
PARAMETER_FIELDS = {"type", "unit", "minimum", "maximum", "default", "required", "enum"}
CONDITION_FIELDS = {"signal", "operator", "value", "parameter_ref"}
SKILL_KINDS = {"process", "embodied_action", "device_control"}
FAILURE_TYPES = {
    "action_failed", "action_cancelled", "action_interrupted",
    "execution_failed", "execution_cancelled", "execution_interrupted",
    "arm_finalization_failed", "arm_finalization_cancelled",
}


def _pick(mapping, fields):
    return {key: value for key, value in mapping.items() if key in fields}


def partner_event_type(event_type, error_code=None):
    if event_type in {"action_started", "action_completed"}:
        return event_type
    if event_type == "action_failed" and error_code == "MOTION_TIMEOUT":
        return "timeout"
    if event_type in FAILURE_TYPES:
        return "action_failed"
    if event_type == "execution_succeeded":
        return "experiment_completed"
    if event_type == "wait_completed":
        return "condition_satisfied"
    return "telemetry_received"


NODE_FIELDS = ("node_id", "device_id", "skill_id", "skill_version")


def execution_event(body, demo=False):
    original = body.get("event_type")
    result = _pick(body, EXECUTION_EVENT_FIELDS)
    result["event_type"] = partner_event_type(original, body.get("error_code"))
    payload = dict(body.get("payload") or {}) if isinstance(body.get("payload"), dict) else {}
    payload["gateway_event_type"] = original
    if demo and "node_id" in result:
        # Demo mode runs our pour graph under the partner's package_id; its nodes and
        # devices are not in the partner's graph, so node steps go out as telemetry
        # with the node identity in the payload. Failures stay action_failed.
        payload["demo_node"] = {field: result.pop(field) for field in NODE_FIELDS if field in result}
        if result["event_type"] != "action_failed":
            result["event_type"] = "telemetry_received"
    result["payload"] = payload
    if result["event_type"] == "action_failed" and "error_code" not in result:
        # Execution-level failures carry the code only in their payload.
        error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        result["error_code"] = error.get("code") or original.upper()
        result["retryable"] = bool(error.get("retryable", False))
    return result


def device_event(body):
    return _pick(body, DEVICE_EVENT_FIELDS)


def _parameter(spec):
    if not isinstance(spec, dict):
        return spec
    result = _pick(spec, PARAMETER_FIELDS)
    if "minimum" not in result and "exclusiveMinimum" in spec:
        result["minimum"] = spec["exclusiveMinimum"]
    if "maximum" not in result and "exclusiveMaximum" in spec:
        result["maximum"] = spec["exclusiveMaximum"]
    if "enum" not in result and "const" in spec:
        result["enum"] = [spec["const"]]
    return result


def _conditions(items):
    return [_pick(item, CONDITION_FIELDS) for item in items or [] if isinstance(item, dict)]


def skill(definition):
    result = _pick(definition, SKILL_FIELDS)
    if result.get("skill_kind") not in SKILL_KINDS:
        result["skill_kind"] = "embodied_action"
    if isinstance(result.get("parameters"), dict):
        result["parameters"] = {name: _parameter(spec) for name, spec in result["parameters"].items()}
    for field in ("preconditions", "completion_conditions", "abort_conditions"):
        if field in result:
            # Free-text preconditions have no {signal, operator, value} form; they stay local.
            result[field] = _conditions(result[field])
    return result


def snapshot(body):
    result = _pick(body, SNAPSHOT_FIELDS)
    result["devices"] = [_pick(device, DEVICE_FIELDS) for device in body.get("devices", [])]
    result["atomic_skills"] = [skill(definition) for definition in body.get("atomic_skills", [])]
    return result


def to_partner(path, body, demo=False):
    """Return the body to send for ``path``; unknown paths pass through unchanged."""
    if not isinstance(body, dict):
        return body
    body = copy.deepcopy(body)
    if path.startswith("/v1/executions/") and path.endswith("/events"):
        return execution_event(body, demo)
    if path == "/v1/integrations/device-state-events":
        return device_event(body)
    if path in {"/v1/integrations/catalog/validate", "/v1/integrations/catalog/snapshots"}:
        return snapshot(body)
    return body
