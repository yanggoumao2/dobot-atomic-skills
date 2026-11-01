"""Explicit supported subset of the partner's draft RuntimeExecutionPackage 2.1."""

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from ..atomic_graph import AtomicNode


DEVICES = {"dobot-nova-left": "left", "dobot-nova-right": "right"}
OBJECTS = {"left": "A", "right": "B"}
SKILLS = {"embodied.pick", "embodied.place"}

# The registered chemistry atoms, published under the same RuntimeExecutionPackage
# envelope as embodied.pick/place. The id is the chemistry skill id with a dobot.
# prefix, so the mapping is mechanical and auditable; the parameters, defaults and
# ranges are NOT restated here - they come from task_skills/chemistry/catalog.json
# and are validated by task_skills.chemistry.registry, so there is exactly one
# validator for them.
CHEMISTRY_SKILL_PREFIX = "dobot."
CHEMISTRY_UPSTREAM_SKILLS = {
    "arm.move_joint", "arm.move_linear", "arm.move_to_home",
    "gripper.set_position", "arm.rotate_about_pivot",
}
CHEMISTRY_SKILLS = {CHEMISTRY_SKILL_PREFIX + name: name for name in CHEMISTRY_UPSTREAM_SKILLS}
# Guidance skills name a key experimental operation. They never reach the
# executor: Catalog.validate expands them into dobot.* atoms first.
GUIDANCE_SKILLS = {"embodied.pour"}
EXECUTABLE_SKILLS = SKILLS | set(CHEMISTRY_SKILLS)
FAMILIES = {"embodied", "chemistry"}


class ContractError(Exception):
    def __init__(self, code, message, path="", status=422, details=None):
        super().__init__(message)
        self.status = status
        self.body = {"code": code, "message": message, "path": path,
                     "severity": "blocking", "retryable": False, "details": details or {}}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def identifier(value, path):
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ContractError("INVALID_GRAPH", "Expected a nonempty ID of at most 200 characters", path)
    return value


def number(value, path, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError("INVALID_GRAPH", "Expected a finite number", path)
    if not minimum <= value <= maximum:
        raise ContractError("DEVICE_PARAMETER_OUT_OF_RANGE", "Value outside the supported range", path,
                            details={"minimum": minimum, "maximum": maximum, "actual": value})
    if not math.isfinite(value):
        raise ContractError("INVALID_GRAPH", "Expected a finite number", path)
    return float(value)


@dataclass
class Plan:
    package: Dict[str, Any]
    graph_id: str
    fingerprint: str
    order: List[str]
    dependencies: Dict[str, List[str]]
    arms: Dict[str, str]
    operations: Dict[str, List[AtomicNode]]
    family: str = "embodied"


def _chemistry_contract(profile=None, catalog=None):
    """Publish the chemistry atoms as a second skill group in this catalogue.

    The ids, parameters, defaults, ranges and pose whitelist are read from
    ``task_skills/chemistry/`` instead of being restated here: one catalogue, one
    validator. Imported lazily because the gateway must never pull in a module
    that opens a device on import, and the chemistry package splits its pure
    registry from its hardware driver for exactly that reason.
    """
    from ..chemistry.registry import load_catalog as load_chemistry_catalog
    from ..chemistry.registry import load_profile as load_chemistry_profile

    chemistry_catalog = load_chemistry_catalog() if catalog is None else catalog
    chemistry_profile = load_chemistry_profile() if profile is None else profile
    skills = []
    for entry in chemistry_catalog.get("skills", []):
        name = entry.get("id")
        if name not in CHEMISTRY_UPSTREAM_SKILLS:
            continue
        contract = entry.get("contract", {})
        aliases = entry.get("cn_aliases") or [name]
        skills.append({
            "skill_id": CHEMISTRY_SKILL_PREFIX + name, "version": entry.get("version", "1.0.0"),
            "name_zh": aliases[0], "aliases_zh": list(aliases),
            "lifecycle": "approved", "skill_kind": "atomic_action", "implements": [],
            "method": {"family": "registered_atom", "mode": "calibrated_pose"},
            "required_capabilities": ["arm_motion" if name.startswith("arm.") else "gripper_actuation"],
            "required_sensors": ["controller_state", "tcp_pose", "joint_position", "gripper_position"],
            "parameters": entry.get("parameters", {}),
            "applicable_device_types": ["embodied_executor"],
            "preconditions": list(contract.get("preconditions", [])),
            "completion_conditions": [{"signal": "motion_sequence_completed", "operator": "eq", "value": True}],
            "abort_conditions": [{"signal": "controller_alarm", "operator": "eq", "value": True}],
            "execution_semantics": {"idempotent_start": True, "supports_pause": False,
                                    "supports_cancel": True,
                                    "maximum_retries": int(entry.get("retry_count", 0))},
            "verification": {"level": "command_trace", "physical_success_verified": False},
            "resources": list(entry.get("resources", [])),
            "evidence": list(entry.get("evidence", [])),
            "object_held_fact": bool(contract.get("object_held_fact", False)),
            "warning": contract.get("warning"),
        })
    skills.sort(key=lambda item: item["skill_id"])
    return {"profile": chemistry_profile, "catalog": chemistry_catalog,
            "poses_digest": digest(chemistry_profile.get("poses", {})), "skills": skills}


def _pour_guidance_skill():
    return {
        "skill_id": "embodied.pour", "version": "1.0.0", "name_zh": "倾倒液体",
        "aliases_zh": ["倾倒液体", "把烧杯中的液体倒入离心管"],
        "lifecycle": "approved", "skill_kind": "embodied_action", "implements": [],
        "method": {"family": "guided_choreography", "mode": "bimanual_calibrated_pose"},
        "required_capabilities": ["arm_motion", "gripper_actuation"],
        "required_sensors": ["controller_state", "tcp_pose", "joint_position", "gripper_position"],
        "parameters": {
            "source_object": {"type": "string", "required": True, "enum": ["beaker"]},
            "target_object": {"type": "string", "required": True, "enum": ["tube"]},
            "tilt_deg": {"type": "number", "default": 80, "exclusiveMinimum": 0, "maximum": 90},
            "motion_speed": {"type": "integer", "default": 15, "minimum": 1, "maximum": 35},
        },
        "applicable_device_types": ["embodied_executor"],
        "preconditions": ["both grippers are open: run prepare_open_grippers first"],
        "completion_conditions": [{"signal": "motion_sequence_completed", "operator": "eq", "value": True}],
        "abort_conditions": [{"signal": "controller_alarm", "operator": "eq", "value": True}],
        "execution_semantics": {"idempotent_start": True, "supports_pause": False,
                                "supports_cancel": True, "maximum_retries": 0},
        "verification": {"level": "command_trace", "physical_success_verified": False},
        "warning": "Uses both arms; poses, order and timing are chosen by this gateway.",
    }


class Catalog:
    def __init__(self, scene, mode="mock", chemistry=None):
        self.scene = scene
        self.mode = mode
        self.chemistry = _chemistry_contract() if chemistry is None else chemistry
        self.revision = "dobot-" + digest({"scene": asdict(scene), "profile": "dobot-dag/1",
                                           "mode": mode, "skill_version": "1.0.0",
                                           "cross_arm_motion_concurrency": 2,
                                           "chemistry_catalog": self.chemistry["catalog"],
                                           "chemistry_poses": self.chemistry["poses_digest"],
                                           "guidance_skills": _pour_guidance_skill()})[:20]

    def _chemistry_operation(self, skill, arm, params, node_id, timeout, path):
        """Normalize one dobot.* node through the chemistry registry.

        The chemistry package owns the parameter contract (defaults, ranges and the
        taught pose whitelist), so its ``normalize_parameters`` is the only place the
        rules live. Its errors are mapped onto this envelope's error codes rather
        than passed through raw.
        """
        from ..chemistry.registry import SkillValidationError, normalize_parameters

        if not isinstance(params, dict):
            raise ContractError("INVALID_GRAPH", "Expected a parameters object",
                                path + ".skill_invocation.parameters")
        name = CHEMISTRY_SKILLS[skill]
        try:
            normalized = normalize_parameters(name, arm, params, self.chemistry["profile"])
        except SkillValidationError as error:
            code = "DEVICE_PARAMETER_OUT_OF_RANGE" if " must be in " in str(error) else "INVALID_GRAPH"
            raise ContractError(code, str(error), path + ".skill_invocation.parameters") from error
        return AtomicNode(node_id + ":0", 0, "chemistry_atom", arm,
                          parameters={"skill": name, "parameters": normalized},
                          timeout_seconds=timeout)

    @staticmethod
    def _expand_guidance(package):
        """Replace guidance nodes with atoms; every later check sees only atoms."""
        nodes = package.get("nodes")
        if not isinstance(nodes, list) or not any(
                isinstance(node, dict) and isinstance(node.get("skill_invocation"), dict)
                and node["skill_invocation"].get("skill_id") in GUIDANCE_SKILLS for node in nodes):
            return package
        from ..chemistry.choreography import ChoreographyError, choreograph

        try:
            expanded, lineage = choreograph(package)
        except ChoreographyError as error:
            raise ContractError("INVALID_GRAPH", str(error), "nodes") from error
        expanded["guidance_lineage"] = lineage
        return expanded

    def snapshot(self):
        devices = []
        for device, arm in DEVICES.items():
            devices.append({
                "device_id": device, "name": "Dobot Nova " + arm, "device_type": "embodied_executor",
                "manufacturer": "Dobot", "capability_version": "1.0.0",
                "capabilities": ["pick", "place", "atomic_actions"],
                "supported_skill_ids": sorted(EXECUTABLE_SKILLS | GUIDANCE_SKILLS),
                "parameter_ranges": {}, "precision": {},
                "completion_tolerances": {"tcp_position_mm": 1.0, "joint_angle_deg": 1.0},
                "sensors": ["controller_state", "tcp_pose", "joint_position", "gripper_position"],
                "resource_pool": "dobot-bimanual-rig", "concurrency_limit": 1,
                "status": "idle" if self.mode == "mock" else "offline",
                "control_interface": {"protocol": "vendor_adapter", "supports_pause": False,
                                      "supports_cancel": True},
                "bindings": {"object_ids": [OBJECTS[arm]],
                             "location_ids": ["object:" + OBJECTS[arm], "target:target_" + OBJECTS[arm]]},
            })
        skills = []
        for skill in sorted(SKILLS):
            location = "source_ref" if skill == "embodied.pick" else "destination_ref"
            skills.append({
                "skill_id": skill, "version": "1.0.0", "name_zh": "定点抓取" if skill.endswith("pick") else "定点放置",
                "lifecycle": "approved", "skill_kind": "embodied_action", "implements": [],
                "method": {"family": "calibrated_manipulation", "mode": "fixed_pose"},
                "required_capabilities": [skill.split(".")[1]],
                "required_sensors": ["controller_state", "tcp_pose", "gripper_position"],
                "parameters": {"object_id": {"type": "string", "required": True},
                               location: {"type": "string", "required": True}},
                "applicable_device_types": ["embodied_executor"],
                "preconditions": [],
                "completion_conditions": [{"signal": "motion_sequence_completed", "operator": "eq", "value": True}],
                "abort_conditions": [{"signal": "controller_alarm", "operator": "eq", "value": True}],
                "execution_semantics": {"idempotent_start": True, "supports_pause": False,
                                        "supports_cancel": True, "maximum_retries": 0},
                "verification": {"level": "command_trace", "physical_success_verified": False},
            })
        skills.extend(self.chemistry["skills"])
        skills.append(_pour_guidance_skill())
        return {"schema_version": "1.0", "catalog_revision": self.revision,
                "provider": "dobot-embodied-team", "devices": devices, "atomic_skills": skills,
                "execution_profile": {"name": "dobot-dag/1", "schema_version": "2.1", "mode": self.mode,
                    "node_types": ["action", "wait"], "edge_types": ["success"],
                    "wait_shape": {"duration_s": "number 0..60"},
                    "edge_shape": {"source": "node_id", "target": "node_id", "edge_type": "success"},
                    "supports_resume": False, "maximum_nodes": 100,
                    "cross_arm_motion_concurrency": 2,
                    "requires_initial_home": self.mode == "live",
                    "condition_ids": ["motion_sequence_completed", "wait_elapsed"],
                    "hardware_connected": False}}

    def validate(self, package, initial_state=None, check_state=True):
        if not isinstance(package, dict):
            raise ContractError("INVALID_GRAPH", "graph/package must be an object", "graph")
        try:
            canonical(package)
        except (ValueError, TypeError, RecursionError) as error:
            raise ContractError("INVALID_GRAPH", "Package must contain finite JSON values") from error
        package = copy.deepcopy(package)
        if package.get("schema_version") != "2.1" or package.get("graph_version") != "0.1":
            raise ContractError("UNSUPPORTED_SCHEMA", "Supported schema_version=2.1, graph_version=0.1")
        package = self._expand_guidance(package)
        graph_id = identifier(package.get("package_id"), "package_id")
        identifier(package.get("experiment_id"), "experiment_id")
        if package.get("graph_id", graph_id) != graph_id:
            raise ContractError("INVALID_GRAPH", "This profile uses graph_id=package_id", "graph_id")
        nodes = package.get("nodes")
        edges = package.get("edges")
        if not isinstance(nodes, list) or not 1 <= len(nodes) <= 100 or not isinstance(edges, list) or len(edges) > 1000:
            raise ContractError("INVALID_GRAPH", "Expected 1..100 nodes and at most 1000 edges")
        ids, arms, operations, invocation_keys = {}, {}, {}, set()
        families = set()
        for index, node in enumerate(nodes):
            path = "nodes[%d]" % index
            if not isinstance(node, dict):
                raise ContractError("INVALID_GRAPH", "Node must be an object", path)
            node_id = identifier(node.get("node_id"), path + ".node_id")
            if node_id in ids:
                raise ContractError("INVALID_GRAPH", "Duplicate node_id", path)
            ids[node_id] = node
            if node.get("status") != "pending":
                raise ContractError("UNSUPPORTED_RESUME", "Fresh submissions require pending nodes", path + ".status")
            for field in ("preconditions", "abort_conditions"):
                if node.get(field, []) != []:
                    raise ContractError("UNSUPPORTED_CONDITION", "Extra conditions require a registered evaluator", path + "." + field)
            policy = node.get("on_fail", {})
            if not isinstance(policy, dict):
                raise ContractError("INVALID_GRAPH", "on_fail must be an object", path + ".on_fail")
            if policy.get("strategy", "safe_stop") not in {"safe_stop", "decision_agent"}:
                raise ContractError("UNSUPPORTED_RECOVERY", "Unsupported on_fail strategy", path + ".on_fail")
            if "max_retries" in policy and (type(policy["max_retries"]) is not int or not 0 <= policy["max_retries"] <= 10):
                raise ContractError("INVALID_GRAPH", "max_retries must be an integer 0..10", path + ".on_fail")
            allowed = policy.get("allowed_actions", [])
            if not isinstance(allowed, list) or any(x not in {"retry", "resume_from", "revise_graph", "human_confirmation", "safe_stop"} for x in allowed):
                raise ContractError("UNSUPPORTED_RECOVERY", "Unknown recovery action", path + ".on_fail")
            if node.get("recovery_policy") not in (None, "stop_and_preserve_evidence", "decision_agent"):
                raise ContractError("UNSUPPORTED_RECOVERY", "Only stop or external decision policies are supported", path)
            timeout = number(node.get("timeout_s", 180), path + ".timeout_s", 0.01, 900)
            kind = node.get("node_type")
            if kind == "wait":
                if "skill_invocation" in node:
                    raise ContractError("INVALID_GRAPH", "wait nodes cannot also invoke a skill", path)
                seconds = number(node.get("duration_s"), path + ".duration_s", 0, 60)
                if seconds > timeout:
                    raise ContractError("INVALID_GRAPH", "Wait exceeds node timeout", path)
                arms[node_id] = "system"
                operations[node_id] = [AtomicNode(node_id + ":0", 0, "wait", "system", parameters={"seconds": seconds}, timeout_seconds=timeout)]
                expected_condition = "wait_elapsed"
            elif kind == "action":
                call = node.get("skill_invocation")
                if not isinstance(call, dict):
                    raise ContractError("INVALID_GRAPH", "action requires skill_invocation", path)
                skill = call.get("skill_id")
                if skill not in EXECUTABLE_SKILLS or call.get("skill_version") != "1.0.0":
                    raise ContractError("UNSUPPORTED_SKILL",
                                        "Executable 1.0.0 skills: " + ", ".join(sorted(EXECUTABLE_SKILLS)), path)
                device = call.get("device_id")
                if device not in DEVICES:
                    raise ContractError("DEVICE_INCOMPATIBLE", "Unknown registered device", path)
                arm = DEVICES[device]
                arms[node_id] = arm
                key = identifier(call.get("idempotency_key"), path + ".skill_invocation.idempotency_key")
                if key in invocation_keys:
                    raise ContractError("IDEMPOTENCY_CONFLICT", "Node invocation keys must be unique", path, 409)
                invocation_keys.add(key)
                params = call.get("parameters")
                if skill in CHEMISTRY_SKILLS:
                    families.add("chemistry")
                    operations[node_id] = [self._chemistry_operation(skill, arm, params, node_id, timeout, path)]
                else:
                    families.add("embodied")
                    location_key = "source_ref" if skill.endswith("pick") else "destination_ref"
                    if not isinstance(params, dict) or set(params) != {"object_id", location_key}:
                        raise ContractError("INVALID_GRAPH", "Expected object_id and " + location_key, path + ".skill_invocation.parameters")
                    obj = OBJECTS[arm]
                    if params["object_id"] != obj or params[location_key] not in {"object:" + obj, "target:target_" + obj}:
                        raise ContractError("DEVICE_INCOMPATIBLE", "Object/location is not calibrated for this device", path)
                    ref = params[location_key]
                    if skill.endswith("pick"):
                        specs = [("open_gripper", {}), ("move_pose", {"pose_ref": ref + ":transport", "motion": "linear"}),
                                 ("move_pose", {"pose_ref": ref, "motion": "linear"}), ("close_gripper", {}),
                                 ("move_pose", {"pose_ref": ref + ":transport", "motion": "linear"})]
                    else:
                        specs = [("move_pose", {"pose_ref": ref + ":transport", "motion": "linear"}),
                                 ("move_pose", {"pose_ref": ref, "motion": "linear"}), ("open_gripper", {}),
                                 ("move_pose", {"pose_ref": ref + ":transport", "motion": "linear"})]
                    operations[node_id] = [AtomicNode(node_id + ":" + str(i), i, op, arm, parameters=p, timeout_seconds=timeout)
                                           for i, (op, p) in enumerate(specs)]
                expected_condition = "motion_sequence_completed"
            else:
                raise ContractError("UNSUPPORTED_NODE_TYPE", "Supported nodes are action and wait", path)
            if node.get("completion_condition", expected_condition) != expected_condition:
                raise ContractError("UNSUPPORTED_CONDITION", "Completion condition is not supported", path)
        if len(families) > 1:
            raise ContractError("INVALID_GRAPH",
                                "A package must stay inside one skill family: either embodied.pick/place "
                                "or the dobot.* atomic skills, never both", "nodes")
        family = "chemistry" if families == {"chemistry"} else "embodied"
        dependencies = {node_id: [] for node_id in ids}
        for index, edge in enumerate(edges):
            path = "edges[%d]" % index
            if not isinstance(edge, dict) or set(edge) != {"source", "target", "edge_type"}:
                raise ContractError("INVALID_GRAPH", "Expected source/target/edge_type edge", path)
            if edge["edge_type"] != "success":
                raise ContractError("UNSUPPORTED_EDGE_TYPE", "Only success dependencies are supported", path)
            if edge["source"] not in ids or edge["target"] not in ids or edge["source"] == edge["target"]:
                raise ContractError("INVALID_GRAPH", "Invalid edge endpoint", path)
            if edge["source"] in dependencies[edge["target"]]:
                raise ContractError("INVALID_GRAPH", "Duplicate edge", path)
            dependencies[edge["target"]].append(edge["source"])
        roots = {node_id for node_id, deps in dependencies.items() if not deps}
        entries = package.get("entry_node_ids")
        if entries is None:
            entries = [package.get("entry_node_id")]
        if not isinstance(entries, list) or any(not isinstance(x, str) for x in entries) or len(set(entries)) != len(entries) or set(entries) != roots:
            raise ContractError("INVALID_GRAPH", "Entry nodes must exactly match dependency roots", "entry_node_ids")
        if "entry_node_id" in package and package["entry_node_id"] not in roots:
            raise ContractError("INVALID_GRAPH", "entry_node_id must be a root", "entry_node_id")
        order, ancestors = [], {}
        remaining = set(ids)
        while remaining:
            ready = sorted(x for x in remaining if set(dependencies[x]).issubset(ancestors))
            if not ready:
                raise ContractError("INVALID_GRAPH", "Cycles are not supported by this execution profile", "edges")
            for node_id in ready:
                ancestors[node_id] = set(dependencies[node_id])
                for dep in dependencies[node_id]:
                    ancestors[node_id].update(ancestors[dep])
                order.append(node_id)
                remaining.remove(node_id)
        for index, first in enumerate(order):
            for second in order[index + 1:]:
                if arms[first] != "system" and arms[first] == arms[second] and first not in ancestors[second]:
                    raise ContractError("RESOURCE_CONFLICT", "Same-device actions need explicit dependency order", "edges", 409,
                                        {"nodes": [first, second]})
        # Validate the whole registered object's logical lifecycle, not just parameter
        # shapes. The dobot.* atoms move a tool, not a registered scene object, so the
        # held/location model does not describe them; they are skipped rather than
        # forced into it.
        if family == "embodied":
            state = initial_state or {"left": {"held": False, "location": "object:A"},
                                      "right": {"held": False, "location": "object:B"}}
            held = {arm: item["held"] for arm, item in state.items()}
            locations = {arm: item["location"] for arm, item in state.items()}
            for node_id in order:
                node = ids[node_id]
                if node["node_type"] != "action":
                    continue
                arm, call = arms[node_id], node["skill_invocation"]
                if call["skill_id"].endswith("pick"):
                    if check_state and (held[arm] or call["parameters"]["source_ref"] != locations[arm]):
                        raise ContractError("INVALID_OBJECT_STATE", "Pick requires an unheld object at the registered source", "nodes")
                    held[arm] = True
                else:
                    if check_state and not held[arm]:
                        raise ContractError("INVALID_OBJECT_STATE", "Place requires a preceding successful pick", "nodes")
                    held[arm] = False
                    locations[arm] = call["parameters"]["destination_ref"]
            if check_state and any(held.values()):
                raise ContractError("INVALID_OBJECT_STATE", "This fresh-run profile must end with objects released", "nodes")
        return Plan(package, graph_id, digest(package), order, dependencies, arms, operations, family)
