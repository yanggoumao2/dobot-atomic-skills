"""Expand a guidance graph of key experimental operations into dobot.* atoms.

Input and output are both RuntimeExecutionPackage 2.1. A guidance node invoking
``embodied.pour`` becomes a dependency subgraph across both arms; ``wait`` nodes
and existing ``dobot.*`` nodes pass through unchanged. Poses, order and staggers
come from fixed tables here, never from the guidance graph.
"""

from __future__ import annotations

import copy
from typing import Any

POUR_SKILL = "embodied.pour"
DEVICE_BY_ARM = {"left": "dobot-nova-left", "right": "dobot-nova-right"}
# Only these two have taught approach/grasp/work poses, and the pivot atom only
# accepts right.beaker.work, so this is the one pour the rig can perform.
OBJECTS = {"beaker": "right", "tube": "left"}
ACTION_FAILURE_POLICY = {"strategy": "decision_agent", "max_retries": 0,
                         "allowed_actions": ["safe_stop", "human_confirmation"]}
WAIT_FAILURE_POLICY = {"strategy": "safe_stop", "max_retries": 0}
POUR_PARAMETERS = {"source_object", "target_object", "tilt_deg", "motion_speed"}
GRIPPER_SPEED = 10
POUR_SPEED = 15


class ChoreographyError(ValueError):
    """The guidance graph cannot be turned into a safe atomic graph."""


def _ref(arm: str, thing: str, role: str) -> str:
    return f"{arm}.{thing}.{role}"


def _pour_steps(source: str, target: str, tilt: float, speed: int) -> list[dict[str, Any]]:
    pourer, holder = OBJECTS[source], OBJECTS[target]
    s = lambda role: _ref(pourer, source, role)
    t = lambda role: _ref(holder, target, role)

    def joint(name, arm, to, start, grip, needs):
        parameters = {"target_ref": to, "grip_requirement": grip,
                      "speed": speed, "tilt_limit_deg": 1}
        if start:
            parameters["source_ref"] = start
        return _atom(name, arm, "arm.move_joint", parameters, needs)

    def linear(name, arm, start, to, grip, needs):
        return _atom(name, arm, "arm.move_linear",
                     {"source_ref": start, "target_ref": to, "grip_requirement": grip,
                      "speed": speed, "tilt_limit_deg": 1}, needs)

    def grip(name, arm, position, at, release, needs):
        return _atom(name, arm, "gripper.set_position",
                     {"position": position, "speed": GRIPPER_SPEED,
                      "allow_release": release, "at_ref": at}, needs)

    return [
        joint(f"{holder}_{target}_approach", holder, t("approach"), None, "open", []),
        joint(f"{pourer}_{source}_approach", pourer, s("approach"), None, "open", []),
        linear(f"{holder}_{target}_insert", holder, t("approach"), t("grasp"), "open",
               [f"{holder}_{target}_approach"]),
        linear(f"{pourer}_{source}_insert", pourer, s("approach"), s("grasp"), "open",
               [f"{pourer}_{source}_approach"]),
        grip(f"{holder}_{target}_close", holder, 0, t("grasp"), False,
             [f"{holder}_{target}_insert"]),
        grip(f"{pourer}_{source}_close", pourer, 0, s("grasp"), False,
             [f"{pourer}_{source}_insert"]),
        joint(f"{pourer}_{source}_to_work", pourer, s("work"), s("grasp"), "clamped",
              [f"{pourer}_{source}_close"]),
        _wait("work_stagger_8s", 8,
              [f"{holder}_{target}_close", f"{pourer}_{source}_close"]),
        joint(f"{holder}_{target}_to_work", holder, t("work"), t("grasp"), "clamped",
              [f"{holder}_{target}_close", "work_stagger_8s"]),
        _atom("pour_into_tube", pourer, "arm.rotate_about_pivot",
              {"source_ref": s("work"), "tilt_deg": tilt, "segments": 16, "speed": POUR_SPEED,
               "cp": 100, "offset_mm": [0.0, 0.0, 100.0]},
              [f"{holder}_{target}_to_work", f"{pourer}_{source}_to_work"]),
        joint(f"{pourer}_{source}_untip", pourer, s("work"), None, "clamped", ["pour_into_tube"]),
        _wait("return_stagger_5s", 5, [f"{pourer}_{source}_untip"]),
        joint(f"{holder}_{target}_return", holder, t("grasp"), t("work"), "clamped",
              [f"{pourer}_{source}_untip"]),
        grip(f"{holder}_{target}_release", holder, 255, t("grasp"), True,
             [f"{holder}_{target}_return"]),
        linear(f"{holder}_{target}_retreat", holder, t("grasp"), t("approach"), "open",
               [f"{holder}_{target}_release"]),
        joint(f"{pourer}_{source}_return", pourer, s("grasp"), s("work"), "clamped",
              [f"{pourer}_{source}_untip", "return_stagger_5s"]),
        grip(f"{pourer}_{source}_release", pourer, 255, s("grasp"), True,
             [f"{pourer}_{source}_return"]),
        linear(f"{pourer}_{source}_retreat", pourer, s("grasp"), s("approach"), "open",
               [f"{pourer}_{source}_release"]),
        _atom(f"{holder}_arm_home", holder, "arm.move_to_home", {"speed": speed},
              [f"{holder}_{target}_retreat", f"{pourer}_{source}_retreat"]),
        _atom(f"{pourer}_arm_home", pourer, "arm.move_to_home", {"speed": speed},
              [f"{holder}_{target}_retreat", f"{pourer}_{source}_retreat"]),
    ]


def _atom(name, arm, skill, parameters, needs):
    return {"name": name, "arm": arm, "skill": "dobot." + skill,
            "parameters": parameters, "needs": needs}


def _wait(name, seconds, needs):
    return {"name": name, "seconds": float(seconds), "needs": needs}


def _render(step, node_id, run_id):
    if "seconds" in step:
        return {"node_id": node_id, "node_type": "wait", "status": "pending",
                "duration_s": step["seconds"], "timeout_s": 60.0,
                "on_fail": dict(WAIT_FAILURE_POLICY)}
    return {"node_id": node_id, "node_type": "action", "status": "pending", "timeout_s": 180,
            "skill_invocation": {
                "skill_id": step["skill"], "skill_version": "1.0.0",
                "device_id": DEVICE_BY_ARM[step["arm"]],
                "idempotency_key": f"{run_id}:{node_id}",
                "parameters": copy.deepcopy(step["parameters"])},
            "on_fail": copy.deepcopy(ACTION_FAILURE_POLICY)}


def _pour_arguments(node):
    call = node.get("skill_invocation") or {}
    parameters = call.get("parameters")
    if call.get("skill_version") != "1.0.0" or not isinstance(parameters, dict):
        raise ChoreographyError(f"{node['node_id']}: expected {POUR_SKILL}@1.0.0 with parameters")
    unknown = set(parameters) - POUR_PARAMETERS
    if unknown:
        raise ChoreographyError(f"{node['node_id']}: unknown parameter(s) {sorted(unknown)}")
    source, target = parameters.get("source_object"), parameters.get("target_object")
    if (source, target) != ("beaker", "tube"):
        raise ChoreographyError(f"{node['node_id']}: no taught pour from {source!r} into {target!r}")
    tilt = parameters.get("tilt_deg", 80)
    speed = parameters.get("motion_speed", 15)
    if isinstance(tilt, bool) or not isinstance(tilt, (int, float)) or not 0 < tilt <= 90:
        raise ChoreographyError(f"{node['node_id']}: tilt_deg must be in (0, 90]")
    if type(speed) is not int or not 1 <= speed <= 35:
        raise ChoreographyError(f"{node['node_id']}: motion_speed must be an integer in 1..35")
    return source, target, tilt, speed


def _node_ids_ordered(guidance, predecessors):
    pending, ordered = {n["node_id"] for n in guidance["nodes"]}, []
    while pending:
        ready = sorted(n for n in pending if predecessors[n].isdisjoint(pending))
        if not ready:
            raise ChoreographyError("guidance graph has a cycle")
        ordered.extend(ready)
        pending.difference_update(ready)
    return ordered


def _require_serial_pours(pours, predecessors):
    """Every pour uses both arms, so any two pours must be ordered by the graph."""
    def before(first, second, seen=()):
        return first in predecessors[second] or any(
            before(first, middle, seen + (second,))
            for middle in predecessors[second] if middle not in seen)

    for index, first in enumerate(pours):
        for second in pours[index + 1:]:
            if not before(first, second) and not before(second, first):
                raise ChoreographyError(f"{first} and {second} need both arms but are unordered")


def choreograph(guidance: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Return the atomic package and a guidance-node -> atomic-node lineage map."""
    if guidance.get("schema_version") != "2.1" or guidance.get("graph_version") != "0.1":
        raise ChoreographyError("expected schema_version 2.1 and graph_version 0.1")
    run_id = guidance.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ChoreographyError("run_id is required")
    nodes_valid = isinstance(guidance.get("nodes"), list) and all(
        isinstance(node, dict) and isinstance(node.get("node_id"), str) for node in guidance["nodes"])
    edges_valid = isinstance(guidance.get("edges", []), list) and all(
        isinstance(edge, dict) for edge in guidance.get("edges", []))
    if not nodes_valid or not edges_valid:
        raise ChoreographyError("nodes and edges must be arrays of objects with string node ids")
    nodes = {node["node_id"]: node for node in guidance["nodes"]}
    if len(nodes) != len(guidance["nodes"]):
        raise ChoreographyError("duplicate guidance node_id")
    predecessors = {node_id: set() for node_id in nodes}
    for edge in guidance.get("edges", []):
        if edge.get("edge_type") != "success" or edge.get("source") not in nodes \
                or edge.get("target") not in nodes:
            raise ChoreographyError(f"unsupported edge {edge!r}")
        predecessors[edge["target"]].add(edge["source"])

    pours = [n for n, node in nodes.items()
             if (node.get("skill_invocation") or {}).get("skill_id") == POUR_SKILL]
    _require_serial_pours(pours, predecessors)

    atoms: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    lineage: dict[str, list[str]] = {}
    entries: dict[str, list[str]] = {}
    exits: dict[str, list[str]] = {}
    for guidance_id in _node_ids_ordered(guidance, predecessors):
        node = nodes[guidance_id]
        skill = (node.get("skill_invocation") or {}).get("skill_id", "")
        if guidance_id in pours:
            source, target, tilt, speed = _pour_arguments(node)
            steps = _pour_steps(source, target, tilt, speed)
            ids = {step["name"]: f"{guidance_id}.{step['name']}" for step in steps}
            atoms.extend(_render(step, ids[step["name"]], run_id) for step in steps)
            for step in steps:
                edges.extend({"source": ids[need], "target": ids[step["name"]], "edge_type": "success"}
                             for need in step["needs"])
            needed = {need for step in steps for need in step["needs"]}
            entries[guidance_id] = [ids[s["name"]] for s in steps if not s["needs"]]
            exits[guidance_id] = [ids[s["name"]] for s in steps if s["name"] not in needed]
            lineage[guidance_id] = list(ids.values())
        elif node.get("node_type") == "wait" or skill.startswith("dobot."):
            atoms.append(copy.deepcopy(node))
            entries[guidance_id] = exits[guidance_id] = [guidance_id]
            lineage[guidance_id] = [guidance_id]
        else:
            raise ChoreographyError(f"{guidance_id}: no procedure for skill {skill!r}")

    for edge in guidance.get("edges", []):
        edges.extend({"source": tail, "target": head, "edge_type": "success"}
                     for tail in exits[edge["source"]] for head in entries[edge["target"]])
    targets = {edge["target"] for edge in edges}
    package = {key: copy.deepcopy(value) for key, value in guidance.items()
               if key not in {"nodes", "edges", "entry_node_id", "entry_node_ids"}}
    package.update(nodes=atoms, edges=edges,
                   entry_node_ids=sorted(a["node_id"] for a in atoms if a["node_id"] not in targets))
    return package, lineage
