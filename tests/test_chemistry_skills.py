"""Offline contracts for the chemistry atomic-skill library.

Nothing here connects to the rig. The live driver is exercised only for its
import safety, its fail-closed behaviour and its pure helpers; every claim about
real motion still needs an on-site run.
"""

import copy
import importlib
import json
import math
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from task_skills.chemistry import geometry
from task_skills.chemistry.driver import (
    ARRIVAL_TOL_DEG,
    LiveDriver,
    _Feedback,
    angle_delta,
    forward_frame,
    inverse_frame,
    j5_margin,
    joint_frame,
    linear_frame,
    parse_frame,
    parse_single,
)
from task_skills.chemistry.graph import (
    GraphValidationError,
    MockDriver,
    execute_graph,
    unauthorised_gripper_openings,
    validate_graph,
)
from task_skills.chemistry.recipes import compile_recipe, recipe_names
from task_skills.chemistry.registry import (
    SkillValidationError,
    derive_resources,
    invoke,
    load_catalog,
    load_profile,
    normalize_parameters,
)
from task_skills.layout import physical_nova_home_degrees


class CatalogTests(unittest.TestCase):
    def test_catalog_lists_six_minimal_atoms_with_unique_ids(self):
        catalog = load_catalog()
        skills = catalog["skills"]
        self.assertEqual(len(skills), 6)
        ids = [skill["id"] for skill in skills]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sorted(ids), sorted([
            "arm.move_joint", "arm.move_linear", "arm.move_to_home",
            "gripper.set_position", "arm.rotate_about_pivot", "control.hold"]))
        for skill in skills:
            self.assertEqual(skill["kind"], "atom")
            self.assertEqual(skill["version"], "1.0.0")
            self.assertTrue(skill["cn_aliases"])
            self.assertEqual(skill["retry_count"], 0)

    def test_catalog_implementation_targets_exist(self):
        for skill in load_catalog()["skills"]:
            parts = skill["implementation"].split(".")
            module_name, class_name, method = ".".join(parts[:-2]), parts[-2], parts[-1]
            self.assertTrue(module_name and class_name and method, skill["implementation"])
            module = importlib.import_module(module_name)
            target = getattr(module, class_name, None)
            self.assertIsNotNone(target, skill["implementation"])
            self.assertTrue(callable(getattr(target, method, None)), skill["implementation"])

    def test_catalog_never_claims_physical_success(self):
        policy = load_catalog()["execution_policy"]
        self.assertFalse(policy["physical_success_claim"])
        self.assertFalse(policy["parallel_motion_claim"])
        self.assertFalse(policy["gripper_position_proves_object_held"])
        self.assertEqual(policy["retry_count"], 0)


class ProfileTests(unittest.TestCase):
    def test_profile_covers_both_arms_and_six_named_poses(self):
        profile = load_profile()
        self.assertEqual(set(profile["arms"]), {"left", "right"})
        expected = {
            "left.tube.approach", "left.tube.grasp", "left.tube.work",
            "right.beaker.approach", "right.beaker.grasp", "right.beaker.work",
        }
        self.assertEqual(set(profile["poses"]), expected)
        for reference, pose in profile["poses"].items():
            self.assertEqual(pose["arm"], reference.split(".")[0])
            self.assertEqual(len(pose["joints"]), 6)
            self.assertEqual(len(pose["xyz"]), 3)
            self.assertEqual(len(pose["rpy"]), 3)
            for value in (*pose["joints"], *pose["xyz"], *pose["rpy"]):
                self.assertTrue(math.isfinite(value), reference)

    def test_profile_records_the_measured_gripper_convention(self):
        gripper = load_profile()["gripper"]
        self.assertEqual(gripper["open"], 255)
        self.assertEqual(gripper["closed_target"], 0)
        self.assertIn("not object-presence", gripper["evidence"])

    def test_pour_offset_is_declared_an_estimate(self):
        pour = load_profile()["pour"]
        self.assertEqual(pour["source_ref"], "right.beaker.work")
        self.assertEqual(pour["offset_evidence"], "operator_approximation_not_calibrated")
        self.assertFalse(pour["liquid_stream_verified"])
        self.assertFalse(pour["cp_path_accuracy_verified"])


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile()

    def test_defaults_are_applied(self):
        parameters = normalize_parameters("arm.move_joint", "left",
                                          {"target_ref": "left.tube.approach"}, self.profile)
        self.assertEqual(parameters["speed"], 20)
        self.assertEqual(parameters["grip_requirement"], "any")
        self.assertEqual(parameters["tilt_limit_deg"], 1)

    def test_pour_offset_defaults_from_profile(self):
        parameters = normalize_parameters("arm.rotate_about_pivot", "right",
                                          {"source_ref": "right.beaker.work"}, self.profile)
        self.assertEqual(parameters["offset_mm"], self.profile["pour"]["offset_mm"])
        self.assertEqual(parameters["tilt_deg"], 80)
        self.assertEqual(parameters["segments"], 16)

    def test_catalog_declared_defaults_match_what_the_code_applies(self):
        """The catalog publishes the defaults; normalise_parameters hardcodes them.

        Two sources of truth drift silently, so pin them to each other.
        """
        required = {
            "arm.move_joint": ("left", {"target_ref": "left.tube.approach"}),
            "arm.move_linear": ("left", {"source_ref": "left.tube.approach",
                                         "target_ref": "left.tube.grasp"}),
            "arm.move_to_home": ("left", {}),
            "gripper.set_position": ("left", {"position": 255}),
            "arm.rotate_about_pivot": ("right", {"source_ref": "right.beaker.work"}),
            "control.hold": ("left", {"seconds": 5}),
        }
        for skill in load_catalog()["skills"]:
            arm, minimal = required[skill["id"]]
            applied = normalize_parameters(skill["id"], arm, dict(minimal), self.profile)
            for name, spec in skill["parameters"].items():
                if "default" not in spec:
                    continue
                self.assertIn(name, applied, f"{skill['id']}.{name} was not defaulted")
                self.assertEqual(applied[name], spec["default"],
                                 f"{skill['id']}.{name} default drifted from the catalog")

    def test_unknown_and_missing_parameters_are_rejected(self):
        with self.assertRaises(SkillValidationError):
            normalize_parameters("control.hold", "left", {"seconds": 1, "extra": 1}, self.profile)
        with self.assertRaises(SkillValidationError):
            normalize_parameters("control.hold", "left", {}, self.profile)

    def test_non_finite_and_out_of_range_parameters_are_rejected(self):
        for value in (float("nan"), float("inf"), -1, 601):
            with self.assertRaises(SkillValidationError):
                normalize_parameters("control.hold", "left", {"seconds": value}, self.profile)

    def test_pose_reference_must_belong_to_the_requested_arm(self):
        with self.assertRaises(SkillValidationError):
            normalize_parameters("arm.move_joint", "right",
                                 {"target_ref": "left.tube.approach"}, self.profile)

    def test_rotate_is_bound_to_the_right_beaker_work_pose(self):
        with self.assertRaises(SkillValidationError):
            normalize_parameters("arm.rotate_about_pivot", "left",
                                 {"source_ref": "right.beaker.work"}, self.profile)
        with self.assertRaises(SkillValidationError):
            normalize_parameters("arm.rotate_about_pivot", "right",
                                 {"source_ref": "right.beaker.grasp"}, self.profile)

    def test_unregistered_skill_is_rejected(self):
        with self.assertRaises(SkillValidationError):
            normalize_parameters("arm.fly", "left", {}, self.profile)

    def test_resources_include_the_shared_workspace_and_gripper_port(self):
        self.assertIn("workspace:shared",
                      derive_resources("arm.move_joint", "left", self.profile))
        self.assertIn("serial:COM8", derive_resources("gripper.set_position", "left", self.profile))
        self.assertIn("serial:COM7", derive_resources("gripper.set_position", "right", self.profile))


class GraphContractTests(unittest.TestCase):
    def test_every_recipe_validates_and_uses_only_registered_atoms(self):
        profile = load_profile()
        registered = {skill["id"] for skill in load_catalog()["skills"]}
        for name in recipe_names():
            normalized = validate_graph(compile_recipe(name), profile)
            for node in normalized["nodes"]:
                self.assertIn(node["skill"], registered)
                self.assertTrue(node["resources"])

    def test_bimanual_recipe_keeps_both_arms_ordered(self):
        normalized = validate_graph(compile_recipe("dual_prepare"))
        order = normalized["topological_order"]
        arms = [node["arm"] for node in normalized["nodes"] if node["id"] == order[-1]]
        self.assertTrue(order)
        self.assertEqual(len(arms), 1)

    def test_graph_rejects_unknown_fields_and_bad_versions(self):
        graph = compile_recipe("left_tube_to_work")
        graph["extra"] = 1
        with self.assertRaises(GraphValidationError):
            validate_graph(graph)
        graph = compile_recipe("left_tube_to_work")
        graph["schema_version"] = "2.0.0"
        with self.assertRaises(GraphValidationError):
            validate_graph(graph)

    def test_graph_rejects_a_cycle(self):
        graph = compile_recipe("left_tube_to_work")
        graph["nodes"][0]["depends_on"] = [graph["nodes"][-1]["id"]]
        with self.assertRaises(GraphValidationError):
            validate_graph(graph)

    def test_graph_rejects_dangling_dependencies_and_unordered_same_arm_nodes(self):
        graph = compile_recipe("left_tube_to_work")
        graph["nodes"][1]["depends_on"] = ["nowhere"]
        with self.assertRaises(GraphValidationError):
            validate_graph(graph)
        graph = compile_recipe("left_tube_to_work")
        graph["nodes"][-1]["depends_on"] = []
        with self.assertRaises(GraphValidationError):
            validate_graph(graph)

    def test_graph_rejects_duplicate_node_ids(self):
        graph = compile_recipe("left_tube_to_work")
        graph["nodes"][1]["id"] = graph["nodes"][0]["id"]
        with self.assertRaises(GraphValidationError):
            validate_graph(graph)

    def test_graph_accepts_no_python_shell_or_condition_fields(self):
        graph = compile_recipe("left_tube_to_work")
        graph["nodes"][0]["command"] = "print(1)"
        with self.assertRaises(GraphValidationError):
            validate_graph(graph)


class _ClosingMockDriver(MockDriver):
    """MockDriver that records whether graph.py released the session."""

    def __init__(self, preflight_ok=True, confirm_result=True):
        self.closed = 0
        self.preflight_ok = preflight_ok
        self.confirm_result = confirm_result

    def preflight(self, graph, profile):
        if self.preflight_ok:
            return {"simulated": True, "nodes_checked": len(graph["nodes"])}
        return {"success": False, "error": "synthetic preflight failure"}

    def confirm(self):
        return self.confirm_result

    def close(self):
        self.closed += 1
        return {"closed": True}


class ExecutionContractTests(unittest.TestCase):
    def test_mock_run_reports_simulated_and_never_physical_success(self):
        report = execute_graph(compile_recipe("dual_prepare"), MockDriver())
        self.assertEqual(report["status"], "simulated")
        self.assertFalse(report["physical_success_verified"])
        self.assertTrue(all(node["status"] == "completed" for node in report["nodes"]))

    def test_close_hook_runs_on_success_decline_and_preflight_failure(self):
        for driver, expected_status in ((_ClosingMockDriver(), "simulated"),
                                        (_ClosingMockDriver(confirm_result=False), "aborted"),
                                        (_ClosingMockDriver(preflight_ok=False), "failed_preflight")):
            report = execute_graph(compile_recipe("left_tube_to_work"), driver)
            self.assertEqual(report["status"], expected_status)
            self.assertEqual(driver.closed, 1, expected_status)

    def test_failure_stops_the_graph_and_blocks_the_rest(self):
        class _FailingDriver(MockDriver):
            def execute(self, node, profile):
                if node["skill"] == "gripper.set_position":
                    raise RuntimeError("synthetic gripper failure")
                return super().execute(node, profile)

            def stop(self, used_arms):
                return {"simulated": True, "stopped": list(used_arms)}

        report = execute_graph(compile_recipe("left_tube_to_work"), _FailingDriver())
        self.assertEqual(report["status"], "failed")
        failed_at = next(index for index, node in enumerate(report["nodes"])
                         if node["status"] == "failed")
        # The approach is a move, so it completes; the first gripper action is the
        # close, because preparing the jaws is a separate authorised graph now.
        self.assertEqual(report["nodes"][failed_at]["id"], "left_tube_close")
        self.assertTrue(all(node["status"] == "completed"
                            for node in report["nodes"][:failed_at]))
        self.assertTrue(all(node["status"] == "blocked"
                            for node in report["nodes"][failed_at + 1:]))
        self.assertIn("stop", report)

    def test_invoke_dispatches_a_single_atom(self):
        report = invoke("control.hold", "left", {"seconds": 1}, MockDriver(), load_profile())
        self.assertEqual(report["status"], "simulated")
        self.assertEqual(len(report["nodes"]), 1)


class GeometryTests(unittest.TestCase):
    def test_euler_round_trip_including_the_pitch_degeneracy(self):
        for rpy in ((0, 0, 0), (12.5, -33.25, 78.125), (-5.776, -90.0, 0.0),
                    (148.3891, -90.0, 24.8571), (-0.904786, 89.819982, -52.195747)):
            matrix = geometry.euler_zyx_to_matrix(rpy)
            recovered = geometry.matrix_to_euler_zyx(matrix)
            again = geometry.euler_zyx_to_matrix(recovered)
            for row in range(3):
                for column in range(3):
                    self.assertAlmostEqual(again[row][column], matrix[row][column], places=12,
                                           msg=str(rpy))

    def test_tool_axes_are_unit_columns(self):
        rpy = (-0.904786, 89.819982, -52.195747)
        matrix = geometry.euler_zyx_to_matrix(rpy)
        for index, axis in enumerate("XYZ"):
            vector = geometry.tool_axis(rpy, axis)
            self.assertAlmostEqual(math.dist(vector, (0, 0, 0)), 1.0, places=12)
            for row in range(3):
                self.assertAlmostEqual(vector[row], matrix[row][index], places=12)

    def test_tilt_and_horizontal_measures(self):
        self.assertAlmostEqual(geometry.tilt_from_vertical_deg((0, 0, 0), "X"), 90.0, places=9)
        self.assertAlmostEqual(geometry.tilt_from_vertical_deg((0, -90, 0), "X"), 0.0, places=9)
        self.assertAlmostEqual(geometry.tilt_from_vertical_deg((0, 90, 0), "X"), 0.0, places=9)
        self.assertAlmostEqual(geometry.tilt_from_vertical_deg((-175.763, 89.5, 170.29), "X"),
                               0.5, places=6)
        leveled = (12.0, -90.0, 7.0)
        self.assertAlmostEqual(geometry.angle_from_horizontal_deg(geometry.tool_axis(leveled, "Y")),
                               0.0, places=9)

    def test_rodrigues_rotation_is_about_the_requested_axis(self):
        matrix = geometry.rotation_about_axis_deg((0, 0, 1), 90.0)
        rotated = geometry.matrix_vector(matrix, (1, 0, 0))
        self.assertAlmostEqual(rotated[0], 0.0, places=12)
        self.assertAlmostEqual(rotated[1], 1.0, places=12)
        self.assertAlmostEqual(rotated[2], 0.0, places=12)

    def test_pivot_offset_uses_the_start_orientation(self):
        profile = load_profile()
        start = profile["poses"]["right.beaker.work"]
        centre = geometry.pivot_from_tool_offset(start["xyz"], start["rpy"],
                                                 profile["pour"]["offset_mm"])
        offset_vector = [centre[axis] - start["xyz"][axis] for axis in range(3)]
        self.assertAlmostEqual(math.dist(offset_vector, (0, 0, 0)), 100.0, places=6)
        # Tool X is vertical in the upright carry pose, so Tool Z - the axis the
        # profile offsets along - lies in the horizontal plane.
        self.assertAlmostEqual(offset_vector[2], 100.0 * geometry.tool_axis(start["rpy"], "Z")[2],
                               places=6)
        # 0.18 deg of start tilt, not exactly zero: the taught pose is upright to
        # within the field scripts' 1 deg start tolerance.
        self.assertLess(abs(geometry.tool_axis(start["rpy"], "Z")[2]), 5e-3)

    def test_pour_waypoints_hold_the_estimated_centre_and_reach_the_tilt(self):
        profile = load_profile()
        start = profile["poses"]["right.beaker.work"]
        offset = profile["pour"]["offset_mm"]
        waypoints = geometry.rotate_about_pivot(start["xyz"], start["rpy"], offset,
                                                profile["pour"]["tilt_deg"],
                                                profile["pour"]["segments"], "Y")
        self.assertEqual(len(waypoints), profile["pour"]["segments"] + 1)
        for axis in range(3):
            self.assertAlmostEqual(waypoints[0]["xyz"][axis], start["xyz"][axis], places=9)
            self.assertAlmostEqual(waypoints[0]["reference_rpy"][axis], start["rpy"][axis], places=9)
        self.assertAlmostEqual(waypoints[-1]["reference_tool_x_tilt_deg"], 80.0, places=6)
        centre = geometry.pivot_from_tool_offset(start["xyz"], start["rpy"], offset)
        for waypoint in waypoints:
            revived = geometry.pivot_from_tool_offset(waypoint["xyz"], waypoint["reference_rpy"], offset)
            self.assertAlmostEqual(math.dist(revived, centre), 0.0, places=9)
            self.assertLess(geometry.angle_from_horizontal_deg(
                geometry.tool_axis(waypoint["reference_rpy"], "Y")), 0.1)

    def test_rotation_is_only_about_the_requested_world_axis(self):
        profile = load_profile()
        start = profile["poses"]["right.beaker.work"]
        waypoints = geometry.rotate_about_pivot(start["xyz"], start["rpy"],
                                                profile["pour"]["offset_mm"], 80.0, 16, "Y")
        normal = waypoints[0]["jaw_normal_world"]
        for waypoint in waypoints:
            self.assertAlmostEqual(math.dist(waypoint["jaw_normal_world"], normal), 0.0, places=12)

    def test_geometry_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            geometry.rotation_about_axis_deg((0, 0, 0), 10.0)
        with self.assertRaises(ValueError):
            geometry.tool_axis((0, 0, 0), "W")
        with self.assertRaises(ValueError):
            geometry.rotate_about_pivot((0, 0, 0), (0, 0, 0), (0, 0), 80.0)
        with self.assertRaises(ValueError):
            geometry.rotate_about_pivot((0, 0, 0), (0, 0, 0), (0, 0, 100), 0.0)
        with self.assertRaises(ValueError):
            geometry.rotate_about_pivot((0, 0, 0), (0, 0, 0), (0, 0, 100), 95.0)
        with self.assertRaises(ValueError):
            geometry.rotate_about_pivot((0, 0, 0), (0, 0, 0), (0, 0, float("nan")), 80.0)
        with self.assertRaises(ValueError):
            geometry.angle_from_horizontal_deg((0, 0, 0))


class DriverSafetyTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile()
        self.loopback = copy.deepcopy(self.profile)
        for arm in self.loopback["arms"].values():
            arm["ip"] = "127.0.0.1"      # never the real rig

    def test_construction_is_inert(self):
        driver = LiveDriver()
        self.assertEqual(driver._rigs, {})
        self.assertIsNone(driver._lease)
        self.assertFalse(Path(driver._lock_path).exists())
        self.assertEqual(driver.close(), {"lease_released": False, "closed": []})

    def test_confirm_fails_closed_without_a_callable(self):
        self.assertFalse(LiveDriver(allow_live=True).confirm())
        self.assertTrue(LiveDriver(allow_live=True, confirmation=lambda: True).confirm())
        self.assertFalse(LiveDriver(allow_live=True, confirmation=lambda: False).confirm())

    def test_hardware_actions_are_refused_without_allow_live(self):
        profile = self.profile
        driver = LiveDriver(profile=profile)
        node = {"id": "hold", "skill": "control.hold", "arm": "left", "parameters": {"seconds": 1}}
        with self.assertRaises(RuntimeError):
            driver.execute(node, profile)
        report = driver.preflight(compile_recipe("left_tube_to_work"), profile)
        self.assertFalse(report["success"])
        self.assertFalse(report["hardware_commands_sent"])
        self.assertIn("allow_live", report["note"])

    def test_preflight_fails_closed_and_commands_nothing_when_unreachable(self):
        driver = LiveDriver(profile=self.loopback, allow_live=True)
        report = driver.preflight(compile_recipe("left_tube_to_work"), self.loopback)
        self.assertFalse(report["success"])
        self.assertFalse(report["hardware_commands_sent"])
        self.assertTrue(report["error"])
        driver.close()

    def test_stop_without_any_open_session_is_honest(self):
        driver = LiveDriver(profile=self.loopback, allow_live=True)
        result = driver.stop(["left", "right"])
        self.assertEqual(result["stopped"], {})
        self.assertFalse(result["gripper_released"])
        self.assertFalse(result["disable_robot_called"])

    def test_frame_builders_match_the_controllers_echo_format(self):
        self.assertEqual(joint_frame((1.0, -2.5, 3.0, 4.0, 5.0, 6.0)),
                         "JointMovJ(1.000000,-2.500000,3.000000,4.000000,5.000000,6.000000)")
        self.assertEqual(linear_frame((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
                         "MovL(1.000000,2.000000,3.000000,4.000000,5.000000,6.000000)")
        self.assertTrue(inverse_frame((1, 2, 3), (4, 5, 6), (7, 8, 9, 10, 11, 12), 0, 0)
                        .startswith("InverseSolution(1.000000000,2.000000000,3.000000000,"))
        self.assertTrue(forward_frame((1, 2, 3, 4, 5, 6), 0, 0)
                        .startswith("PositiveSolution(1.000000000,"))

    def test_dashboard_frame_parsing(self):
        status, values, message = parse_frame("0,{1.5,-2.5,3.5,4.5,5.5,6.5},GetAngle();")
        self.assertEqual(status, 0)
        self.assertEqual(values, (1.5, -2.5, 3.5, 4.5, 5.5, 6.5))
        self.assertTrue(message)
        status, values, _ = parse_frame("-2,{not tcp mode}")
        self.assertEqual(status, -2)
        self.assertIsNone(values)
        status, values, _ = parse_frame("garbage")
        self.assertIsNone(status)
        self.assertIsNone(values)
        status, values, _ = parse_frame("0,{1,2,3,4,5},GetAngle();")
        self.assertIsNone(values)
        self.assertEqual(parse_single("0,{5},RobotMode();"), 5)
        self.assertIsNone(parse_single("0,{},RobotMode();"))

    def test_angle_and_singularity_helpers(self):
        self.assertAlmostEqual(angle_delta(359.0, 1.0), -2.0, places=9)
        self.assertAlmostEqual(angle_delta(-179.0, 179.0), 2.0, places=9)
        self.assertAlmostEqual(j5_margin(-176.2025), 3.7975, places=6)
        self.assertAlmostEqual(j5_margin(180.0), 0.0, places=9)

    def test_feedback_decoder_rejects_a_zeroed_packet(self):
        import struct as struct_module
        from task_skills.chemistry.driver import FEEDBACK_MAGIC, FEEDBACK_PACKET

        record = bytearray(FEEDBACK_PACKET)
        struct_module.pack_into("<Q", record, 0, FEEDBACK_PACKET)
        record[48:56] = FEEDBACK_MAGIC
        self.assertIsNone(_Feedback._decode(bytes(record)))

    def test_feedback_decoder_accepts_a_well_formed_packet(self):
        import struct as struct_module
        from task_skills.chemistry.driver import FEEDBACK_MAGIC, FEEDBACK_PACKET

        record = bytearray(FEEDBACK_PACKET)
        struct_module.pack_into("<Q", record, 0, FEEDBACK_PACKET)
        record[48:56] = FEEDBACK_MAGIC
        struct_module.pack_into("<Q", record, 24, 5)
        struct_module.pack_into("<6d", record, 432, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        struct_module.pack_into("<6d", record, 624, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
        struct_module.pack_into("<B", record, 1026, 1)
        feedback = object.__new__(_Feedback)
        feedback.sock = mock.Mock()
        feedback.sock.recv.side_effect = [b"partial" + record[:500], bytes(record[500:])]
        decoded = feedback._read_packet()
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded["robot_mode"][0], 5)
        self.assertEqual(decoded["q_actual"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(decoded["pose_actual"], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    def test_rotate_atom_is_bound_to_the_right_arm_at_runtime(self):
        driver = LiveDriver(profile=self.loopback, allow_live=True)
        node = {"id": "pour", "skill": "arm.rotate_about_pivot", "arm": "left",
                "parameters": {"source_ref": "right.beaker.work"}}
        with self.assertRaises(RuntimeError):
            driver.execute(node, self.loopback)


class _FakeGripper:
    def __init__(self, start):
        self.position = start
        self.moves = []

    def get_current_position(self):
        return self.position

    def move(self, position, speed, force):
        self.moves.append((position, speed, force))
        self.position = position
        return (0, 0)


class GripperAtomTests(unittest.TestCase):
    def _driver(self, start, parameters):
        profile = copy.deepcopy(load_profile())
        driver = LiveDriver(profile=profile, allow_live=True)
        fake = _FakeGripper(start)
        driver._ensure_session = lambda *args, **kwargs: None      # type: ignore[assignment]
        driver._gripper = lambda rig, prof: fake                   # type: ignore[assignment]
        node = {"id": "g", "skill": "gripper.set_position", "arm": "left", "parameters": parameters}
        return driver, fake, node, profile

    def test_closing_an_open_gripper_is_allowed_and_reports_no_note(self):
        parameters = {"position": 0, "speed": 10, "allow_release": False}
        driver, fake, node, profile = self._driver(255, parameters)
        result = driver.execute(node, profile)
        self.assertTrue(result["success"])
        self.assertFalse(result["opened"])
        self.assertEqual(result["notes"], [])
        self.assertEqual(fake.moves, [(0, 10, 0)])
        self.assertFalse(result["physical_success_claimed"])

    def test_closing_a_not_open_gripper_is_reported_not_refused(self):
        parameters = {"position": 0, "speed": 10, "allow_release": False}
        driver, fake, node, profile = self._driver(88, parameters)
        result = driver.execute(node, profile)
        self.assertTrue(result["success"])
        self.assertTrue(result["notes"])
        self.assertIn("not a fresh grasp", result["notes"][0])

    def test_release_needs_allow_release_and_then_moves(self):
        parameters = {"position": 255, "speed": 30, "allow_release": False}
        driver, fake, node, profile = self._driver(0, parameters)
        with self.assertRaises(RuntimeError):
            driver.execute(node, profile)
        self.assertEqual(fake.moves, [])
        parameters = {"position": 255, "speed": 30, "allow_release": True}
        driver, fake, node, profile = self._driver(0, parameters)
        result = driver.execute(node, profile)
        self.assertTrue(result["opened"])
        self.assertEqual(fake.moves, [(255, 30, 0)])

    def test_hold_does_not_power_up_the_arm(self):
        driver = LiveDriver(profile=load_profile(), allow_live=True)
        recorded = {}

        def fake_session(profile, arm, speed, **kwargs):
            recorded.update(kwargs)
            return None

        driver._ensure_session = fake_session                       # type: ignore[assignment]
        node = {"id": "h", "skill": "control.hold", "arm": "left", "parameters": {"seconds": 0.3}}
        result = driver.execute(node, load_profile())
        self.assertTrue(result["success"])
        self.assertEqual(recorded, {"configure": False})


EXAMPLE_GRAPH = (Path(__file__).resolve().parent.parent
                 / "examples" / "chemistry" / "right_beaker_pour_into_left_tube.json")

# Both arms read exactly this way in the live preflight on 2026-10-15: the
# controller's parked joint pose, which is not any taught reference. The pose
# triples are stand-ins, because only the controller knows where those joints
# put the tool; the joints alone already miss every taught reference by ~180 deg
# on J6, which is what the checks below depend on.
HOME_STATE = {
    "left": {"joints": [90.0, 0.0, 90.0, 0.0, -90.0, -90.0], "mode": 5, "source": "dashboard",
             "pose": [0.0, 0.0, 350.0, 0.0, 90.0, 0.0]},
    "right": {"joints": [-90.0, 0.0, -90.0, 0.0, 90.0, 90.0], "mode": 5, "source": "dashboard",
              "pose": [0.0, 0.0, 350.0, 0.0, 90.0, 0.0]},
}


class _OfflineArm:
    def __init__(self, arm):
        self.arm = arm


class _OfflinePreflightDriver(LiveDriver):
    """A LiveDriver whose controller calls are answered locally.

    This exists only to drive the preflight state machine offline. The FK and IK
    answers are scaffold: FK echoes whatever IK was just asked for, so a passing
    preflight here says nothing about real kinematics. What it does exercise is
    which state each node is checked against.
    """

    def __init__(self, profile, home):
        super().__init__(profile=profile, allow_live=True)
        self._home = home
        self._pending = None

    def _rig(self, arm, profile, *, with_motion=True):
        return self._rigs.setdefault(arm, _OfflineArm(arm))

    def _read_state(self, rig):
        return copy.deepcopy(self._home[rig.arm])

    def _taught_pose(self, profile, joints):
        wanted = [round(value, 6) for value in joints]
        for pose in profile["poses"].values():
            if [round(value, 6) for value in pose["joints"]] == wanted:
                return list(pose["xyz"]) + list(pose["rpy"])
        return None

    def _forward_solution(self, rig, joints, profile):
        if self._pending is not None:
            xyz, rpy = self._pending
            self._pending = None
            return {"xyz": list(xyz), "rpy": list(rpy)}
        if ([round(value, 6) for value in joints]
                == [round(value, 6) for value in physical_nova_home_degrees(rig.arm)]):
            # Home is a library constant, not a taught pose, so the scaffold has to
            # answer for it separately. The pose below is a stand-in.
            return {"xyz": list(self._home[rig.arm]["pose"][:3]),
                    "rpy": list(self._home[rig.arm]["pose"][3:])}
        taught = self._taught_pose(profile, joints)
        if taught is None:
            raise RuntimeError("offline preflight has no forward solution for these joints")
        return {"xyz": taught[:3], "rpy": taught[3:]}

    def _inverse_solution(self, rig, xyz, rpy, seed, profile):
        self._pending = (list(xyz), list(rpy))
        return list(seed)


class PreflightStateThreadingTests(unittest.TestCase):
    """A whole-graph preflight must judge each node where it will actually run.

    Checking every node against one frozen pre-run snapshot demands that an arm
    sit at several taught poses at once, so no reachable start configuration can
    pass. The declared start references must still be enforced, which is why a
    linear move with no earlier node reaching its source is refused.
    """

    def setUp(self):
        self.profile = load_profile()
        self.document = json.loads(EXAMPLE_GRAPH.read_text(encoding="utf-8"))
        self.graph = validate_graph(self.document, self.profile)
        self.driver = _OfflinePreflightDriver(self.profile, HOME_STATE)

    def tearDown(self):
        self.driver.close()

    def _state_at(self, arm, reference):
        """The threaded state an arm is in once the graph has reached ``reference``."""
        pose = self.profile["poses"][reference]
        return {"joints": list(pose["joints"]), "mode": 5, "source": "dashboard",
                "pose": list(pose["xyz"]) + list(pose["rpy"])}

    def test_preflight_threads_the_simulated_state_through_the_graph(self):
        report = self.driver.preflight(self.graph, self.profile)
        self.assertTrue(report["success"], report.get("error"))
        self.assertFalse(report["hardware_commands_sent"])
        node_checks = [entry for entry in report["checks"] if entry.get("node") not in
                       ("<left state>", "<right state>")]
        self.assertEqual(len(node_checks), len(self.graph["nodes"]))
        self.assertEqual([entry["node"] for entry in node_checks],
                         self.graph["topological_order"])

    def test_every_declared_start_reference_is_checked_against_the_threaded_state(self):
        report = self.driver.preflight(self.graph, self.profile)
        observed = {entry["node"]: entry["checks"]["source"]
                    for entry in report["checks"] if entry.get("checks", {}).get("source")}
        self.assertEqual(observed["left_tube_insert"]["source_ref"], "left.tube.approach")
        self.assertTrue(observed["left_tube_insert"]["observed"])
        self.assertEqual(observed["left_tube_insert"]["max_joint_error_deg"], 0.0)
        self.assertEqual(observed["left_tube_insert"]["position_error_mm"], 0.0)

    def test_a_linear_move_without_its_predecessor_is_still_refused(self):
        document = {"schema_version": "1.0.0", "name": "orphan_linear",
                    "description": "a straight move whose start nothing reaches",
                    "nodes": [{"id": "left_tube_insert", "skill": "arm.move_linear",
                               "arm": "left", "depends_on": [],
                               "parameters": {"source_ref": "left.tube.approach",
                                              "target_ref": "left.tube.grasp"}}]}
        report = self.driver.preflight(validate_graph(document, self.profile), self.profile)
        self.assertFalse(report["success"])
        self.assertIn("does not match left.tube.approach", report["error"])
        self.assertFalse(report["hardware_commands_sent"])

    def test_the_pour_leaves_the_arm_on_the_tipped_waypoint(self):
        node = next(entry for entry in self.graph["nodes"] if entry["id"] == "pour_into_tube")
        state = self._state_at("right", "right.beaker.work")
        entry = self.driver._preflight_node(node, self.profile, copy.deepcopy(state))
        final = entry["checks"]["final_waypoint"]
        self.driver._advance_state(node, self.profile, state, entry)
        self.assertEqual(state["joints"], [round(value, 6) for value in final["joints"]])
        self.assertEqual(state["pose"], [round(value, 6) for value in final["pose"]])
        upright = self.profile["poses"]["right.beaker.work"]
        self.assertNotEqual(state["pose"][3:], [round(value, 6) for value in upright["rpy"]])

    def test_a_dwell_and_a_gripper_action_leave_the_simulated_state_alone(self):
        state = copy.deepcopy(HOME_STATE["left"])
        before = copy.deepcopy(state)
        for node in ({"skill": "control.hold", "arm": "left", "parameters": {"seconds": 5}},
                     {"skill": "gripper.set_position", "arm": "left",
                      "parameters": {"position": 255}}):
            self.driver._advance_state(node, self.profile, state, {"checks": {}})
        self.assertEqual(state, before)

    def test_an_unregistered_skill_cannot_be_smuggled_past_the_advance(self):
        state = copy.deepcopy(HOME_STATE["left"])
        with self.assertRaises(RuntimeError):
            self.driver._advance_state({"skill": "arm.fly", "arm": "left", "parameters": {}},
                                       self.profile, state, {"checks": {}})


class GripperAuthorisationTests(unittest.TestCase):
    """Opening the jaws needs explicit authorisation, so it gets its own graph.

    The driver refuses any command that widens the jaws unless the node sets
    allow_release=True, because position feedback cannot prove the jaws are empty.
    Such a node is therefore unsatisfiable, not dangerous - but it fails on
    hardware, mid-graph, so validate_graph rejects it up front.
    """

    def setUp(self):
        self.profile = load_profile()

    def _document(self, nodes):
        return {"schema_version": "1.0.0", "name": "auth_probe", "nodes": nodes}

    def test_a_first_wide_open_without_authorisation_is_rejected(self):
        document = self._document([{"id": "left_tube_open", "skill": "gripper.set_position",
                                    "arm": "left", "depends_on": [],
                                    "parameters": {"position": 255, "allow_release": False}}])
        with self.assertRaises(GraphValidationError) as caught:
            validate_graph(document, self.profile)
        self.assertIn("allow_release=True", str(caught.exception))
        self.assertIn("left_tube_open", str(caught.exception))

    def test_reopening_after_a_close_without_authorisation_is_rejected(self):
        document = self._document([
            {"id": "close", "skill": "gripper.set_position", "arm": "left", "depends_on": [],
             "parameters": {"position": 0, "allow_release": False}},
            {"id": "reopen", "skill": "gripper.set_position", "arm": "left",
             "depends_on": ["close"],
             "parameters": {"position": 200, "allow_release": False}},
        ])
        with self.assertRaises(GraphValidationError) as caught:
            validate_graph(document, self.profile)
        self.assertIn("reopen", str(caught.exception))

    def test_the_same_open_is_accepted_when_it_is_authorised(self):
        document = self._document([{"id": "left_tube_open", "skill": "gripper.set_position",
                                    "arm": "left", "depends_on": [],
                                    "parameters": {"position": 255, "allow_release": True}}])
        normalized = validate_graph(document, self.profile)
        self.assertEqual(unauthorised_gripper_openings(normalized, self.profile), [])

    def test_the_preparation_recipe_opens_both_jaws_with_authorisation(self):
        normalized = validate_graph(compile_recipe("prepare_open_grippers"), self.profile)
        nodes = normalized["nodes"]
        self.assertEqual(len(nodes), 2)
        self.assertEqual({node["arm"] for node in nodes}, {"left", "right"})
        for node in nodes:
            self.assertEqual(node["skill"], "gripper.set_position")
            self.assertEqual(node["parameters"]["position"], 255)
            self.assertTrue(node["parameters"]["allow_release"])

    def test_no_shipped_graph_commands_an_unauthorised_opening(self):
        paths = sorted(EXAMPLE_GRAPH.parent.glob("*.json"))
        self.assertTrue(paths)
        for path in paths:
            document = json.loads(path.read_text(encoding="utf-8"))
            normalized = validate_graph(document, self.profile)
            self.assertEqual(unauthorised_gripper_openings(normalized, self.profile),
                             [], f"{path.name} has an unsatisfiable gripper node")
        for name in recipe_names():
            normalized = validate_graph(compile_recipe(name), self.profile)
            self.assertEqual(unauthorised_gripper_openings(normalized, self.profile), [], name)

    def test_task_graphs_close_and_release_but_never_prepare_the_jaws(self):
        """Preparation is a separate authorised step; task graphs may only close or release."""
        for name in ("left_tube_to_work", "dual_prepare", "right_beaker_to_work"):
            normalized = validate_graph(compile_recipe(name), self.profile)
            gripper_nodes = [node for node in normalized["nodes"]
                             if node["skill"] == "gripper.set_position"]
            self.assertTrue(gripper_nodes, name)
            for node in gripper_nodes:
                self.assertFalse(node["parameters"]["allow_release"],
                                 f"{name}:{node['id']} releases inside a task graph")
                self.assertLess(node["parameters"]["position"], 250,
                                f"{name}:{node['id']} opens the jaws inside a task graph")

    def test_the_task_graph_that_was_run_live_now_has_no_opening_node(self):
        normalized = validate_graph(json.loads(EXAMPLE_GRAPH.read_text(encoding="utf-8")),
                                    self.profile)
        gripper_nodes = [node for node in normalized["nodes"]
                         if node["skill"] == "gripper.set_position"]
        self.assertEqual(sorted(node["id"] for node in gripper_nodes),
                         ["left_tube_close", "left_tube_release",
                          "right_beaker_close", "right_beaker_release"])


class _RecordingHomeDriver(LiveDriver):
    """A LiveDriver whose session, frames, speeds and arrivals stay local.

    Only the hardware boundary is replaced. The sequence under test - session,
    speed configuration, JointMovJ frame, arrival wait - is the real code path.
    """

    #: Where the read-only probe says each arm is, as an offset from the
    #: calibrated home joints. Away from home by default, so a test that does not
    #: care about the probe still exercises the real move path.
    probe_offset_deg = 30.0
    probe_mode = 5

    def __init__(self, profile, failing=()):
        super().__init__(profile=profile, allow_live=True, confirmation=lambda: True)
        self.sent = []
        self.session_speeds = []
        self.configured_speeds = []
        self.arrivals = []
        self.failing = set(failing)

    def _rig(self, arm, profile, *, with_motion=True):
        return self._rigs.setdefault(arm, _OfflineArm(arm))

    def _read_state(self, rig):
        home = physical_nova_home_degrees(rig.arm)
        return {"source": "offline", "mode": self.probe_mode,
                "joints": [value + self.probe_offset_deg for value in home],
                "pose": [0.0] * 6}

    def _ensure_session(self, profile, arm, speed, *, configure=True):
        self.session_speeds.append((arm, speed))
        return _OfflineArm(arm)

    def _configure_speed(self, rig, speed, *, linear):
        self.configured_speeds.append((rig.arm, speed, "linear" if linear else "joint"))

    def _send_motion(self, rig, command, label):
        if rig.arm in self.failing:
            raise RuntimeError(f"injected transport failure on the {rig.arm} arm")
        self.sent.append((rig.arm, command, label))
        return {"command": command, "expected_ack": f"0,{{}},{command};",
                "raw_reply": f"0,{{}},{command};", "result": "acknowledged"}

    def _wait_for_joints(self, rig, target, label, timeout=None):
        self.arrivals.append((rig.arm, list(target), label, timeout))
        return {"mode": 5, "worst_joint_error_deg": 0.0}


class HomingOperationTests(unittest.TestCase):
    """Homing exists in both forms, and neither can be pointed somewhere else."""

    def setUp(self):
        self.profile = load_profile()

    def test_the_home_atom_carries_no_target_of_its_own(self):
        """A graph may decide *when* to park, never *where* "parked" is."""
        skills = {skill["id"]: skill for skill in load_catalog()["skills"]}
        self.assertIn("arm.move_to_home", skills)
        self.assertNotIn("target_ref", skills["arm.move_to_home"]["parameters"])
        with self.assertRaises(SkillValidationError):
            normalize_parameters("arm.move_to_home", "left",
                                 {"target_ref": "left.tube.work"}, self.profile)

    def test_the_primitive_recipes_never_smuggle_in_a_home_move(self):
        """Parking the arms belongs to the task graph, not to grasp/carry/pour steps."""
        for name in recipe_names():
            for node in compile_recipe(name)["nodes"]:
                self.assertNotEqual(node["skill"], "arm.move_to_home", name)

    def test_both_forms_of_homing_share_one_implementation(self):
        """The atom and the session utility must not be able to drift apart."""
        driver = _RecordingHomeDriver(self.profile)
        node = {"id": "h", "skill": "arm.move_to_home", "arm": "left",
                "parameters": {"speed": 20}}
        atom = driver.execute(node, self.profile)
        self.assertTrue(atom["success"])
        self.assertEqual(driver.sent,
                         [("left", joint_frame(physical_nova_home_degrees("left")), "home on left")])
        self.assertEqual(driver.arrivals,
                         [("left", list(physical_nova_home_degrees("left")),
                           "arrival at home on left", None)])
        self.assertEqual(driver.session_speeds, [("left", 20)])
        self.assertEqual(driver.configured_speeds, [("left", 20, "joint")])

        utility = _RecordingHomeDriver(self.profile)
        utility.home(("left",), self.profile)
        self.assertEqual(utility.sent, driver.sent)
        self.assertEqual([entry[1] for entry in utility.arrivals],
                         [entry[1] for entry in driver.arrivals])

    def test_home_refuses_without_allow_live(self):
        driver = LiveDriver(profile=self.profile)
        self.assertEqual(driver._rigs, {})
        with self.assertRaises(RuntimeError):
            driver.home(("left", "right"), self.profile)

    def test_home_refuses_without_on_site_confirmation(self):
        for confirmation in (None, lambda: False):
            driver = LiveDriver(profile=self.profile, allow_live=True, confirmation=confirmation)
            with self.assertRaises(RuntimeError):
                driver.home(("left", "right"), self.profile)
            self.assertEqual(driver._rigs, {}, "no connection may be opened before consent")

    def test_home_drives_both_arms_to_the_calibrated_joints_at_the_profile_speed(self):
        driver = _RecordingHomeDriver(self.profile)
        report = driver.home(("left", "right"), self.profile)
        self.assertTrue(report["success"], report)
        self.assertEqual(report["failed_arms"], [])
        self.assertFalse(report["grippers_commanded"])
        self.assertFalse(report["arms_disabled"])
        self.assertFalse(report["physical_success_claimed"])
        self.assertEqual([entry[0] for entry in driver.sent], ["left", "right"],
                         "arms are homed one at a time, in the order requested")
        for arm, command, _ in driver.sent:
            self.assertEqual(command, joint_frame(physical_nova_home_degrees(arm)))
            self.assertIn("JointMovJ", command)
        for arm, target, _, _ in driver.arrivals:
            self.assertEqual(target, list(physical_nova_home_degrees(arm)))
        expected_speed = {arm: self.profile["arms"][arm]["speed_default"] for arm in ("left", "right")}
        self.assertEqual(dict(driver.session_speeds), expected_speed)
        self.assertEqual({arm: speed for arm, speed, _ in driver.configured_speeds}, expected_speed)
        self.assertTrue(all(mode == "joint" for _, _, mode in driver.configured_speeds))
        for arm in ("left", "right"):
            self.assertTrue(report["arms"][arm]["success"])
            self.assertEqual(report["arms"][arm]["target_joints"],
                             list(physical_nova_home_degrees(arm)))

    def test_home_speed_follows_the_profile_instead_of_hardcoding_it(self):
        profile = copy.deepcopy(self.profile)
        profile["arms"]["right"]["speed_default"] = 4
        driver = _RecordingHomeDriver(profile)
        driver.home(("right",), profile)
        self.assertEqual(driver.session_speeds, [("right", 4)])
        self.assertEqual(driver.configured_speeds, [("right", 4, "joint")])

    def test_home_skips_an_arm_already_at_the_calibrated_joints(self):
        """Parked is a state, not a gesture: an arm already there is left untouched."""
        driver = _RecordingHomeDriver(self.profile)
        driver.probe_offset_deg = 0.0
        report = driver.home(("left",), self.profile)
        self.assertTrue(report["success"], report)
        entry = report["arms"]["left"]
        self.assertTrue(entry["skipped"])
        self.assertEqual(entry["observed"]["worst_joint_error_deg"], 0.0)
        self.assertEqual(entry["observed"]["mode"], 5)
        self.assertEqual(driver.sent, [], "a parked arm must not be commanded again")
        self.assertEqual(driver.session_speeds, [],
                         "a look must not open or configure a session")
        self.assertEqual(driver.configured_speeds, [])
        self.assertIsNone(driver._lease, "a look must not take the device lease")

    def test_home_skips_only_within_the_arrival_tolerance(self):
        """The bar for "already parked" is the bar a fresh arrival would have to clear."""
        parked = _RecordingHomeDriver(self.profile)
        parked.probe_offset_deg = ARRIVAL_TOL_DEG * 0.8
        parked.home(("left",), self.profile)
        self.assertEqual(parked.sent, [])

        away = _RecordingHomeDriver(self.profile)
        away.probe_offset_deg = ARRIVAL_TOL_DEG * 1.2
        away.home(("left",), self.profile)
        self.assertEqual([entry[0] for entry in away.sent], ["left"])

    def test_home_does_not_treat_an_arm_that_is_not_idle_as_parked(self):
        driver = _RecordingHomeDriver(self.profile)
        driver.probe_offset_deg = 0.0
        driver.probe_mode = 9
        driver.home(("left",), self.profile)
        self.assertEqual([entry[0] for entry in driver.sent], ["left"],
                         "matching joints are not enough: RobotMode 5 is the idle proof")

    def test_one_failing_arm_does_not_cancel_the_other(self):
        driver = _RecordingHomeDriver(self.profile, failing={"left"})
        report = driver.home(("left", "right"), self.profile)
        self.assertFalse(report["success"])
        self.assertEqual(report["failed_arms"], ["left"])
        self.assertIn("injected transport failure", report["arms"]["left"]["error"])
        self.assertIs(report["arms"]["left"]["success"], False)
        self.assertTrue(report["arms"]["right"]["success"])
        self.assertEqual([entry[0] for entry in driver.sent], ["right"])

    def test_home_never_touches_a_gripper_even_when_an_arm_is_mid_pour(self):
        """A tipped arm is homed by joints alone; the jaws keep holding whatever they hold."""
        driver = _RecordingHomeDriver(self.profile)
        driver.home(("right",), self.profile)
        expected = self.profile["arms"]["right"]["speed_default"]
        self.assertEqual(driver.session_speeds, [("right", expected)])
        self.assertEqual(driver.configured_speeds, [("right", expected, "joint")])
        self.assertNotIn("gripper", " ".join(command for _, command, _ in driver.sent))

    def test_home_reports_transport_failure_per_arm_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            loopback = copy.deepcopy(self.profile)
            for arm in loopback["arms"].values():
                arm["ip"] = "127.0.0.1"      # never the real rig
            driver = LiveDriver(profile=loopback, allow_live=True, confirmation=lambda: True,
                                lock_path=Path(directory) / "device-owner")
            report = driver.home(("left", "right"), loopback)
            self.assertFalse(report["success"])
            self.assertEqual(report["failed_arms"], ["left", "right"])
            self.assertTrue(all(entry["error"] for entry in report["arms"].values()))
            # The read-only probe is what fails here, before any session or
            # command, so the failure must not leave rig ownership behind either
            # - it never took any. close() stays idempotent and says so.
            self.assertIsNone(driver._lease)
            self.assertEqual(driver.close()["lease_released"], False)


class HomeCliTests(unittest.TestCase):
    def test_the_home_command_refuses_to_run_without_live(self):
        import contextlib
        import io
        from task_skills.chemistry.cli import main

        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["home"])
        self.assertEqual(code, 2)
        self.assertIn("--live", stderr.getvalue())

    def test_the_home_command_rejects_an_empty_arm_list(self):
        import contextlib
        import io
        from task_skills.chemistry.cli import main

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(["home", "--live", "--arms", ",,"])
        self.assertEqual(code, 2)
        self.assertIn("no arm", stderr.getvalue())


class _ConcurrentRecordingDriver(MockDriver):
    """MockDriver with a real duration, so wall-clock overlap becomes visible.

    An instant driver cannot show concurrency: two zero-length intervals never
    intersect. Sleeping here is what turns "the scheduler said it dispatched both"
    into "both arms were measurably under command at once".
    """

    def __init__(self, seconds=0.1, fail_on=None):
        self.seconds = seconds
        self.fail_on = fail_on
        self.started = {}
        self.finished = {}
        self.in_flight = []
        self.max_in_flight_per_arm = {}
        self.stops = []
        self._lock = threading.Lock()

    def execute(self, node, profile):
        with self._lock:
            self.started[node["id"]] = time.monotonic()
            self.in_flight.append(node["arm"])
            self.max_in_flight_per_arm[node["arm"]] = max(
                self.max_in_flight_per_arm.get(node["arm"], 0),
                self.in_flight.count(node["arm"]))
        try:
            if node["id"] == self.fail_on:
                raise RuntimeError("synthetic failure")
            time.sleep(self.seconds)
            return {"simulated": True, "skill": node["skill"], "arm": node["arm"]}
        finally:
            with self._lock:
                self.in_flight.remove(node["arm"])
                self.finished[node["id"]] = time.monotonic()

    def stop(self, used_arms):
        self.stops.append(sorted(used_arms))
        return {"simulated": True, "stopped": sorted(used_arms)}


class ConcurrentDagTests(unittest.TestCase):
    """What the DAG permits, the scheduler must actually use."""

    def setUp(self):
        self.profile = load_profile()
        self.graph = json.loads(EXAMPLE_GRAPH.read_text(encoding="utf-8"))

    def _report(self, **kwargs):
        self.driver = _ConcurrentRecordingDriver(**kwargs)
        return execute_graph(self.graph, self.driver, self.profile)

    def test_the_two_arms_really_move_at_the_same_time(self):
        report = self._report(seconds=0.1)
        self.assertEqual(report["status"], "simulated")
        self.assertEqual(report["execution"]["mode"], "concurrent_dag")
        self.assertEqual(report["execution"]["workers"], 2)
        pairs = report["execution"]["simultaneous_pairs"]
        for expected in (["left_tube_approach", "right_beaker_approach"],
                         ["left_tube_insert", "right_beaker_insert"],
                         ["left_tube_close", "right_beaker_close"]):
            self.assertIn(expected, pairs)
        self.assertEqual(self.driver.max_in_flight_per_arm, {"left": 1, "right": 1})

    def test_nothing_ever_overlaps_on_the_same_arm(self):
        report = self._report(seconds=0.05)
        by_id = {node["id"]: node for node in report["nodes"]}
        for first, second in report["execution"]["simultaneous_pairs"]:
            self.assertNotEqual(by_id[first]["arm"], by_id[second]["arm"],
                                f"{first} and {second} share an arm and must not overlap")

    def test_every_declared_dependency_is_respected_in_wall_time(self):
        """Simultaneity must never come from skipping a dependency."""
        report = self._report(seconds=0.03)
        by_id = {node["id"]: node for node in report["nodes"]}
        for node in self.graph["nodes"]:
            for dependency in node["depends_on"]:
                self.assertLessEqual(
                    by_id[dependency]["finished_after_seconds"],
                    by_id[node["id"]]["started_after_seconds"],
                    f"{node['id']} started before its dependency {dependency} finished")

    def test_the_ordered_phases_stay_ordered(self):
        """The spec's ordering survives the concurrency: these phases never overlap."""
        report = self._report(seconds=0.03)
        by_id = {node["id"]: node for node in report["nodes"]}
        self.assertLessEqual(by_id["right_beaker_to_work"]["started_after_seconds"],
                             by_id["left_tube_to_work"]["started_after_seconds"],
                             "the beaker must set off for the work pose before the tube does")
        self.assertLessEqual(by_id["left_tube_to_work"]["finished_after_seconds"],
                             by_id["pour_into_tube"]["started_after_seconds"],
                             "the pour must not start before the tube has arrived")
        # The return is staggered rather than serial: the left departs, and the
        # right departs a fixed 5 s later so the two returns overlap.
        self.assertLessEqual(by_id["right_beaker_untip"]["finished_after_seconds"],
                             by_id["left_tube_return"]["started_after_seconds"],
                             "the left arm departs to put the tube back first")
        self.assertLessEqual(by_id["left_tube_return"]["started_after_seconds"],
                             by_id["right_beaker_return"]["started_after_seconds"],
                             "the right arm departs after the stagger, not before")
        # The shared reset only starts once both arms are clear.
        for retreat in ("left_tube_retreat", "right_beaker_retreat"):
            self.assertLessEqual(by_id[retreat]["finished_after_seconds"],
                                 by_id["left_arm_home"]["started_after_seconds"],
                                 f"the reset started before {retreat} finished")

    def test_the_carry_out_is_staggered_the_way_the_return_is(self):
        """Right departs for the work pose first; the left follows 8 s later.

        Same shape as the return: the dwell and the arm that moves first share one
        trigger, so the 8 s runs from "the right arm set off" rather than from
        "the right arm arrived" - which is what "8 s after" has to mean when the
        carry takes an unknown amount of time.
        """
        nodes = {node["id"]: node for node in self.graph["nodes"]}
        stagger = nodes["work_stagger_8s"]
        self.assertEqual(stagger["skill"], "control.hold")
        self.assertEqual(stagger["parameters"]["seconds"], 8)
        self.assertEqual(stagger["arm"], "left", "the arm that waits owns the dwell")
        self.assertEqual(nodes["right_beaker_to_work"]["depends_on"], ["right_beaker_close"])
        self.assertEqual(stagger["depends_on"], ["left_tube_close", "right_beaker_close"])
        self.assertEqual(nodes["left_tube_to_work"]["depends_on"],
                         ["left_tube_close", "work_stagger_8s"])
        self.assertEqual(nodes["pour_into_tube"]["depends_on"],
                         ["left_tube_to_work", "right_beaker_to_work"])

    def test_the_carry_out_stagger_ticks_while_the_right_arm_is_already_moving(self):
        report = self._report(seconds=0.1)
        self.assertIn(["right_beaker_to_work", "work_stagger_8s"],
                      report["execution"]["simultaneous_pairs"],
                      "the 8 s must run while the right arm is already carrying, "
                      "not after it has parked")

    def test_the_two_returns_are_staggered_and_the_two_resets_are_simultaneous(self):
        report = self._report(seconds=0.1)
        pairs = report["execution"]["simultaneous_pairs"]
        self.assertIn(["left_tube_return", "return_stagger_5s"], pairs,
                      "the stagger must tick while the left arm is already moving")
        self.assertIn(["left_arm_home", "right_arm_home"], pairs,
                      "the final reset must be simultaneous, not one arm after the other")
        by_id = {node["id"]: node for node in report["nodes"]}
        self.assertEqual(by_id["left_arm_home"]["started_after_seconds"],
                         by_id["right_arm_home"]["started_after_seconds"])

    def test_the_stagger_and_the_shared_reset_are_expressed_as_dependencies(self):
        nodes = {node["id"]: node for node in self.graph["nodes"]}
        stagger = nodes["return_stagger_5s"]
        self.assertEqual(stagger["skill"], "control.hold")
        self.assertEqual(stagger["parameters"]["seconds"], 5)
        self.assertEqual(stagger["arm"], "right")
        # The dwell and the left arm's departure share one trigger, so the 5 s is
        # measured from "the left set off", not from "the left arrived".
        self.assertEqual(nodes["left_tube_return"]["depends_on"], ["right_beaker_untip"])
        self.assertEqual(stagger["depends_on"], ["right_beaker_untip"])
        self.assertEqual(nodes["right_beaker_return"]["depends_on"],
                         ["right_beaker_untip", "return_stagger_5s"])
        # A shared barrier is what makes the two home moves start together.
        for home in ("left_arm_home", "right_arm_home"):
            self.assertEqual(nodes[home]["skill"], "arm.move_to_home")
            self.assertEqual(nodes[home]["depends_on"],
                             ["left_tube_retreat", "right_beaker_retreat"])

    def test_a_single_arm_graph_has_nothing_to_run_in_parallel(self):
        driver = _ConcurrentRecordingDriver(seconds=0.02)
        report = execute_graph(compile_recipe("left_tube_to_work"), driver, self.profile)
        self.assertEqual(report["execution"]["simultaneous_pairs"], [])
        self.assertEqual(report["execution"]["workers"], 1)
        self.assertEqual(driver.max_in_flight_per_arm, {"left": 1})

    def test_the_mock_driver_honours_a_declared_dwell(self):
        """A simulated run has to keep the graph's own timing, or a 5 s / 8 s
        stagger would be invisible to whoever is checking the graph without arms."""
        graph = {"schema_version": "1.0.0", "name": "dwell_probe",
                 "nodes": [{"id": "wait", "skill": "control.hold", "version": "1.0.0",
                            "arm": "left", "parameters": {"seconds": 0.4}, "depends_on": []}]}
        started = time.monotonic()
        report = execute_graph(graph, MockDriver(), self.profile)
        self.assertEqual(report["status"], "simulated")
        self.assertGreaterEqual(time.monotonic() - started, 0.4)

    def test_a_failure_stops_dispatch_and_leaves_the_sibling_its_own_outcome(self):
        report = self._report(seconds=0.15, fail_on="right_beaker_insert")
        by_id = {node["id"]: node for node in report["nodes"]}
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error_node"], "right_beaker_insert")
        self.assertEqual(by_id["right_beaker_insert"]["status"], "failed")
        # The left insert was already under command; a stop is a stop, not a rollback.
        self.assertEqual(by_id["left_tube_insert"]["status"], "completed")
        self.assertEqual(self.driver.max_in_flight_per_arm, {"left": 1, "right": 1})
        # Nothing may be reported as attempted that never was.
        for node_id in ("left_tube_close", "right_beaker_close", "left_tube_to_work",
                        "pour_into_tube", "right_beaker_retreat"):
            self.assertEqual(by_id[node_id]["status"], "blocked", node_id)
            self.assertEqual(by_id[node_id]["blocked_by"], "right_beaker_insert")
        self.assertEqual(self.driver.stops, [["left", "right"]])


class _FakeRig:
    def __init__(self, arm):
        self.arm = arm
        self.configured = False
        self.dashboard = _FakeDashboard()
        self.motion = None
        self.feedback = None
        self.gripper = None


class _FakeDashboard:
    def __getattr__(self, name):
        return lambda *args: f"0,{{}},{name}();"


class ConcurrentSessionTests(unittest.TestCase):
    def test_both_arms_opening_at_once_take_the_lease_exactly_once(self):
        """The lease is one resource for the whole machine, not one per arm."""
        created = []

        class _Lease:
            def __init__(self, path=None, purpose=None):
                self.acquired = False
                self.owner = f"fake-{len(created)}"
                created.append(self)

            def acquire(self):
                if any(other.acquired for other in created):
                    raise RuntimeError("Dobot rig is owned or quarantined")
                time.sleep(0.05)          # widen the window the lock has to close
                self.acquired = True
                return self

            def release(self):
                self.acquired = False

        with tempfile.TemporaryDirectory() as directory:
            profile = copy.deepcopy(load_profile())
            driver = LiveDriver(profile=profile, allow_live=True,
                                lock_path=Path(directory) / "device-owner")
            driver._rig = lambda arm, prof, with_motion=True: driver._rigs.setdefault(  # type: ignore[assignment]
                arm, _FakeRig(arm))
            barrier = threading.Barrier(2)
            errors = []

            def open_arm(arm):
                try:
                    barrier.wait(timeout=5)
                    driver._ensure_session(profile, arm, 15)
                except Exception as error:  # noqa: BLE001
                    errors.append(f"{arm}: {type(error).__name__}: {error}")

            with mock.patch("task_skills.chemistry.driver.DeviceLease", _Lease):
                threads = [threading.Thread(target=open_arm, args=(arm,))
                           for arm in ("left", "right")]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=10)

            self.assertEqual(errors, [])
            self.assertEqual(len(created), 1,
                             "each arm must not take its own lease on the shared rig")
            self.assertTrue(driver._rigs["left"].configured)
            self.assertTrue(driver._rigs["right"].configured)
