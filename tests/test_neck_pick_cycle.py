"""Focused simulation checks for the commissioned single-start neck grasp."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from chem_demo.neck_pick_cycle import (
    NeckPickCycle,
    NeckPickRefused,
    HOVER_Z,
    START_RPY,
    preflight,
    validate_profile,
    validate_start,
)
from task_skills.run_cartesian_jog import JogRefused, SimArm


HOME = [117.6, -344.8, 415.5]
PHOTO_XYZ = [300.0, -160.0, 535.0]
TARGET_XY = [265.0, -381.0]


def commissioned_profile() -> dict:
    return {
        "neck_grasp": {
            "arm": "left",
            "frame": "base",
            "units": "mm",
            "tcp_z": 360.0,
            "hover_z": HOVER_Z,
            "rpy": list(START_RPY),
            "gripper": {
                "open": 255,
                "target": 0,
                "speed": 5,
                "free_closed_final": 4,
                "loaded_closed_final": 94,
            },
            "commissioning": {
                "confirmed": True,
                "actual_closed_around_neck": True,
                "evidence": "simulated operator-confirmed neck closure",
                "verified_tcp_z_mm": 360.0,
                "z_soft_floor_mm": 350.0,
                "z_range_mm": {"min": 350.0, "max": HOVER_Z},
            },
        }
    }


class NeckArm(SimArm):
    """SimArm with explicit compliant-close feedback from the commissioning profile."""

    def __init__(self, xyz=HOME, *, final=94):
        super().__init__(xyz, START_RPY)
        self.close_final = final
        self.grip_calls: list[tuple[int, int]] = []

    def grip_to(self, position: int, speed: int) -> dict:
        self.grip_calls.append((position, speed))
        before = self.gripper_pos
        self.gripper_pos = self.close_final
        return {"start": before, "target": position, "final": self.close_final,
                "samples": [self.close_final] * 5,
                "travel": abs(before - self.close_final)}


def cycle_for(arm: NeckArm, *, localization=None, wrist=None, verify=None, stop_pressed=lambda: False, lift_mm=15):
    observations = iter(({"id": "before"}, {"id": "after"}))
    calls = {"localize": [], "wrist": [], "verify": []}

    def localize(stage, pose):
        calls["localize"].append((stage, list(pose)))
        return localization or {"ok": True, "xy": TARGET_XY, "source": "sim top observation"}

    def check_wrist(pose, target):
        calls["wrist"].append((list(pose), dict(target)))
        if callable(wrist):
            return wrist(pose, target)
        return wrist or {"ok": True, "center_error_mm": 0.4}

    def verify_held(before, after):
        calls["verify"].append((before, after))
        if callable(verify):
            return verify(before, after)
        return verify if verify is not None else {"held": True, "source": "jaw-relative sim evidence"}

    cycle = NeckPickCycle(
        arm,
        commissioned_profile(),
        localize=localize,
        check_wrist=check_wrist,
        read_wrist=lambda: next(observations),
        verify_held=verify_held,
        lift_mm=lift_mm,
        stop_pressed=stop_pressed,
        out=lambda _line: None,
    )
    return cycle, calls


class NeckPickCycleTest(unittest.TestCase):
    def test_requested_30_mm_lift_from_normal_start(self):
        arm = NeckArm()
        cycle, calls = cycle_for(arm, lift_mm=30)
        self.assertTrue(cycle.run()["result"]["success"])
        self.assertEqual(len(calls["localize"]), 1)
        self.assertAlmostEqual(arm.pose[2], cycle.profile["neck_grasp"]["tcp_z"] + 30)

    def test_preflight_accepts_only_home_or_high_hover(self):
        arm = NeckArm()
        config = preflight(commissioned_profile(), arm.read())
        self.assertEqual(config["start_kind"], "home")

        high = NeckArm([HOME[0], HOME[1], HOVER_Z])
        self.assertEqual(validate_start(high.read()), "high_hover")
        target_hover = NeckArm([TARGET_XY[0], TARGET_XY[1], HOVER_Z])
        self.assertEqual(validate_start(target_hover.read()), "high_hover")
        self.assertEqual(validate_profile(commissioned_profile())["tcp_z"], 360.0)

        invalid = NeckArm([HOME[0] + 20.0, HOME[1], HOVER_Z])
        with self.assertRaisesRegex(NeckPickRefused, "clear start and target areas"):
            validate_start(invalid.read())

    def test_photo_pose_is_an_accepted_platform_start(self):
        photo = NeckArm(PHOTO_XYZ)
        self.assertEqual(validate_start(photo.read()), "photo")
        # +90 degree yaw recorded in poses.json is also accepted; it is rotated before wrist work.
        flipped = NeckArm(PHOTO_XYZ)
        flipped.pose[3:] = [180.0, 0.0, 90.0]
        self.assertEqual(validate_start(flipped.read()), "photo")
        tilted = NeckArm(PHOTO_XYZ)
        tilted.pose[3:] = [150.0, 0.0, -90.0]
        with self.assertRaisesRegex(NeckPickRefused, "orientation"):
            validate_start(tilted.read())

    def test_run_from_photo_pose_goes_straight_to_hover_and_grasps(self):
        arm = NeckArm(PHOTO_XYZ)
        cycle, calls = cycle_for(arm)
        self.assertTrue(cycle.run()["result"]["success"])
        zs = [leg[2] for leg, _ in arm.sent]
        # No intermediate leg at photo height: the first leg already ends at hover over the bottle.
        self.assertAlmostEqual(zs[0], HOVER_Z)
        self.assertIn(HOVER_Z, [round(z, 3) for z in zs])
        self.assertEqual(calls["wrist"][0][0][2], HOVER_Z)

    def test_run_from_photo_pose_with_flipped_yaw_rotates_in_place_first(self):
        arm = NeckArm(PHOTO_XYZ)
        arm.pose[3:] = [180.0, 0.0, 90.0]
        arm.joints = arm.solve(arm.pose[:3], arm.pose[3:], None)
        cycle, _ = cycle_for(arm)
        result = cycle.run()
        steps = [s["step"] for s in result["steps"]]
        self.assertIn("align_yaw_settled", steps)
        self.assertLess(steps.index("align_yaw_settled"), steps.index("translate_at_hover"))
        first_moves = [pose for pose, _ in arm.sent[:3]]
        self.assertTrue(all(abs(p[0] - PHOTO_XYZ[0]) < 1e-6 and abs(p[1] - PHOTO_XYZ[1]) < 1e-6
                            for p in first_moves))

    def test_missing_actual_commissioned_depth_fails_before_any_motion(self):
        arm = NeckArm()
        profile = commissioned_profile()
        del profile["neck_grasp"]["tcp_z"]
        cycle, _ = cycle_for(arm)
        cycle.profile = profile

        with self.assertRaisesRegex(NeckPickRefused, "tcp_z"):
            cycle.run()

        self.assertEqual(arm.sent, [])
        self.assertEqual(arm.grip_calls, [])

    def test_nonzero_close_target_is_rejected_before_motion(self):
        arm = NeckArm()
        profile = commissioned_profile()
        profile["neck_grasp"]["gripper"]["target"] = 1
        cycle, _ = cycle_for(arm)
        cycle.profile = profile

        with self.assertRaisesRegex(NeckPickRefused, "full-close target 0"):
            cycle.run()

        self.assertEqual(arm.sent, [])
        self.assertEqual(arm.grip_calls, [])

    def test_fresh_target_is_approached_at_hover_then_wrist_checked_then_lifted(self):
        with tempfile.TemporaryDirectory() as temporary:
            arm = NeckArm()
            cycle, calls = cycle_for(arm)
            cycle.log_path = Path(temporary) / "cycle.json"
            result = cycle.run()
            saved = json.loads(cycle.log_path.read_text(encoding="utf-8"))

        self.assertTrue(result["result"]["success"])
        self.assertEqual(saved["result"], result["result"])
        self.assertEqual(result["result"]["status"], "held_and_verified")
        self.assertEqual(arm.grip_calls, [(0, 5)])
        self.assertEqual(calls["localize"][0][0], "start")
        self.assertAlmostEqual(calls["wrist"][0][1]["z"], HOVER_Z)
        self.assertEqual(calls["verify"], [({"id": "before"}, {"id": "after"})])
        step_names = [step["step"] for step in result["steps"]]
        self.assertLess(step_names.index("gripper_closed"), step_names.index("wrist_before_lift_captured"))
        self.assertLess(step_names.index("wrist_before_lift_captured"), step_names.index("short_verification_lift"))
        targets = [pose[:3] for pose, _rpy in arm.sent]
        self.assertEqual(targets, [
            [HOME[0], HOME[1], HOVER_Z],
            [*TARGET_XY, HOVER_Z],
            [*TARGET_XY, 360.0],
            [*TARGET_XY, 375.0],
        ])
        self.assertTrue(all(_rpy == list(START_RPY) for _pose, _rpy in arm.sent))
        self.assertEqual(arm.gripper_pos, 94)

    def test_localization_outside_clear_patch_fails_before_any_motion(self):
        arm = NeckArm()
        cycle, _ = cycle_for(arm, localization={"ok": True, "xy": [265.0, -330.0]})

        with self.assertRaisesRegex(NeckPickRefused, "outside the commissioned"):
            cycle.run()

        self.assertEqual(arm.sent, [])
        self.assertEqual(arm.grip_calls, [])

    def test_wrist_gate_failure_prevents_descent_and_close(self):
        arm = NeckArm()
        cycle, _ = cycle_for(arm, wrist={"ok": False, "reason": "misaligned"})

        with self.assertRaisesRegex(NeckPickRefused, "wrist alignment gate"):
            cycle.run()

        self.assertEqual([target[2] for target, _ in arm.sent], [HOVER_Z, HOVER_Z])
        self.assertEqual(arm.grip_calls, [])

    def test_hover_offset_is_corrected_then_grasped_at_the_corrected_xy(self):
        true_xy = [TARGET_XY[0] + 5.0, TARGET_XY[1] - 3.0]   # bottle is really here

        def wrist(pose, target):
            correction = [true_xy[0] - pose[0], true_xy[1] - pose[1]]
            ok = math.hypot(*correction) < 4.0
            return {"ok": ok, "reason": None if ok else "off", "correction_xy_mm": correction}

        arm = NeckArm()
        cycle, calls = cycle_for(arm, wrist=wrist)
        result = cycle.run()
        self.assertTrue(result["result"]["success"])
        self.assertEqual(len(calls["wrist"]), 2)
        descend = [pose[:3] for pose, _ in arm.sent][3]
        self.assertAlmostEqual(descend[0], true_xy[0], places=3)
        self.assertAlmostEqual(descend[1], true_xy[1], places=3)
        self.assertAlmostEqual(descend[2], 360.0)

    def test_hover_correction_that_does_not_converge_is_refused_before_descent(self):
        def wrist(pose, target):          # always says "6 mm further": the camera mapping is wrong
            return {"ok": False, "reason": "off", "correction_xy_mm": [6.0, 0.0]}

        arm = NeckArm()
        cycle, _ = cycle_for(arm, wrist=wrist)
        with self.assertRaisesRegex(NeckPickRefused, "not converging"):
            cycle.run()
        self.assertTrue(all(abs(target[2] - HOVER_Z) < 1e-6 for target, _ in arm.sent))
        self.assertEqual(arm.grip_calls, [])

    def test_hover_correction_never_wanders_far_from_the_top_camera_target(self):
        def wrist(pose, target):
            return {"ok": False, "reason": "off", "correction_xy_mm": [40.0, 0.0]}

        arm = NeckArm()
        cycle, _ = cycle_for(arm, wrist=wrist)
        with self.assertRaisesRegex(NeckPickRefused, "from the top-camera target|not converging"):
            cycle.run()
        self.assertEqual(arm.grip_calls, [])

    def test_empty_feedback_code_stays_closed_and_does_not_lift(self):
        arm = NeckArm(final=4)
        cycle, calls = cycle_for(arm)
        result = cycle.run()

        self.assertFalse(result["result"]["success"])
        self.assertEqual(result["result"]["status"], "empty_close")
        self.assertEqual(result["result"]["gripper_released"], False)
        self.assertEqual(arm.sent[-1][0][2], 360.0)
        self.assertEqual(arm.gripper_pos, 4)
        self.assertEqual(calls["verify"], [])

    def test_uncertain_lift_verdict_keeps_the_gripper_closed(self):
        arm = NeckArm()
        cycle, _ = cycle_for(arm, verify={"held": None, "reason": "ambiguous"})

        result = cycle.run()

        self.assertFalse(result["result"]["success"])
        self.assertEqual(result["result"]["status"], "hold_uncertain")
        self.assertFalse(result["result"]["gripper_released"])
        self.assertEqual(arm.gripper_pos, 94)
        self.assertEqual(arm.sent[-1][0][2], 375.0)

    def test_wrist_camera_lost_after_close_still_lifts_and_reports_uncertain(self):
        # 2026-11-23 16:19: the left D405 dropped off USB right after the close; the cycle stopped
        # with the bottle clamped on the table. Now it lifts and leaves the verdict to the jaw code.
        arm = NeckArm()
        cycle, calls = cycle_for(arm)

        def lost():
            raise RuntimeError("Device disconnected. Failed to reconnect: No device connected5000")

        cycle.read_wrist = lost
        result = cycle.run()

        self.assertEqual(result["result"]["status"], "hold_uncertain")
        self.assertFalse(result["result"]["gripper_released"])
        self.assertEqual(calls["verify"], [])
        self.assertEqual(arm.sent[-1][0][2], 375.0)  # the verification lift still happened
        self.assertIn("Device disconnected", result["held_verification"]["reason"])
        self.assertEqual(len(result["wrist_camera_errors"]), 1)

    def test_operator_stop_during_wrist_capture_is_not_swallowed(self):
        arm = NeckArm()
        cycle, _ = cycle_for(arm)

        def stopped():
            raise KeyboardInterrupt

        cycle.read_wrist = stopped
        with self.assertRaises(KeyboardInterrupt):
            cycle.run()

    def test_first_closed_verification_uses_current_xy_without_relocalizing_or_regripping(self):
        arm = NeckArm([TARGET_XY[0], TARGET_XY[1], 360.0], final=94)
        arm.gripper_pos = 94
        cycle, calls = cycle_for(arm)
        contact_pose = [TARGET_XY[0] - 0.5, TARGET_XY[1] - 0.5, 359.5, *START_RPY]

        result = cycle.verify_first_closed(contact_pose)

        self.assertTrue(result["result"]["success"])
        self.assertEqual(result["operation"], "verify_first_closed")
        self.assertEqual(arm.sent, [([TARGET_XY[0], TARGET_XY[1], 375.0], list(START_RPY))])
        self.assertEqual(arm.grip_calls, [])
        self.assertEqual(arm.gripper_pos, 94)
        self.assertEqual(calls["localize"], [])
        self.assertEqual(calls["wrist"], [])
        self.assertEqual(calls["verify"], [({"id": "before"}, {"id": "after"})])
        self.assertEqual(result["closed_contact"]["signed_xyz_delta_mm"], [0.5, 0.5, 0.5])

    def test_first_closed_verification_rejects_wrong_loaded_code_and_pose_discontinuity(self):
        contact = [*TARGET_XY, 360.0, *START_RPY]
        wrong_grip = NeckArm([TARGET_XY[0], TARGET_XY[1], 360.0], final=4)
        wrong_grip.gripper_pos = 4
        cycle, _ = cycle_for(wrong_grip)
        with self.assertRaisesRegex(NeckPickRefused, "loaded-close code"):
            cycle.verify_first_closed(contact)
        self.assertEqual(wrong_grip.sent, [])
        self.assertEqual(wrong_grip.grip_calls, [])

        moved_arm = NeckArm([TARGET_XY[0] + 2.0, TARGET_XY[1], 360.0], final=94)
        moved_arm.gripper_pos = 94
        cycle, _ = cycle_for(moved_arm)
        with self.assertRaisesRegex(NeckPickRefused, "signed XYZ deltas"):
            cycle.verify_first_closed(contact)
        self.assertEqual(moved_arm.sent, [])
        self.assertEqual(moved_arm.grip_calls, [])

    def test_first_closed_verifier_error_is_logged_after_lift_and_keeps_closed(self):
        arm = NeckArm([TARGET_XY[0], TARGET_XY[1], 360.0], final=94)
        arm.gripper_pos = 94

        def fail_verification(_before, _after):
            raise RuntimeError("synthetic verifier failure")

        cycle, _ = cycle_for(arm, verify=fail_verification)
        with self.assertRaisesRegex(RuntimeError, "synthetic verifier"):
            cycle.verify_first_closed([*TARGET_XY, 360.0, *START_RPY])

        self.assertEqual(cycle.log["verification_stage"], "held_verifier")
        self.assertEqual(cycle.log["result"]["status"], "verification_failed_after_lift")
        self.assertFalse(cycle.log["result"]["gripper_released"])
        self.assertEqual(arm.gripper_pos, 94)
        self.assertEqual(arm.sent[-1][0], [TARGET_XY[0], TARGET_XY[1], 375.0])

    def test_ik_refusal_prechecks_the_whole_hover_route_before_motion(self):
        class UnreachableTargetArm(NeckArm):
            def plan(self, xyz, rpy, seed):
                if xyz[0] > HOME[0] + 30:
                    raise RuntimeError("synthetic unreachable target")
                return super().plan(xyz, rpy, seed)

        arm = UnreachableTargetArm()
        cycle, _ = cycle_for(arm)
        with self.assertRaisesRegex(JogRefused, "synthetic unreachable"):
            cycle.run()
        self.assertEqual(arm.sent, [])


if __name__ == "__main__":
    unittest.main()
