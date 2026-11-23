"""Offline checks for guided and replayed neck commissioning."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from chem_demo.neck_commission import CommissioningError, commission, is_neck_grasp_commissioned


class FakeArm:
    arm = "left"

    def __init__(self, *, loaded_final: int = 44, free_final: int = 3, unstable_free: bool = False):
        self.pose = [100.0, -200.0, 485.0, 180.0, 0.0, -90.0]
        self.speed = 8
        self.grip = 255
        self.loaded_final = loaded_final
        self.free_final = free_final
        self.unstable_free = unstable_free
        self.speed_changes: list[int] = []
        self.grip_calls: list[tuple[int, int]] = []
        self.events: list[tuple[str, int | float, float]] = []
        self.stop_calls = 0

    def read(self) -> dict[str, Any]:
        return {"pose": list(self.pose), "mode": 5}

    def gripper_position(self) -> int:
        return self.grip

    def set_linear_speed(self, percent: int) -> None:
        self.speed = percent
        self.speed_changes.append(percent)

    def grip_to(self, position: int, speed: int) -> dict[str, Any]:
        self.grip_calls.append((position, speed))
        self.events.append(("grip", position, self.pose[2]))
        start = self.grip
        if position == 255:
            final = 255
        elif self.pose[2] < 480:
            final = self.loaded_final
        else:
            final = self.free_final
        self.grip = final
        samples = [final] * 5
        if self.unstable_free and position == 0 and self.pose[2] >= 480:
            samples = [final, final + 2, final, final + 1, final]
        return {"start": start, "target": position, "final": final,
                "samples": samples, "travel": abs(final - start)}

    def stop(self) -> None:
        self.stop_calls += 1


class FakeJog:
    def __init__(self, arm: FakeArm):
        self.arm = arm
        self.xyz = list(arm.pose[:3])
        self.rpy = list(arm.pose[3:])
        self.limits = SimpleNamespace(z_min=100.0)
        self.checked: list[list[float]] = []
        self.routed: list[list[float]] = []

    def check_route(self, legs):
        self.checked.extend([list(leg) for leg in legs])
        return [[0.0] * 6 for _ in legs]

    def run_route(self, legs, finals):
        for leg in legs:
            self.routed.append(list(leg))
            self.arm.events.append(("z", leg[2], self.arm.pose[2]))
            self.xyz = list(leg)
            self.arm.pose = [*leg, *self.rpy]


class ScriptedAsk:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        try:
            return next(self.answers)
        except StopIteration as error:
            raise AssertionError(f"Unexpected prompt: {prompt}") from error


class NeckCommissionTests(unittest.TestCase):
    PROFILE = {"profile_id": "test-profile", "arms": {"left": {"ip": "ignored"}}}

    def commission_once(self, path: Path, *, loaded_final=44, free_final=3):
        arm = FakeArm(loaded_final=loaded_final, free_final=free_final)
        jog = FakeJog(arm)
        ask = ScriptedAsk(["y", "0", "d", "d", "c", "y", "y"])
        output: list[str] = []
        result = commission(
            arm, jog, profile=self.PROFILE, profile_path=path,
            same_bottle_designation="same empty bottle A",
            ask=ask, out=output.append, sleep=lambda _: None,
        )
        return result, arm, jog, ask, output

    def test_manual_teach_measures_free_first_and_returns_closed_at_contact(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "commissioned.json"
            path.write_text(json.dumps({"profile_id": "test-profile", "diagnostics": {"keep": True}}),
                            encoding="utf-8")
            result, arm, jog, ask, _ = self.commission_once(path)
            saved = json.loads(path.read_text(encoding="utf-8"))
            neck = result["neck_grasp"]
            self.assertEqual(saved["diagnostics"], {"keep": True})
            self.assertEqual(neck["tcp_z"], 465.0)
            self.assertEqual(neck["hover_z"], 485.0)
            self.assertEqual(neck["rpy"], [180.0, 0.0, -90.0])
            self.assertEqual(neck["gripper"], {
                "open": 255, "target": 0, "speed": 5,
                "free_closed_final": 3, "loaded_closed_final": 44,
            })
            commissioning = neck["commissioning"]
            self.assertEqual(commissioning["commanded_width"]["servo_position_counts"], 0)
            self.assertEqual(commissioning["observed_width"]["loaded_servo_position_counts"], 44)
            self.assertEqual(commissioning["free_baseline"]["measured_at"],
                             "verified_open_air_hover_before_Z")
            self.assertEqual(commissioning["same_bottle_designation"], "same empty bottle A")
            self.assertTrue(commissioning["pre_stability"]["stable"])
            self.assertTrue(commissioning["post_stability"]["stable"])
            self.assertEqual(commissioning["z_range_mm"], {"min": 350.0, "max": 485.0})
            self.assertNotIn("pose_samples", commissioning["pre_stability"])
            self.assertNotIn("xyz", neck)
            self.assertNotIn("xy", neck)
            self.assertEqual([target for target, _ in arm.grip_calls], [0, 255, 0])
            self.assertTrue(all(speed == 5 for _, speed in arm.grip_calls))
            self.assertEqual(arm.events[:3], [
                ("grip", 0, 485.0), ("grip", 255, 485.0), ("z", 475.0, 485.0),
            ])
            self.assertEqual(arm.speed_changes, [5, 8])
            self.assertTrue(is_neck_grasp_commissioned(path, "test-profile"))
            self.assertFalse(is_neck_grasp_commissioned(path, "different-profile"))
            self.assertEqual(len(jog.checked), len(jog.routed))
            self.assertEqual([leg[2] for leg in jog.routed], [475.0, 465.0])
            self.assertTrue(all(leg[:2] == [100.0, -200.0] for leg in jog.routed))
            self.assertEqual(arm.pose, [100.0, -200.0, 465.0, 180.0, 0.0, -90.0])
            self.assertEqual(arm.grip, 44)
            self.assertIn("y 接受並保持閉合", ask.prompts[-1])

    def test_accepted_contact_z_is_prompt_free_deferred_and_returns_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "commissioned.json"
            arm = FakeArm()
            jog = FakeJog(arm)
            candidate = commission(arm, jog, profile=self.PROFILE, profile_path=path,
                                   same_bottle_designation="same empty bottle A", ask=self.fail,
                                   out=lambda _: None, accepted_contact_z=465.0, defer_save=True,
                                   sleep=lambda _: None)
            self.assertFalse(path.exists())
            self.assertEqual([leg[2] for leg in jog.routed], [475.0, 465.0])
            self.assertEqual([position for position, _ in arm.grip_calls], [0, 255, 0])
            self.assertEqual(arm.pose, [100.0, -200.0, 465.0, 180.0, 0.0, -90.0])
            self.assertEqual(arm.grip, 44)
            commissioning = candidate["neck_grasp"]["commissioning"]
            self.assertEqual(commissioning["contact_depth_source"],
                             "prior_operator_approved_contact_z_replayed")
            self.assertFalse(commissioning["contact_visual_confirmation"])
            self.assertFalse(commissioning["fresh_operator_confirmation"])
            self.assertIn("No new operator confirmation", commissioning["evidence"])

    def test_prior_contact_z_uses_checked_steps_and_records_truthful_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            arm = FakeArm()
            jog = FakeJog(arm)
            candidate = commission(arm, jog, profile=self.PROFILE,
                                   profile_path=Path(temp) / "commissioned.json",
                                   same_bottle_designation="same empty bottle A", ask=self.fail,
                                   out=lambda _: None, accepted_contact_z=464.0, defer_save=True,
                                   sleep=lambda _: None)
            self.assertEqual([leg[2] for leg in jog.routed], [475.0, 465.0, 464.0])
            evidence = candidate["neck_grasp"]["commissioning"]["evidence"]
            self.assertIn("No new operator confirmation", evidence)
            self.assertNotIn("visually confirmed", evidence)

    def test_replayed_depth_cannot_persist_before_verified_lift(self):
        with tempfile.TemporaryDirectory() as temp:
            arm = FakeArm()
            jog = FakeJog(arm)
            with self.assertRaisesRegex(CommissioningError, "defer persistence"):
                commission(arm, jog, profile=self.PROFILE,
                           profile_path=Path(temp) / "commissioned.json",
                           same_bottle_designation="same empty bottle A", ask=self.fail,
                           out=lambda _: None, accepted_contact_z=465.0,
                           defer_save=False, sleep=lambda _: None)
            self.assertEqual(jog.routed, [])
            self.assertEqual(arm.grip_calls, [])
            self.assertEqual(arm.speed_changes, [])

    def test_low_start_resume_is_refused_before_any_motion(self):
        with tempfile.TemporaryDirectory() as temp:
            arm = FakeArm()
            arm.pose[2] = 363.0
            jog = FakeJog(arm)
            with self.assertRaisesRegex(CommissioningError, "low-start resume"):
                commission(arm, jog, profile=self.PROFILE,
                           profile_path=Path(temp) / "commissioned.json",
                           same_bottle_designation="same empty bottle A", ask=self.fail,
                           out=lambda _: None, sleep=lambda _: None)
            self.assertEqual(jog.routed, [])
            self.assertEqual(arm.grip_calls, [])
            self.assertEqual(arm.speed_changes, [])

    def test_unstable_free_feedback_refuses_before_any_z_move(self):
        with tempfile.TemporaryDirectory() as temp:
            arm = FakeArm(unstable_free=True)
            jog = FakeJog(arm)
            ask = ScriptedAsk(["y"])
            with self.assertRaisesRegex(CommissioningError, "not stable"):
                commission(arm, jog, profile=self.PROFILE,
                           profile_path=Path(temp) / "commissioned.json",
                           same_bottle_designation="same empty bottle A", ask=ask,
                           out=lambda _: None, sleep=lambda _: None)
            self.assertEqual(jog.routed, [])
            self.assertEqual([target for target, _ in arm.grip_calls], [0, 255])
            self.assertEqual(arm.grip, 255)

    def test_o_opens_only_at_supported_station_and_retunes_z_in_one_mm_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            arm = FakeArm()
            jog = FakeJog(arm)
            ask = ScriptedAsk(["y", "0", "d", "d", "c", "y", "o", "d", "c", "y", "y"])
            result = commission(arm, jog, profile=self.PROFILE,
                                profile_path=Path(temp) / "commissioned.json",
                                same_bottle_designation="same empty bottle A", ask=ask,
                                out=lambda _: None, sleep=lambda _: None)
            self.assertEqual([leg[2] for leg in jog.routed], [475.0, 465.0, 464.0])
            self.assertEqual([target for target, _ in arm.grip_calls], [0, 255, 0, 255, 0])
            self.assertEqual(arm.pose[2], 464.0)
            self.assertEqual(arm.grip, 44)
            self.assertEqual(result["neck_grasp"]["commissioning"]["retune_count"], 1)

    def test_loaded_free_mismatch_returns_no_candidate_and_keeps_supported_contact_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "commissioned.json"
            arm = FakeArm(loaded_final=5, free_final=3)
            jog = FakeJog(arm)
            ask = ScriptedAsk(["y", "0", "d", "d", "c", "y", "y"])
            with self.assertRaisesRegex(CommissioningError, "not distinguishable"):
                commission(arm, jog, profile=self.PROFILE, profile_path=path,
                           same_bottle_designation="same empty bottle A", ask=ask,
                           out=lambda _: None, sleep=lambda _: None)
            self.assertFalse(path.exists())
            self.assertEqual(arm.pose[2], 465.0)
            self.assertEqual(arm.grip, 5)
            self.assertEqual([target for target, _ in arm.grip_calls], [0, 255, 0])

    def test_valid_prior_commissioning_is_not_overwritten_or_moved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "commissioned.json"
            first, _, _, _, _ = self.commission_once(path)
            before = path.read_bytes()
            arm = FakeArm()
            jog = FakeJog(arm)
            with self.assertRaisesRegex(CommissioningError, "already valid"):
                commission(arm, jog, profile=self.PROFILE, profile_path=path,
                           same_bottle_designation="same empty bottle A", ask=self.fail,
                           out=lambda _: None, sleep=lambda _: None)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(arm.grip_calls, [])
            self.assertEqual(arm.speed_changes, [])
            self.assertEqual(first["profile_id"], "test-profile")

    def test_abort_after_free_baseline_does_not_move_z_or_write(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "commissioned.json"
            arm = FakeArm()
            jog = FakeJog(arm)
            ask = ScriptedAsk(["y", "q"])
            with self.assertRaisesRegex(CommissioningError, "aborted"):
                commission(arm, jog, profile=self.PROFILE, profile_path=path,
                           same_bottle_designation="same empty bottle A", ask=ask,
                           out=lambda _: None, sleep=lambda _: None)
            self.assertFalse(path.exists())
            self.assertEqual(jog.routed, [])
            self.assertEqual([target for target, _ in arm.grip_calls], [0, 255])
            self.assertEqual(arm.speed_changes, [5, 8])


if __name__ == "__main__":
    unittest.main()
