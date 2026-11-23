"""The first verified lift must use the very same closure that taught the height."""
import tempfile
import unittest
from pathlib import Path

from chem_demo.neck_commission import commission
from chem_demo.neck_pick_cycle import NeckPickCycle
from task_skills.run_cartesian_jog import Jog, Limits, SimArm


class FieldArm(SimArm):
    arm = "left"
    speed = 5

    def __init__(self):
        super().__init__([265.87, -381.89, 484.86], [179.99, .047, -89.94])
        self.commands = []

    def set_linear_speed(self, speed):
        self.speed = speed

    def grip_to(self, target, speed):
        self.commands.append((target, self.pose[2]))
        start = self.gripper_pos
        self.gripper_pos = 254 if target == 255 else (2 if self.pose[2] > 480 else 103)
        return dict(start=start, target=target, final=self.gripper_pos, samples=[self.gripper_pos] * 5)


class FirstPickTest(unittest.TestCase):
    def test_accepted_field_height_flows_to_loaded_lift_without_reopening(self):
        with tempfile.TemporaryDirectory() as folder:
            arm = FieldArm()
            path = Path(folder) / "commissioned.json"
            jog = Jog(arm, Limits(arm.pose[:3], 2, 350, 486), record_path=path.with_suffix(".csv"))
            candidate = commission(arm, jog, profile={"profile_id": "bench"}, profile_path=path,
                                   same_bottle_designation="blue-rim bottle", ask=self.fail,
                                   out=lambda _: None, accepted_contact_z=358.86, defer_save=True,
                                   sleep=lambda _: None)
            self.assertFalse(path.exists())
            contact = arm.read()["pose"]
            cycle = NeckPickCycle(arm, candidate, localize=self.fail, check_wrist=self.fail,
                                  read_wrist=lambda: arm.read(), verify_held=lambda a, b: {"held": True},
                                  out=lambda _: None)
            self.assertTrue(cycle.verify_first_closed(contact)["result"]["held"])
            self.assertEqual([c[0] for c in arm.commands], [0, 255, 0])
            self.assertAlmostEqual(arm.commands[-1][1], 358.86)
            self.assertAlmostEqual(arm.pose[2], 373.86)
