from dataclasses import dataclass
import time
from typing import List

import numpy as np

# On this rig, forward from the Nova bases toward the operator is taken as
# negative base Y. Shift the existing allowed region by 150 mm; retain its
# width, X bounds and table clearance. Confirm this direction onsite.
WORKSPACE_X_RANGES_MM = {"left": (-450, 300), "right": (-250, 450)}
WORKSPACE_Y_SHIFT_MM = -150
WORKSPACE_Y_RANGE_MM = tuple(y + WORKSPACE_Y_SHIFT_MM for y in (-750, -160))
WORKSPACE_Z_MIN_MM = 40


def is_within_safe_position(position, x_range, y_range, z_min):
    return (
        x_range[0] <= position[0] <= x_range[1]
        and y_range[0] <= position[1] <= y_range[1]
        and position[2] > z_min
    )


def is_within_workspace(position_mm, side):
    return is_within_safe_position(position_mm, WORKSPACE_X_RANGES_MM[side],
                                   WORKSPACE_Y_RANGE_MM, WORKSPACE_Z_MIN_MM)


def check_pose_protection(positions, velocities, nova_flags):
    """Check only the Nova sides actively receiving arm commands."""
    flags = np.asarray(nova_flags)
    if flags.shape != (2,):
        raise ValueError(f"Expected two Nova flags, got shape {flags.shape}")

    protected = False
    warnings = []
    positions_mm = {key: value * 1000 for key, value in positions.items()}
    sides = (
        ("left", bool(flags[0])),
        ("right", bool(flags[1])),
    )

    for side, active in sides:
        if not active:
            continue
        tip_names = (f"{side}_left", f"{side}_right")
        if any(velocities[name][2] < -1 for name in tip_names):
            warnings.append(
                f"[Warn]:The {side} robot speed of the TCP is moving too fast!"
            )
            warnings.append(f"delta_{tip_names[0]}: {velocities[tip_names[0]][2]}")
            protected = True
        if not all(
            is_within_workspace(positions_mm[name], side)
            for name in tip_names
        ):
            warnings.append(f"[Warn]:The {side} arm is out of the safe zone!")
            protected = True

    for warning in warnings:
        print(warning)
    return protected


def consume_servo_warmup(follower_ref, master_now, remaining_cycles):
    follower_ref = np.asarray(follower_ref, dtype=float)
    master_now = np.asarray(master_now, dtype=float)
    if follower_ref.shape != (14,) or master_now.shape != (14,):
        raise ValueError("Expected 14-D follower and master samples")
    if remaining_cycles <= 0:
        return None, master_now.copy(), 0
    return follower_ref.copy(), master_now.copy(), int(remaining_cycles) - 1


@dataclass(frozen=True)
class MasterRunawayResult:
    tripped: bool
    joint_indices: List[int]
    deltas: np.ndarray
    allowed_delta_rad: float
    elapsed_s: float


class MasterRunawayGuard:
    """Detect implausible master-arm speed without accumulating motion."""

    def __init__(
        self,
        max_speed_rad_s: float,
        min_delta_rad: float,
        required_violation_cycles: int = 2,
    ) -> None:
        if max_speed_rad_s <= 0:
            raise ValueError("max_speed_rad_s must be positive")
        if min_delta_rad < 0:
            raise ValueError("min_delta_rad must be non-negative")
        if required_violation_cycles < 1:
            raise ValueError("required_violation_cycles must be positive")
        self.max_speed_rad_s = float(max_speed_rad_s)
        self.min_delta_rad = float(min_delta_rad)
        self.required_violation_cycles = int(required_violation_cycles)
        self._previous = None
        self._previous_timestamp = None
        self._violation_cycles = 0

    def reset(self, sample, timestamp=None) -> None:
        sample = self._validate_sample(sample)
        if timestamp is None:
            timestamp = time.monotonic()
        self._previous = sample.copy()
        self._previous_timestamp = float(timestamp)
        self._violation_cycles = 0

    def clear(self) -> None:
        self._previous = None
        self._previous_timestamp = None
        self._violation_cycles = 0

    def check(self, sample, arm_flags, timestamp=None) -> MasterRunawayResult:
        sample = self._validate_sample(sample)
        flags = np.asarray(arm_flags)
        if flags.shape != (2,):
            raise ValueError(f"Expected two arm flags, got shape {flags.shape}")
        if timestamp is None:
            timestamp = time.monotonic()
        timestamp = float(timestamp)

        if self._previous is None:
            self.reset(sample, timestamp)
            return MasterRunawayResult(
                tripped=False,
                joint_indices=[],
                deltas=np.zeros(14),
                allowed_delta_rad=self.min_delta_rad,
                elapsed_s=0.0,
            )

        elapsed_s = timestamp - self._previous_timestamp
        if elapsed_s < 0:
            raise ValueError("timestamp must be monotonic")
        deltas = np.abs(sample - self._previous)
        allowed_delta_rad = max(
            self.min_delta_rad,
            self.max_speed_rad_s * elapsed_s,
        )

        joint_indices = []
        if bool(flags[0]):
            joint_indices.extend(
                int(index)
                for index in np.where(deltas[0:6] > allowed_delta_rad)[0]
            )
        if bool(flags[1]):
            joint_indices.extend(
                int(index + 7)
                for index in np.where(deltas[7:13] > allowed_delta_rad)[0]
            )

        if joint_indices:
            self._violation_cycles += 1
        else:
            self._violation_cycles = 0
        tripped = self._violation_cycles >= self.required_violation_cycles

        # Advance the reference every cycle. The old control loop omitted
        # this update and eventually compared all motion to the unlock pose.
        self._previous = sample.copy()
        self._previous_timestamp = timestamp
        return MasterRunawayResult(
            tripped=tripped,
            joint_indices=joint_indices,
            deltas=deltas,
            allowed_delta_rad=allowed_delta_rad,
            elapsed_s=elapsed_s,
        )

    @staticmethod
    def _validate_sample(sample) -> np.ndarray:
        sample = np.asarray(sample, dtype=float)
        if sample.shape != (14,):
            raise ValueError(f"Expected 14-D master sample, got shape {sample.shape}")
        return sample
