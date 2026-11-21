"""Pure-Python simulated gripper model for the Nova2 demo primitive."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Any

STROKE_M = 0.08
DEFAULT_SPEED_M_S = 0.08
GRASP_UNKNOWN = 0
MOTION_MOVING = 1
MOTION_STOPPED = 2
_EPSILON_M = 1e-9
_SIMULATED_MESSAGE = "synthetic jaw motion accepted; contact is not modeled."


class MockGripper:
    """Model one simulated jaw pair with time-based synthetic width feedback."""

    def __init__(
        self,
        side: str,
        *,
        active: bool = True,
        monotonic: Callable[[], float] = time.monotonic,
        time_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        normalized_side = str(side).strip().lower()
        if normalized_side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        self.side = normalized_side
        self._monotonic = monotonic
        self._time_ns = time_ns
        self._lock = threading.RLock()
        self._width_m = STROKE_M
        self._target_width_m = STROKE_M
        self._updated_at = monotonic()
        self._motion_state = MOTION_STOPPED
        self._active = active

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def activate(self) -> None:
        """Enable motion after the provider's ACTIVATE lifecycle command."""
        with self._lock:
            self._active = True

    def deactivate(self) -> None:
        """Freeze current width and disable motion until the next activation."""
        with self._lock:
            self._advance_locked()
            self._target_width_m = self._width_m
            self._motion_state = MOTION_STOPPED
            self._active = False

    def info(self) -> dict[str, Any]:
        """Return fixed mock capabilities; no force or contact sensing exists."""
        return {
            "ok": True,
            "message": "synthetic jaw width only; no contact or force sensing",
            "grip_effort_configurable": False,
            "move_speed_configurable": False,
            "grip_speed_configurable": False,
            "open_speed_configurable": False,
            "opening_width_range_valid": True,
            "min_opening_width_m": 0.0,
            "max_opening_width_m": STROKE_M,
            "gripping_force_range_valid": False,
            "min_gripping_force_n": 0.0,
            "max_gripping_force_n": 0.0,
            "grip_with_expected_object_width_supported": False,
            "grip_without_expected_object_width_supported": True,
        }

    def open(self, relative_speed: float = 0.0) -> dict[str, Any]:
        """Dispatch simulated opening to the configured 0.08 m endpoint."""
        error = self._relative_setting_error("relative_speed", relative_speed)
        if error:
            return self._rejected(error)
        return self._move_to(STROKE_M)

    def move(
        self, opening_width_m: float, relative_speed: float = 0.0
    ) -> dict[str, Any]:
        """Dispatch a simulated absolute clear-width target."""
        error = self._relative_setting_error("relative_speed", relative_speed)
        if error:
            return self._rejected(error)
        try:
            width = float(opening_width_m)
        except (TypeError, ValueError):
            return self._rejected("opening_width_m must be a finite number")
        if not math.isfinite(width):
            return self._rejected("opening_width_m must be a finite number")
        if not 0.0 <= width <= STROKE_M:
            return self._rejected(
                f"opening_width_m must be between 0 and {STROKE_M:.2f} m"
            )
        return self._move_to(width)

    def stop(self) -> dict[str, Any]:
        """Halt the simulated jaw movement without changing its current width."""
        with self._lock:
            if not self._active:
                return self._rejected("mock gripper is inactive")
            self._advance_locked()
            self._target_width_m = self._width_m
            self._motion_state = MOTION_STOPPED
        return {"ok": True, "message": "simulated jaw motion stopped"}

    def grip(
        self,
        relative_effort: float = 0.0,
        relative_speed: float = 0.0,
        expected_object_width_valid: bool = False,
        expected_object_width_m: float = 0.0,
    ) -> dict[str, Any]:
        """Close the synthetic jaws; this model never detects object contact."""
        error = self._relative_setting_error("relative_effort", relative_effort)
        if error:
            return self._rejected(error)
        error = self._relative_setting_error("relative_speed", relative_speed)
        if error:
            return self._rejected(error)
        if not isinstance(expected_object_width_valid, bool):
            return self._rejected("expected_object_width_valid must be a boolean")
        if expected_object_width_valid:
            return self._rejected(
                "expected object width is unsupported; this mock has no object model"
            )
        # The no-width form only closes the jaws. It has no object or contact model.
        return self._move_to(0.0)

    def state(self) -> dict[str, Any]:
        """Return synthetic width and motion with contact kept explicitly unknown."""
        with self._lock:
            self._advance_locked()
            timestamp_ns = max(0, int(self._time_ns()))
            return {
                "opening_width_valid": True,
                "opening_width_m": self._width_m,
                "grasp_state": GRASP_UNKNOWN,
                "stamp_sec": timestamp_ns // 1_000_000_000,
                "stamp_nanosec": timestamp_ns % 1_000_000_000,
                "motion_state": self._motion_state,
            }

    @staticmethod
    def _relative_setting_error(name: str, value: float) -> str | None:
        try:
            setting = float(value)
        except (TypeError, ValueError):
            return f"{name} must be a finite number in [0, 1]"
        if not math.isfinite(setting) or not 0.0 <= setting <= 1.0:
            return f"{name} must be a finite number in [0, 1]"
        if setting != 0.0:
            return f"{name} is not configurable by this mock gripper; use 0"
        return None

    @staticmethod
    def _rejected(message: str) -> dict[str, Any]:
        return {"ok": False, "message": message}

    def _move_to(self, width_m: float) -> dict[str, Any]:
        with self._lock:
            if not self._active:
                return self._rejected("mock gripper is inactive")
            self._advance_locked()
            self._target_width_m = width_m
            self._motion_state = (
                MOTION_MOVING
                if abs(self._target_width_m - self._width_m) > _EPSILON_M
                else MOTION_STOPPED
            )
        return {"ok": True, "message": _SIMULATED_MESSAGE}

    def _advance_locked(self) -> None:
        now = self._monotonic()
        elapsed_s = max(0.0, now - self._updated_at)
        self._updated_at = max(self._updated_at, now)
        difference_m = self._target_width_m - self._width_m
        remaining_m = abs(difference_m)
        travel_m = min(remaining_m, DEFAULT_SPEED_M_S * elapsed_s)
        if travel_m:
            self._width_m += math.copysign(travel_m, difference_m)
        if abs(self._target_width_m - self._width_m) <= _EPSILON_M:
            self._width_m = self._target_width_m
            self._motion_state = MOTION_STOPPED
        else:
            self._motion_state = MOTION_MOVING
