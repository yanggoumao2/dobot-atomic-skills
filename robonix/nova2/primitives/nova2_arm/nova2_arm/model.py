"""Pure model for the Robonix Nova2 mock arm."""

from __future__ import annotations

import importlib.util
import math
import threading
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any


class CommandValidationError(ValueError):
    """Raised when a JointState command is incomplete or malformed."""


@dataclass(frozen=True)
class JointStateFrame:
    """A consistent snapshot of the six named arm joints in radians."""

    name: tuple[str, ...]
    position: tuple[float, ...]
    velocity: tuple[float, ...] = ()
    effort: tuple[float, ...] = ()


@dataclass(frozen=True)
class ArmSettings:
    """Validated per-instance settings accepted by the package lifecycle."""

    side: str
    feedback_rate_hz: float = 10.0

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ArmSettings":
        """Parse the package config and reject unsupported keys."""
        if not isinstance(config, Mapping):
            raise ValueError("config must be a mapping")

        unknown = set(config) - {"side", "feedback_rate_hz"}
        if unknown:
            raise ValueError(f"unsupported config key(s): {', '.join(sorted(map(str, unknown)))}")

        side = config.get("side")
        if side not in ("left", "right"):
            raise ValueError("config.side must be exactly 'left' or 'right'")

        rate = config.get("feedback_rate_hz", 10.0)
        if isinstance(rate, bool) or not isinstance(rate, Real):
            raise ValueError("config.feedback_rate_hz must be a finite number from 1 to 50")
        rate = float(rate)
        if not math.isfinite(rate) or not 1.0 <= rate <= 50.0:
            raise ValueError("config.feedback_rate_hz must be a finite number from 1 to 50")

        return cls(side=side, feedback_rate_hz=rate)


def joint_names(side: str) -> tuple[str, ...]:
    """Return the six canonical joint names for one isolated arm instance."""
    if side not in ("left", "right"):
        raise ValueError("side must be exactly 'left' or 'right'")
    return tuple(f"{side}_joint{index}" for index in range(1, 7))


def _default_home_positions(side: str) -> tuple[float, ...]:
    """Read calibrated home angles directly without importing task_skills."""
    layout_path = Path(__file__).resolve().parents[5] / "task_skills" / "layout.py"
    if not layout_path.is_file():
        raise RuntimeError(f"Nova2 home layout is missing: {layout_path}")

    spec = importlib.util.spec_from_file_location("_nova2_arm_layout", layout_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load arm home constants from {layout_path}")
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)

    try:
        positions = tuple(float(value) for value in layout.physical_nova_home_radians(side))
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"cannot read the {side} arm home pose from {layout_path}") from error
    if len(positions) != 6 or not all(math.isfinite(value) for value in positions):
        raise RuntimeError(f"invalid home positions for {side} arm in {layout_path}")
    return positions


def _joint_limits(names: tuple[str, ...]) -> tuple[tuple[float, float], ...]:
    """Load the side's six revolute limits from the package's Nova2 URDF."""
    urdf_path = Path(__file__).resolve().parents[3] / "urdf" / "nova2_dual.urdf"
    if not urdf_path.is_file():
        raise RuntimeError(f"Nova2 URDF is missing: {urdf_path}")
    try:
        root = ET.parse(urdf_path).getroot()
    except (ET.ParseError, OSError) as error:
        raise RuntimeError(f"cannot read joint limits from {urdf_path}") from error

    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in names:
            continue
        limit = joint.find("limit")
        if joint.get("type") != "revolute" or limit is None:
            raise RuntimeError(f"{urdf_path} has no revolute limit for {name}")
        try:
            lower = float(limit.attrib["lower"])
            upper = float(limit.attrib["upper"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{urdf_path} has invalid limits for {name}") from error
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise RuntimeError(f"{urdf_path} has invalid limits for {name}")
        limits[name] = (lower, upper)

    if set(limits) != set(names):
        missing = sorted(set(names) - set(limits))
        raise RuntimeError(f"{urdf_path} is missing limits for: {', '.join(missing)}")
    return tuple(limits[name] for name in names)


def _joint_values(
    values: Any,
    field: str,
    *,
    required: bool,
    expected_count: int,
) -> tuple[float, ...]:
    if values is None:
        values = ()
    if isinstance(values, (str, bytes, bytearray)):
        raise CommandValidationError(f"{field} must be a numeric array")

    try:
        items = tuple(values)
    except TypeError as error:
        raise CommandValidationError(f"{field} must be a numeric array") from error

    if not required and not items:
        return ()
    if len(items) != expected_count:
        requirement = "exactly" if required else "empty or exactly"
        raise CommandValidationError(
            f"{field} must contain {requirement} {expected_count} values"
        )

    result: list[float] = []
    for index, value in enumerate(items):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise CommandValidationError(f"{field}[{index}] must be a finite number")
        number = float(value)
        if not math.isfinite(number):
            raise CommandValidationError(f"{field}[{index}] must be a finite number")
        result.append(number)
    return tuple(result)


def _require_empty(values: Any, field: str) -> None:
    if values is None:
        return
    if isinstance(values, (str, bytes, bytearray)):
        raise CommandValidationError(f"{field} is unsupported; only position commands are accepted")
    try:
        items = tuple(values)
    except TypeError as error:
        raise CommandValidationError(f"{field} must be empty") from error
    if items:
        raise CommandValidationError(f"{field} is unsupported; only position commands are accepted")


class ArmModel:
    """Thread-safe, hardware-free model of one Nova2 arm."""

    def __init__(
        self,
        side: str,
        *,
        home_positions: Any = None,
    ) -> None:
        self.side = side
        self.name = joint_names(side)
        self._limits = _joint_limits(self.name)
        initial = (
            _default_home_positions(side)
            if home_positions is None
            else _joint_values(home_positions, "home_positions", required=True, expected_count=6)
        )
        if len(initial) != 6 or not all(math.isfinite(value) for value in initial):
            raise ValueError("home_positions must contain six finite numbers")
        self._validate_limits(tuple(float(value) for value in initial))
        self._lock = threading.RLock()
        self._position = tuple(float(value) for value in initial)

    def _validate_limits(self, positions: tuple[float, ...]) -> None:
        for name, value, (lower, upper) in zip(self.name, positions, self._limits):
            if value < lower or value > upper:
                raise CommandValidationError(
                    f"{name} position {value} is outside URDF range [{lower}, {upper}] rad"
                )

    def apply_command(
        self,
        names: Any,
        positions: Any,
        *,
        velocity: Any = (),
        effort: Any = (),
    ) -> JointStateFrame:
        """Validate a complete target, then apply all six positions together.

        Names may arrive in any order. Every accepted command must include all
        six canonical names and positions. This mock rejects all velocity and
        effort commands because it models position targets only.
        """
        if isinstance(names, (str, bytes, bytearray)):
            raise CommandValidationError("name must contain the six canonical joint names")
        try:
            received_names = tuple(names)
        except TypeError as error:
            raise CommandValidationError(
                "name must contain the six canonical joint names"
            ) from error

        if len(received_names) != 6 or any(
            not isinstance(name, str) or not name for name in received_names
        ):
            raise CommandValidationError("name must contain exactly six non-empty joint names")
        if len(set(received_names)) != 6:
            raise CommandValidationError("name must not contain duplicate joint names")
        if set(received_names) != set(self.name):
            raise CommandValidationError(
                f"name must contain exactly {', '.join(self.name)}"
            )

        received_positions = _joint_values(
            positions, "position", required=True, expected_count=6
        )
        _require_empty(velocity, "velocity")
        _require_empty(effort, "effort")
        by_name = dict(zip(received_names, received_positions))
        candidate = tuple(by_name[name] for name in self.name)
        self._validate_limits(candidate)

        with self._lock:
            self._position = candidate
            return JointStateFrame(name=self.name, position=self._position)

    def apply_joint_state(self, message: Any) -> JointStateFrame:
        """Apply a generated JointState-shaped object."""
        try:
            return self.apply_command(
                message.name,
                message.position,
                velocity=message.velocity,
                effort=message.effort,
            )
        except AttributeError as error:
            raise CommandValidationError(
                "command must provide name, position, velocity, and effort fields"
            ) from error

    def feedback(self) -> JointStateFrame:
        """Return a stable snapshot in canonical name order."""
        with self._lock:
            return JointStateFrame(name=self.name, position=self._position)
