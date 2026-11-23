"""Platform start ("photo") poses shared by the chem-demo cycles.

The photo pose is where each arm waits before every top-camera snapshot. It is
high enough to clear the instruments on the white platform, so it is now the
start and return pose of the chem-demo flows instead of the low home pose.
The numbers live in poses.json (verified on the real robot 2026-11-20).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

POSES_PATH = Path(__file__).with_name("poses.json")

# Left wrist camera calibration, hover references and the commissioned neck
# grasp all use this yaw. poses.json records +90 for the left photo pose; the
# gripper is symmetric so the top camera does not care, and cycles rotate the
# wrist in place before any wrist-camera work.
LEFT_WRIST_RPY = (180.0, 0.0, -89.94)
RIGHT_WRIST_RPY = (180.0, 0.0, 89.97)


def photo_pose(arm: str) -> tuple[list[float], list[float]]:
    """Return (xyz, rpy) of the photo pose for ``left`` or ``right`` (base frame, mm/deg)."""
    if arm not in ("left", "right"):
        raise ValueError(f"unknown arm {arm!r}")
    entry = json.loads(POSES_PATH.read_text(encoding="utf-8"))["photo"][arm]
    xyz = [float(v) for v in entry["xyz"]]
    rpy = [float(v) for v in entry["rpy"]]
    if len(xyz) != 3 or len(rpy) != 3:
        raise ValueError(f"poses.json photo.{arm} must hold 3 xyz and 3 rpy numbers")
    return xyz, rpy


def near_photo(arm: str, pose: Sequence[float], *, xy_tol: float = 5.0, z_tol: float = 5.0) -> bool:
    """True if a measured 6-vector TCP pose sits at the photo position (orientation not checked)."""
    xyz, _ = photo_pose(arm)
    return (abs(pose[2] - xyz[2]) <= z_tol and
            ((pose[0] - xyz[0]) ** 2 + (pose[1] - xyz[1]) ** 2) ** 0.5 <= xy_tol)


def handoff_station() -> dict | None:
    """Stored handoff station: {"left_tcp": [x, y, z], "approach_offset_deg": a, ...} or None.

    It is found once by the full reachability search (run_handoff --preflight --search) and then
    reused, so every run hands over at the same place. The bottle position still comes from the
    top camera and the final right-hand alignment from the right wrist camera.
    """
    entry = json.loads(POSES_PATH.read_text(encoding="utf-8")).get("handoff_station")
    if entry is None:
        return None
    point = [float(v) for v in entry["left_tcp"]]
    if len(point) != 3:
        raise ValueError("poses.json handoff_station.left_tcp must hold 3 numbers")
    return {**entry, "left_tcp": point, "approach_offset_deg": float(entry["approach_offset_deg"])}


def save_handoff_station(left_tcp: Sequence[float], approach_offset_deg: float, source: str,
                         pitch_down_deg: float = 0.0) -> None:
    """Write the handoff station into poses.json, keeping every other entry."""
    data = json.loads(POSES_PATH.read_text(encoding="utf-8"))
    data["handoff_station"] = {"left_tcp": [round(float(v), 2) for v in left_tcp],
                               "approach_offset_deg": float(approach_offset_deg),
                               "pitch_down_deg": float(pitch_down_deg), "source": str(source)}
    POSES_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
