"""Validated selection of teleoperation cameras; no device access."""
import configparser
from dataclasses import dataclass
from pathlib import Path

CAMERA_NAMES = ("top", "left", "right")
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "scripts/dobot_config/dobot_settings.ini"


@dataclass(frozen=True)
class CameraSettings:
    serials: dict
    fps: int

    @property
    def recording_enabled(self):
        # The dataset writer requires all three views. Never write fake views.
        return set(self.serials) == set(CAMERA_NAMES)


def load_camera_settings(path=DEFAULT_PATH):
    config = configparser.ConfigParser()
    with open(path, encoding="utf-8") as handle:
        config.read_file(handle)
    names = [s.strip() for s in config.get("CAMERA", "enabled", fallback="top,left,right").split(",")]
    if not names or any(n not in CAMERA_NAMES for n in names) or len(set(names)) != len(names):
        raise ValueError("CAMERA.enabled must contain distinct top/left/right names")
    serials = {name: config.get("CAMERA", name).strip() for name in CAMERA_NAMES if name in names}
    if any(not s or not s.isdigit() for s in serials.values()) or len(set(serials.values())) != len(serials):
        raise ValueError("Enabled cameras must have distinct numeric serial numbers")
    fps = config.getint("CAMERA", "fps", fallback=90)
    if fps not in (15, 30, 60, 90):
        raise ValueError("CAMERA.fps must be 15, 30, 60 or 90")
    return CameraSettings(serials=serials, fps=fps)
