"""Validated selection of teleoperation cameras; no device access."""
import configparser
from dataclasses import dataclass, field
from pathlib import Path
from dobot_control.config import config_path as get_config_path

CAMERA_NAMES = ("top", "left", "right")
DEFAULT_PATH = get_config_path()


@dataclass(frozen=True)
class CameraSettings:
    serials: dict
    fps: int
    flips: dict = field(default_factory=lambda: {'top': True, 'left': False, 'right': True})
    depth_enabled: bool = True

    @property
    def recording_enabled(self):
        # Save only configured views, including single-camera teleoperation.
        return bool(self.serials)


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
    flips = {name: config.getboolean('CAMERA', name + '_flip', fallback=name != 'left')
             for name in CAMERA_NAMES}
    return CameraSettings(serials=serials, fps=fps, flips=flips,
                          depth_enabled=config.getboolean('CAMERA', 'depth_enabled', fallback=True))
