"""One configuration source, selected before starting a process."""
import configparser
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def config_path():
    override = os.environ.get("DOBOT_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform.startswith("linux"):
        return ROOT / "config/linux.local.ini"
    return ROOT / "scripts/dobot_config/dobot_settings.ini"

def read_config():
    config = configparser.ConfigParser()
    with config_path().open(encoding="utf-8") as f:
        config.read_file(f)
    return config

def robot_ip(side):
    if side not in ("left", "right"):
        raise ValueError(f"Unknown side: {side}")
    config = read_config()
    # Legacy Windows profiles predate ROBOTS.
    fallback = {"right": "192.168.5.1", "left": "192.168.5.2"}[side]
    value = config.get("ROBOTS", side + "_ip", fallback=fallback).strip()
    if not value:
        raise ValueError(f"Fill [ROBOTS] {side}_ip in {config_path()}")
    return value

def validate_hardware_config():
    import ipaddress
    c = read_config()
    ips = [str(ipaddress.IPv4Address(robot_ip(s))) for s in ("right", "left")]
    if len(set(ips)) != 2:
        raise ValueError("Robot IPs must be distinct")
    ports = [c.get(s, "port").strip() for s in ("HAND_LEFT", "HAND_RIGHT", "GRIPPER_LEFT", "GRIPPER_RIGHT")]
    if any(not p for p in ports) or len(set(os.path.realpath(p) for p in ports)) != 4:
        raise ValueError("Fill four distinct master/gripper serial ports")
    if sys.platform.startswith("linux") and any(not p.startswith("/dev/") for p in ports):
        raise ValueError("Linux serial ports must be /dev/ paths; prefer /dev/serial/by-id")
    for s in ("HAND_LEFT", "HAND_RIGHT"):
        if not c.getboolean(s, "absolute_mapping_verified", fallback=False):
            raise ValueError(f"Verify joint signs/offsets/routing before enabling {s}")
    from dobot_control.cameras.settings import load_camera_settings
    load_camera_settings(config_path())
    return c

def hardware_metadata():
    c = read_config()
    ips = [robot_ip(s) for s in ("right", "left")]
    return dict(channel_order=[label for ip in ips for label in (ip + " J1..J6", ip + " gripper")],
                physical_side_note="vector order: physical RIGHT, physical LEFT",
                gripper_ports={robot_ip(s): c.get("GRIPPER_" + s.upper(), "port") + "/ID" + c.get("GRIPPER_" + s.upper(), "id") for s in ("right", "left")})
