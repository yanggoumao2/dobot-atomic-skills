"""Offline environment/configuration checks; never constructs a robot driver.

--inventory additionally enumerates USB serial ports and RealSense devices.
It does not open a serial port, start a camera stream, reset a device, connect
to a robot IP, or send motion/enable commands. This is not a hardware acceptance.
"""
import argparse
import configparser
import importlib
import importlib.metadata
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", action="store_true")
    args = parser.parse_args()
    failures = []
    pending = []
    print(f"Python: {sys.version.split()[0]} | {sys.executable}")
    print(f"Project: {ROOT}")
    modules = {
        "numpy": "numpy", "cv2": "opencv-python", "tyro": "tyro",
        "serial": "pyserial", "zmq": "pyzmq", "requests": "requests",
        "pyrealsense2": "pyrealsense2",
    }
    for module, package in modules.items():
        try:
            importlib.import_module(module)
            print(f"[OK] {package}=={importlib.metadata.version(package)}")
        except Exception as exc:
            failures.append(f"{module}: {type(exc).__name__}: {exc}")
    for module in ("tkinter", "experiments.launch_nodes", "experiments.run_control"):
        try:
            importlib.import_module(module)
            print(f"[OK] import {module} (no main/constructor called)")
        except Exception as exc:
            failures.append(f"{module}: {type(exc).__name__}: {exc}")

    config = configparser.ConfigParser()
    config.read(ROOT / "scripts/dobot_config/dobot_settings.ini", encoding="utf-8")
    ports = []
    cameras = []
    try:
        for section in ("HAND_LEFT", "HAND_RIGHT", "GRIPPER_LEFT", "GRIPPER_RIGHT"):
            port = config.get(section, "port")
            ports.append(port.upper())
            print(f"[CONFIG] {section}.port = {port}")
            if sys.platform == "win32" and not re.fullmatch(r"COM[1-9]\d*", port, re.I):
                pending.append(f"{section}: replace {port!r} with the measured Windows COM port")
        if len(set(ports)) != 4:
            pending.append("The four configured serial ports are not distinct")
        from dobot_control.cameras.settings import load_camera_settings
        camera_settings = load_camera_settings()
        cameras = list(camera_settings.serials.values())
        print(f"[CONFIG] enabled cameras: {camera_settings.serials}, fps={camera_settings.fps}")
        print(f"[CONFIG] three-view recording enabled: {camera_settings.recording_enabled}")
    except (configparser.Error, ValueError) as exc:
        failures.append(f"INI: {exc}")

    if args.inventory:
        try:
            from serial.tools.list_ports import comports
            available = list(comports())
            print("[INVENTORY] serial ports (enumeration only):")
            for port in available:
                print(f"  {port.device}: {port.description} | {port.hwid}")
            if not available:
                print("  none")
            present = {port.device.upper() for port in available}
            for port in ports:
                if port not in present:
                    pending.append(f"Configured port not enumerated: {port}")
        except Exception as exc:
            failures.append(f"Serial inventory: {exc}")
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            present_cameras = [
                device.get_info(rs.camera_info.serial_number)
                for device in ctx.query_devices()
            ]
            print(f"[INVENTORY] RealSense serial numbers: {present_cameras}")
            for serial in cameras:
                if serial not in present_cameras:
                    pending.append(f"Configured camera not enumerated: {serial}")
        except Exception as exc:
            failures.append(f"RealSense inventory: {exc}")

    for message in failures:
        print(f"[FAIL] {message}")
    for message in pending:
        print(f"[PENDING] {message}")
    print("No robot connection or hardware motion was attempted.")
    print("Exit 0 = software/static checks pass; 1 = environment failure; 2 = configuration pending.")
    print("Even exit 0 does not verify wiring, calibration, firmware, camera streaming or safe motion.")
    return 1 if failures else 2 if pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
