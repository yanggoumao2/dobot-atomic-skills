"""Optional read-only DOBOT CR/Nova V3 GetPose() collector (no motion/enable commands).

Protocol reference: https://github.com/Dobot-Arm/TCP-IP-Python-V3/blob/main/dobot_api.py
Controller compatibility and selected base/TCP must be checked on the real device.
"""
import argparse
import csv
import json
import math
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_pose(reply):
    match = re.fullmatch(r"\s*([+-]?\d+)\s*,\s*\{([^{}]*)\}\s*,\s*GetPose\(\)\s*;\s*", reply)
    if not match:
        raise ValueError("Unrecognized GetPose reply; check the controller protocol version")
    if int(match[1]) != 0:
        raise ValueError(f"Controller GetPose error {match[1]}")
    values = [float(v.strip()) for v in match[2].split(",")]
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise ValueError("Expected six finite pose values")
    return values


def read_pose(ip, port=29999, timeout=3.):
    with socket.create_connection((ip, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(b"GetPose()")
        message = bytearray()
        while b";" not in message:
            chunk = sock.recv(1024)
            if not chunk:
                raise ValueError("Controller closed connection before full reply")
            message.extend(chunk)
            if len(message) > 8192:
                raise ValueError("Oversized controller reply")
    raw = message.decode("utf-8").strip()
    return parse_pose(raw), raw


FIELDS = ["corner_id", "split", "x", "y", "z", "unit", "robot_id", "base_frame", "tcp_name",
          "host_receive_utc", "rx_raw", "ry_raw", "rz_raw", "getpose_reply"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ip", required=True)
    p.add_argument("--port", type=int, default=29999)
    p.add_argument("--robot-id", required=True)
    p.add_argument("--base-frame", required=True)
    p.add_argument("--tcp-name", required=True)
    p.add_argument("--confirm-base-tcp", action="store_true", help="Operator checked returned pose against pendant base/TCP values")
    p.add_argument("--unit", choices=["mm", "m"], required=True, help="Verified GetPose XYZ unit; rotation is stored raw and unused")
    p.add_argument("--corner-id", type=int, required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--out", required=True, help="Dedicated CSV; rows append, duplicate IDs rejected")
    args = p.parse_args()
    try:
        if not args.confirm_base_tcp:
            raise ValueError("Check GetPose XYZ against pendant base/TCP coordinates, then use --confirm-base-tcp")
        if not 0 <= args.corner_id < 88:
            raise ValueError("corner-id must be 0..87")
        path = Path(args.out)
        existing = []
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != FIELDS:
                    raise ValueError("Use a new dedicated collector CSV (not the manual template)")
                existing = list(reader)
            for row in existing:
                if int(row["corner_id"]) == args.corner_id:
                    raise ValueError("This corner ID is already recorded")
                for key in ("unit", "robot_id", "base_frame", "tcp_name"):
                    if row[key] != getattr(args, key):
                        raise ValueError(f"Cannot mix {key} in one CSV")
        pose, raw = read_pose(args.ip, args.port)
        record = dict(zip(FIELDS, [args.corner_id, args.split, *pose[:3], args.unit,
                                  args.robot_id, args.base_frame, args.tcp_name,
                                  datetime.now(timezone.utc).isoformat(), *pose[3:], raw]))
        path.parent.mkdir(parents=True, exist_ok=True)
        header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if header:
                writer.writeheader()
            writer.writerow(record)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
