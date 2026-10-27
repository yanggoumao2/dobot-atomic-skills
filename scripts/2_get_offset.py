"""Read-only master mapping audit. Never writes motors or the active INI.

Stop teleoperation first. --reference-positioned includes candidate offsets
for review; it does NOT verify joint order, signs, or the physical posture.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from third_party.DynamixelSDK.python.src.dynamixel_sdk.port_handler import PortHandler
from third_party.DynamixelSDK.python.src.dynamixel_sdk.packet_handler import PacketHandler
from scripts.manipulate_utils import load_ini_data_hands

REPO = Path(__file__).resolve().parents[1]
INI_PATH = REPO / "scripts/dobot_config/dobot_settings.ini"


class ReadOnlyPositionPort(PortHandler):
    def __init__(self, name, ids):
        super().__init__(name)
        self.allowed_ids = set(ids)

    def writePort(self, packet):
        if (len(packet) != 14 or list(packet[:4]) != [255, 255, 253, 0]
                or packet[4] not in self.allowed_ids or packet[7] != 2
                or list(packet[8:12]) != [140, 0, 4, 0]):
            raise RuntimeError("Blocked packet: only position READ is permitted")
        return super().writePort(packet)


def capture(section, config, reference_positioned=False):
    ids = list(config.joint_ids)
    if len(ids) != 6 or len(set(ids)) != 6:
        raise ValueError("Expected six distinct arm joint IDs")
    signs = np.asarray(config.joint_signs, dtype=float)
    offsets = np.asarray(config.joint_offsets, dtype=float)
    reference = np.asarray(config.start_joints, dtype=float)
    if (signs.shape != (6,) or not np.isin(signs, [-1, 1]).all()
            or offsets.shape != (6,) or reference.shape != (6,)
            or not np.isfinite(offsets).all() or not np.isfinite(reference).all()):
        raise ValueError("Invalid calibration configuration")
    port = ReadOnlyPositionPort(config.port, ids)
    handler = PacketHandler(2.0)
    samples = []
    try:
        if not port.setBaudRate(config.baud_rate):
            raise RuntimeError(f"{section}: cannot open {config.port}")
        for sample_index in range(5):
            sample = []
            for servo_id in ids:
                value, comm, error = handler.read4ByteTxRx(port, servo_id, 140)
                if comm != 0 or error != 0:
                    raise RuntimeError(f"{section} ID{servo_id}: READ failed comm={comm} error={error}")
                if not 0 <= value <= 0xFFFFFFFF:
                    raise RuntimeError("Invalid raw position word")
                signed = value - 0x100000000 if value & 0x80000000 else value
                sample.append(signed * np.pi / 2048)
            samples.append(sample)
            if sample_index < 4:
                time.sleep(.1)
    finally:
        port.closePort()
    samples = np.asarray(samples)
    if np.max(np.ptp(samples, axis=0)) > .01:
        raise RuntimeError(f"{section}: master moved during sampling; hold stationary and repeat")
    raw = samples.mean(axis=0)
    mapped = (raw - offsets) * signs
    result = {
        "section": section, "port": config.port, "joint_ids": ids,
        "nova_side": "right" if section == "HAND_LEFT" else "left",
        "raw_rad": raw.tolist(), "mapped_deg": np.rad2deg(mapped).tolist(),
        "reference_deg": np.rad2deg(reference).tolist(),
        "offsets_rad": offsets.tolist(), "signs": signs.tolist(),
        "sample_span_rad": np.ptp(samples, axis=0).tolist(),
        "mapping_verified": False,
    }
    if reference_positioned:
        result["candidate_offsets_rad"] = (raw - reference * signs).tolist()
        result["candidate_notice"] = "Assumes confirmed reference AND signs; not proof of either. INI unchanged."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("HAND_LEFT", "HAND_RIGHT", "both"), default="both")
    parser.add_argument("--reference-positioned", action="store_true",
                        help="Include candidate offsets ONLY at a physically confirmed reference posture")
    parser.add_argument("--output", type=Path, help="New JSON output path (will not overwrite)")
    args = parser.parse_args()
    _, hands = load_ini_data_hands()
    before = hashlib.sha256(INI_PATH.read_bytes()).hexdigest()
    sections = ("HAND_LEFT", "HAND_RIGHT") if args.section == "both" else (args.section,)
    report = {"timestamp": datetime.now().astimezone().isoformat(), "ini_sha256": before,
              "read_only": True, "hands": [capture(s, hands[s], args.reference_positioned) for s in sections]}
    if hashlib.sha256(INI_PATH.read_bytes()).hexdigest() != before:
        raise RuntimeError("INI changed concurrently; discard this capture")
    output = args.output or REPO.parent / "manuals/equipment" / (
        "master-mapping-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {output}. No motor writes, INI edits or verification flags applied.")


if __name__ == "__main__":
    main()
