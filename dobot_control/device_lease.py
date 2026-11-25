"""A cooperative rig owner shared by the gateway, atomic CLI and ZMQ server."""

import json
import os
import socket
from pathlib import Path
from uuid import uuid4


class DeviceLease:
    def __init__(self, path=None, purpose="gateway"):
        self.path = Path(path or os.environ.get(
            "DOBOT_DEVICE_LOCK_PATH",
            str(Path(__file__).resolve().parents[2] / ".local/runtime/device-owner"),
        ))
        self.owner = uuid4().hex
        self.acquired = False
        self.purpose = purpose
        self.scene_marker = self.path.with_name(self.path.name + ".scene-invalid")
        self.scene_stamp = self.path.with_name(self.path.name + ".scene.json")

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
        except FileExistsError as error:
            raise RuntimeError("Dobot rig is owned or quarantined; inspect the device-owner directory") from error
        self.acquired = True
        try:
            (self.path / "owner.json").write_text(json.dumps({
                "owner": self.owner, "pid": os.getpid(), "host": socket.gethostname(),
                "purpose": self.purpose,
            }), encoding="utf-8")
            if self.purpose == "legacy-control":
                self.scene_marker.write_text("Legacy control may have changed objects; onsite reset required.\n", encoding="utf-8")
        except Exception:
            # Retain the directory if ownership could not be written safely.
            raise
        return self

    def release(self):
        if not self.acquired:
            return
        record = self.path / "owner.json"
        if json.loads(record.read_text(encoding="utf-8"))["owner"] != self.owner:
            raise RuntimeError("Device ownership changed unexpectedly")
        record.unlink()
        self.path.rmdir()
        self.acquired = False

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()
