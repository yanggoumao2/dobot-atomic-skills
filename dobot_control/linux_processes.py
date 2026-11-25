"""Linux /proc guards for local operator tools; never stop a process."""
import json
import os
from pathlib import Path
import socket
import time

TOKENS = ("launch_nodes.py", "experiments.launch_nodes", "run_control.py", "experiments.run_control", "5_camera_read", "gripper_panel.py", "tools.gripper_panel", "act_real_test")

def control_processes(proc_root=Path("/proc")):
    found = []
    if not proc_root.is_dir():
        raise RuntimeError("Linux /proc unavailable; cannot confirm device ownership")
    for directory in proc_root.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            args = (directory / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise RuntimeError("Cannot inspect /proc; check process visibility") from exc
        if args and "python" in Path(args[0]).name and any(any(token in arg for token in TOKENS) for arg in args[1:]):
            found.append(dict(pid=int(directory.name), argv=args))
    return found

def assert_stopped(all_control=False):
    active = control_processes()
    if not all_control:
        active = [p for p in active if any(any(t in a for t in ("run_control", "5_camera_read", "act_real_test")) for a in p["argv"])]
    if active:
        raise RuntimeError(f"Exit control processes first: {active}")

def archive_lock(path):
    assert_stopped(all_control=True)
    path = Path(path)
    if not path.exists():
        return "No device lock"
    record = json.loads((path / "owner.json").read_text(encoding="utf-8"))
    pid = record.get("pid")
    if record.get("host") != socket.gethostname() or not isinstance(pid, int) or pid <= 0:
        raise RuntimeError("Lock owner is not a verified local process")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError("Lock owner is still alive")
    target = path.with_name(path.name + f".closed-{time.time_ns()}")
    path.rename(target)
    return str(target)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("list", "no-b", "no-control", "archive"))
    parser.add_argument("--lock")
    args = parser.parse_args()
    if args.action == "list": print(control_processes())
    elif args.action == "archive":
        if not args.lock: parser.error("--lock required")
        print(archive_lock(args.lock))
    else: assert_stopped(args.action == "no-control")
