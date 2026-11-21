"""Exercise both simulated arm/gripper pairs against a private Robonix Atlas."""

import argparse
import json
import math
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import grpc
import yaml
from google.protobuf.empty_pb2 import Empty
from robonix_api import ATLAS
from robonix_api.codegen import ensure_proto_gen


ROOT = Path(__file__).resolve().parents[1]
DRIVER = "robonix/lifecycle/driver"
ARM = "robonix/primitive/arm/"
GRIPPER = "robonix/primitive/gripper/"


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def call(provider, contract, request, transport="grpc"):
    with ATLAS.connect_channel(consumer_id="nova2_test", provider_id=provider,
                               contract_id=contract, transport=transport) as channel:
        return channel.call(request, timeout=5)


def state(provider, contract, message_type):
    # Cancel after one frame. Both state contracts are open-ended streams.
    with ATLAS.connect_capability(consumer_id="nova2_test", provider_id=provider,
                                  contract_id=contract, transport="grpc") as channel:
        with grpc.insecure_channel(channel.endpoint, options=[("grpc.enable_http_proxy", 0)]) as wire:
            route = "/" + channel.params.service_name + "/" + channel.params.method.split("/")[-1]
            read = wire.unary_stream(route, request_serializer=Empty.SerializeToString,
                                     response_deserializer=message_type.FromString)
            frames = read(Empty(), timeout=5)
            try:
                return next(frames)
            finally:
                frames.cancel()


def exercise():
    from gripper_pb2 import GripperState
    from sensor_msgs_pb2 import JointState

    def joints(side):
        return state(side + "_arm", ARM + "joint_states", JointState)

    def jaws(side):
        sample = state(side + "_gripper", GRIPPER + "state", GripperState)
        check(sample.opening_width_valid, f"{side} opening width invalid")
        check(sample.grasp_state == 0, f"{side} reported contact in an empty gripper model")
        return sample.opening_width_m

    left, right = joints("left"), joints("right")
    check(list(left.name) == [f"left_joint{i}" for i in range(1, 7)], "left arm joint mapping")
    check(list(right.name) == [f"right_joint{i}" for i in range(1, 7)], "right arm joint mapping")
    target = list(left.position)
    target[0] += 0.05
    call("left_arm", ARM + "joint_command", [{"name": list(left.name), "position": target}])
    left_feedback = joints("left")
    check(list(left_feedback.position) == target, "left arm feedback missed command")
    check(list(joints("right").position) == list(right.position), "cross-arm state change")

    left_width = jaws("left")
    info = call("right_gripper", GRIPPER + "info", {}, "mcp")
    check(info.ok and info.opening_width_range_valid, "missing mock jaw travel")
    for op, args in (
        ("grip", {"relative_effort": 0.0, "relative_speed": 0.0,
                  "expected_object_width_valid": False, "expected_object_width_m": 0.0}),
        ("stop", {}),
        ("open", {"relative_speed": 0.0}),
    ):
        reply = call("right_gripper", GRIPPER + op, args, "mcp")
        check(reply.ok, f"{op}: {reply.message}")
    goal = info.max_opening_width_m / 2
    result = call("right_gripper", GRIPPER + "move",
                  {"opening_width_m": goal, "relative_speed": 0.0}, "mcp")
    check(result.ok, result.message)
    deadline = time.monotonic() + 10
    right_width = jaws("right")
    while not math.isclose(right_width, goal, abs_tol=1e-6):
        check(time.monotonic() < deadline, "right gripper missed target")
        time.sleep(0.05)
        right_width = jaws("right")
    check(jaws("left") == left_width, "cross-gripper state change")
    return {"left_joint1_rad": left_feedback.position[0], "right_opening_m": right_width}


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run(atlas_bin, capabilities):
    manifest = yaml.safe_load((ROOT / "robonix_manifest.yaml").read_text(encoding="utf-8"))
    entries = manifest["primitive"]
    for entry in entries:
        if ensure_proto_gen(ROOT / entry["path"]) is None:
            raise RuntimeError("Run rbnx build for the deployment before this test")

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        endpoint = f"127.0.0.1:{reservation.getsockname()[1]}"
    os.environ.update(ROBONIX_ATLAS=endpoint, ROBONIX_PROVIDER_BIND_HOST="127.0.0.1",
                      ROBONIX_ADVERTISE_HOST="127.0.0.1", NO_PROXY="localhost,127.0.0.1",
                      no_proxy="localhost,127.0.0.1")
    with tempfile.TemporaryDirectory() as logs, ExitStack() as stack:
        def start(command, name, cwd, env):
            output = stack.enter_context(open(Path(logs) / (name + ".log"), "w+", encoding="utf-8"))
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
            stack.callback(stop_process, process)
            return process, output

        env = {**os.environ, "SCRIBE_LOG_DIR": logs}
        atlas, _ = start([atlas_bin, "--listen", endpoint, "--capabilities", str(capabilities)],
                         "atlas", logs, env)
        with grpc.insecure_channel(endpoint, options=[("grpc.enable_http_proxy", 0)]) as wire:
            grpc.channel_ready_future(wire).result(timeout=10)
        check(atlas.poll() is None and ATLAS.query_contract(DRIVER), "private Atlas did not load contracts")
        for entry in entries:
            name, package = entry["name"], ROOT / entry["path"]
            generated = package / "rbnx-build/codegen"
            package_manifest = yaml.safe_load((package / "package_manifest.yaml").read_text(encoding="utf-8"))
            child_env = {**env, "RBNX_INSTANCE_NAME": name, "ROBONIX_DRIVER_CONTRACT_ID": DRIVER,
                         "RBNX_PACKAGE_ROOT": str(package)}
            if os.name == "nt":
                command = [sys.executable, "-m", package.name + ".driver"]
                child_env["PYTHONPATH"] = os.pathsep.join(map(str, (
                    ROOT, generated / "proto_gen", generated / "robonix_mcp_types",
                )))
            else:
                command = shlex.split(package_manifest["start"])
            child, output = start(command, name, package, child_env)
            expected = {cap["name"] for cap in package_manifest["capabilities"]} | {DRIVER}
            deadline = time.monotonic() + 30
            while True:
                providers = ATLAS.query_primitives(id=name)
                if providers and {c.contract_id for c in providers[0].capabilities} == expected:
                    break
                if child.poll() is not None or time.monotonic() >= deadline:
                    output.seek(0)
                    raise RuntimeError(f"{name} registration failed\n{output.read()}")
                time.sleep(0.1)
            for command in (0, 1):
                reply = call(name, DRIVER, {"command": command, "config_json": json.dumps(entry["config"])})
                check(reply.ok, f"{name} lifecycle {command}: {reply.error}")
        result = exercise()
        for entry in reversed(entries):
            for command in (2, 3):
                reply = call(entry["name"], DRIVER, {"command": command, "config_json": "{}"})
                check(reply.ok, reply.error)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-bin", default="robonix-atlas")
    parser.add_argument("--capabilities", type=Path, required=True, help="Robonix source capabilities directory")
    args = parser.parse_args()
    print(json.dumps(run(args.atlas_bin, args.capabilities.resolve()), indent=2))
