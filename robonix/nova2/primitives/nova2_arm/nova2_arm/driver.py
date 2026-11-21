"""Robonix mock arm primitive backed by the pure Python model."""

from __future__ import annotations

import logging
import threading
import time
from typing import NoReturn

import grpc
from google.protobuf.empty_pb2 import Empty

import sensor_msgs_pb2  # type: ignore
from nova2_provider import Nova2Primitive
from robonix_api import Err, Ok
from nova2_arm.model import ArmModel, ArmSettings, CommandValidationError, JointStateFrame

primitive = Nova2Primitive(id="nova2_arm", namespace="robonix/primitive/arm")
log = logging.getLogger("nova2_arm")
logging.basicConfig(level=logging.INFO, format="[nova2_arm] %(levelname)s %(message)s")

_guard = threading.RLock()
_activation_stop = threading.Event()
_activation_stop.set()
_activation_generation = 0
_model: ArmModel | None = None
_settings: ArmSettings | None = None
_active = False


def _abort(context, status: grpc.StatusCode, message: str) -> NoReturn:
    context.abort(status, message)
    raise RuntimeError(message)


def _joint_state_message(frame: JointStateFrame):
    """Convert a pure model snapshot into the generated standard JointState."""
    message = sensor_msgs_pb2.JointState()
    now_ns = time.time_ns()
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(now_ns, 1_000_000_000)
    message.name.extend(frame.name)
    message.position.extend(frame.position)
    message.velocity.extend(frame.velocity)
    message.effort.extend(frame.effort)
    return message


@primitive.grpc("robonix/primitive/arm/joint_command")
def joint_command(commands, context):
    """Accept complete six-joint position targets from a client stream.

    Every frame is validated before any of its positions are applied. If a
    later frame is invalid, earlier accepted frames remain applied.
    """
    global _activation_generation
    with _guard:
        if not _active or _model is None:
            _abort(
                context,
                grpc.StatusCode.FAILED_PRECONDITION,
                "arm is not ACTIVE",
            )
        generation = _activation_generation

    accepted = 0
    for command in commands:
        if not context.is_active():
            return Empty()
        with _guard:
            if (
                not _active
                or _model is None
                or generation != _activation_generation
            ):
                _abort(
                    context,
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "arm lifecycle changed during joint_command stream",
                )
            try:
                _model.apply_joint_state(command)
            except CommandValidationError as error:
                _abort(context, grpc.StatusCode.INVALID_ARGUMENT, str(error))
        accepted += 1

    with _guard:
        if not _active or generation != _activation_generation:
            _abort(
                context,
                grpc.StatusCode.FAILED_PRECONDITION,
                "arm lifecycle changed during joint_command stream",
            )

    if accepted == 0:
        _abort(
            context,
            grpc.StatusCode.INVALID_ARGUMENT,
            "joint_command stream must contain at least one JointState",
        )
    return Empty()


@primitive.grpc("robonix/primitive/arm/joint_states")
def joint_states(_request, context):
    """Stream the six modeled positions while this arm instance is ACTIVE."""
    with _guard:
        if not _active or _model is None:
            _abort(
                context,
                grpc.StatusCode.FAILED_PRECONDITION,
                "arm is not ACTIVE",
            )
        generation = _activation_generation
        stop_event = _activation_stop

    while context.is_active() and not stop_event.is_set():
        with _guard:
            if (
                not _active
                or _model is None
                or _settings is None
                or generation != _activation_generation
            ):
                return
            frame = _model.feedback()
            period_s = 1.0 / _settings.feedback_rate_hz

        if stop_event.is_set():
            return
        yield _joint_state_message(frame)
        if stop_event.wait(period_s):
            return


@primitive.on_init
def init(config: dict):
    """Validate this instance and seed feedback from the repository home pose."""
    global _model, _settings, _active, _activation_stop, _activation_generation
    try:
        settings = ArmSettings.from_mapping(config)
        model = ArmModel(settings.side)
    except (TypeError, ValueError, RuntimeError, OSError) as error:
        return Err(str(error))

    with _guard:
        _activation_stop.set()
        _activation_stop = threading.Event()
        _activation_stop.set()
        _activation_generation += 1
        _model = model
        _settings = settings
        _active = False

    log.info(
        "initialized side=%s joints=%s rate=%.1f Hz",
        settings.side,
        ", ".join(model.name),
        settings.feedback_rate_hz,
    )
    return Ok()


@primitive.on_activate
def activate():
    """Allow commands and state streams after lifecycle activation."""
    global _active, _activation_stop, _activation_generation
    with _guard:
        if _model is None or _settings is None:
            return Err("arm must be initialized before activation")
        _activation_stop.set()
        _activation_stop = threading.Event()
        _activation_generation += 1
        _active = True
    return Ok()


@primitive.on_deactivate
def deactivate():
    """Stop accepting commands and wake streams from this activation."""
    global _active, _activation_generation
    with _guard:
        _active = False
        _activation_generation += 1
        _activation_stop.set()
    return Ok()


@primitive.on_shutdown
def shutdown():
    """Stop feedback streams before the process exits."""
    global _active, _activation_generation
    with _guard:
        _active = False
        _activation_generation += 1
        _activation_stop.set()
    return Ok()


def main() -> int:
    primitive.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
