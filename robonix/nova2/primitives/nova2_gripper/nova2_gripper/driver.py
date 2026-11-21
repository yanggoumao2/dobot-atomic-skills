#!/usr/bin/env python3
"""Native Linux Robonix primitive exposing the Nova2 mock gripper."""

import threading

from robonix_api import Err, Ok
from nova2_provider import Nova2Primitive

import builtin_interfaces_pb2
import gripper_pb2
from gripper_mcp import (
    GetGripperInfo_Request,
    GetGripperInfo_Response,
    Grip_Request,
    Grip_Response,
    Move_Request,
    Move_Response,
    Open_Request,
    Open_Response,
    Stop_Request,
    Stop_Response,
)

from .model import MockGripper

gripper = Nova2Primitive(
    id="nova2_gripper",
    namespace="robonix/primitive/gripper",
)

_model_lock = threading.RLock()
_model: MockGripper | None = None
_state_stop = threading.Event()
_state_stop.set()
_STATE_INTERVAL_S = 0.1


def _current_model() -> MockGripper | None:
    with _model_lock:
        return _model


def _uninitialized(message_type):
    return message_type(ok=False, message="mock gripper is not initialized")


def _dispatch_motion(message_type, method_name: str, **kwargs):
    with _model_lock:
        if _model is None:
            return _uninitialized(message_type)
        if not _model.is_active:
            return message_type(ok=False, message="mock gripper is inactive")
        return message_type(**getattr(_model, method_name)(**kwargs))


@gripper.mcp("robonix/primitive/gripper/info")
def info(_request: GetGripperInfo_Request) -> GetGripperInfo_Response:
    """Describe this simulated gripper; width is synthetic and contact is unavailable."""
    model = _current_model()
    if model is None:
        return _uninitialized(GetGripperInfo_Response)
    return GetGripperInfo_Response(**model.info())


@gripper.mcp("robonix/primitive/gripper/open")
def open_jaws(request: Open_Request) -> Open_Response:
    """Open the synthetic jaws; the response confirms dispatch only."""
    return _dispatch_motion(
        Open_Response, "open", relative_speed=request.relative_speed
    )


@gripper.mcp("robonix/primitive/gripper/move")
def move_jaws(request: Move_Request) -> Move_Response:
    """Move to a synthetic absolute jaw opening width in metres."""
    return _dispatch_motion(
        Move_Response,
        "move",
        opening_width_m=request.opening_width_m,
        relative_speed=request.relative_speed,
    )


@gripper.mcp("robonix/primitive/gripper/stop")
def stop_jaws(_request: Stop_Request) -> Stop_Response:
    """Stop simulated jaw movement without opening or disabling the gripper."""
    return _dispatch_motion(Stop_Response, "stop")


@gripper.mcp("robonix/primitive/gripper/grip")
def grip_jaws(request: Grip_Request) -> Grip_Response:
    """Close the synthetic jaws without an expected object width."""
    return _dispatch_motion(
        Grip_Response,
        "grip",
        relative_effort=request.relative_effort,
        relative_speed=request.relative_speed,
        expected_object_width_valid=request.expected_object_width_valid,
        expected_object_width_m=request.expected_object_width_m,
    )


@gripper.grpc(
    "robonix/primitive/gripper/state",
    description=(
        "Stream synthetic gripper width and motion state. Contact remains UNKNOWN."
    ),
)
def stream_state(_request, context):
    """Stream timestamped synthetic jaw width without reporting contact."""
    with _model_lock:
        if _state_stop.is_set() or _model is None or not _model.is_active:
            return
        stream_stop = _state_stop
    while context.is_active() and not stream_stop.is_set():
        with _model_lock:
            if (
                _state_stop is not stream_stop
                or _model is None
                or not _model.is_active
            ):
                return
            state = _model.state()
        yield gripper_pb2.GripperState(
            opening_width_valid=state["opening_width_valid"],
            opening_width_m=state["opening_width_m"],
            grasp_state=state["grasp_state"],
            stamp=builtin_interfaces_pb2.Time(
                sec=state["stamp_sec"],
                nanosec=state["stamp_nanosec"],
            ),
            motion_state=state["motion_state"],
        )
        if stream_stop.wait(_STATE_INTERVAL_S):
            return


@gripper.on_init
def init(config: dict):
    """Select the left or right simulated jaw model for this instance."""
    global _model
    _state_stop.set()
    with _model_lock:
        if _model is not None:
            _model.deactivate()
        _model = None
    try:
        model = MockGripper(side=config.get("side", "left"), active=False)
    except (AttributeError, TypeError, ValueError) as exc:
        return Err(str(exc))
    with _model_lock:
        _model = model
    return Ok()


@gripper.on_activate
def activate():
    """Enable simulated operations and allow new state streams."""
    global _state_stop
    with _model_lock:
        if _model is None:
            return Err("mock gripper is not initialized")
        _model.activate()
        _state_stop.set()
        _state_stop = threading.Event()
    return Ok()


@gripper.on_deactivate
def deactivate():
    """Stop motion and state streams before the provider becomes inactive."""
    with _model_lock:
        if _model is not None:
            _model.deactivate()
        _state_stop.set()
    return Ok()


@gripper.on_shutdown
def shutdown():
    """Stop active state streams and release the mock model."""
    global _model
    with _model_lock:
        if _model is not None:
            _model.deactivate()
        _state_stop.set()
        _model = None
    return Ok()


if __name__ == "__main__":
    gripper.run()
