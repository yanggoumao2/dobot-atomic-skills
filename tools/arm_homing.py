"""Single physical arm homing; imports and construction never open a device."""
from contextlib import ExitStack
import math
import logging
import re
import time

from dobot_control.layout import physical_nova_home_degrees, physical_nova_ip, TELEOP_NOVA_SIDES

SPEED_RAD_S = 0.15
PERIOD_S = 0.05
TOLERANCE_DEG = 0.1


def close_connection(connection):
    try:
        connection.close()
    except Exception:
        logging.exception("Closing arm connection failed")


class StopUnconfirmed(RuntimeError):
    """Keep the shared lease when a sent motion could not be stopped."""


def values(reply, count):
    if not isinstance(reply, str) or not re.match(r"^\s*0\s*,", reply):
        raise RuntimeError(f"控制柜命令未成功：{reply!r}")
    match = re.search(r"\{([^{}]*)\}", reply)
    try:
        result = [float(x.strip()) for x in match.group(1).split(",") if x.strip()]
    except (ValueError, AttributeError) as exc:
        raise RuntimeError(f"控制柜回复无法解析：{reply!r}") from exc
    if len(result) != count or not all(math.isfinite(x) for x in result):
        raise RuntimeError(f"控制柜回复数据无效：{reply!r}")
    return result


def connect_dashboard(ip):
    from dobot_control.robots.dobot_api import DobotApiDashboard
    return DobotApiDashboard(ip, 29999)


def connect_motion(ip):
    from dobot_control.robots.dobot_api import DobotApiMove
    return DobotApiMove(ip, 30003)


def validate_path(side, current_deg, target_deg):
    import numpy as np
    from experiments.run_control import validate_absolute_alignment
    slot = TELEOP_NOVA_SIDES.index(side)
    current = np.zeros(14)
    current[slot*7:slot*7+6] = np.deg2rad(current_deg)
    target = current.copy()
    target[slot*7:slot*7+6] = np.deg2rad(target_deg)
    flags = [int(i == slot) for i in range(2)]
    accepted, detail = validate_absolute_alignment(current, target, "Nova 2", flags)
    if not accepted:
        raise RuntimeError(f"归位路径未通过工作区/关节保护，请现场调整姿态。拒绝点：{detail}")


class ArmHoming:
    def __init__(self, dashboard_factory=connect_dashboard, motion_factory=connect_motion,
                 path_check=validate_path, clock=time.monotonic, sleep=time.sleep):
        self.dashboard_factory, self.motion_factory = dashboard_factory, motion_factory
        self.path_check, self.clock, self.sleep = path_check, clock, sleep

    def run(self, side, cancel):
        ip = physical_nova_ip(side)
        target = list(physical_nova_home_degrees(side))
        sent = False
        with ExitStack() as resources:
            dashboard = self.dashboard_factory(ip)
            resources.callback(close_connection, dashboard)
            try:
                mode = values(dashboard.RobotMode(), 1)[0]
                if mode not in (4, 5):
                    raise RuntimeError(f"控制柜状态 {mode:g} 不允许归位，请先处理报警/拖拽/运行状态")
                current = values(dashboard.GetAngle(), 6)
                self.path_check(side, current, target)
                if cancel.is_set():
                    raise RuntimeError("归位已取消，未发送运动")
                if max(abs(a-b) for a,b in zip(current,target)) <= TOLERANCE_DEG:
                    return current
                if mode == 4:
                    values(dashboard.EnableRobot(), 0)
                    deadline = self.clock() + 3
                    while values(dashboard.RobotMode(), 1)[0] != 5:
                        if cancel.is_set() or self.clock() > deadline:
                            raise RuntimeError("机器人未在 3 秒内完成使能")
                        self.sleep(.05)
                motion = self.motion_factory(ip)
                resources.callback(close_connection, motion)
                max_delta = max(abs(a-b) for a,b in zip(current,target))
                steps = max(1, math.ceil(math.radians(max_delta) / (SPEED_RAD_S*PERIOD_S)))
                start, last_command = current[:], current[:]
                for index in range(1, steps+1):
                    if cancel.is_set():
                        raise RuntimeError("归位已停止")
                    if values(dashboard.RobotMode(), 1)[0] not in (5, 7):
                        raise RuntimeError("控制柜状态异常，归位中止")
                    actual = values(dashboard.GetAngle(), 6)
                    if max(abs(a-b) for a,b in zip(actual,last_command)) > 5:
                        raise RuntimeError("实际关节角落后指令超过 5°，归位中止")
                    self.path_check(side, actual, actual)
                    command = [a+(b-a)*index/steps for a,b in zip(start,target)]
                    sent = True  # a failed transport can have partially sent the command
                    motion.ServoJ(*command, t=PERIOD_S)
                    last_command = command
                    self.sleep(PERIOD_S)
                deadline = self.clock() + 10
                settled = 0
                while self.clock() < deadline:
                    if cancel.is_set():
                        raise RuntimeError("归位已停止")
                    mode = values(dashboard.RobotMode(), 1)[0]
                    if mode not in (5, 7):
                        raise RuntimeError("控制柜状态异常，无法确认归位")
                    actual = values(dashboard.GetAngle(), 6)
                    self.path_check(side, actual, actual)
                    at_target = mode == 5 and max(abs(a-b) for a,b in zip(actual,target)) <= TOLERANCE_DEG
                    settled = settled+1 if at_target else 0
                    if settled >= 3:
                        return actual
                    self.sleep(.05)
                raise TimeoutError("归位指令发送后 10 秒内未确认到位")
            except Exception as exc:
                if sent:
                    try:
                        values(dashboard.ResetRobot(), 0)
                        if values(dashboard.RobotMode(), 1)[0] not in (4, 5):
                            raise RuntimeError("停止后状态未静止")
                    except Exception as stop_error:
                        raise StopUnconfirmed(
                            f"{exc}；停止未确认，请现场检查，设备锁已保留。停止错误：{stop_error}") from exc
                raise
