"""Single gripper recovery through the A service's existing serial connection."""
import os
import subprocess
import time

from dobot_control.layout import physical_nova_ip


def require_teleop_stopped():
    # B does not claim an ACT session. Check it at the A service as well as
    # rejecting inference session ownership in the RPC dispatcher.
    if os.name != 'nt':
        from dobot_control.linux_processes import assert_stopped
        assert_stopped()
        return
    script = r"""
$ErrorActionPreference = 'Stop'
$running = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'run_control\.py'
})
if ($running.Count) { exit 2 }
"""
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
                            capture_output=True, timeout=8,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode == 2:
        raise RuntimeError('请先停止录制、关闭跟随，并 Ctrl+C 退出终端 B；终端 A 保持运行')
    if result.returncode:
        raise RuntimeError('无法确认终端 B 已退出，请检查本机进程查询权限')


def recover_gripper(bimanual, side, action):
    if side not in ('left', 'right') or action not in ('open', 'close'):
        raise ValueError('请选择实体左/右夹爪和全开/全闭')
    require_teleop_stopped()
    ip = physical_nova_ip(side)
    robot = next((robot for robot in (bimanual._robot_l, bimanual._robot_r)
                  if robot.robot_ip == ip), None)
    if robot is None or not robot._use_gripper:
        raise RuntimeError(f'{side} 夹爪未连接，请检查终端 A 初始化结果')
    target = 255 if action == 'open' else 0
    # The gripper's move/read methods already share the driver's RLock with
    # ACT feedback polling. Never reopen or disconnect A's serial connection.
    robot.gripper.move(target, 10, 1)
    robot._last_gripper_cmd = target
    deadline = time.monotonic() + 10
    while True:
        position = robot.gripper.get_current_position()
        if abs(position - target) <= 5:
            return dict(side=side, action=action, position=position, target=target, speed=10)
        if time.monotonic() >= deadline:
            raise TimeoutError(f'夹爪 10 秒内未到位，读回 {position}，目标 {target}，请检查实机')
        time.sleep(0.1)
