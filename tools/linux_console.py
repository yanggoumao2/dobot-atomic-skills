"""Bash equivalents of the operator console actions."""
import os
import shlex
import shutil
import subprocess
from dobot_control.config import config_path

def commands(project, episode, root, python, lock):
    q = lambda value: shlex.quote(str(value))
    py = q(python) + " -X utf8 -B "
    guard = py + "-m dobot_control.linux_processes "
    base = ("set -e\ncd " + q(root) + "\nexport DOBOT_CONFIG=" + q(config_path()) +
            "\nexport DOBOT_DEVICE_LOCK_PATH=" + q(lock) + "\nexport DOBOT_ACT_RECORDING=1\n")
    data = root.parent / "datasets" / project / "collect_data"
    definitions = [
        ("processes", "查看控制进程", guard + "list", False),
        ("clear_lock", "归档旧锁", guard + "archive --lock " + q(lock), True),
        ("start_a", "启动终端 A", guard + "no-control\n" + py + "-m experiments.launch_nodes", True),
        ("start_b", "启动终端 B", guard + "no-b\nunset DOBOT_ALLOW_SAFE_POSE_INIT DOBOT_RECORD_QUEUE_SIZE\n" + py + "-m experiments.run_control --show-img --project-name " + q(project), True),
        ("home_service", "通过 A 双臂归位", guard + "no-b\n" + py + "-m tools.home_service", True),
        ("camera", "相机预览", guard + "no-b\n" + py + "-m scripts.5_camera_read", False),
        ("convert", "导出 ACT 数据集", guard + "no-b\n" + py + "-m dobot_control.act_dataset convert " + q(data) + " --output " + q(data.parent / "act_v1") + " --fps 15", False),
    ]
    if episode:
        ep = data / episode
        definitions += [("check", "检查 episode", py + "-m dobot_control.act_dataset check " + q(ep), False),
                        ("preview", "生成回看视频", py + "-m dobot_control.act_dataset preview " + q(ep) + " --output " + q(ep / "preview.mp4"), False)]
    descriptions = {
        "processes": "查看控制进程、PID 和命令行。",
        "clear_lock": "先退出 A/B 与其他控制程序并确认设备停止；仅归档本机已退出进程的旧锁。",
        "start_a": "会使能两臂并让两只夹爪闭合再张开。清空活动范围，等待 6001 服务就绪后再开 B。",
        "start_b": "启动遥操与示教采集；默认检查初始姿态并开启配置的相机预览。",
        "home_service": "保持 A 运行并退出 B/ACT 执行程序。确认两臂回程路径空闲，夹爪保持开度。",
        "camera": "先退出 B 与其他相机预览；Q / Esc 退出。",
        "convert": "先退出 B；仅导出质量合格且标为 success 的记录，act_v1 必须为新目录。",
        "check": "检查 episode 完整性、正常收尾、反馈和时间戳。",
        "preview": "生成 preview.mp4，输出文件必须不存在。",
    }
    return [dict(id=i, title=t, description=descriptions[i], confirm=c, code=base + s) for i,t,s,c in definitions]

def launch(command, root):
    if command["id"] in ("start_a", "start_b", "camera"):
        terminal = shutil.which("x-terminal-emulator")
        if not terminal or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("需要 Linux 桌面终端；请在现场终端粘贴所显示的 Bash 命令")
        subprocess.Popen([terminal, "-e", "bash", "-c", command["code"] + '\nstatus=$?; echo "Exit: $status"; read -r -p "Press Enter to close"'], cwd=root)
        return "已打开 Linux 终端，请在窗口查看结果并用 Ctrl+C 正常退出"
    result = subprocess.run(["bash", "-c", command["code"]], cwd=root, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout or f"Exit {result.returncode}")
    return result.stdout or "完成"
