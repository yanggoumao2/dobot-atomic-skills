"""Local gripper and single-arm home panel. No device access on window open."""
from __future__ import annotations

import configparser
from dataclasses import dataclass
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dobot_control.config import config_path as get_config_path
INI_PATH = get_config_path()
SPEED = 10
TARGETS = {"open": 255, "close": 0}  # Confirmed by the operator.
SECTIONS = {"left": "GRIPPER_LEFT", "right": "GRIPPER_RIGHT"}
SIDE_NAMES = {"left": "左夹爪", "right": "右夹爪"}


@dataclass(frozen=True)
class Endpoint:
    port: str
    servo_id: int
    positions: tuple[int, int]


def load_endpoints(path=INI_PATH):
    config = configparser.ConfigParser()
    with open(path, encoding="utf-8") as handle:
        config.read_file(handle)
    endpoints = {}
    for side, section in SECTIONS.items():
        positions = tuple(int(x.strip()) for x in config.get(section, "pos").split(","))
        if len(positions) != 2 or positions[0] >= positions[1]:
            raise ValueError(f"[{section}] pos 必须是递增的两个行程值")
        port = config.get(section, "port").strip()
        if not port:
            raise ValueError(f"[{section}] 缺少串口")
        endpoints[side] = Endpoint(port, config.getint(section, "id"), positions)
    if endpoints["left"].port.upper() == endpoints["right"].port.upper():
        raise ValueError("两只夹爪不能配置为同一串口")
    return endpoints


def make_gripper(endpoint):
    from dobot_control.gripper.dobot_gripper import DobotGripper
    return DobotGripper(port=endpoint.port, id_name=endpoint.servo_id,
                        servo_pos=list(endpoint.positions))


def make_lease():
    from dobot_control.device_lease import DeviceLease
    path = os.environ.get("DOBOT_DEVICE_LOCK_PATH") or str(
        ROOT.parent / ".local/runtime/device-owner")
    return DeviceLease(path=path, purpose="legacy-control")


class GripperControl:
    def __init__(self, endpoints, gripper_factory=make_gripper, lease_factory=make_lease,
                 clock=time.monotonic, sleep=time.sleep, homing_factory=None):
        self.endpoints = endpoints
        self._factory, self._lease = gripper_factory, lease_factory
        self._clock, self._sleep = clock, sleep
        self._busy = threading.Lock()
        self._homing_factory = homing_factory
        self._cancel = threading.Event()

    def cancel_home(self):
        self._cancel.set()

    def prepare_home(self):
        self._cancel.clear()

    def home(self, side):
        from tools.arm_homing import ArmHoming, StopUnconfirmed
        if side not in SIDE_NAMES:
            raise ValueError("请选择左臂或右臂")
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("正在执行动作，请等待完成")
        lease, acquired, keep = None, False, False
        try:
            lease = self._lease()
            try:
                lease.acquire()
                acquired = True
            except RuntimeError as exc:
                raise RuntimeError("设备被占用或存在保留锁，请退出 A/B 并按手册 11.2 处理旧锁") from exc
            runner = (self._homing_factory or ArmHoming)()
            try:
                return runner.run(side, self._cancel)
            except StopUnconfirmed:
                keep = True
                raise
        finally:
            if acquired and not keep:
                lease.release()
            self._busy.release()

    def execute(self, side, action):
        if side not in self.endpoints or action not in TARGETS:
            raise ValueError("请选择左/右夹爪与全开/全闭")
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("夹爪正在执行动作，请等待到位")
        try:
            endpoint = self.endpoints[side]
            try:
                lease = self._lease()
                lease.acquire()
            except RuntimeError as exc:
                raise RuntimeError(
                    "设备被占用或存在保留锁。请退出遥操 A/B；若保留锁仍在，"
                    "按操作手册 11.2 核对并归档旧锁。") from exc
            # Acquire ownership before constructing a driver (constructor writes settings).
            try:
                gripper = self._factory(endpoint)
                try:
                    target = TARGETS[action]
                    gripper.move(target, SPEED, 1)
                    deadline = self._clock() + 10.0
                    while True:
                        position = gripper.get_current_position()
                        if abs(position - target) <= 5:
                            return position
                        if self._clock() >= deadline:
                            raise TimeoutError(f"10 秒内未到位，当前位置 {position}；请检查夹爪")
                        self._sleep(0.1)
                finally:
                    gripper.disconnect()
            finally:
                lease.release()
        finally:
            self._busy.release()


class GripperPanel:
    def __init__(self, root, control):
        self.root, self.control = root, control
        self.results = queue.Queue()
        self.busy = False
        self.side = tk.StringVar(value="")
        self.status = tk.StringVar(value="先选择一只夹爪，再点击全开或全闭。")
        root.title("Dobot · 夹爪与从臂归位")
        root.geometry("580x610")
        root.resizable(False, False)
        root.configure(bg="#f4f6fa")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("Panel.TFrame", background="#f4f6fa")
        style.configure("Panel.TLabel", background="#f4f6fa", foreground="#26354b", font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", background="#f4f6fa", foreground="#152238", font=("Microsoft YaHei UI", 20, "bold"))
        style.configure("Side.TRadiobutton", padding=16, font=("Microsoft YaHei UI", 12), background="white")
        style.configure("Action.TButton", font=("Microsoft YaHei UI", 14, "bold"), padding=(16, 17))
        panel = ttk.Frame(root, style="Panel.TFrame", padding=26)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="夹爪与从臂归位", style="Title.TLabel").pack(anchor="w")
        ttk.Label(panel, text="低速单次动作  ·  速度 10  ·  到位后自动断开串口", style="Panel.TLabel").pack(anchor="w", pady=(6, 18))
        choices = ttk.Frame(panel, style="Panel.TFrame")
        choices.pack(fill="x")
        self.selectors = []
        for i, side in enumerate(("left", "right")):
            endpoint = control.endpoints[side]
            button = ttk.Radiobutton(choices, text=f"{SIDE_NAMES[side]}\n{endpoint.port}  /  ID {endpoint.servo_id}",
                                     variable=self.side, value=side, command=self.select,
                                     style="Side.TRadiobutton")
            button.grid(row=0, column=i, sticky="ew", padx=(0, 10) if i == 0 else (0, 0))
            choices.columnconfigure(i, weight=1)
            self.selectors.append(button)
        actions = ttk.Frame(panel, style="Panel.TFrame")
        actions.pack(fill="x", pady=18)
        self.buttons = {}
        for i, (action, title) in enumerate((("open", "全 开"), ("close", "全 闭"))):
            button = ttk.Button(actions, text=title, command=lambda a=action: self.start(a),
                                state="disabled", style="Action.TButton")
            button.grid(row=0, column=i, sticky="ew", padx=(0, 10) if i == 0 else (0, 0))
            actions.columnconfigure(i, weight=1)
            self.buttons[action] = button
        ttk.Label(panel, text="从臂回到初始位置 · 0.15 rad/s", style="Panel.TLabel").pack(anchor="w", pady=(0,8))
        homing = ttk.Frame(panel, style="Panel.TFrame")
        homing.pack(fill="x", pady=(0,10))
        self.home_buttons = {}
        for i, side in enumerate(("left", "right")):
            button = ttk.Button(homing, text=f"{'左' if side == 'left' else '右'}臂慢速归位",
                                command=lambda s=side: self.start_home(s), style="Action.TButton")
            button.grid(row=0,column=i,sticky="ew",padx=(0,10) if i==0 else (0,0))
            homing.columnconfigure(i,weight=1)
            self.home_buttons[side] = button
        self.stop_button = ttk.Button(panel,text="停止归位",command=self.control.cancel_home,state="disabled")
        self.stop_button.pack(anchor="w",pady=(0,12))
        ttk.Label(panel, textvariable=self.status, wraplength=482, style="Panel.TLabel").pack(anchor="w")
        ttk.Label(panel, text="先退出遥操 A/B；归位前清空路径与夹爪持物。", style="Panel.TLabel").pack(side="bottom", anchor="w", pady=(12, 0))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._poll_id = root.after(80, self.poll)

    def select(self):
        if not self.busy:
            for button in self.buttons.values():
                button.state(["!disabled"])
            self.status.set(f"已选择{SIDE_NAMES[self.side.get()]} · 全开 255 / 全闭 0")

    def start(self, action):
        if self.busy or self.side.get() not in SIDE_NAMES:
            return
        side = self.side.get()
        title = "全开" if action == "open" else "全闭"
        self.launch(lambda: self.control.execute(side, action), f"{SIDE_NAMES[side]}正在{title}…",
                    lambda position: f"{SIDE_NAMES[side]}已{title} · 读回位置 {position}")

    def start_home(self, side):
        if self.busy:
            return
        name = "左臂" if side == "left" else "右臂"
        self.control.prepare_home()
        self.launch(lambda: self.control.home(side), f"{name}正在检查路径并慢速归位…",
                    lambda angles: f"{name}已归位 · J1～J6：" + ", ".join(f"{v:.2f}°" for v in angles), homing=True)

    def launch(self, operation, progress, completion, homing=False):
        self.busy = True
        for button in [*self.selectors, *self.buttons.values(), *self.home_buttons.values()]:
            button.state(["disabled"])
        self.stop_button.state(["!disabled"] if homing else ["disabled"])
        self.status.set(progress)
        def worker():
            try:
                result = operation()
                self.results.put((True, completion(result)))
                logging.info("%s result=%s", progress, result)
            except Exception as exc:
                logging.exception("%s failed", progress)
                self.results.put((False, f"动作未完成：{exc}"))
        threading.Thread(target=worker, name="gripper-command", daemon=False).start()

    def poll(self):
        try:
            success, message = self.results.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            self.status.set(message)
            for button in [*self.selectors, *self.home_buttons.values()]:
                button.state(["!disabled"])
            for button in self.buttons.values():
                button.state(["!disabled"] if self.side.get() in SIDE_NAMES else ["disabled"])
            self.stop_button.state(["disabled"])
        self._poll_id = self.root.after(80, self.poll)

    def close(self):
        if self.busy:
            self.status.set("正在执行动作；归位可点“停止归位”，等待结果后关闭窗口。")
            return
        self.root.after_cancel(self._poll_id)
        self.root.destroy()


def main():
    root = tk.Tk()
    root.withdraw()
    try:
        log = ROOT / ".local/gripper-panel.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(log, maxBytes=512_000, backupCount=2, encoding="utf-8")
        logging.basicConfig(level=logging.INFO, handlers=[handler],
                            format="%(asctime)s %(levelname)s %(message)s")
        GripperPanel(root, GripperControl(load_endpoints()))
    except Exception as exc:
        messagebox.showerror("夹爪控制启动失败", str(exc), parent=root)
        root.destroy()
        return 1
    root.deiconify()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
