# Nova2 Linux 接线、遥操与 ACT 示教操作手册

版本：2026-11-24。适用对象：首次接手本装置的研究人员。所有 Bash 命令在 Linux 主机执行；除注明管理员安装步骤外，使用普通用户 yxy。

## 1. 主机与设备表

主机沿用 ACT 训练工作站：yxy@192.168.5.100，Ubuntu 20.04.6，双 Xeon Gold 6240、376 GiB 内存、双 Quadro RTX 8000（每卡 48 GiB）。这些配置来自项目内 11 月 13 日主机记录。训练环境位于 `/home/yxy/data/projects/dobot-act/env`，Python 3.10.13、PyTorch 2.5.1+cu121。

| 接线对象 | Linux 配置段 | 地址或内容 |
|---|---|---|
| 实体右 Nova2 控制柜 | ROBOTS.right_ip | 192.168.5.1 |
| 实体左 Nova2 控制柜 | ROBOTS.left_ip | 192.168.5.2 |
| 工作站机器人网卡 | 系统网络设置 | 192.168.5.100/24，沿用原配置 |
| 右侧从臂夹爪 USB–RS485 | GRIPPER_RIGHT | ID22；port 待填 |
| 左侧从臂夹爪 USB–RS485 | GRIPPER_LEFT | ID21；port 待填 |
| 主手配置 HAND_RIGHT | HAND_RIGHT | 关节 ID11,12,14,15,16,17；append13；port 待填 |
| 主手配置 HAND_LEFT | HAND_LEFT | 关节 ID1,2,4,5,6,7；append3；port 待填 |
| 顶视/左腕/右腕 RealSense | CAMERA | 序列号待填；默认启用 top,left |

主手配置名不代表操作者站位。源码路由为 HAND_RIGHT→实体右 Nova，HAND_LEFT→实体左 Nova；先按配置 ID 与 USB 身份识别，再现场逐轴核对。14 维数据顺序为 RIGHT、LEFT。

## 2. 从接线开始

1. 先固定两臂底座、工具与相机，清理工作区，确认急停可触及。接线时按设备手册断电，主手有支撑，夹爪行程内无手指或物体。
2. 两控制柜 Ethernet 接同一机器人网络交换机，交换机接工作站原机器人网口。保持两臂 .1/.2 和工作站 .100；不要给机器人专用网卡新增默认网关。
3. 两只主手 USB 转串口接工作站；两只夹爪沿用原 USB–RS485 接线及供电。主手与夹爪的转接器分开标记，保持原 A/B、供电极性与接地，不根据线色猜测针脚。
4. 顶视和腕部 RealSense 接 USB 3.x，优先分配独立高速端口；固定腕部线缆并预留关节运动余量。
5. 给转接器贴 HAND_LEFT、HAND_RIGHT、GRIPPER_LEFT、GRIPPER_RIGHT 标签。记录转接器序列号、从臂侧别、舵机 ID、相机视角与序列号。
6. 使用控制柜的既有操作端确认 Nova 固件为 V3，项目要求至少 3.5.8.1；设置 TCP/IP 控制模式。A 服务启动会使能两臂并让夹爪低速闭合、再张开。

查看已有网络配置：

```bash
ip -br address
ip route
ping -c 2 192.168.5.1
ping -c 2 192.168.5.2
```

TCP 端口 29999（Dashboard）、30003（运动）、30004（反馈）是协议端口，保留固定值；6001 是主机内 A/B 通讯端口。不要同时运行其他占用机器人连接的控制程序。

## 3. 部署代码与环境

把本次修改后的整个代码目录复制到 `/home/yxy/data/projects/dobot-teleop`。本次交付在 Windows 的 `E:\Dobot\dobot-atomic-skills`；GitHub 分支只有在推送本次改动后才包含迁移代码。数据集、模型和虚拟环境不随源码复制。

首次由管理员安装系统组件：

```bash
sudo apt-get update
sudo apt-get install python3-venv python3-tk libgl1 libglib2.0-0 xterm
sudo usermod -aG dialout "$USER"
```

退出桌面会话并重新登录后，`id -nG` 应包含 dialout。RealSense 安装对应 librealsense udev 规则，采用 [Intel 官方 Linux 安装说明](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)。USB 权限变更后重新插拔相机。遥操窗口依赖 Linux 本机桌面 DISPLAY/Wayland；只通过无图形 SSH 启动无法显示相机预览。

```bash
cd /home/yxy/data/projects/dobot-teleop
bash scripts/linux/setup.sh
source scripts/linux/activate.sh
```

安装脚本使用原训练环境的 Python 创建 `.venv-teleop`，安装带 GUI 的 OpenCV、RealSense、串口、ZeroMQ、数据转换依赖。原训练环境的 headless OpenCV 保持独立。若训练 Python 不在默认路径，设置 `DOBOT_BASE_PYTHON=/实际路径/python` 后运行 setup。若 venv 提示 tkinter 不可用，需为该 Python 安装匹配的 Tk 支持；网页控制台内部也使用夹爪面板模块。

安装需要主机能访问软件源；原主机曾通过笔记本网络代理联网，需要时沿用现场网络方式。不要重新安装 GPU 驱动或覆盖 ACT 环境。

## 4. 填写本机配置

setup 首次生成 `config/linux.local.ini`，重复运行不会覆盖。此文件被 Git 忽略。

```bash
python -m scripts.1_find_port
ls -l /dev/serial/by-id/
python -c "import pyrealsense2 as rs; print([(d.get_info(rs.camera_info.name),d.get_info(rs.camera_info.serial_number)) for d in rs.context().query_devices()])"
nano config/linux.local.ini
```

逐一插拔转接器辨认 USB 身份，把四个不同的 `/dev/serial/by-id/...` 路径填写到四个 port 字段。没有唯一序列号的设备可用 by-path，但需固定 USB 插口；不要把不稳定的 ttyUSB 编号当长期身份。

相机通过遮挡镜头确认视角，填写 top、left；right 未启用可为空。默认 640×480、15 FPS、RGB，depth_enabled=false；保留 top/left 的 180°翻转后再在预览核对。按实际任务修改 ACT_TASK；当前模板为左臂试管插放、top/left 两视角。

主手 2 Mbps、夹爪 1 Mbps，夹爪行程与关节偏置沿用原装置参考值。两个 `absolute_mapping_verified` 初始为 0，完成下一节后才改 1。

可选 FTDI 低延迟：检查 `readlink -f /dev/serial/by-id/<实际设备>`，找到对应 ttyUSB 名称，再查看 `/sys/bus/usb-serial/devices/ttyUSBx/latency_timer`。管理员可为该转接器设置专用 udev 规则将属性设为 1；程序不保存 sudo 密码、不修改设备为 777，不支持该属性的设备直接跳过。

## 5. 主手映射与初始姿态

迁移后仍需现场复核 USB 路由与标定。A/B 均退出，先在原控制端查看从臂角度，以独立姿态参照摆放主手，支撑好主手，不通过遥操把从臂拉到未知目标。

```bash
source scripts/linux/activate.sh
python -m scripts.2_get_offset --section HAND_RIGHT --output /tmp/master-right.json
python -m scripts.2_get_offset --section HAND_LEFT --output /tmp/master-left.json
```

脚本只读主手编码器，报告 raw_rad、mapped_deg、offsets_rad、signs。计算关系为 `mapped=(raw-offset)*sign`。仅当已独立确认主手处于参考姿态时，增加 `--reference-positioned` 可生成候选偏置；程序不自动写配置。

逐轴核对六轴顺序、正负方向、参考姿态和另一独立姿态；重启后复读。把检查结果、实际侧别和日期保存到项目记录，验收的一侧才设置 `absolute_mapping_verified=1`。若同一批硬件未拆改，原偏置可作为起点；出现系统性角度误差时先查方向、路由和姿态。

已知参考 home（角度）：右臂 `[90,0,90,0,-90,-90]`，左臂 `[-90,0,-90,0,90,90]`。回程路径必须由现场人员确认。B 默认只检查初始姿态，不自动归位。需要归位时在 A 已启动、B 已退出的条件下使用控制台“通过 A 双臂归位”，或明确执行 `python -m tools.home_service`。

## 6. 启动遥操

终端 A：

```bash
source /home/yxy/data/projects/dobot-teleop/scripts/linux/activate.sh
python -m tools.check_teleop_local --inventory
python -m experiments.launch_nodes --hostname 127.0.0.1 --robot-port 6001
```

等待右、左两路夹爪初始化完成、服务显示监听 6001。配置不完整时程序在建连前拒绝。若显示设备锁占用，按第 9 节处理。

终端 B（新的本机桌面终端）：

```bash
source /home/yxy/data/projects/dobot-teleop/scripts/linux/activate.sh
python -m experiments.run_control --hostname 127.0.0.1 --robot-port 6001 --show-img --project-name tube_insert_linux_v1
```

预览应为顶视与实体左腕。确认按钮线程就绪、姿态检查通过。手稳住主手，每次只开启一侧，等待 ALIGN_DONE 后做小范围动作并观察从臂和夹爪侧别。

| 按钮操作 | 功能 |
|---|---|
| A 按下并释放 <0.5 秒 | 主手力矩锁定/解锁 |
| A 按下并释放 >1 秒 | 对应侧跟随开/关，开启执行对齐门控 |
| A 持续 0.5～1 秒 | 当前代码可能无动作，避免使用这个区间 |
| 任一 B，已有跟随 | 启动/停止共享录制会话 |

A 短按的主手锁定与停止从臂跟随是不同状态。观察 SERVO_OFF 与实体静止来确认跟随已停。保护报错时保存日志，按报错检查主手串口、相机、角度和工作区。

## 7. 采集、标注、导出和训练

按 B 开始，完成动作后按 B 结束，观察 RECORD_START/RECORD_STOP。试管任务要完整包含插入、松爪、撤离。正常退出 B 后检查数据，异步队列需要时间落盘。

默认数据位置是仓库上级的 `datasets/<project>/collect_data/<episode>`。若采用建议路径，就是 `/home/yxy/data/projects/datasets/...`。可通过 B 的 `--save-data-path` 指定其他位置；网页控制台的数据操作使用默认位置。

```bash
raw=/home/yxy/data/projects/datasets/tube_insert_linux_v1/collect_data
episode="$raw/<实际episode目录名>"
python -m dobot_control.act_dataset check "$episode"
python -m dobot_control.act_dataset preview "$episode" --output "$episode/preview.mp4"
python -m dobot_control.act_dataset label "$episode" --status success --note '试管插入、松爪后稳定、左臂撤离'
# 失败示范按实际情况标 failure，并填写原因。
dataset=/home/yxy/data/projects/dobot-act/datasets/tube_insert_linux_v1
python -m dobot_control.act_dataset convert "$raw" --output "$dataset" --fps 15
python -m tools.prepare_training "$dataset"
```

导出目录必须是新目录；仅导出质量通过且标 success 的 episode。同一数据版本要求相机、物理绑定、图像处理与配置一致。至少两个成功 episode 才能形成非空训练/验证划分；正式训练需要覆盖实际工作区域的充分示范。prepare_training 只用 train split 计算归一化，不覆盖已有统计。

训练在另一个终端使用原 ACT 环境：

```bash
source /home/yxy/data/projects/dobot-act/activate.sh
cd /home/yxy/data/projects/dobot-teleop
export PYTHONPATH="$PWD:/home/yxy/data/projects/dobot-act/src/act:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
python -m scripts.train_act_host   --dataset /home/yxy/data/projects/dobot-act/datasets/tube_insert_linux_v1   --run /home/yxy/data/projects/dobot-act/runs/tube_insert_linux_v1_10k   --steps 10000 --batch-size 8 --workers 4
```

这是启动示例，步骤数按研究计划决定；本次迁移未执行训练。当前训练入口针对 top/left 两相机、14 维状态/动作、chunk30、FP32。产物包括 config.json、splits.json、normalization.json、metrics.jsonl、best.pt、last.pt。训练过程不依赖机器人在线。

## 8. 图形入口与夹爪恢复

```bash
source scripts/linux/activate.sh
python -m tools.open_console
# 或直接显示服务日志：
python -m tools.local_console --no-browser
# 独立夹爪/Tk 面板：
python -m tools.gripper_panel
```

浏览器打开 `http://127.0.0.1:8765`。控制台提供进程列表、旧锁归档、A/B 启动、归位、相机预览、episode 检查/回看/导出和夹爪操作。硬件操作按页面提示确认。Linux 启动 A/B 需要桌面 `x-terminal-emulator`；不可用时复制页面 Bash 命令到本机终端运行。

A 运行时恢复夹爪：先停止录制、关闭跟随并退出 B，在控制台选择“通过 A”的恢复操作，复用 A 已占用的串口。独立夹爪面板需要退出 A/B 并取得共享设备锁。不要同时用两个进程打开同一串口。

## 9. 停机和旧锁处理

依次停止录制、分别长按 A 关闭跟随、观察实体静止、支撑并锁住主手；Ctrl+C 退出 B，等异步写盘关闭，再退出 A。需要撤销电机使能时使用控制柜已有操作端，紧急情况使用实体急停。

旧 A 服务退出后保留设备锁，要求现场确认设备停止后归档：

```bash
python -m dobot_control.linux_processes list
python -m dobot_control.linux_processes archive --lock "$DOBOT_DEVICE_LOCK_PATH"
```

归档工具检查本机主机名、持锁 PID 已退出、控制进程已关闭，并保留 owner.json。外来锁、缺损记录或仍存活 PID 会拒绝自动归档；先查明来源。设备锁路径须在所有终端一致。

## 10. 常见问题

| 现象 | 处理 |
|---|---|
| 找不到 linux.local.ini | 运行 setup 或从 example 复制，确认 DOBOT_CONFIG |
| 串口 Permission denied | 确认 dialout 组已重新登录生效，检查 by-id 目标与 udev |
| RealSense 未枚举/启动失败 | 检查 USB3、供电、规则、序列号、其他占用相机的进程 |
| cv2.imshow 报错 | 使用 .venv-teleop 的 opencv-python 和本机桌面，不能用训练环境的 headless 版本 |
| tkinter 导入失败 | 给所用 Python 安装匹配 Tk 组件，不同 Python 的系统 Tk 未必互通 |
| 地址连不上 | 检查原机器人网卡 .100/24、网线/交换机与两控制柜 TCP/IP 模式 |
| 夹爪恢复被拒绝 | 正常退出 B/ACT 实验，再使用 A 内恢复；检查 /proc 可见性 |
| ALIGN_REJECTED_UNSAFE / MASTER_DROP_LOCK | 停跟随，检查主手角度、接线、标定与物理路径 |
| RECORD_WRITER_FAILURE | 停录制，检查磁盘空间、权限和 episode 质量报告 |
| GPU OOM | 保持原训练环境，降低 batch，检查是否有其他 GPU 任务 |

## 11. 新人交接记录

记录实际 USB by-id 路径、相机序列号及视角、主手配置与实体侧别、标定检查日期、固件版本、接线照片、操作人、数据目录和训练 run。首次空载完成单侧、双侧、夹爪、短 episode 回看后再进入正式采集。

参考：[越疆 Nova 产品与资料](https://www.dobot-robots.com/products/nova-series/dobot-nova-series.html)、[越疆官方 V3 Python SDK](https://github.com/Dobot-Arm/TCP-IP-Python-V3)。本项目保留已用的 V3 通讯方式；V4 SDK 对应不同固件，不能直接替换其命令与反馈结构。
