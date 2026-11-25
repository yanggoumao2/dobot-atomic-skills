# 项目说明：Nova2 遥操、示教数据与 ACT

## 系统职责

A（experiments.launch_nodes）独占两臂 TCP 与夹爪串口，通过本机 ZeroMQ 6001 暴露状态、控制和反馈。B（experiments.run_control）读取两主手、按钮与相机，规划关节目标，通过 A 执行，并将图像/状态/动作写入 episode。网页控制台只负责现场入口与受控操作。ACT 训练读取已导出的 HDF5，与实时遥操进程分开。

```text
主手两路 USB → B 按钮/绝对映射/安全规划 → ZeroMQ → A → Nova TCP + 两路夹爪 USB
RealSense → B 帧快照/异步写盘 → 原始 episode → 质检/人工标注 → HDF5 → ACT 训练
```

## 代码地图

| 目录/模块 | 职责 |
|---|---|
| dobot_control/config.py | 单一配置路径、启动前完整性检查、实际硬件元数据 |
| dobot_control/layout.py | 物理侧别、固定向量顺序、参考 home |
| dobot_control/device_lease.py | 跨进程设备所有权目录锁 |
| dobot_control/linux_processes.py | Linux /proc 检查、归档本机已退出进程的旧锁 |
| dobot_control/absolute_teleop.py、teleop_safety.py | 对齐、步长/速度、姿态/工作区/主手突变保护 |
| dobot_control/robots/ | Nova V3 SDK、驱动、双臂组合、ZeroMQ |
| dobot_control/dynamixel/、gripper/ | 主手和夹爪驱动、串口超时与反馈 |
| dobot_control/cameras/ | RealSense、相机配置与方向 |
| dobot_control/act_recording.py、async_writer.py | 时间戳、采样节奏、反馈、异步落盘与关闭 |
| dobot_control/act_dataset.py | check、preview、label、convert、训练窗口读取 |
| tools/local_console.py、linux_console.py | Windows PowerShell / Linux Bash 控制台 |
| tools/prepare_training.py | 已导出数据的训练集归一化统计 |
| scripts/train_act_host.py、evaluate_act_host.py | 原训练主机 ACT 训练/评估 |
| scripts/linux/ | Ubuntu 环境安装与统一激活 |
| config/linux.example.ini | Linux 模板；IP 固定，USB/相机身份留空 |
| third_party/ | 主手/夹爪 SDK 与许可证 |

## 平台与配置

配置优先级：DOBOT_CONFIG → Linux config/linux.local.ini → Windows scripts/dobot_config/dobot_settings.ini。选择环境变量后再启动进程；配置变更后完整重启 A/B。Windows/Linux 共用控制算法，不复制两套算法。

实体右臂为 .1、左臂为 .2。HALF0=RIGHT，HALF1=LEFT，各 6 个关节+1 个夹爪。关节 rad；夹爪归一化 0=闭、1=开，驱动 0～255。主手输入按 HAND_LEFT、HAND_RIGHT 排列，master_to_nova_action 交换两个半向量得到从臂顺序；按钮和夹爪使能使用同一映射。

设备锁默认位于仓库上级 .local/runtime/device-owner，所有操作入口应使用同一 DOBOT_DEVICE_LOCK_PATH。旧任务编排模块中的 layout/device_lease 已迁入控制核心。

## 功能对照

| 功能 | Windows | Linux |
|---|---|---|
| A/B 双臂遥操、主手按钮、绝对跟随与保护 | 共用 | 共用 |
| 夹爪反馈/独立面板/通过 A 恢复 | PowerShell 进程检查 | /proc 检查 |
| 单/多相机预览、翻转、RGB/深度配置 | 共用 | 共用，需 GUI OpenCV |
| 异步采集、质量检查、标注、回看、HDF5 | 共用 | 共用 |
| 网页控制台与归位 | PowerShell | Bash / 桌面终端 |
| ACT 训练/评估 | 保留入口 | 原主机 CUDA 环境 |
| 既有 ACT 真机实验工具 | 保留 Windows 前台/空格门控 | 离线 replay；live 仍要求 Windows 前台门控 |

ACT 真机推理实验属于保留的历史工具，Linux 遥操、采集和训练采用上表共享链路。Windows 快捷启动 .cmd 保留。

## 数据契约

原始 episode 包含 topImg/leftImg（按启用相机）、observation/*.pkl、episode.json、closed.json。元数据写入当前串口绑定、相机序列号、图像处理、任务和硬件布局版本。closed 表示录制结束状态；检查工具判定样本完整性、时间与反馈质量。标签由操作者回看后给出。

HDF5 包含 observations/qpos、qvel、images/<camera>、action。保持状态/动作同行对齐，图像 RGB；尾部动作窗口用 padding mask。导出生成 manifest、整条 episode 的 train/validation split。归一化只使用训练集，避免验证数据泄漏。

当前训练脚本使用 top/left、14 维、chunk30。新增相机/任务时先明确数据契约，再同步训练入口。改变标定、串口身份、视角或图像处理后用新数据版本，转换器会拒绝混合不兼容配置。

## 开发入口与测试

离线模拟命令 `python tools/simulate_linux.py`：临时 Linux 配置、模拟 /proc、硬件替身，加共享遥操回归。具体结果见 migration 报告。硬件连接参数和现场标定由操作手册接续。

新增代码优先放入 dobot_control 独立模块；入口不应在 import 时连接硬件。保持设备所有权和启动前配置检查。GUI/操作系统差异放在平台辅助模块，算法和数据层保持共用。

## 官方接口依据

[越疆官方 V3 Python SDK](https://github.com/Dobot-Arm/TCP-IP-Python-V3)支持 CR/Nova V3；本项目继续使用 Python socket 与 NumPy 反馈解析，Linux 不需要 Windows DLL。项目内固件门槛为 V3≥3.5.8.1。[V4 官方仓库](https://github.com/Dobot-Arm/TCP-IP-Python-V4)对应 V4 固件，升级需要单独核对指令和反馈协议。
