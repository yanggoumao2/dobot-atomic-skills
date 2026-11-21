# Dobot Nova2 双臂 · Robonix 离线部署

本部署在同一台电脑上运行 Atlas、Soma、两条六轴 Nova2 模拟臂和两个模拟平行夹爪。四个实例各自持有内存状态，关节和夹爪命令只作用于对应实例。整个部署只用本地软件包和回环地址，无需 Hub 账号、模型密钥、ROS 2 或机器人连接。

| 实例 | Robonix 标准能力 | 输入和反馈 |
| --- | --- | --- |
| `left_arm`、`right_arm` | `robonix/primitive/arm/{joint_command,joint_states}` | gRPC 客户端流输入、服务端流反馈。每帧使用 `sensor_msgs/JointState`，六个关节的名称带左右前缀，位置单位为 rad。 |
| `left_gripper`、`right_gripper` | `robonix/primitive/gripper/{info,open,move,grip,stop,state}` | 前五项为 MCP 调用；`state` 为 gRPC 服务端流。开口单位为 m，模拟全行程为 0.08 m。命令应答表示受理，到位由状态流确认。 |

部署固定对照 Robonix 源码 `fb76dc9c2766f71936c7f20b39f09994965e1176`。线上手册的示例修订较早，独立夹爪接口以该源码中的 IDL 为准。[官方接入指南](https://book.robonix.ai/integration-guide/vendor-onboarding) · [标准原语契约](https://github.com/syswonder/robonix/tree/fb76dc9c2766f71936c7f20b39f09994965e1176/capabilities/primitive)

## 本地快速检查

从本仓库根目录运行。Windows 和 Linux 都可使用 Python 3.10+，测试需要 PyYAML 和 pytest。

```bash
python -m pytest -q robonix/nova2
```

## 在 Windows 上运行协议验收

需要 Git、Rust 的 MSVC 工具链、Visual Studio C++ 构建工具、CMake 和 uv。以下命令从本仓库根目录的 PowerShell 运行，使用临时目录存放上游源码和 Python 环境。首次构建需要下载依赖。脚本会编译官方 Atlas、生成协议绑定并通过回环地址调用四个模拟实例。

```powershell
$src = Join-Path $env:TEMP 'robonix-fb76dc9'
git clone https://github.com/syswonder/robonix.git $src
git -C $src checkout --detach fb76dc9c2766f71936c7f20b39f09994965e1176
git -C $src submodule update --init capabilities/lib/common_interfaces capabilities/lib/rcl_interfaces capabilities/lib/unique_identifier_msgs
uv venv --python 3.10 "$src/.venv"
$python = "$src/.venv/Scripts/python.exe"
uv pip install --python $python "${src}/pylib/robonix-api[codegen]"
& ./robonix/nova2/test-wire.ps1 -RobonixSource $src -Python $python
```

## 在 Linux 上构建和启动

在有 Git、Rust、C/C++ 构建工具、Python 3.10+ 和 uv 的 Linux 主机上安装固定版本。下列命令在独立 Python 环境中安装该版本的 SDK 和代码生成依赖。后续终端也应激活同一环境。

```bash
git clone https://github.com/syswonder/robonix.git
cd robonix
git checkout --detach fb76dc9c2766f71936c7f20b39f09994965e1176
git submodule update --init --recursive
uv venv --seed .venv
source .venv/bin/activate
uv pip install './pylib/robonix-api[codegen]'
make install
rbnx setup "$PWD"
```

进入 **本仓库** 的 `robonix/nova2` 目录。先构建，再做进程间验收，最后启动部署。

```bash
rbnx build -f robonix_manifest.yaml
python tests/wire_smoke.py --capabilities "$(rbnx path capabilities)"
rbnx boot -f robonix_manifest.yaml
```

`wire_smoke.py` 启动自己的临时 Atlas 和四个模拟进程。它按清单执行初始化与激活，移动左臂第一关节 0.05 rad、让右爪到达 0.04 m，检查另外一条臂和另一只夹爪保持原值，最后停用实例并退出。成功时打印实际关节角和开口。该验收可独立于常驻的部署运行。

部署运行后，在另一终端执行 `rbnx caps -v`，应能看到四个实例各自的能力及共享的 `robonix/lifecycle/driver`。在启动终端按 Ctrl+C 结束。四个实例的配置在 [部署清单](robonix_manifest.yaml)，整机部件与能力归属在 [Soma 描述](soma.yaml)。

## 模型和适用范围

单臂连杆的关节变换取自越疆 Nova2 描述，初始关节角沿用本项目的左右臂归位配置。双臂安装间距 0.60 m、夹爪外形和 80 mm 开口用于模拟。URDF 中固定张开的夹爪只用于展示，模拟运动后的开口以 `gripper/state` 为准。模型来源和关节限位记在 [URDF](urdf/nova2_dual.urdf)。

本阶段覆盖能力发现、生命周期、关节指令和夹爪指令的反馈。机械臂按目标直接更新关节位置，夹爪模拟空载开合。接触、持物、碰撞、笛卡尔轨迹和相机均在本次验证范围之外。该版本的 Soma 运行状态监视器读取 ROS 2 关节反馈，本部署的实时关节值从 Atlas 发现的 gRPC 状态流读取。
