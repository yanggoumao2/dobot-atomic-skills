# Linux Docker 部署与离线测试

此镜像封装 Python3.10、遥操/夹爪/RealSense/GUI/采集转换依赖和测试工具。基础系统 Ubuntu22.04，可在原 Ubuntu20.04 Linux Docker 主机上运行；训练继续使用原 dobot-act CUDA 环境。模型权重、数据、硬件本机配置独立挂载。

本次已构建镜像与测试结果见 [发布报告](../docs/migration/20261125-linux-docker-release.zh-CN.md)。

## 获取镜像

在 GitHub 的 Linux Docker release 成功运行页面，下载 artifact `dobot-teleop-linux-amd64`，解压外层 ZIP，得到 `dobot-teleop-linux-amd64.tar.gz`。镜像适用于 amd64/x86_64；Actions artifact 保留30天，也可从源码重新构建。

```bash
docker load -i dobot-teleop-linux-amd64.tar.gz
docker run --rm dobot-teleop:linux
```

默认命令执行全部离线测试和完整合成数据流程，使用 Xvfb 虚拟显示，无设备挂载。它包含网页 HTTP 接口、A/B 真实本机 RPC、合成图像/状态的异步录制、质检、回看、标注、HDF5 导出、训练集归一化与 ACT 输入加载。测试不运行 GPU 优化训练。

自行构建：

```bash
git clone --branch linux https://github.com/HYJn14/dobot-atomic-skills.git
cd dobot-atomic-skills
docker build -t dobot-teleop:linux .
```

## 现场运行准备

先完成 [Linux 接线手册](TELEOP_LINUX_SETUP.zh-CN.md) 的网卡、USB 权限、相机和主手标定。主机仍为192.168.5.100，右臂 .1、左臂 .2。Docker Engine 与 Compose v2 在主机预先安装。

在仓库根目录创建挂载目录：

```bash
mkdir -p ../datasets ../.local/runtime .local scripts/logs
cp -n config/linux.example.ini config/linux.local.ini
```

四个主机串口在 Compose 中映射为容器固定名称。因此填写 linux.local.ini：

| 配置段 | 容器 port |
|---|---|
| HAND_LEFT | /dev/dobot-hand-left |
| HAND_RIGHT | /dev/dobot-hand-right |
| GRIPPER_LEFT | /dev/dobot-gripper-left |
| GRIPPER_RIGHT | /dev/dobot-gripper-right |

相机序列号、主手核验标志按现场填写。四个真实 USB by-id 路径仅设置在以下主机环境变量，不能把表中的容器名称填成主机设备来源：

```bash
export HAND_LEFT_DEVICE=/dev/serial/by-id/<现场主手LEFT设备>
export HAND_RIGHT_DEVICE=/dev/serial/by-id/<现场主手RIGHT设备>
export GRIPPER_LEFT_DEVICE=/dev/serial/by-id/<现场夹爪LEFT设备>
export GRIPPER_RIGHT_DEVICE=/dev/serial/by-id/<现场夹爪RIGHT设备>
export UID_LOCAL=$(id -u)
export GID_LOCAL=$(id -g)
export DIALOUT_GID=$(getent group dialout | cut -d: -f3)
export DOBOT_HOSTNAME=$(hostname)
export XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}
```

在本机 X11/XWayland 桌面终端执行，确认 DISPLAY 和 XAUTHORITY 指向有效会话。将实际值写到 Git 忽略的 `.env` 也可供 Compose 使用。主机 udev 应允许当前用户访问 RealSense USB，容器保留该用户 UID/GID 和 dialout 组。

Compose 使用 host 网络，让网页仍监听主机 `127.0.0.1:8765`，两臂 TCP 地址直接可达。使用 host PID 和相同 hostname，让控制进程检查及锁归档识别主机和容器内进程；共享锁目录与主机原生运行保持一致。四个串口显式映射，RealSense 使用 USB 总线与字符设备189权限，未开启 privileged。

## 启动控制台

```bash
docker compose --profile hardware up -d
```

在主机浏览器访问 `http://127.0.0.1:8765`。界面自动显示 Bash、容器数据路径和容器串口名。点击启动 A/B 时在现场桌面打开容器内 xterm；相机预览也显示在该桌面。

也可在两个终端分别执行：

```bash
docker compose exec teleop python -m experiments.launch_nodes
# A服务就绪后，在另一个终端：
docker compose exec teleop python -m experiments.run_control --show-img --project-name tube_insert_linux_v1
```

原始数据在容器 `/workspace/datasets`，对应主机仓库上级 `datasets`。锁对应主机 `../.local/runtime`；日志分别映射到 `.local` 和 `scripts/logs`。

正常停录、关闭跟随、退出 B、退出 A 后再停止容器：

```bash
docker compose --profile hardware down
```

若容器异常退出，先现场确认设备停止，再由控制台/手册步骤处理旧锁。不要让主机原生 A/B 与容器 A/B 同时控制同一套设备。

## ACT训练衔接

容器输出 HDF5 与训练统计可直接供主机原 ACT 环境读取；路径使用挂载后的主机数据路径。详细命令见 Linux 手册第7节。镜像不替换原训练 CUDA/PyTorch 环境。

## CI与证据

`.github/workflows/linux-docker.yml` 在 linux 分支推送后：构建镜像 → 容器内离线设备模拟测试 → docker save → 上传镜像和日志。镜像仅在测试成功后导出。

参考：[Docker Compose 配置](https://docs.docker.com/reference/compose-file/services/)、[GitHub Actions artifacts](https://docs.github.com/en/actions/tutorials/store-and-share-data)。
