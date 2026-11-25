# Nova2 双臂遥操与 ACT 示教训练

Windows / Linux 共享一套遥操、采集与数据处理代码。实体右臂 192.168.5.1、左臂 192.168.5.2；14 维数据顺序始终为右臂六关节+夹爪、左臂六关节+夹爪。

- [Docker 镜像获取、离线测试与现场部署](manuals/DOCKER_LINUX.zh-CN.md)
- **新研究人员从这里开始：[Linux 接线与操作手册](manuals/TELEOP_LINUX_SETUP.zh-CN.md)**
- [项目结构、功能、数据与开发说明](docs/PROJECT_GUIDE.zh-CN.md)
- [本次迁移变更及模拟测试结果](docs/migration/20261124-linux-migration.zh-CN.md)
- [Windows 操作手册](TELEOP_WINDOWS_SETUP.zh-CN.md)、[ACT 数据采集说明](manuals/ACT_DEMONSTRATION_COLLECTION.zh-CN.md)

## Linux 快速入口

仓库建议放置 `/home/yxy/data/projects/dobot-teleop`。先按手册接线、安装依赖、填写配置并完成主手映射复核：

```bash
bash scripts/linux/setup.sh
source scripts/linux/activate.sh
python -m tools.check_teleop_local --inventory
# 终端 A：会使能两臂并初始化夹爪
python -m experiments.launch_nodes
# 另一个终端，重新 source 后：
python -m experiments.run_control --show-img --project-name tube_insert_linux_v1
```

网页控制台：`python -m tools.open_console`，地址 `http://127.0.0.1:8765`。Linux 使用 Bash/本机桌面终端，Windows 使用 PowerShell。配置通过 `DOBOT_CONFIG` 统一选择。Linux 模板的串口和相机序列号为空，机器人 IP 已填入原装置地址。

## 范围

保留主手按钮、绝对关节跟随、对齐与运动保护、夹爪、单/多相机预览、异步录制、ACT 质量检查/标注/HDF5 导出、训练/评估，以及 Windows ACT 实验工具。删除独立任务 DAG、中文任务解析、化学/烧杯任务和上游 HTTP 执行网关；详见迁移清单。

2026-11-25：[完整流程与 Docker 发布](docs/migration/20261125-linux-docker-release.zh-CN.md)已通过，Linux 容器160项通过、7项Windows专属跳过，并完成合成数据全流程。

## 离线验证

```bash
python tools/simulate_linux.py
```

2026-11-24 已在 Windows 上完成一次 Linux 分支模拟及共享逻辑回归：145 项通过。Linux 分支采用临时 `/dev` 配置、模拟 `/proc` 和设备替身；测试报告说明覆盖范围。

## 来源

基础代码来自 Dobot X-Trainer，参见 [原始说明](docs/upstream/README.dobot.md)、[LICENSE](LICENSE)、[第三方许可](THIRD-PARTY-LICENSES)。Nova V3 TCP/IP 参考 [越疆官方 SDK](https://github.com/Dobot-Arm/TCP-IP-Python-V3)。
