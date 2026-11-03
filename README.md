# Dobot 双臂原子技能编排

基于越疆 X-Trainer / Nova 平台，将抓取、放置和等待组织为可执行的依赖图，提供真机适配器、执行记录和面向上游规划团队的 HTTP 执行网关。

当前链路：**上游生成执行图 → 网关校验 → 原子技能执行 → 状态查询与事件回传**。现场人工入口同时保留中文指令编排。

## 已有功能

- 单臂 A/B 抓放、垂直接近与抬升、运输高度和归位流程。
- 双臂依赖调度、等待、保持与返程；自然语言示例可编译为 43 个底层节点。
- 外部执行包校验、每请求新执行的单次提交、面向自动重试的显式幂等提交、两阶段预检/提交、查询、事件和取消。
- 持久幂等、设备所有权、场景版本及事件 outbox。
- 控制器位姿/关节收敛、夹爪行程反馈、失败停止和执行日志。

外部合同使用 `RuntimeExecutionPackage 2.1`、`graph_version=0.1`、`dobot-dag/1`。目前支持 `embodied.pick@1.0.0`、`embodied.place@1.0.0`、`wait` 和 success AND 依赖；位置由本机场景标定解析。Mock 与 live 网关都允许左右臂独立技能并行，同臂动作须在依赖图中显式串行；跨臂并行不提供碰撞检测。

## 本地单臂与双臂操作

在仓库根目录运行，先预览执行图。以下命令不连接机械臂：

```powershell
# 左臂：A → 目标 A
python -X utf8 -B -m task_skills.run_task "把物体A抓取放到目标A" --dry-run

# 右臂：B → 目标 B
python -X utf8 -B -m task_skills.run_task "把物体B抓取放到目标B" --dry-run

# 双臂顺序执行：左臂完成后，再执行右臂
python -X utf8 -B -m task_skills.run_task "把物体A抓取放到目标A，然后把物体B抓取放到目标B" --dry-run

# 双臂协调：并行抓取、左臂保持 5 秒、放置后右臂取回 B
$task = '用左臂把物体A拿起来，然后右臂把物体B拿起来。' +
        '左臂拿起物体A以后保持5秒，右臂直接送到目标B。' +
        '然后5s后左臂再把物体A送到目标A。' +
        '随后右臂把物体B从目标B拿起来放回物体B原来的位置。'
python -X utf8 -B -m task_skills.run_task "$task" --dry-run
```

真机执行使用已配置硬件依赖的 Windows 环境：完成现场检查后，去掉 `--dry-run`，核对打印的执行图，再按提示输入大写 `YES`。该入口直接使用设备适配器，不需要启动 HTTP 网关或 `launch_nodes.py`。固定绑定为左臂 A、右臂 B。

[本地单臂与双臂操作说明](docs/local-control.md)包含完整真机命令、设备锁、初始场景、归位语义和结果文件说明。

## 快速开始：Mock 联调

使用 Python 3.10 或更新版本，在仓库根目录运行。HTTP Mock 网关只需 Python 标准库，无需机械臂或完整训练环境。

```powershell
git clone https://github.com/yanggoumao2/dobot-atomic-skills.git
cd dobot-atomic-skills
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -B -m task_skills.run_gateway --mode mock --port 8080 --data-dir ./gateway-data/mock-demo
```

Linux 可使用 `python3 -m venv .venv`、`source .venv/bin/activate`。在另一个已激活同一环境的终端提交七节点示例：

```powershell
python -B -m task_skills.gateway.demo_client --output ./gateway-data/demo-request.json
```

两阶段提交使用 `--two-phase`；第一次演示会改变持久化逻辑场景，重复演示前按[接口说明](docs/execution-api/README.md)关闭网关并复位 mock 场景，或使用新的 mock 数据目录。

## 离线验证

```powershell
python -B -m unittest discover -s tests -p test_execution_gateway.py -q
python -B -m unittest discover -s tests -p test_atomic_runner_unittest.py -q
```

遥操作回归还需要 NumPy（安装到项目虚拟环境）：

```powershell
python -m pip install numpy
python -B -m unittest discover -s tests -p test_teleop_regressions.py -q
python -B tests/gateway_cli_smoke.py --output ./gateway-data/cli-smoke
```

`requirements.txt` 保留 X-Trainer 原始完整环境依赖，包含相机、仿真与训练相关组件；Mock 联调无需安装该列表。

## 相机到基座标定 UI

仓库内置独立的七步中文标定向导，复用现有 Nova2 固定棋盘算法，支持左右臂分别标定、后台任务与日志、像素查询、独立验收和证据导出。它不修改执行网关、现有示教位姿或 scene/chemistry profile。

Windows 在仓库根目录运行（创建隔离的 `.venv-calibration`，固定 NumPy 1.26.4 + OpenCV 4.8.1.78）：

```powershell
.\Start-Calibration.ps1
```

也可手工启动：

```powershell
py -3.12 -m venv .venv-calibration
.\.venv-calibration\Scripts\python.exe -m pip install -r requirements-calibration.txt
.\.venv-calibration\Scripts\python.exe -X utf8 -B -m task_skills.run_calibration_ui --open-browser
```

默认只监听 `127.0.0.1:8765`，数据写入被 Git 忽略的 `calibration-data/`。启动不会连接相机或机器人；相机预览/采集与只读 `GetPose()` 必须由操作者在页面中显式触发。模拟任务全程标记 `synthetic_only`，不能作为真机验收或用于真实控制。完整步骤、安全边界、集成 API 与测试命令见 [标定 UI 文档](docs/calibration-ui.md)。

## 文档与入口

| 内容 | 位置 |
| --- | --- |
| 当前 HTTP 合同、双方操作、幂等与现场恢复 | [docs/execution-api/README.md](docs/execution-api/README.md) |
| 原子技能执行接口与联调手册（14 页 PDF） | [联调手册 v1.1](docs/execution-api/Dobot_原子技能执行接口与联调手册_v1.1.pdf) |
| 外部运行包示例 | [runtime-package.example.json](docs/execution-api/runtime-package.example.json) |
| 受限 profile Schema | [runtime-package.profile.schema.json](docs/execution-api/runtime-package.profile.schema.json) |
| 中文指令与底层执行图 | [task_skills/README.md](task_skills/README.md) |
| HTTP 网关 | `python -m task_skills.run_gateway` |
| 指令预览 | `python -m task_skills.run_task "把物体A抓取放到目标A" --dry-run` |
| 场景配置 | `task_skills/scene_config.json` |
| 控制与串口配置 | `dobot_control/`、`scripts/dobot_config/dobot_settings.ini` |

## 真机与验证记录

2026-09-25，单臂 A、单臂 B 和双臂 43 节点保持/返程任务已完成控制器执行，并由现场操作者确认实际抓取、转运、放置与松爪。10 月新增 HTTP 网关完成离线与 Mock 验证，真实上游联调及网关真机验收待做。执行文件的证据等级仍为 `command_trace`，物理成功依赖现场或独立传感器确认。

配置中的场景坐标、Tool 0、左右臂与夹爪绑定来自当前装置，移植时须重新标定。本装置夹爪约定为 **0 = 夹紧，255 = 张开**。Windows 真机入口要求显式 live 参数、现场令牌、场景复位及设备锁；具体步骤见接口说明。配置文件不含本机口令，`COMPUTER.passcode` 已替换为占位符；旧 Linux 串口权限工具如需使用，须在本地配置。访问令牌通过环境变量注入。

## 源码范围与来源

本仓库以 2026-10-12 工作镜像为来源，包含截至 10 月 5 日的执行网关改动、原子技能、相关底层控制、遥操作/采集支撑模块、SDK 和测试。上传目录为独立快照；模型训练目录、模型权重、数据集、现场运行日志、数据库、缓存及原 Git 历史未纳入。

基础控制工程来自 Dobot X-Trainer，原始说明见 [Dobot README](docs/upstream/README.dobot.md)。保留 [LICENSE](LICENSE)、[第三方说明](THIRD-PARTY-LICENSES)及 SDK 各自许可证。
