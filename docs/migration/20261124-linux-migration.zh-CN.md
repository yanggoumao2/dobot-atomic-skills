# Linux 迁移记录与模拟测试

2026-11-24。当前工作目录 E:\Dobot\dobot-atomic-skills，原分支 ACT_Train。迁移基于本地已有 ACT/夹爪修复，保留原有未提交工作。

## 本次改动

1. 新增 Linux 配置模板和统一 DOBOT_CONFIG；按用户确认填入右 .1、左 .2，主机沿用 .100；串口、相机留空，主手标定核验标记为 0。
2. A/B 启动前检查配置完整性；相机、夹爪、主手、校准工具、采集元数据统一读取所选配置。
3. layout/device_lease 从 task_skills 迁入 dobot_control；统一设备锁默认目录。
4. Linux 控制台 Bash 启动、Linux /proc 检查、旧锁归档、A 内夹爪恢复、显式服务归位；Windows PowerShell 入口保留。
5. 串口发现改只读，取消自动写舵机/自动写配置和 sudo 密码 chmod；FTDI 使用串口真实路径解析，权限失败给出提示。
6. 新增独立 Linux 遥操 venv 安装/激活脚本，使用 Python3.10；训练沿用原 ACT CUDA 环境。
7. 新增通用导出数据训练归一化入口、Linux 新人手册、项目说明，重写 README。
8. 精简默认 requirements.txt，移除不属于遥操/采集的仿真、ADB、ROS 等全量依赖；新增 .gitattributes 固定 shell 脚本 LF 换行。
9. 修正标定报告及 Windows 按钮手册中遗留的左右侧别说明，与当前主手路由一致。
10. 删除独立原子任务、DAG、网关、化学/烧杯流程、上传脚本及专属测试/文档。完整逐文件列表见 removed-files-20261124.txt。现场 artifacts、原始数据、模型与历史运行记录保持原位置。

## 一次模拟测试

命令：`E:/Dobot/.venv/Scripts/python.exe -B tools/simulate_linux.py`。
结果：**145 tests，12.502 秒，OK**。
日志：`E:/Dobot/linux-simulation-20261124.log`。

运行宿主为 Windows，当前机器没有已安装的 WSL Linux 环境。Linux 专项用临时 /dev/serial/by-id 配置、sys.platform=linux 分支、模拟 /proc 和进程探测验证；共享回归使用机器人、串口、相机替身。测试涵盖空配置在建连前拒绝、重复端口、Linux 命令生成、进程发现、恢复互斥、旧锁归档，以及遥操映射/保护、夹爪、相机、录制、ACT 数据转换、归位和 Dashboard 解析。

该次运行验证 Python 逻辑与平台分支；Ubuntu 二进制依赖安装、桌面窗口、USB 权限与真实控制周期由现场部署步骤验证。本次未运行机器人运动或 ACT 训练。

## 接手顺序

阅读 manuals/TELEOP_LINUX_SETUP.zh-CN.md → 复制本次源码到主机 → setup → 接线/填写 USB 和相机身份 → 现场主手映射复核 → A/B → 短示范回看 → 正式采集。GitHub 发布状态以最终交付说明为准。
