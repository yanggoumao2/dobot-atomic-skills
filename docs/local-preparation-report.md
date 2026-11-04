# 本机遥操准备记录

> 这是目录迁移前的历史准备记录。仓库现已移至 `E:\Dobot\dobot-atomic-skills`，解释器仍为 `E:\Dobot\.venv\Scripts\python.exe`；当前操作入口见 `E:\Dobot\TELEOP_WINDOWS_SETUP.zh-CN.md`，最新环境检查见 `E:\Dobot\manuals\LOCAL_ENVIRONMENT.md`。下面的旧工作目录用于保留当时的记录，不作为新命令路径。

日期：2026-10-20（北京时间）。工作目录：`E:\Dobot`。

## 源码基线

- Origin：`https://github.com/yanggoumao2/dobot-atomic-skills.git`
- Branch：`main`
- Commit：`c402e9c1242e18eee87369accb140253bca495ac`
- 提交日期：2026-10-15
- 克隆前目录为空。未创建提交，未推送远端。

## 软件环境

- Git：2.54.0.windows.1。
- 原全局 Python：`C:\miniconda3\python.exe`，3.13.12。
- 本次基础解释器：`C:\Users\Lenovo\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe`。
- 本次项目解释器：`E:\Dobot\.venv\Scripts\python.exe`，3.11.16，64 位。
- 安装文件：根目录 `requirements-teleop-windows.txt`。
- 精确依赖快照：根目录 `requirements-teleop-windows.lock.txt`。
- `.local/` 保存临时目录与安装缓存，已加入 `.gitignore`。

初次 `venv` 的 ensurepip 阶段失败；把当前进程 TEMP/TMP 指向项目 `.local/tmp` 后，执行 `python -m ensurepip --upgrade --default-pip` 成功。依赖最终安装成功。

## 已执行验证

| 检查 | 结果与边界 |
| --- | --- |
| `python -m pip check` | `No broken requirements found.` |
| 核心库与 `tkinter` 导入 | 成功 |
| `experiments.launch_nodes`、`experiments.run_control` 导入 | 成功；未调用 main 或驱动构造函数 |
| 两个入口的 `--help` | 成功；端口默认 6001、主机默认 127.0.0.1 |
| `test_teleop_regressions.py` | 20 项通过 |
| `test_gripper_diagnostics.py` | 4 项通过 |
| INI 检查 | 四路串口仍为 Linux `/dev/ttyUSB*`，Windows 配置待现场填写 |
| 串口枚举 | 本次进程返回空列表 |
| RealSense 设备枚举 | 本次进程返回空列表 |

设备枚举未打开串口、未启动相机流、未复位设备。空列表不是对设备物理缺失的最终判定，仍可能与未连接、驱动或工具运行权限有关。

## 本次没有执行的动作

未更改网卡或控制柜 IP，未连接控制柜 TCP/HTTP 服务，未使能或运动实体机器人，未写入舵机设置，未进行相机取流或现场标定。上游运动源码和 `dobot_settings.ini` 保持克隆版本。

因此当前状态是**软件环境已准备、离线检查通过，现场硬件遥操待验证**。按根目录 `TELEOP_WINDOWS_SETUP.zh-CN.md` 的顺序继续。
