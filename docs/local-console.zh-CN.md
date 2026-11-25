# 本地 ACT 采集控制台

更新于 2026-11-12。双击仓库根目录 `open_local_console.cmd`，自动打开浏览器。保留弹出的网页服务终端。

本机桌面已添加“Nova2控制台”快捷方式，使用 `tools/open_console.py`。双击后复用已运行的控制台，或在后台启动网页服务，再打开浏览器。机械臂图标取自 [Lucide robot-arm](https://github.com/lucide-icons/lucide/blob/main/icons/robot-arm.svg)，配色版 SVG、Windows ICO 及许可证保存在 `tools/console_web/icons`。

网页地址为 <http://127.0.0.1:8765>。手动启动可使用以下命令。

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B tools\local_console.py
```

端口被占用时，可先打开上述地址核对已有控制台，或通过 `--port 8766` 启动另一个端口。每次操作仅使用一个控制台。

## ACT 采集流程

1. 默认项目为当前 ACT 手册采用的 `tube_insert_cup_2cam_rgb_marked_v2`，录制目录为 `E:\Dobot\datasets\tube_insert_cup_2cam_rgb_marked_v2\collect_data`。可填写项目名后点“更新指令”，所有命令统一使用该项目。页面打开后只读取配置。
2. 如遇旧锁，停止设备、退出 A/B 和其他控制程序，点“归档旧锁”。操作会核对本机持锁 PID 已退出，将锁目录重命名为带时间戳的归档目录，保留 scene-invalid 标记。持锁进程仍运行、所有者缺失或来自其他主机时，页面报告失败，需要按原手册 11.2 核对。
3. 清空夹爪活动范围，点“启动终端 A”。该入口沿用服务原有使能和夹爪初始化动作。在新 PowerShell 中等待 `Starting robot server on port 6001`。
4. 点“启动终端 B”。A/B 均设置 `DOBOT_ACT_RECORDING=1`，B 清除旧录制队列大小覆盖及自动归位授权，使用所选项目名。相机读取当前 INI，当前为 top、left 两路，640×480、15 fps、RGB-only。
5. 需要归位时，先停录、关闭两侧跟随，Ctrl+C 退出 B。A 可继续运行，点击“通过 A 双臂归位”，确认整条回程路径空闲后执行。该入口复用 A，调用原 `robot_pose_init()`，发送结束后需现场确认两臂姿态。完成后再启动 B。
6. 录制按主手 B 键开始/结束。先结束录制并等待落盘，再关闭跟随。长按 A 超过 1 秒切换该侧跟随。

每张指令卡可“复制指令”，也可展开查看完整 PowerShell 内容。执行按钮使用同一份生成内容，限定为页面列出的操作。A/B 和相机预览在独立可见 PowerShell 中运行，页面反馈“窗口已打开”后，以对应终端日志判断是否成功。其他指令在执行结束后向页面返回结果。遇到原生 Python 退出错误时，页面显示失败。

## 夹爪与单臂归位

2026-11-17 新增“A 运行时恢复夹爪”。先停止录制、关闭跟随并退出 B 和 ACT 执行程序，保持新版 A 运行。在夹爪控制区勾选该选项，再选择左/右夹爪并点全开或全闭。操作通过 A 的 6001 服务复用串口，速度 10，全开 255、全闭 0，读取反馈确认到位。B 运行或 ACT 会话占用时拒绝操作。失败显示具体原因，操作结束后 A 继续提供服务。

首次使用此功能需正常退出并重启 A，加载新增的 recover_gripper 服务接口，同时重启网页服务。若网页服务是后台运行的，可在 PowerShell 用下列命令只结束控制台服务，再双击桌面“Nova2控制台”。

```powershell
Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'local_console\.py'
} | ForEach-Object { Stop-Process -Id $_.ProcessId }
```

使用原有独立控制模式时取消勾选“A 运行时恢复夹爪”，按下面的流程操作。

退出 A/B 后，选择左/右夹爪，再点全开或全闭。全开 255，全闭 0，速度 10。串口和 ID 读取当前 INI，当前实体左为 COM3 / ID21，实体右为 COM4 / ID22。

“左臂归位”和“右臂归位”独立控制所选臂。先确认回程路径空闲、夹爪无持物。实体左 .2、实体右 .1，目标 home 读取 `task_skills/layout.py`。沿用原窗口的路径预检、0.15 rad/s 插值、到位读回和设备锁。归位时可点“停止归位”，停止确认失败会保留设备锁。停止响应取决于当前网络调用返回，异常运动按现场急停流程处理。

页面操作互斥，动作期间其他执行按钮禁用。关闭浏览器后网页服务与已打开的 A/B 仍继续运行。停止录制、关闭跟随并 Ctrl+C 退出 B 后，再 Ctrl+C 退出 A，最后在网页服务终端 Ctrl+C 退出。详细控制条件见 `docs/gripper-panel.zh-CN.md` 和 ACT 采集手册。

## 数据检查与导出

填写 episode 目录名并更新指令后，会出现“检查 episode”和“生成回看视频”。目录名指 collect_data 下的单条目录名称。预览生成 preview.mp4，已存在时会报错。

“导出 ACT 数据集”在退出 B 后执行，使用 15 Hz，输出当前项目的 act_v1。输出目录需要尚未存在，重复导出可复制命令后将输出改为 act_v2 等新目录。仅 success 且质量合格的 episode 进入导出。标注方式沿用 `manuals/ACT_DEMONSTRATION_COLLECTION.zh-CN.md` 第 5 节。

## 实现与验证

后端使用 Python 标准库 HTTP 服务，仅监听 127.0.0.1。浏览器请求验证本机 Host、Origin 和会话令牌。页面资产全部位于 `tools/console_web`，无需额外安装前端包。日志位于仓库 `.local/operator-console.log`。设备动作复用 `tools/gripper_panel.py` 和 `tools/arm_homing.py`。

```powershell
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m unittest tests.test_local_console tests.test_arm_homing tests.test_gripper_panel tests.test_gripper_diagnostics
```
