> 2026-10-27 发布说明：用户已确认当前本机版本调试成功；当前提交的 INI 两侧 absolute_mapping_verified 均为 1，并保留新偏置。下文是截至 2026-10-25 的历史正文，其中待调试、verified=0、未更新偏置等状态已被本次用户确认及当前配置取代。本次发布未操作硬件。详情见 [当前版本说明](../docs/windows-teleop.md)。

# 新对话交接：Nova 2 主从映射现场核验

更新：2026-10-25（补充标定手册入口；控制代码及 INI 没有在本次文档更新中改变）。本文件针对当前本地修改后的代码；不要重新安装、重新克隆或覆盖当前工作树。

**最新操作入口：** 主手册第 7.1.1～7.1.10 现已详细说明“以对应 Nova 的实际 J1～J6 为参考”的标定流程，包含独立主手摆位、逐轴方向核对、从臂实测角与只读 raw 数据计算偏置、配置保存、第二姿态和重启复测。现有 `--reference-positioned` 仅使用 INI start_joints，不查询从臂；新流程改用文档内离线计算命令接收 DobotStudio 实测角。未做现场采样、未写新偏置，两侧 verified 仍为 0。下文 10 月 23 日代码校验清单中的文档哈希可能已变，代码哈希仍可作当时检查点。

## 可直接发送给新对话

> 工作区是 E:\Dobot，仓库 E:\Dobot\dobot-atomic-skills。请先读 E:\Dobot\manuals\HANDOFF_NEXT_CHAT.zh-CN.md、MAPPING_AUDIT_FIX_20261023.zh-CN.md 和主手册第 7.1 节。当前已能主从跟随，但左从臂绝对对齐曾向外运动到 Y 边界。不要修改安全边界或先提高限速，先核验实体主手与从臂的关节顺序、零位、方向和对应关系。两侧 absolute_mapping_verified 当前为 0；只读诊断工具已准备，禁止凭任意姿态重算偏置或直接改成 1。请从当前代码继续，保存现场采样与结论，完成后更新手册和修复日志。

## 1. 用户目标和约束

- Windows 本机，只运行 GitHub Python 遥操；不需要官方 X-Trainer UI、训练环境、CUDA 或新装 Conda。
- 长按后先以 0.15 rad/s 缓慢对齐主手映射的绝对六轴角，随后正常绝对跟随（当前上限 4 rad/s，仍受保护）。用户不要每次用当前位置重置相对偏置。
- **不要修改安全边界。** 先排查主从对应关系；若排除，再查目标本身和模型/安装/工具问题。
- 助手本轮未操作硬件。启动 A 会使能机器人并低速依次动作左/右夹爪，启动 B 会操作主手/初始化，均不是只读检查。禁止用启动入口当普通测试。

## 2. 当前现场/环境

| 项目 | 已知信息与证据限制 |
| --- | --- |
| Python | E:\Dobot\.venv\Scripts\python.exe，3.11；不要切到 PowerShell 的 base 环境解释器 |
| 仓库 | E:\Dobot\dobot-atomic-skills；基线 c402e9c1242e18eee87369accb140253bca495ac，存在多轮未提交改动，不得 reset/clean/pull 覆盖 |
| Nova | 代码左 .1=192.168.5.1，右 .2=192.168.5.2；前期查到两柜 Nova 2 / 3.5.8.5；实体绑定应追线复核 |
| 主手 hand=1 | HAND_RIGHT，COM5，2 Mbps，六轴 ID11,12,14,15,16,17，append13、grip18；用户本轮实体左主手对应这一按键事件 → 左 Nova 六轴 |
| 主手 hand=0 | HAND_LEFT，COM6，2 Mbps，六轴 ID1,2,4,5,6,7，append3、grip8 → 右 Nova 六轴；实体侧未独立确认 |
| 夹爪 | GRIPPER_LEFT COM3 ID21，GRIPPER_RIGHT COM4 ID22；实体左右待核对 |
| 相机 | 仅顶部 D405 218622278417，15 fps；30 fps 曾间歇超时；侧相机/三视角采集不用 |
| 进度 | 用户已跑通 9.1/9.2，短按/长按事件及主从运动曾工作；本轮代码尚未进行真机复测 |
| 现在的停点 | 两侧 absolute_mapping_verified=0，长按会拒绝自动对齐；需现场映射证据，不是新安装故障 |

配置文件：`E:\Dobot\dobot-atomic-skills\scripts\dobot_config\dobot_settings.ini`。旧 offsets/signs 保留原值。

## 3. 问题证据与诊断边界

用户 18:27:40 日志 hand=1，18:27:41 ALIGN_START flags=[1,0]，18:27:46 因 `left_left.y≈-157.615 mm` 越过 -160 mm 边界而 SERVO_REJECTED。Z 速度约 +0.033/+0.047 m/s，因此本次不是向下速度超限。重试重复同目标，主循环仍运行。

实际代码映射：`raw/2048*pi`（有符号位置）→ `(raw_rad-offset)*sign` → `master_to_nova_action` 交换两组。主手读取还使用 alpha=0.99 的平滑，静止后与原始映射值接近。当前 run_control 构造 DobotAgent 时没有传 start_joints，所以 DynamixelRobot 中可选的启动 2π 偏置调整分支未启用，不能把该分支当本次根因。参考角理论上与各自 Nova home 匹配；这只能证明配置意图，**不能证明实体主手在参考姿态时真的输出这些值**。没有同姿态采样、逐关节方向或厂家标定证据，不能宣布“偏置错”是已确认根因，也不能猜着互换左右/翻转符号。

最新修复记录含重试角度和 FK 离线复现。边界和阈值未改；`dobot_control/teleop_safety.py` 本轮无 diff。

## 4. 从这里开始下一步

1. 先确认 A/B 是否还在运行、设备是否静止。让用户按主手册退出；不要盲目终止进程、清锁或自动启动新控制进程。旧锁按主手册 11.2 核验归档，不改锁路径绕过。
2. 追线确认 COM5/COM6 实体侧、关节 ID 顺序、Nova IP 实体侧。统一用“实体侧 / 配置段 / COM / hand索引 / Nova侧”记录，不能只写“左”。
3. 主手有支撑、停止遥操后运行只读诊断；本工具不会解锁，需要现场用已确认方式摆放，不强扳锁定关节：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m scripts.2_get_offset --section HAND_RIGHT
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m scripts.2_get_offset --section HAND_LEFT
```

每次生成独立 JSON：`E:\Dobot\manuals\equipment\master-mapping-*.json`，含时间、INI SHA256、raw_rad、mapped_deg、reference_deg、ID、目标 Nova 侧。当前没有本轮真实采样报告，只有离线测试输出，不能混淆。

4. 获取厂家参考姿态依据，或记录已独立确认的实体六轴角度；在相同参考姿态比较映射值，并逐关节小幅改变、停稳重采，验证关节顺序及正方向。若无法确认参考姿态，需要用户/厂家提供依据，不能假定“看起来一样”或“放在当前位置”就是零位。
5. 只有确认处于 INI start_joints 对应姿态后，才可加 `--reference-positioned` 输出 candidate_offsets_rad。该工具只产生候选，不改配置。候选数学回代成功不能证明真实方向。
6. 有证据再备份并审查修改 INI；复采参考姿态和另一姿态验证。记录验收来源后，仅将已通过侧 absolute_mapping_verified 设 1；未验证侧保持 0。不要改 home、夹爪、工具或边界作为补偿。
7. 若对应关系通过，继续核查目标是否真实超出工作区、FK 的基坐标/安装方向/工具几何。当前 gripper 观测含固定占位值，不能宣称 FK 工具点等于真实工具 TCP；查证后再做针对性修正。
8. 映射和路径确认后由现场人员按 9.1→9.2 小范围验收。新代码对齐前采样检查；ALIGN_REJECTED_UNSAFE 会记录首个越界点并停止双侧跟随。不要反复长按或扩大边界。首次保持主手静止到 ALIGN_DONE，再单关节/单侧、停止重开，最后双侧。

## 5. 关键代码与本轮变更

以下路径均相对 `E:\Dobot\dobot-atomic-skills`：

| 文件 | 用途 |
| --- | --- |
| experiments/run_control.py | 按钮、主循环、映射验收门控、validate_absolute_alignment、慢速/正常跟随、TCP 步长约束、拒绝后复位 |
| dobot_control/absolute_teleop.py | 纯计算绝对规划器与 CommandInterval；不自动取最近 2π 等价角；目标可随主手更新 |
| scripts/manipulate_utils.py | INI 加载、六轴/夹爪映射、初始化 home |
| dobot_control/robots/dynamixel.py | 主手 raw→角度的 offsets/signs；搜索 get_joint_state 确认当前版本 |
| dobot_control/dynamixel/driver.py | 原始读取、主手按钮与力矩/PID；构造该驱动会写硬件 |
| scripts/2_get_offset.py | 新的只读 READ 140 诊断及偏置候选；不写 INI、不写电机 |
| dobot_control/teleop_safety.py | 原有空间、向下速度与突变保护；本轮未修改边界 |
| dobot_control/robots/dobot.py | 从臂 GetAngle、工具/使能及夹爪启动，真实构造有动作 |
| experiments/launch_nodes.py | 终端 A，设备锁和双臂服务 |
| task_skills/device_lease.py | 设备独占锁；退出后可能保留，需核验状态后归档 |
| scripts/logs/teleop_runtime.log | 现有运行日志，后续保留新日志；不要无故清空 |

前期还修过按钮无效样本形状/线程故障、cycle_elapsed、参考更新、串口短回包、单相机/15 fps、夹爪低速左后右启动。不要回滚这些累积修改。本轮两种拒绝分支都关闭双侧跟随并重锁主手，避免按钮计数复位后另一侧仍继续运行。

## 6. 离线验证与检查点

81 项离线测试通过；测试输出中的故障警告来自模拟注入。运行命令（硬件接口均模拟，不运行真实 A/B）：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m unittest tests.test_master_calibration tests.test_absolute_teleop tests.test_servo_enable_cycle tests.test_button_monitor_safety tests.test_camera_selection tests.test_feetech_reply_validation tests.test_gripper_startup tests.test_gripper_diagnostics tests.test_teleop_regressions
```

测试输出：`E:\Dobot\manuals\equipment\mapping-fix-tests-20261023.txt`。
本轮代码检查点清单：`E:\Dobot\manuals\equipment\mapping-code-checkpoint-20261023.json`（路径、SHA256、基线及工作树状态）。先核对文件是否在后续对话又被修改，不要把此清单当已提交 Git 版本。没有承诺所有未来真机错误均不存在。

主手册：`E:\Dobot\TELEOP_WINDOWS_SETUP.zh-CN.md`；本轮记录：`E:\Dobot\manuals\MAPPING_AUDIT_FIX_20261023.zh-CN.md`。后续每次改动写明依据、离线验证和现场未验收项；更新本交接文件，使下一次不用重做历史排查。
