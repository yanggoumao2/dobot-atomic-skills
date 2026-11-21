# Windows 本机双臂遥操部署与操作手册

更新：2026-11-05（北京时间）。对照源码：分支 `windows_teleop`，以 `a97d247` 为基线并包含本地的主手串口恢复、工作区 X 平移及 J6 参考角修正。手册更新与离线修复验证未启动实体控制程序或改动 INI。

## 0. 当前状态与使用入口

2026-11-05 状态：用户确认最新版代码已经跑通。实际录制根目录为 `E:\Dobot\datasets\dataset_package_test\collect_data`，已发现三段同名 JPG/PKL 配对录制（3046、465、985 对）。本页原有待验收说明保留作逐项记录边界，不代表用户本次确认被撤销。

用户已确认 10 月版本的遥操可以成功运行。11 月 2 日版本已修改工作区、跟随限速和录制，清理前 90 项相关离线测试通过；当前 X 平移范围、J6 修正及真机录制仍待现场验收，不能把旧版本调通等同于本轮真机验收。

| 项目 | 当前版本 |
| --- | --- |
| 工作区 / 仓库 / Python | `E:\Dobot` / `E:\Dobot\dobot-atomic-skills` / `E:\Dobot\.venv\Scripts\python.exe` |
| 代码仓库 | `https://github.com/HYJn14/dobot-atomic-skills`，分支 `windows_teleop`；原上游 `yanggoumao2` 保留作来源 |
| 主手验证标志 | INI 两侧 `absolute_mapping_verified = 1`；已有新偏置，日常启动不要重置或重算 |
| 正常跟随 / 慢速对齐 | 默认 `3.2 rad/s` / `0.15 rad/s` |
| 工作区 | 实体右臂 .1（代码 left）X `[-300,450]`、实体左臂 .2（代码 right）X `[-400,300]`、两侧 Y `[-750,-160]`、Z **严格大于 40**，单位 mm |
| 工作区变更含义 | 撤销 Y 平移；.1 的 X 加 150 mm，.2 的 X 减 150 mm，区间宽度不变；允许区平移本身不自动移动机器人 |
| 启用相机 | 仅顶部 D405 `218622278417`，640×480 RGB/Depth，15 fps |
| 录制 | 单相机可录；已开启跟随后，任一 B 完整按下并释放开始，再按一次停止 |
| 输出根目录 | `E:\Dobot\datasets\dataset_package_test\collect_data` |
| 测试脚本清理 | 9 份本地调试测试已移出；上游回归测试和环境检查工具保留 |

**实体与代码通道：** 用户已根据坐标图确认实体右臂为 .1（代码名 left），朝操作者为该臂 +X；实体左臂 .2（代码名 right）方向相反。现已按此修正 X 并恢复原 Y；不交换 IP 或通道名称。另据原始标定记录修正 .1 对应 HAND_RIGHT 的 J6 参考角符号，解决 ALIGN_START 近半圈旋转，详情见 `docs/workspace-x-wrist-correction-20261102.zh-CN.md`。现场仍需复测范围和姿态。

日常操作：核对第 6 节配置和第 7.2 节姿态 → 按第 9 节先 A 后 B 启动 → 第 10.1 节跟随 → 第 10.2 节录制 → 第 11 节退出。**第 7.1 节仅供换设备、改装或发现映射异常后的重新标定，不是每次开机的必做步骤。**

本手册以仓库内版本为维护源，工作区根目录副本同步更新。路径、设备参数与标定值只对应当前装置，迁移机器时须重新核对。历史交接文档与旧日志只供追溯，不能覆盖本文当前状态。

## 1. 软件分工与资料

- Windows 本机运行仓库的 Python 遥操，不要求官方 X-Trainer 图形 UI、Ubuntu、训练环境、CUDA 或新装 Conda。
- DobotStudio Pro 用于控制柜配置、报警处理和读取关节角；进入 Python 遥操前结束其控制连接，不同时发送运动命令。
- 本机曾登记 DobotStudio Pro `2.8.3-x86-64`，路径 `E:\Dobot\apps\DobotStudioPro\DobotStudio Pro\DobotStudio Pro.exe`。这是历史安装记录，不是本次重新检查安装的结果。
- 两柜历史 HTTP 读取为 Nova 2 / V3.5.8.5；当前源码要求 V3，版本不低于 3.5.8.1 且低于 V4。不要因软件版本更新而顺便升级控制柜固件。
- 官方资料在 `E:\Dobot\manuals`：`XTrainer-Light-V1.3-20240729-cn.pdf`、`XTrainer-UI-V1.1-20260818-cn.pdf`、`XTrainer-FAQ-cn.pdf`、`DobotStudio-Pro-2.8-CR-Nova-cn.pdf`。这些 PDF、设备照片和采样记录不是当前 Git 仓库的完整交付内容。

## 2. 代码与依赖准备（已有本机环境不需重装）

本机代码位于仓库子目录，`.venv` 位于工作区根目录。不要在已有目录重新克隆或拉取覆盖未提交修改。

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
git status --short --branch
git remote -v
git rev-parse HEAD
E:\Dobot\.venv\Scripts\python.exe --version
E:\Dobot\.venv\Scripts\python.exe -m pip check
```

当前依赖以仓库中的 `requirements-teleop-windows.lock.txt` 为精确版本快照；`requirements-teleop-windows.txt` 为遥操依赖子集。确需恢复依赖时：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -m pip install -r requirements-teleop-windows.lock.txt
E:\Dobot\.venv\Scripts\python.exe -m pip check
```

全新机器才按以下布局准备代码和 Python 3.11 x64 环境，再填写本机硬件配置；已有本机不要重复执行：

```powershell
git clone --branch windows_teleop https://github.com/HYJn14/dobot-atomic-skills.git E:\Dobot\dobot-atomic-skills
# 用本机真实的 Python 3.11 x64 路径替换下一行路径：
& '实际Python3.11目录\python.exe' -m venv E:\Dobot\.venv
```

仓库中的源码、Windows 依赖和本手册已提交过；`E:\Dobot\tools\setup-teleop.ps1` 等工作区级脚本、虚拟环境和官方 PDF 不随克隆自动获得。不要把虚拟环境目录直接搬到其他电脑。

## 3. 软件检查与保留的回归入口

环境或依赖变化后可执行以下软件检查；不构造真实驱动或发送机器人命令：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B tools\check_teleop_local.py
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B experiments\launch_nodes.py --help
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B experiments\run_control.py --help
```

环境检查退出 0 表示软件/有限静态检查通过，1 表示环境失败，2 表示配置待处理。工具中的旧输出标签 `three-view recording enabled` 实际取当前 `recording_enabled`；单相机下为 True 表示已启用视角可录，不代表有三台相机。

保留的相关回归命令（只在需要回归时执行，不需每次启动运行）：

```powershell
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m unittest tests.test_teleop_regressions tests.test_gripper_diagnostics
```

9 份本地调试测试已按用户要求移出，旧文档中的 `tests.test_absolute_teleop` 等命令不再可用。清理前 90 项测试通过的历史记录见 `docs/teleop-fixes-20261102.zh-CN.md`；本次手册更新只做文档与源码核对，没有重跑硬件或功能测试。

## 4. 接线与设备枚举

保持两臂、夹爪及主手有可靠支撑，急停可用，清空运动区域。确认四路目标串口和顶部相机的实体标签，不按枚举顺序推断左右。

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B tools\check_teleop_local.py --inventory
```

`--inventory` 仅枚举串口和 RealSense，不打开串口、不启动图像流、不复位设备。它不证明接线、实体左右、标定或运动正确。Viewer、串口调试工具与遥操不要同时占用设备。

## 5. 控制网络与控制柜

| 设备/端口 | 配置或用途 |
| --- | --- |
| 代码左 Nova / 代码右 Nova | `192.168.5.1` / `192.168.5.2` |
| 电脑有线网卡 | 当前工作网段为 `192.168.5.x`；例如未占用的 `192.168.5.10/24`，不得与 .1/.2 重复 |
| 29999 / 30003 / 30004 | Dashboard / 运动（含 ServoJ）/ 状态反馈 |
| 22000 | HTTP 固件与型号查询 |
| 127.0.0.1:6001 | 本机 ZMQ 服务，两个终端一致，不是控制柜端口 |

按匹配控制柜固件的 DobotStudio 手册设置 TCP/IP 二次开发模式；两柜不能使用相同 IP。如需改 IP，先隔离设置并记录原值，不能只改 `task_skills/layout.py`：`dobot.py` 的夹爪绑定、`run_control.py` 的查询地址也使用该网段。

连接诊断可使用 `Test-NetConnection 192.168.5.1 -Port 29999` 等；TCP 可达不等于协议、报警或运动链路通过。保持 Python 服务仅监听 `127.0.0.1`。

## 6. 当前现场配置与路由

配置文件：`E:\Dobot\dobot-atomic-skills\scripts\dobot_config\dobot_settings.ini`。运行中不改配置；更改前停止 A/B 并备份。

| 输入/设备 | 当前配置 | 六轴目标 / 夹爪通道 |
| --- | --- | --- |
| HAND_LEFT / hand=0 | COM6，2 Mbps，J1～J6 IDs `1,2,4,5,6,7`，append3，grip8，verified=1 | 六轴 → 右 Nova .2；夹爪 → .1 通道 |
| HAND_RIGHT / hand=1 | COM5，2 Mbps，J1～J6 IDs `11,12,14,15,16,17`，append13，grip18，verified=1 | 六轴 → 左 Nova .1；夹爪 → .2 通道 |
| GRIPPER_LEFT | COM3，ID21，行程 `2048,3052` | 绑定 .1 |
| GRIPPER_RIGHT | COM4，ID22，行程 `2048,3052` | 绑定 .2 |
| CAMERA | enabled=top，fps=15，top=218622278417 | 侧相机配置不启用，不要求接入 |

配置名称 LEFT/RIGHT 不等于实体左右。INI 的 `HAND_RIGHT.physical_side=left`；`HAND_LEFT.physical_side=unverified` 是尚未补全的标签字段，不能把 verified=1 推导为该标签已独立核验。夹爪通道也需用实物接线确认，不从六轴交叉路由推断。

当前 offsets（rad）原样保留，不作为重新标定的抄写模板：

```ini
[HAND_LEFT]
joint_offsets = 3.97528188, 3.93619470, 3.10403946, 3.84568984, 3.74518476, 3.21602006
joint_signs = 1, 1, -1, -1, -1, 1
[HAND_RIGHT]
joint_offsets = 0.76471876, 3.87023353, 3.16534002, 3.88864130, 4.12106872, 3.12238856
joint_signs = 1, 1, -1, -1, -1, 1
```

`scripts.1_find_port` 会构造驱动、写力矩/PID 等并更新 INI，不是普通只读扫描；日常启动无需运行。Windows 不需填写 `COMPUTER.passcode`。

## 7. 标定维护与启动姿态

### 7.1 仅在需要重新标定时：以对应从臂 J1～J6 为参考

当前两侧已有偏置且 verified=1。仅在换装、方向/零位异常或现场证据要求重新标定时，完整执行 7.1.1～7.1.10。先处理 COM5→.1，再处理 COM6→.2；不互抄偏置、不凭任意姿态计算偏置。以下重标定流程中的置 0、保存候选、重新验收只适用于本次改动侧。

#### 7.1.1 标定目标：同一个关节姿态应输出相同的 J1～J6

你要求的“主手按和从臂相同的坐标标定”，在当前程序中落实为：

```text
主手与对应从臂处于经核对的相同六轴关节姿态时：
主手标定后的 [J1,J2,J3,J4,J5,J6] = 从臂实际反馈的 [J1,J2,J3,J4,J5,J6]
主手某轴沿从臂该轴的正方向变化 +Δ°，输出也应变化 +Δ°。
```

以 **DobotStudio 的关节坐标 J1～J6（单位 °）** 为参考，不用 TCP 的 X/Y/Z/Rx/Ry/Rz，不改控制柜的机械零点，不用重新创建用户坐标系或工具坐标系。主手编码器原始读数与 Nova 角度不相等是正常现象，偏置负责换算。两个设备的底座位置、连杆长度不同，“同关节角”也不表示末端在工作台上占据同一个 XYZ 点。

当前代码关系是：

```text
q_master_rad = (raw_rad - joint_offsets) * joint_signs
joint_offsets = raw_rad_at_reference - q_slave_reference_rad * joint_signs
q_slave_reference_rad = DobotStudio实际关节角(°) × π / 180
```

因此必须先把主手各关节真正摆到与参考角对应的姿态，再计算偏置。把主手随意停在某个位置后强行赋予从臂的数值，只是人为重定义零位，不能达到你要求的实体姿态一致。一次同姿态只能确定偏置，还要逐轴和第二姿态检查方向、关节顺序及角度比例。

依据：官方轻量版手册印刷页 28～29（本地 PDF 阅读器第 33～34 页）也是“独立解锁主手 → 与从手初始姿态一致 → 锁定并退出摆位脚本 → 保存同步基准 → 启动遥操”的顺序。本文命令与偏置保存方式以本地改造后的代码为准，不直接执行官方旧文件名 `2_gello_get_offset.py`。本地资料：[X-Trainer 轻量版手册](E:/Dobot/manuals/XTrainer-Light-V1.3-20240729-cn.pdf)、[DobotStudio Pro 手册](E:/Dobot/manuals/DobotStudio-Pro-2.8-CR-Nova-cn.pdf)。

#### 7.1.2 先确认一对主从及六个关节

| 本轮处理对象 | 主手配置 / 按键日志 | 主手端口、六轴 ID（依次对应 J1～J6） | 对应从臂 |
| --- | --- | --- | --- |
| 第一对 | HAND_RIGHT / hand=1；用户本轮描述为实体左主手 | COM5；11,12,14,15,16,17 | 代码左 Nova，192.168.5.1 |
| 第二对 | HAND_LEFT / hand=0；实体侧仍需追线核对 | COM6；1,2,4,5,6,7 | 代码右 Nova，192.168.5.2 |

IDs13/3 是附加舵机，18/8 是主手夹爪输入；本次六轴标定不把它们插进 J1～J6 数组。每个 ID 是否确实对应这根物理关节，需要逐轴采样验证，不能只依据数字连续性判断。

给两主手和两 Nova 贴上“配置段/COM/目标 IP”标签。观察者面对两台设备的方向可能不同，不能用“看起来都是顺时针”判断同向；应按各自基座和关节轴的局部方向对照。J1 的基座朝向、J3 的肘部折叠方向、J4～J6 的腕部旋转也要对应，不能只让两只夹爪朝向相同。主手夹爪与 Nova 夹爪的实体绑定另做验收。

#### 7.1.3 停止遥操，备份配置，创建本次记录目录

1. 按第 11.1 节停止跟随、退出 B，再退出 A，确认机器人已静止。标定期间不启动 `launch_nodes.py` / `run_control.py`，也不要把“核验标志为 0”当成启动过程不会操作硬件的保证。
2. 主手和工具有可靠支撑；不强扳锁定的关节。不要拔电让无支撑主手自由下落。
3. 新开一个 PowerShell，作为**采样终端 C**，依次执行。后文 `$calDir`、`$calSection` 等变量都在这个窗口继续使用：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
$calIni = 'E:\Dobot\dobot-atomic-skills\scripts\dobot_config\dobot_settings.ini'
$calDir = Join-Path 'E:\Dobot\manuals\equipment' ('calibration-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $calDir | Out-Null
Copy-Item -LiteralPath $calIni -Destination (Join-Path $calDir 'dobot_settings.before.ini')
$calSection = 'HAND_RIGHT'
$calDir
```

4. 仅在确需重新标定时，将本次要改动侧的 `absolute_mapping_verified` 从 1 改为 0；另一侧保留已验证状态。改动前已有上述备份。不要改相机、夹爪、home 或安全边界。

```powershell
notepad.exe $calIni
```

保存 INI 使用 UTF-8 无 BOM（记事本“另存为”的 UTF-8，不选 UTF-8 带 BOM）。本次每项记录放在 `$calDir`，不要覆盖上次的采样文件。

#### 7.1.4 在 DobotStudio 中确定从臂参考姿态和实际角度

1. 保持 A/B 已退出。在 DobotStudio 连接当前这对的 IP，先确认连接的是哪台实体 Nova。
2. 打开关节点动/关节坐标区域，读取 **J1、J2、J3、J4、J5、J6**。不要抄旁边笛卡尔坐标区域的 XYZ/RxRyRz。
3. 优先使用本仓库已核对的 home 作为标定参考，以便随后启动检查一致。以下只是目标参考，不是要求从当前姿态直接自动跳转：

| 目标 Nova | J1 | J2 | J3 | J4 | J5 | J6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 左 .1 → 配 HAND_RIGHT / COM5 | 90° | 0° | 90° | 0° | −90° | −90° |
| 右 .2 → 配 HAND_LEFT / COM6 | −90° | 0° | −90° | 0° | 90° | 90° |

4. 现场人员先检查当前位置到参考姿态的可行路径，再用 DobotStudio 的低速关节点动/小步寸动逐段摆到参考姿态。观察并避开桌面、另一只臂和相机，不因目标表看起来安全就直接运行自动归位。若参考姿态或路径无法确认安全，停在这里，不修改边界来通过。
5. 到位并静止后，记录**实际反馈角度**和对应 IP、时间，保存关节面板截图。计算偏置时以本次实测六个值为准。例如显示 89.98° 就记 89.98°，不要当作精确 90°。
6. 参考记录存到 `$calDir`。当你摆放主手时，从臂保持这个姿态、不再点动；第二对另记一组角度。

本仓库 home 与官方原示例的左右角度相反，原因和对照见 7.2，不要混用。也不要把两个从臂都设成六个 0°；这不是当前布局的标定参考，且不满足当前 J3 方向检查。

如果确需用其他参考姿态，公式仍可使用该姿态实测角，但必须有独立依据能把主手摆到对应的六轴姿态；且该操作不会改变 9.2 的 home 初始化检查。不能把“任意参考姿态完成零位换算”误认为 home 和运动路径已经验收。

#### 7.1.5 独立解锁主手并摆成相同关节姿态

主手姿态未摆正之前，不运行偏置候选计算，不改成 `absolute_mapping_verified=1`。

本仓库保留官方流程用的 `scripts/3_just_buttonA.py`。经本次静态核对，它的 main 只创建主手对象、读取按键并设置主手力矩，**没有创建 Nova 客户端或发送从臂运动命令**；但构造主手驱动会对两只主手写力矩/PID，不是只读工具。它仍是旧辅助脚本，不含当前遥操入口完整的按钮故障处理，本轮未做其真机验收。若按键异常或线程报错，停止使用，不改用完整遥操来摆主手。

1. 让两只主手都有可靠支撑，再开**独立摆位终端 D**执行。它会打印该辅助进程 PID，请记录；此时不得另开读取同一串口的采样进程：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -c "import os,runpy; print('MASTER_POSE_PID=',os.getpid(),flush=True); runpy.run_module('scripts.3_just_buttonA',run_name='__main__')"
```

2. 双手支撑当前要摆的主手，短按并释放 A（小于 0.5 秒）。看日志对应 hand 索引出现 `unlock`，确认主手可轻柔拖动。不要同时按两边按钮；不要长按或按 B。
3. 逐关节对照从臂：基座朝向、肩/肘弯曲、腕部每个旋转轴都应与已确认参考姿态对应。小主手与大从臂只需关节姿态对应，不需把末端移到空间中同一点。利用厂家姿态图、机械参考标记/工装或可独立测量的角度确定位置。若 J4/J6 的旋转圈数或安装方向辨不清，记录问题并停下，不用“夹爪看起来差不多”代替。
4. 摆好后再次短按 A，确认对应 `lock`。保持支撑，确认实际锁定，再松手。另一只主手保持支撑和锁定。
5. 在 D 按 Ctrl+C 退出。旧脚本有非 daemon 按钮线程，Ctrl+C 后可能仍占用串口；不能以出现 KeyboardInterrupt 就判断已结束。**确认两主手均已锁定、有支撑**后，在 C 用下面命令核对刚才记录的 PID。先检查命令行确实是这次 `scripts.3_just_buttonA`，再结束这一个辅助进程；不要结束所有 Python：

```powershell
$poseProcessId = [int](Read-Host '输入本次 MASTER_POSE_PID')
Get-CimInstance Win32_Process -Filter "ProcessId = $poseProcessId" |
    Select-Object ProcessId,Name,CommandLine
# 仅在上面确认为本次摆位辅助进程且主手已锁定/有支撑时执行下一行：
Stop-Process -Id $poseProcessId -ErrorAction SilentlyContinue
Get-Process -Id $poseProcessId -ErrorAction SilentlyContinue
```

最后无进程结果、串口释放后才采样。若辅助程序因按钮读取等错误退出而不能重新锁定，保持人工/机械支撑，先处理故障，不强行采样或继续标定。进程停止本身不会保证主手锁定。

#### 7.1.6 逐轴验证关节顺序和正方向

这一步验证“同坐标”的方向约定；偏置无法补偿错误的符号或关节顺序。

1. 主手与从臂均在参考姿态，辅助程序已退出。在 C 采一份原始参考数据：

```powershell
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m scripts.2_get_offset --section $calSection --output (Join-Path $calDir ($calSection + '-direction-base.json'))
if ($LASTEXITCODE -ne 0) { throw '采样失败，停止标定并保留现场姿态' }
```

2. 在 DobotStudio 中，现场确认小范围路径安全后，只将从臂某一个关节向正方向寸动一个容易观察的小角度（如约 +2°，这是诊断示例，不是设备统一安全步长）。记录运动前后实际角度，确认其余轴不变，并记录实际正向。
3. 用 D 的独立摆位流程，只调整主手对应的一个关节，使它与从臂新的关节姿态对应。锁定、退出 D，再采该姿态，使用不同文件名，例如 `HAND_RIGHT-J1-positive.json`。J2～J6 逐项重复，每项结束后用受控方式恢复参考姿态；每次读取都在主手和从臂静止时进行。
4. 比较该轴原始读数变化与从臂反馈变化：

```text
Δraw = raw_after - raw_before                  （rad）
Δq_slave = (slave_after_deg - slave_before_deg) × π/180
sign = +1：Δraw 与 Δq_slave 同号
sign = -1：Δraw 与 Δq_slave 异号
角度比例检查：|Δraw| 应与 |Δq_slave| 在实际摆位精度内接近
```

5. 如果动 J2 却主要改变 J4 对应 ID，先查实际关节/ID 顺序，不用改 offset 掩盖。如果数值跳变近一整圈，或比例明显不接近 1，先查编码器模式、安装/传动和读取；不能仅通过翻转符号修复。采样间同一轴应没有无意跨圈。
6. 六轴记录建议填表：`J序号 / servo ID / 从臂前后角 / 主手前后raw / 推导sign / 方向确认方式`。当前两段 signs 都是 `1,1,-1,-1,-1,1`；重新标定时仍应以本次现场逐轴证据核对，不从该数列推断安装方向。只有反复实测证明某轴符号不符，才在本侧 INI 修改对应项，并保持核验标志为 0。
7. 如果需要改变 joint_ids 顺序，应同时重审与该顺序绑定的 offsets、signs、start_joints，不能只交换一列。此类结构不一致先交给调试人员处理，再回到第 1 项；当前手册的自动化步骤不猜测关节重排。

#### 7.1.7 在相同参考姿态采样，用从臂实测角计算候选偏置

1. 两者回到 7.1.4 已确认的同一参考姿态。从 DobotStudio 再读一次实际 J1～J6；确保从臂与主手从此刻到采样结束都不再被移动。方向核验或改 signs 后不要使用旧报告。
2. 锁定并完全退出 D，在 C 采最新数据（不要加 `--reference-positioned`）：

```powershell
$calReferenceFile = Join-Path $calDir ($calSection + '-reference.json')
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m scripts.2_get_offset --section $calSection --output $calReferenceFile
if ($LASTEXITCODE -ne 0) { throw '参考采样失败，不计算偏置' }
```

如果提示文件已存在，不要覆盖；创建新的本轮记录目录或换一个明确的新文件名，重新采样。

3. 下面命令仅做离线计算并保存候选，不连接硬件、不写 INI。提示输入时，粘贴**本次对应从臂实际显示的六个关节角**，按 J1～J6 顺序，用英文逗号分隔，不能带 J1= 等标签：

```powershell
$calReport = Get-Content -LiteralPath $calReferenceFile -Raw -Encoding UTF8 | ConvertFrom-Json
$calHands = @($calReport.hands | Where-Object { $_.section -eq $calSection })
if ($calHands.Count -ne 1) { throw '报告配置段不匹配' }
$calHand = $calHands[0]
$calCulture = [Globalization.CultureInfo]::InvariantCulture
$calReferenceText = Read-Host '输入对应Nova实际J1~J6（度），用英文逗号分隔'
$calParts = @($calReferenceText.Replace('，',',').Trim() -split '[,;\s]+' | Where-Object { $_ -ne '' })
if ($calParts.Count -ne 6) { throw '必须输入六个关节角，不能输入XYZ/RxRyRz' }
$calReferenceDeg = @($calParts | ForEach-Object { [double]::Parse($_,$calCulture) })
$calOffsets = @()
for ($j = 0; $j -lt 6; $j++) {
    if ([double]::IsNaN($calReferenceDeg[$j]) -or [double]::IsInfinity($calReferenceDeg[$j])) { throw '角度必须有限' }
    $calOffsets += [double]$calHand.raw_rad[$j] - ($calReferenceDeg[$j] * [Math]::PI / 180.0) * [double]$calHand.signs[$j]
}
$calOffsetLine = ($calOffsets | ForEach-Object { $_.ToString('F8',$calCulture) }) -join ', '
$calSignLine = ($calHand.signs | ForEach-Object { [int]$_ }) -join ', '
[ordered]@{
    section = $calSection
    source_report = $calReferenceFile
    source_ini_sha256 = $calReport.ini_sha256
    slave_reference_deg = $calReferenceDeg
    joint_ids = $calHand.joint_ids
    joint_signs = $calHand.signs
    candidate_offsets_rad = $calOffsets
    absolute_mapping_verified = $false
} | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath (Join-Path $calDir ($calSection + '-candidate.json')) -Encoding UTF8
"配置段：[$calSection]"
"joint_offsets = $calOffsetLine"
"joint_signs = $calSignLine"
```

4. 检查打印的配置段确实是当前这对主从。公式会把**当前主手摆放**解释成输入的从臂角度，因此输入错 IP、错轴、错单位或摆位不一致都会生成错误偏置，即使数学计算没有报错。
5. 脚本自身 `--reference-positioned` 的候选基于 INI 的 `start_joints`，**不会自动查询 Nova**。本流程使用上述从臂实测角计算，不用该开关。现有 `±1.57 rad` 与精确 `±90°` 差约 0.046°；不要因这个近似差异误判方向错误，也不要把它当作较大姿态不一致的解释。

#### 7.1.8 写入本侧配置，保持未核验，再复读

1. 打开 INI，只替换当前 `[$calSection]` 内的 `joint_offsets`，填入上一步打印的六个数字，单位为 rad。确认 `joint_signs` 与已经核验的方向及本次采样一致。
2. 本侧 `absolute_mapping_verified` 继续为 `0`。不要改另一个主手、夹爪端点、相机或工具；不把候选复制到控制柜机械零点设置。
3. 保存后，在实体姿态没有改变的条件下复读：

```powershell
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m scripts.2_get_offset --section $calSection --output (Join-Path $calDir ($calSection + '-after-offset.json'))
if ($LASTEXITCODE -ne 0) { throw '复读失败，不能放行' }
$calAfter = Get-Content -LiteralPath (Join-Path $calDir ($calSection + '-after-offset.json')) -Raw -Encoding UTF8 | ConvertFrom-Json
$calMapped = @($calAfter.hands[0].mapped_deg)
$calErrors = for ($j = 0; $j -lt 6; $j++) {
    [pscustomobject]@{ Joint=('J'+($j+1)); MasterDeg=$calMapped[$j]; SlaveDeg=$calReferenceDeg[$j]; ErrorDeg=([double]$calMapped[$j]-$calReferenceDeg[$j]) }
}
$calErrors | Format-Table -AutoSize
$calErrors | Export-Csv -LiteralPath (Join-Path $calDir ($calSection + '-reference-errors.csv')) -NoTypeInformation -Encoding UTF8
```

4. 同次采样的数学回代应近乎零误差；重新采样的误差还包含摆位、锁定漂移和编码器量化。以本轮低速调试筛查建议 **各轴 ≤1°** 为起点，超出先查原因；这不是厂家标定精度承诺，也不是自动放行证明。
5. 如果复读角度几乎等于原始大角度，检查是否改错配置段、是否保存以及符号/单位。不要通过改安全边界把它运行起来。

#### 7.1.9 独立复测方向、第二姿态和重启一致性后才放行

1. 不再更改 offset。按 7.1.6 的独立摆位方法，对每个关节分别验证参考点附近的正、负小幅变化，在每个静止点记录从臂实际角与主手 mapped_deg。其他关节应保持对应；误差不能随运动方向系统性增大。
2. 选择另一处现场确认安全、且能独立摆成相同六轴姿态的检查姿态，比较全部六轴。不要通过重新计算 offset 让这个点也强行吻合；若参考点对得上、另一点对不上，说明方向、比例、关节顺序或实体姿态仍有问题。
3. 退出、重新打开只读采样程序后复读，检查映射不依赖某一次进程的临时状态。如果重插 USB/设备重上电，还需重新确认 COM 和编码器角度是否保持，不能复用此前结论。
4. 保存方向表、从臂截图、参考角、候选及复测报告。把 `dobot_settings.before.ini` 留作回退；需要回退时先停止全部控制程序，恢复备份后再次确认两侧核验标志为 0。
5. 只有**实体通道、六轴顺序、正负方向、参考姿态、独立第二姿态**均通过，才将已完成一侧的 `absolute_mapping_verified = 1`，记录日期和证据目录。未完成侧保持 0。这是现场人员的验收记录，程序不会自动认证它。
6. 第二对重复完整流程：在 C 设置 `$calSection = 'HAND_LEFT'`，对应从臂换成 `192.168.5.2`，使用该臂自己的实测六角度和 COM6 报告。偏置与符号分别验收。核验完成后另存一份 `dobot_settings.calibrated.ini` 到记录目录。

#### 7.1.10 标定通过后再启动，首次只做单侧小范围确认

1. 保存从臂与主手当前姿态记录。让从臂停在已核对的启动 home、主手停在对应姿态并有支撑；关闭独立摆位/采样进程，确认串口已释放。
2. 在 DobotStudio 配好 TCP/IP 二次开发模式后退出其控制连接，按第 9.0、9.1、9.2 节顺序启动。不因完成标定而直接开启第 9.3 的自动归位开关；它有独立的路径条件。
3. 比较 B 的映射日志与本次记录，只对已验证侧长按。主手与从臂同姿态时，对齐差应很小；如果出现几十度偏差、持续向外走、反向动作，立即停止跟随，保留日志并回到前面核对，不反复长按。
4. `ALIGN_START` 后保持主手稳定，到 `ALIGN_DONE` 再做小范围单关节移动，依次验证锁定、停止和重启；最后才开双侧。标定改变了角度解释，不会自动验证工作区、TCP、夹爪几何和碰撞。
5. 若 `ALIGN_REJECTED_UNCALIBRATED`，查本侧是否真正完成并登记验收；若 `ALIGN_REJECTED_UNSAFE`，查打印的目标和首个越界点。保持原边界不变。

| 标定记录项目 | 第一对：COM5 → .1 | 第二对：COM6 → .2 |
| --- | --- | --- |
| 实体标签、IP、时间 | 待填 | 待填 |
| 六个关节对应的 ID 已实测 | 待填 | 待填 |
| 从臂参考姿态实际 J1～J6（°） | 待填 | 待填 |
| 主手相同姿态的依据/照片 | 待填 | 待填 |
| 每轴正负方向和比例 | 待填 | 待填 |
| 写入的六个 offsets（rad）与 signs | 待填 | 待填 |
| 参考点复读与第二姿态误差 | 待填 | 待填 |
| 重启复读结果、证据目录 | 待填 | 待填 |
| 本次重新标定后的标志、验收人/日期 | 开始前置 0，通过后登记 1 | 开始前置 0，通过后登记 1 |

### 7.2 从臂启动 home 与工具

| 代码从臂 | home J1～J6（度） |
| --- | --- |
| 左 .1 | `[90, 0, 90, 0, -90, -90]` |
| 右 .2 | `[-90, 0, -90, 0, 90, 90]` |

该左右定义与本地官方轻量版原示例的左右表相反；现场必须按 IP、实体和安装方向确认，不混抄旧表。`start_joints` 中的 ±1.57 rad 是参考近似值，不改变上述 home。

`robot_pose_init()` 默认只检查姿态；六轴差异大于 `1e-6 rad` 时，未显式授权会以 `Safe stop before pose motion` 退出。该开关不影响服务使能或夹爪启动动作。可选授权见 9.3。

服务构造时写入 `SetTool(1,0,0,197,0,0,0)` 并选用 Tool 1；必须与实际末端、负载和安装匹配。保护计算另外使用 FK 模型、约 0.2 m 工具长度及夹爪模型，不等于经标定的实际 TCP 或完整碰撞模型。观测中的夹爪值含占位/指令值，不能当作逐周期实测夹爪反馈。

## 8. 独立相机检查（需要时才运行）

先关闭 Viewer 和占用同一相机的客户端：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m scripts.5_camera_read
# 或有限帧数、不显示的检查：
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m scripts.5_camera_read --frames 180 --no-display
```

当前脚本只打开 `CAMERA.enabled` 里的相机，不调用硬件复位，不连接机器人；Q、Esc 或 Ctrl+C 结束预览，退出时释放相机。15 fps 源于历史 USB 链路超时排查，30 fps 的短测通过不代表持续稳定。

## 9. 启动：先终端 A，再终端 B

### 9.0 每次启动前

1. 确认无另一份遥操、HTTP live 网关、原子任务或其他客户端占用装置；机器人、夹爪、主手及工作台状态可控。
2. 关闭 DobotStudio 的控制连接、相机 Viewer、相机检查和独立主手摆位/采样程序。
3. 确认当前 COM、IP、主从路由、工具、home 和第 0 节工作区方向。INI 当前已为 verified=1，**无需日常改回 0 或重复标定**。
4. 新代码要在停止跟随并退出旧 B 后启动；A 正常运行且没有修改服务代码或配置时，不为重启 B 清理有效设备锁。

### 9.1 终端 A：机器人服务

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
$env:DOBOT_DEVICE_LOCK_PATH = 'E:/Dobot/.local/runtime/device-owner'
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B experiments\launch_nodes.py --hostname 127.0.0.1 --robot-port 6001
```

**该入口会立即操作硬件，没有额外 YES 确认。** 服务申请设备锁，依次构造左 .1 和右 .2：使能机器人；设置 SpeedFactor、AccJ、SpeedJ 为 5；写入 Tool 1；初始化该侧夹爪，再退出拖拽并启动状态监控。

左夹爪 COM3/ID21 完成 `move(0,10,1)` 和 `move(255,10,1)` 后才启动右夹爪 COM4/ID22。每个目标读位置、允许归一化 0～255 量程误差 5、最多等待 10 秒，到位后停 0.5 秒；失败中止后续启动。速度参数 10 不是实测线速度。观察 `[GRIPPER START]`、`[GRIPPER DONE]` 和 `Starting robot server on port 6001`，保持 A 运行。

若出现 `owned or quarantined`，按 11.2 核对旧所有者，不能修改锁路径或盲删锁来绕过。

### 9.2 终端 B：默认不允许自动归位

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
$env:DOBOT_DEVICE_LOCK_PATH = 'E:/Dobot/.local/runtime/device-owner'
Remove-Item Env:DOBOT_ALLOW_SAFE_POSE_INIT -ErrorAction SilentlyContinue
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B experiments\run_control.py --hostname 127.0.0.1 --robot-port 6001 --show-img
```

B 初始化已启用的顶部相机、双主手，查询型号/固件、设置 DO 1/2/3 指示灯并检查 home。主手构造有力矩/PID 写入，因此本入口不是只读模式。

姿态有差异会在归位检查处停止；没有差异才继续。启动正常时应出现：

- `[RECORD] Enabled views: ['top']; output: ...collect_data`：只启用顶部视角，支持录制。
- `[MAPPING] ... absolute_verified=True`：两侧门控取当前 INI；不会自动认证实体标定。
- `robot init success....`、`button thread init success...`、`Ok, let's start`。
- `[TELEOP] ... align=0.15 rad/s, follow=3.2 rad/s ...`：检查实际生效限速。

`--show-img` 仅显示图像；`--no-show-img` 不关闭相机。客户端即使未录制也会创建数据根目录。若异常退出，确认旧进程及线程确已结束再重启，不同时打开第二个 B。

### 9.3 仅在现场确认整条路径后授权归位

审核 `[POSE CHECK]` 中当前、目标、差值和整条路径，确认可以运动后，在 B 设置：

```powershell
$env:DOBOT_ALLOW_SAFE_POSE_INIT = '1'
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B experiments\run_control.py --hostname 127.0.0.1 --robot-port 6001 --show-img
```

若已看到 `robot init success` 和 `Ok, let's start`，随后才出现 `RIGHT master button read failed` / `stale for more than 250 ms`，应转查主手通信，而不是重复授权归位或重算偏置。RIGHT 是配置 HAND_RIGHT / COM5；详见第 12 节。

该分支插值移动两臂到 home，每步最大关节增量默认 0.02 rad、等待 0.05 s；底层 ServoJ 当前传入 0.03。初始化不经正常跟随规划器的完整工作区路径预检，不能把工作区边界视为自动归位的路径保证。此授权与主手 verified 门控是两件事，不因标定为 1 就默认开启。使用后清除该临时变量。

## 10. 跟随与录制

### 10.1 A 键及跟随

| 完整按下并释放 | 当前动作 |
| --- | --- |
| A 持续 <0.5 秒 | 切换相应主手锁定/解锁，不等于从臂停止跟随 |
| A 持续 >1 秒 | 切换该侧跟随；开启时解除该主手力矩锁定，并检查绝对映射和初始路径 |
| A 持续 0.5～1 秒（含端点） | 不满足当前短/长按分支，可能无动作 |
| 任一 B，且已有一侧跟随开启 | 切换一个共享录制会话，详见 10.2 |

握稳并支撑主手，先开启一侧；`ALIGN_START` 后保持稳定，待 `ALIGN_DONE` 再小幅移动。对齐默认 0.15 rad/s，指令和实测六轴误差均在 0.02 rad 内连续 3 次后切换正常跟随。正常默认上限为 3.2 rad/s（原 4.0 的 80%），每次规划的时间跨度最多按 0.05 s 计算，单步上限 0.35 rad，仍受其他保护约束。

再长按该 A 关闭该侧跟随，查看 `SERVO_OFF` 并确认实体静止、主手锁定；两侧分别确认后再依次双侧开启。COM5/hand=1 控制左 Nova 六轴，COM6/hand=0 控制右 Nova 六轴；夹爪路由见第 6 节。

当前边界检查两侧活动臂的模型夹爪点：X、Y 包含端点，Z 严格大于 40 mm。向下 TCP 指令速度保护阈值仍为 1 m/s，与 3.2 rad/s 的关节限速不是同一个量。`TCP_SPEED_LIMITED` 会缩小本周期步长；`ALIGN_REJECTED_UNSAFE`、`SERVO_REJECTED`、`MASTER_DROP_LOCK` 应按原因停查，不反复长按或扩大范围。

`using_sensor_protection=False`，不能由 INI 的 `using_sensor` 推断传感器防跌落保护已启用；另有主手突变保护、按钮/相机失联检查。现有绝对关节映射不为每次开启重置相对零点，也不会自动移动至照片蓝框中心。

### 10.2 B 键录制、保存位置与验收

1. 先完成单侧或双侧正常跟随，等待需要侧 `ALIGN_DONE` 后再录。B 在已有跟随状态时也可能于对齐过程中生效，因此不要把 B 当作“自动等待对齐完成”的按钮。
2. 任一 B 完整按下并释放一次。查看 `[BUTTON_B] action=record_start` 和 `[RECORD_START]` 的 episode 路径；第一次即可启动，不需多按一次。
3. 再完整按下并释放任一 B 停止，查看 `record_stop` 和 `[RECORD_STOP]`。若无跟随，输出 `record_ignored reason=servo_off`，不开始录制。停止全部跟随或发生保护/相关故障时也会清除录制状态，不能要求所有故障路径都打印 `RECORD_STOP`。
4. 停止提交新样本后，异步队列中已有样本还可能继续落盘；`RECORD_STOP.samples` 是已入队数量，不是磁盘写入成功计数。观察文件稳定并正常退出 B；退出时写盘线程最多等待 10 秒，出现 `[RECORD] writer shutdown failed` 不能宣称数据完整。

当前每个会话布局：

```text
E:\Dobot\datasets\dataset_package_test\collect_data\<episode时间编号>\
├─ topImg\0.jpg, 1.jpg, ...
└─ observation\0.pkl, 1.pkl, ...
```

单顶部模式不创建 `leftImg` / `rightImg`。启用并验证更多相机后才创建对应目录。每个 pkl 保存采样时观测、同编号 `control` 动作及 `_teleop_camera_names`、`_teleop_capture_monotonic_s`、`_teleop_source_index`、`_teleop_dropped_before`。图像取相机线程最新 JPEG；15 fps 不等于控制或录制采样率，也不保证不同相机严格同步。不保存深度流为独立图像文件。

仅在停止录制、退出 B 后，可用下面的只读目录检查确认 JPG/PKL 数量，不加载外来 pickle：

```powershell
$episodeDir = Read-Host '粘贴本次 RECORD_START 打印的完整 episode_dir'
Get-ChildItem -LiteralPath (Join-Path $episodeDir 'topImg') -Filter '*.jpg' | Measure-Object
Get-ChildItem -LiteralPath (Join-Path $episodeDir 'observation') -Filter '*.pkl' | Measure-Object
```

正常写完时顶部 JPG 与观测 PKL 应同编号配对、非空；队列溢出会丢弃源采样，`source_index` 可不连续。`RECORD_QUEUE_FULL` / `record_dropped` 表示写盘跟不上；`RECORD_WRITER_FAILURE` 表示写盘失败并停止录制，按错误检查磁盘、权限或图像缺失，不把按键成功当作写盘成功。

`Args.save_data_path` 和 `project_name` 未声明为 CLI 字段，当前不能添加 `--save-data-path` 来改目录。原有训练转换脚本仍带独立路径/三视图假设，单视角录制成功不等于训练转换已适配。

### 10.3 会影响实际行为的环境变量

| 变量 | 默认 / 用途 |
| --- | --- |
| `DOBOT_SERVOJ_MAX_SPEED_RAD_S` | 3.2，正常跟随关节速度上限 |
| `DOBOT_ALIGN_SPEED_RAD_S` | 0.15，慢速对齐 |
| `DOBOT_SERVOJ_MAX_STEP_RAD` | 0.35，规划单步上限 |
| `DOBOT_RECORD_QUEUE_SIZE` | 8，有界异步队列容量 |
| `DOBOT_ALLOW_SAFE_POSE_INIT` | 未设置；1 才显式允许归位 |
| `DOBOT_DEVICE_LOCK_PATH` | 本装置各控制终端统一设为 9.1 的路径 |
| `DOBOT_TELEOP_TRACE` | 默认关闭；1 增加逐命令日志和 I/O |

显式环境变量覆盖代码默认值；若启动日志不是 follow=3.2，先检查 B 终端中的 `DOBOT_SERVOJ_MAX_SPEED_RAD_S`，不要只凭手册认定已限速。工作区边界目前由 `teleop_safety.py` 常量定义，不从 INI 读取。

## 11. 退出与设备锁

### 11.1 正常退出

1. 若在录制，按 B 停止并记录路径，留意队列和写盘错误。
2. 分别长按 A 停止各侧跟随，确认实体静止并支撑、锁定主手。
3. B 按 Ctrl+C，等待进程、相机线程和写盘关闭；不要立即强杀或断电来代替正常关闭。
4. A 按 Ctrl+C，确认服务结束。按现场控制器流程确认停止，必要时禁用机器人；退出 Python 不保证执行 DisableRobot，也不保证夹爪松开。
5. 在 B 清除临时归位开关：

```powershell
Remove-Item Env:DOBOT_ALLOW_SAFE_POSE_INIT -ErrorAction SilentlyContinue
```

若有非预期运动，优先按现场急停流程处置，不等待 Python 正常退出。

### 11.2 `owned or quarantined` 的处理

旧 ZMQ 服务没有经验证的完整关停协议，退出后仍可能保留 `E:\Dobot\.local\runtime\device-owner\owner.json`。

1. 确认现场安全、所有控制程序结束，再读取 owner.json 的 host、pid、purpose。
2. 对照对应主机进程命令行和连接，PID 可能复用，不能只看 PID 存在与否。
3. 仅在确认为已退出进程留下的锁后，将这个确切的 `device-owner` 文件夹重命名为带时间戳的归档；不要删除整个 runtime，也不要改锁路径绕过仍运行的服务。
4. 保留旁边的 `device-owner.scene-invalid`。以后切换 HTTP live 网关时，另按网关流程现场复位；网关的 operator 记录不等于释放 owner 锁。

`run_control.py` 自身不申请设备锁，仍需保证只有一个 B 客户端。

## 12. 故障排查

| 现象 | 当前排查入口 |
| --- | --- |
| `No module named ...` | 使用绝对 `.venv` Python 路径，核对 lock 和 pip check |
| `/dev/ttyUSB...` 或串口打不开 | 当前应为 COM3～COM6；按设备身份核对，不运行自动找端口当只读测试 |
| 夹爪启动超时/读取失败 | 对照 GRIPPER START 的 IP、COM、ID，排查供电与回包，失败不继续下一侧 |
| 相机超时 | 当前 enabled=top、15 fps；检查占用、流配置和 USB 链路 |
| `Safe stop before pose motion` | 归位授权未开启且姿态有差异；审核 home/路径后才按 9.3 处理 |
| `ALIGN_REJECTED_UNCALIBRATED` | 当前读入的那一侧标志为 false；核对实际 INI 和映射证据，不盲设 1 |
| `ALIGN_REJECTED_UNSAFE` / `SERVO_REJECTED` | 查看首个拒绝点与当前 X/Y/Z、分臂 X 范围及 J3 条件，不扩大边界重试 |
| B 无文件 | 先看 `record_ignored`、跟随/对齐状态、RECORD_START 路径和写盘错误；单相机当前支持录制 |
| RECORD_START 有但文件不完整 | 区分入队与写盘成功；检查队列溢出、异步失败和退出等待 |
| 主手通信 -3002 / -3001，按钮或关节超过 250 ms | 损坏回包/接收超时；本地修复已加入协议间清残包、独立按钮轮询及关节时效检查。检查对应 COM 的供电/USB/占用并保留日志，不能扩大超时或跳过失联保护 |
| 主手/相机掉线 | 停止客户端命令，处理设备再显式重启；不要屏蔽线程健康检查 |
| follow 仍为 4.0 | 检查旧 B 是否未退出、终端环境变量是否覆盖 3.2 默认值 |

主日志：`E:\Dobot\dobot-atomic-skills\scripts\logs\teleop_runtime.log`；辅助日志 `scripts\logs\log.txt`。不清空故障日志。固件 HTTP 查询和 ZMQ 接收当前没有统一的请求超时保证，“等待连接”不能仅凭屏幕文字认为已连接。

## 13. 源码与历史记录导航

| 文件 | 核对内容 |
| --- | --- |
| `experiments/launch_nodes.py` | 设备锁、左右服务启动顺序 |
| `experiments/run_control.py` | 按钮、路径预检、跟随、录制与日志 |
| `dobot_control/absolute_teleop.py` | 对齐/跟随规划器及 3.2 默认速度 |
| `dobot_control/teleop_safety.py` | 共用工作区边界、主手突变保护 |
| `dobot_control/cameras/settings.py` | 已启用视角、帧率和单相机录制开关 |
| `dobot_control/robots/dobot.py` | 控制柜、工具、夹爪启动与状态 |
| `scripts/manipulate_utils.py` | 六轴交叉/夹爪通道映射、home 检查 |
| `scripts/2_get_offset.py` | 只读主手采样和候选；不写 INI，不查询从臂参考角 |
| `scripts/5_camera_read.py` | 选定相机预览与有限帧检查，不复位 |
| `scripts/format_obs.py`、`dobot_control/async_writer.py` | pkl 格式、有界队列与失败传播 |
| `docs/workspace-x-wrist-correction-20261102.zh-CN.md` | 当前分臂 X 范围、实体/IP 对应、J6 原始标定证据及回归 |
| `docs/master-serial-recovery-20261102.zh-CN.md` | 主手按钮过期、混合协议收发与 30 项离线回归记录 |
| `docs/teleop-fixes-20261102.zh-CN.md` | 三项修改、90 项历史测试与清理记录 |
| `manuals/HANDOFF_NEXT_CHAT.zh-CN.md` | 历史交接；正文旧状态以本手册为准 |

`task_skills` 原子任务及 HTTP 网关是另一条入口，其夹爪绑定和 Tool 配置不能直接套用于本手册遥操链路。

## 14. 本版现场验收记录

| 项目 | 已知状态 / 本次需登记 |
| --- | --- |
| 原有遥操调通 | 用户于 2026-10-27 确认；不是 11 月新边界/录制验收 |
| 当前偏置和 verified | 本机配置为第 6 节数值，两侧 1；逐项现场原始记录未在本次补造 |
| 新工作区与腕部 | 按用户确认方向 .1 X+150、.2 X-150，Y 恢复；现场复测范围及 J6 姿态 |
| 跟随限速 | 启动日志核对 follow=3.2；记录环境覆盖情况及现场跟随表现 |
| 单相机录制 | 离线真实写盘/读回已过；现场登记 B 首次开始、再次停止、JPG/PKL 配对及退出结果 |
| 停止与设备锁 | 确认静止、主手支撑/锁定、控制器状态和旧锁所有者 |
| 本次文档校验 | 对照源码、INI、脚本路径、文档命令与 Markdown 结构；未启动 A/B、采样或相机流 |
