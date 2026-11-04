> 2026-10-27 发布说明：用户已确认当前本机版本调试成功；当前提交的 INI 两侧 absolute_mapping_verified 均为 1，并保留新偏置。下文是截至 2026-10-25 的历史正文，其中待调试、verified=0、未更新偏置等状态已被本次用户确认及当前配置取代。本次发布未操作硬件。详情见 [当前版本说明](docs/windows-teleop.md)。

# Windows 本机 X-Trainer 主从手遥操部署与操作手册

更新：2026-10-25（北京时间；扩充启动前主手标定，以对应从臂 J1～J6 为参考）
工作区根目录：`E:\Dobot`
GitHub 仓库目录：`E:\Dobot\dobot-atomic-skills`
源码：[yanggoumao2/dobot-atomic-skills](https://github.com/yanggoumao2/dobot-atomic-skills)，提交 `c402e9c1242e18eee87369accb140253bca495ac`。

**已确认方案：你的 Windows 电脑就是主机，使用 GitHub Python 脚本做主从手遥操，不要求官方图形界面，不安装模型训练环境。** DobotStudio Pro 用于配置 Nova 控制柜。照片确认一台为 DOBOT Nova 2（DT-NV020-0）；本次 HTTP 查询确认 `.1/.2` 两柜均返回 Nova 2、控制固件 3.5.8.5。另一台本体 S/N、实体左右绑定、工具及标定适用性仍须现场核对。

**当前进度：用户已运行 9.1/9.2 并实现主手带动从臂；最新问题是左从臂绝对对齐时越过模型工作区边界。当前转回第 7.1 节核验映射，不继续反复长按。** 代码已增加对齐前采样预检；两主手的 `absolute_mapping_verified = 0`，长按会拒绝自动对齐。保留现有偏置、符号和安全边界，尚未通过现场标定验收。本次助手仅修改代码、运行离线模拟，没有操作真机。见 [本轮修复记录](manuals/MAPPING_AUDIT_FIX_20261023.zh-CN.md) 和 [新对话交接文档](manuals/HANDOFF_NEXT_CHAT.zh-CN.md)。

当前使用 **HAND_LEFT / COM6、HAND_RIGHT / COM5（均 2000000 baud）、COM3 / ID21 夹爪、COM4 / ID22 夹爪、顶部 D405 / 218622278417（15 fps）**。配置段名称不是实体左右标签。用户本次实体左主手按键对应日志 `hand=1`，即 COM5 → 左 Nova 六轴；COM6 的实体侧还需独立追线确认。夹爪实体侧仍待确认；侧相机及三视角录制禁用。

## 0. 整理后的目录与已完成事项

```text
E:\Dobot\
├─ README.md                         工作区入口
├─ TELEOP_WINDOWS_SETUP.zh-CN.md      本手册
├─ dobot-atomic-skills\               GitHub 仓库，包含 .git
│  ├─ experiments\                   双臂服务和遥操入口
│  ├─ scripts\dobot_config\          现场硬件配置
│  ├─ requirements-teleop-windows.lock.txt
│  └─ ...                            上游其他源码完整保留
├─ .venv\                            已安装的 Python 3.11.16 环境
├─ apps\
│  └─ DobotStudioPro\                DobotStudio Pro 建议安装位置
├─ installers\                       官方安装包下载到这里
├─ manuals\                          四份官方 PDF、来源及本机环境记录
├─ tools\                            依赖安装与离线检查脚本
├─ .local\                           缓存、备份及共享设备锁
└─ datasets\                         遥操程序首次运行时按需创建
```

代码仓库移动后，虚拟环境 `.venv` 保留原位置，避免路径失效。以后运行 Git 命令先进入 `E:\Dobot\dobot-atomic-skills`；本机解释器始终使用 `E:\Dobot\.venv\Scripts\python.exe`。

| 当前状态 | 说明 |
| --- | --- |
| 已完成 | 代码迁移；独立 Python 环境；遥操依赖安装；官方手册下载 |
| 第 1～3 步已完成 | 2026-10-22 用户确认：方案、软件/环境准备和离线检查无问题；DobotStudio 实际版本、路径仍待登记 |
| 历史软件验证记录 | 2026-10-20：20 项遥操回归、4 项夹爪诊断测试通过；两个启动入口可显示帮助 |
| 已识别的设备 | 照片所示 Nova 2，型号 DT-NV020-0；铭牌明细见第 1.4 节 |
| 历史设备验证 | 四路串口 ID 与读取；两柜 Nova 2 / V3.5.8.5；顶部相机最终采用 15 fps，30 fps 曾超时 |
| 本轮离线验证 | 81 项回归通过；覆盖映射拒绝、路径预检、长按对齐/跟随/重启及只读采样；不代表实体映射验收 |
| 待现场完成 | 物理左右映射、仓库标定/工具/home 适用性、小范围跟随；软件检查通过不等于运动验收 |
| 本助手未执行 | 机器人使能、运动、控制柜设置修改、主手力矩/PID 写入；本次只做硬件读取和相机采集 |

第 1～6、8 步保留供复查；本轮接手从第 7.1 节开始。映射验证完成后才返回第 9.0 节。第 9.3 的自动归位授权与长按对齐核验是两件事。

## 1. 采用哪个 X-Trainer、参考哪些官方步骤

### 1.1 两个软件的分工

| 软件 | 本次用途 | 是否需要一直运行 |
| --- | --- | --- |
| DobotStudio **Pro**（适配 Nova 的版本） | 识别控制柜、设置 IP、查看报警/固件、配置 TCP/IP 二次开发模式 | 设备设置完成后退出控制连接，再运行遥操；故障处理时停止遥操后重新使用 |
| 本仓库的 X-Trainer Python 控制程序 | 读取两主手，控制两 Nova 和夹爪，读取三相机 | 遥操时运行两个 Python 进程 |
| 官网 X-Trainer 图形 UI 包 | 不属于本次部署范围 | 无需下载或安装 |

你已明确选择 GitHub Python 脚本，不要求官方图形界面。GitHub 项目本身已有 Windows 串口适配，安装其依赖后进入脚本遥操的现场配置流程；无需再寻找 Windows UI 包。

已下载的官网 UI 手册仅保留作资料对照；本手册的按钮、归位和映射说明以当前 GitHub 提交为准，不套用官方 UI 的启动或同步流程。

### 1.2 已下载的官方资料

下载入口：[越疆 X-Trainer 下载中心](https://www.dobot.cn/service/download-center?keyword=&type-95%5B%5D=112)。四份原始 PDF 已保存在 `E:\Dobot\manuals`：

| 文件 | 本次用到的内容 |
| --- | --- |
| `XTrainer-Light-V1.3-20240729-cn.pdf` | 印刷页 17～21：纯遥操可省略的训练依赖；21～25：控制柜 IP 和 TCP/IP 模式；26～30：主从手操作 |
| `XTrainer-UI-V1.1-20260818-cn.pdf` | 印刷页 1～2：UI 适用系统；UI 调试与主手操作参考 |
| `XTrainer-FAQ-cn.pdf` | 原版常见问题参考；涉及 Linux 命令时不可直接粘贴进 PowerShell |
| `DobotStudio-Pro-2.8-CR-Nova-cn.pdf` | PDF 第 7～8 页：Windows 安装、自定义安装目录；连接、通讯设置与远程控制章节 |

下载来源记录在 `manuals\official-source-links.json`。原始 PDF 未改写；页码前带“印刷页”指正文页脚数字，和阅读器页序可能不同。

官方文档描述的是其原版示例，本仓库做过修改。本文保留官方的配置顺序，但 Python 版本、串口权限、左右映射、home 角度和启动命令以此仓库实际实现为准，差异在第 5～7 步标明。

### 1.3 你的电脑是否适合当前任务

截图显示：Windows 11 家庭版 25H2 x64、i7-14650HX、16 GB RAM、RTX 5060 Laptop GPU 8 GB。官方轻量版手册的仅遥操参考配置包括 16 GB 内存和较低档位 CPU/GPU；你的配置可以作为本次纯遥操主机的硬件起点。是否流畅仍取决于四路串口、三路相机 USB 带宽和有线网络，需实机验证。

2026-10-20 检测显卡驱动为 `617.42`，GPU 可被系统枚举；当时可用空间约 C 52.8 GB、D 28.2 GB、E 383.4 GB，所以把代码、环境、安装包、数据优先放 E 盘。这是历史快照，空间以实际操作时为准。

上游默认两个 Nova 从臂、两个主手、两个 Feetech 夹爪和三台 RealSense。本地已按你本次需求适配 **仅顶部一台相机**：`CAMERA.enabled = top`、`fps = 15`。`--no-show-img` 仍只关闭显示，不关闭已启用相机。未启用的左右相机不会连接或阻止启动；三视角数据录制暂时禁用。

### 1.4 已收到的设备铭牌（2026-10-22）

来源：你提供的两张照片。原图已复制到工作区，便于后续对照：[铭牌照片 1](manuals/equipment/nova2-nameplate-01.jpg)、[铭牌照片 2](manuals/equipment/nova2-nameplate-02.jpg)。

| 铭牌项目 | 照片可读值 |
| --- | --- |
| 产品名称 | DOBOT Nova 2 |
| 型号 MODEL | DT-NV020-0 |
| 本体重量 WEIGHT | 11 kg |
| 防护等级 ENCLOSURE TYPE | IP54 |
| 最大负载 MAX PAYLOAD | 2 kg |
| 臂长 MAX REACH | 625 mm |
| 额定电压 RATED VOLTAGE | DC 48 V |
| 额定功率 NOMINAL POWER | 100 W |
| 生产日期 PRODUCTION DATE | 2026.01 |
| 序列号 S/N | NV203-4616-0801-00 |

两张照片的 S/N 相同，因此目前只登记一个可识别的设备序列号，不能当作双臂各自的铭牌。照片未标出该设备属于左臂还是右臂；另一台的型号和 S/N 待核对。两张图均为机械臂本体铭牌，不能据此确定控制柜型号、V3/V4 固件、IP、COM 或相机序列号，也不能用生产日期推断固件版本。

**2 kg 是铭牌最大负载，不是要在 DobotStudio 中直接填写的工具负载。** 软件负载参数应按实际夹爪、相机等末端装配及其使用情况核对；本体重量 11 kg 同样不是该输入值。这些铭牌信息也不改变仓库的 home 角度和 Tool 1 配置。

## 2. 安装软件与隔离环境，按以下顺序做

**进度：已完成（用户于 2026-10-22 确认）。本节保留为参考，不要求重新安装。** DobotStudio 的实际版本和程序路径仍可在第 2.3 节的记录栏补填。

### 2.1 先查已有安装，不动其他项目

本次已经找到：

```text
Conda 根目录：C:\miniconda3
已有环境：   C:\miniconda3\envs\gsae
             C:\miniconda3\envs\sae-llm
本项目环境： E:\Dobot\.venv
```

2026-10-20 的安装记录和常见程序目录检查未检出 DobotStudio、独立 X-Trainer 或 RealSense Viewer；这是安装前的历史结果，不能代表你完成第 1～3 步后的状态。需要登记实际安装位置时，在“设置 → 应用 → 已安装的应用”搜索 `Dobot`、`RealSense`，或查看已有快捷方式“属性 → 目标”及“打开文件所在位置”。不需要为查路径而重新安装。

想再次定位 Conda、Python，在 PowerShell 执行：

```powershell
Get-Command conda,python,nvcc,nvidia-smi -ErrorAction SilentlyContinue |
    Select-Object Name,Source
& 'C:\miniconda3\python.exe' -m conda env list
Get-Content -LiteralPath "$env:USERPROFILE\.conda\environments.txt"
E:\Dobot\.venv\Scripts\python.exe -c "import sys; print(sys.executable); print(sys.version)"
```

**建议不在 E 盘重复安装 Anaconda。** 多套发行版不一定必然冲突，但各自添加 PATH、注册默认 Python 或执行 `conda init` 后，容易调用到错误的 base/python/pip。你已经有 Miniconda，它能承担环境管理用途，无需为了手册中出现“Anaconda”就再装完整 Anaconda。现有 `gsae`、`sae-llm` 环境保持原样。

### 2.2 当前推荐：直接复用 E 盘 `.venv`

这是本次已完成且依赖已验证的方案。环境基于已有的 Python 3.11.16 x64，依赖安装在 E 盘；不需要激活 Conda，也不需要改系统 PATH。

```powershell
Set-Location E:\Dobot
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip --version
.\.venv\Scripts\python.exe -m pip check
```

两个路径都应指向 `E:\Dobot\.venv`。以后不要直接使用不带路径的 `pip`；本机全局 `python` 是 Miniconda 的 Python 3.13.12。

此虚拟环境的基础解释器为：

```text
C:\Users\Lenovo\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe
```

不要删除这个基础解释器或移动 `.venv`。若想把基础 Python 也迁到 E 盘，应另建环境并重新验证，而不是拖动 `.venv` 文件夹。

### 2.3 安装 DobotStudio Pro 到 E 盘

1. 先确认控制柜型号、固件大版本。当前仓库要求 V3 且不低于 3.5.8.1；不是任意新版固件都兼容。
2. 官方 X-Trainer 轻量版手册使用 **DobotStudio Pro 2.x.x** 配置 Nova 2。选择供应商确认适配当前 Nova/V3 控制柜的 **Win64** 安装包。不要下载面向 Magician 的其他 DobotStudio 产品。
3. 在[官方下载中心](https://www.dobot.cn/service/download-center?keyword=DobotStudio%20Pro)按名称查找；若页面只显示 4.x，可在[官方产品页的控制软件下载区](https://www.dobot.cn/products/dobot-series/m1-pro.html?lang=cn)找到保留的 `DobotStudio Pro 2.8 (V2.8.3.0) for Win64` / `V2.8.1.0 for Win64` 条目。该链接是官网保留的通用安装包入口，选择前仍要核对 Nova 控制柜兼容性，不能因为入口页面名是 M1 Pro 就选择其四轴手册。
4. 将安装包保存到 `E:\Dobot\installers`。官网如要求正常登录/填写下载信息，由你按实际信息完成。本次未下载或执行该安装程序，也没有伪造可直连的 EXE 地址。
5. 双击安装包 → 选择语言 → 下一步 → **自定义安装**，目标填写 `E:\Dobot\apps\DobotStudioPro`，然后安装。上述自定义安装步骤见本地 DobotStudio 2.8 手册 PDF 第 7～8 页。
6. 若安装包自动追加版本子目录，保留该目录，并在下面记录最终 `DobotStudioPro.exe` 的实际位置；后续从安装器创建的快捷方式启动。
7. 如安装器要求系统组件、驱动或服务写入 C 盘，这是安装器的正常系统级依赖，应用主体放 E 盘并不保证 C 盘零写入。不要把系统目录内容手动搬走。
8. 首次启动，打开“帮助/关于”确认软件版本。此时只验证软件能打开，不使能机器人。网络和控制柜设置按第 5 步继续。

```text
已安装版本：2.8.3-x86-64
程序实际路径："E:\Dobot\apps\DobotStudioPro\DobotStudio Pro\DobotStudio Pro.exe"
对应控制柜型号/固件：____________________
```

官网 4.6/4.7 条目的兼容说明针对相应 V4 控制器，其中 4.7 明确不兼容 V3；不要把“最新版本”理解为本仓库的正确版本，也不要为安装软件顺手升级控制柜固件。

如果本机已经装过 DobotStudio，优先检查已有版本能否适配。不要同时运行两套控制软件或多个可能争用同一服务/端口的版本；安装多个目录并不能自动隔离后台服务。

### 2.4 部署 X-Trainer 脚本遥操功能

本次已经部署到 `E:\Dobot\dobot-atomic-skills`。它通过 Python 运行，不需要再安装一个同名 EXE，也不需要额外的 X-Trainer 应用目录。

先检查代码：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
git remote -v
git rev-parse HEAD
git status --short
```

本机已经有代码，不要重复克隆到这个非空目录。迁移到全新机器时才使用：

```powershell
git clone https://github.com/yanggoumao2/dobot-atomic-skills.git E:\Dobot\dobot-atomic-skills
```

本手册、根目录脚本及 `requirements-teleop-windows*.txt` 是本地新增文件，未提交或推送，换机器时要另外复制。

安装/恢复依赖时，在根目录执行：

```powershell
Set-Location E:\Dobot
.\.venv\Scripts\python.exe -m pip install -r .\dobot-atomic-skills\requirements-teleop-windows.lock.txt
.\.venv\Scripts\python.exe -m pip check
```

也可以使用本次提供的安装脚本。它只创建/检查 Python 环境和安装库，不连接硬件：

```powershell
Set-Location E:\Dobot
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\setup-teleop.ps1
```

这里的执行策略参数仅作用于这次子进程，不修改机器永久执行策略。如果组织策略阻止脚本，使用上面的直接 Python 命令，不需要修改系统策略。

`setup-teleop.ps1` 在 `.venv` 已存在时复用它；不存在时尝试从已核实的本机 Python 3.11 路径创建。换机器可传入 `-BasePython '实际的Python3.11路径\python.exe'`。若创建时仅 ensurepip 失败，可在 PowerShell 运行：

```powershell
New-Item -ItemType Directory -Force E:\Dobot\.local\tmp | Out-Null
$env:TEMP = 'E:\Dobot\.local\tmp'
$env:TMP = $env:TEMP
E:\Dobot\.venv\Scripts\python.exe -m ensurepip --upgrade --default-pip
```

| 包 | 本机锁定版本 | 用途 |
| --- | --- | --- |
| numpy | 1.26.4 | 关节与图像数组 |
| opencv-python | 4.10.0.84 | 图像显示/编码 |
| tyro | 0.8.8 | CLI 参数解析 |
| pyserial | 3.5 | 主手/夹爪串口 |
| pyzmq | 27.2.0 | 两个本机进程之间通信 |
| requests | 2.34.2 | 固件与型号查询 |
| pyrealsense2 | 2.59.1.11053 | 相机 Python 接口 |

其他传递依赖见 lock 文件。`tkinter` 随基础 Python 提供；Dynamixel/Feetech SDK 已在仓库 `third_party` 中，被本仓库按路径导入。

官方旧版 SDK 示例使用 Python 3.8.10，并包含安装训练模块的命令；本 GitHub 快照带有新代码，不能照抄其 Python 3.8 环境。本次选择 3.11，已经通过本仓库离线验证。原 `requirements.txt` 包含当前遥操不需要的仿真/训练组件，使用本地整理的遥操 lock 文件。

### 2.5 可选：坚持使用 Conda 时，复用 C 盘管理器，在 E 盘创建环境

**这是替代方案，本机当前不需要执行，也不要与 `.venv` 混着装。** 不重复安装 Anaconda，仅用现有 Miniconda 在项目目录创建新的独立环境：

指定目录的机制参考 [Conda 官方环境管理说明](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#specifying-a-location-for-an-environment)。

```powershell
& 'C:\miniconda3\python.exe' -m conda create --prefix E:\Dobot\envs\xtrainer-py311 python=3.11 pip
& 'E:\Dobot\envs\xtrainer-py311\python.exe' -m pip install -r E:\Dobot\dobot-atomic-skills\requirements-teleop-windows.lock.txt
& 'E:\Dobot\envs\xtrainer-py311\python.exe' -m pip check
& 'C:\miniconda3\python.exe' -m conda run --prefix E:\Dobot\envs\xtrainer-py311 python -c "import sys; print(sys.executable)"
```

这会把环境放 E 盘，但 Conda 管理器和默认下载缓存仍可能在 C 盘。若希望该次命令的缓存也放 E 盘，可在执行 create 前设置：

```powershell
$env:CONDA_PKGS_DIRS = 'E:\Dobot\.local\conda-pkgs'
```

只有显式选择这个方案时，才把后文所有 `E:\Dobot\.venv\Scripts\python.exe` 替换为 `E:\Dobot\envs\xtrainer-py311\python.exe`。根目录两个辅助 PS1 脚本默认绑定 `.venv`；选 Conda 时用直接 Python 命令，不要再运行绑定 `.venv` 的安装脚本。

IDE 中也要选择同一个解释器。VS Code 选择“Python: Select Interpreter → Enter interpreter path”；PyCharm 添加 Existing Interpreter，填写实际 `python.exe`。不要选择 `base`、`gsae` 或 `sae-llm`。

### 2.6 CUDA、cuDNN、PyTorch：本阶段不安装

官方轻量版手册印刷页 17 允许纯遥操跳过 CUDA/cuDNN，印刷页 20 允许跳过 PyTorch。对应提示已核对 PDF 页面位置；不是连串口/相机依赖也一并跳过。

因此不在 E 盘重复安装 CUDA，不降级现有显卡驱动，也不安装旧训练示例的 `torch==2.0.1/cu118`。你这次不需要训练功能，没必要为 RTX 5060 与旧训练栈做兼容适配。

如需定位已有 CUDA，仅做以下检查：

```powershell
Get-Command nvcc,nvidia-smi -ErrorAction SilentlyContinue | Select-Object Name,Source
Get-ChildItem Env:CUDA* -ErrorAction SilentlyContinue
Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' -ErrorAction SilentlyContinue
nvidia-smi
& 'C:\miniconda3\python.exe' -m conda list --prefix C:\miniconda3\envs\gsae |
    Select-String 'cuda|cudnn|torch'
& 'C:\miniconda3\python.exe' -m conda list --prefix C:\miniconda3\envs\sae-llm |
    Select-String 'cuda|cudnn|torch'
```

本次未在命令路径、CUDA_PATH、标准目录和安装记录中找到独立 CUDA Toolkit，**不能由此排除某个 Conda 环境或非标准目录中有 CUDA 运行库**。`nvidia-smi` 顶部的 CUDA Version 表示驱动支持的 CUDA 版本，不是 Toolkit 的安装证明；查 Toolkit 看 `nvcc --version` 和实际安装目录。

字段含义参考 [NVIDIA 的 nvidia-smi 说明](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。

系统级 Toolkit、显卡驱动与 Python 环境内的 CUDA 运行库不是一回事；以后确需训练时，另建训练环境再匹配版本。不要在这次遥操环境里执行卸载现有 CUDA/torch 的命令。

## 3. 离线检查：在接机器人之前执行

**进度：已完成（用户于 2026-10-22 确认）。下面命令留作环境变更后的复查，本次直接继续第 4 步。**

先在根目录执行：

```powershell
Set-Location E:\Dobot
.\.venv\Scripts\python.exe -X utf8 -B .\dobot-atomic-skills\tools\check_teleop_local.py
```

再进入仓库验证帮助和回归测试：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B experiments\launch_nodes.py --help
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B experiments\run_control.py --help
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests -p test_teleop_regressions.py -q
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests -p test_gripper_diagnostics.py -q
```

本次检查工具不会构造硬件驱动、连接控制柜、复位相机或发送机器人动作。退出码 `0` 表示软件/有限静态检查通过，`1` 表示环境失败，`2` 表示现场配置待处理。原始 INI 中的 `/dev/ttyUSB*` 在 Windows 上出现待配置提示是预期结果。

**验收：** `pip check` 无依赖冲突；两个 `--help` 正常；当前提交的两组测试分别通过 20 项和 4 项。通过后进入接线步骤。

## 4. 安装设备驱动并逐一接线

**本节接线已完成，保留供复查；当前交接从第 9.0 节开始。** 现场标签仍应记录 S/N 为 `NV203-4616-0801-00` 的实体 Nova 2 位于操作者面对工作台时的哪一侧，再核对另一台铭牌。此处“左/右”先作为现场标签，后续仍需与代码通道逐一对应，不由照片视角推定。

接线时填写以下对照表。控制柜型号/固件在第 5 步通过其铭牌及 DobotStudio 补齐；IP 以实测为准，暂不把仓库默认 `.1/.2` 记为已确认值。

| 设备 | 已知信息 | 本步骤需记录 |
| --- | --- | --- |
| 照片所示从臂 | Nova 2 / DT-NV020-0 / NV203-4616-0801-00 | 实体左或右：____；对应控制柜标签：____ |
| 另一台从臂 | 铭牌未提供 | 型号：____；S/N：____；实体位置：____；控制柜标签：____ |
| 左/右主手 | 两路目标串口待枚举 | 左 COM：____；右 COM：____ |
| 两只夹爪 | 两路目标串口待枚举 | 实体标签→COM：____、____；随后按第 6 步核对配置段绑定 |
| 顶部/左侧/右侧相机 | 三个型号和序列号待枚举 | top：____；left：____；right：____ |

完成标记后按下面顺序接线；驱动如果已正常识别就不用重装。此步骤不运行 `launch_nodes.py`、`run_control.py`、找端口或标定驱动脚本。

1. 按实际 X-Trainer 手册确认供电、急停和机械固定；首次接线时保持机器人未使能。
2. 将电脑有线网卡和两个控制柜接入套装控制网络。可以使用交换机，具体接口按设备标签和手册确认，不能凭外观把网口与其他接口混接。
3. 连接两个主手和两个夹爪的 USB 转接器。打开 Windows“设备管理器 → 端口（COM 和 LPT）”，记录每个接口的 COM 号与物理标签。
4. 如果没有 COM 口或有黄色警告，查看转接器硬件 ID，安装设备供应商针对该转接器的驱动。仓库未规定现场 USB 转串口芯片型号，不要任意猜装 CH340、FTDI 等驱动。
5. 本次按用户要求仅连接顶部 RealSense，侧相机无需连接；优先使用 USB 3.x 链路。按 [RealSense 官方 Windows 发行说明](https://github.com/realsenseai/librealsense/blob/master/doc/distribution_windows.md)获取 Viewer/SDK 工具，用 Viewer 确认序列号、RGB 和深度出图。本次已用 Python 验证当前 USB 2.1 链路下 30 fps 可出流。
6. 如无法访问相机，检查 Windows 相机隐私设置是否允许桌面应用访问；这是 [RealSense 官方安装排障说明](https://github.com/realsenseai/librealsense/blob/master/doc/installation_windows.md)列出的检查项。
7. 关闭 Viewer 和其他占用相机、串口、控制柜的程序，再运行下面的枚举。

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B tools\check_teleop_local.py --inventory
```

`--inventory` 仅枚举串口和 RealSense 设备，不打开串口，不启动图像流，不复位设备。设备列表为空时，只能说明本次进程未枚举到设备，还需排查连接、驱动和运行环境权限。

**验收：** 找到并标记四路目标串口和本次启用的顶部相机。电脑可能还有其他 COM 设备，不要把它们误填入配置。以后启用三视角时再补齐另外两台相机。

## 5. 设置有线网络与控制器模式

### 5.1 仓库默认地址

| 设备 | 默认地址 | 来源 |
| --- | --- | --- |
| 物理左 Nova | `192.168.5.1` | `task_skills/layout.py` |
| 物理右 Nova | `192.168.5.2` | `task_skills/layout.py` |
| 电脑有线网卡 |  `192.168.5.10/24` | 本文配置示例；须先确认未占用 |
| 本机 ZMQ 服务 | `127.0.0.1:6001` | 两个 `experiments` 入口默认参数 |

在 Windows 有线网卡 IPv4 属性中填写未占用的 `192.168.5.x` 地址，子网掩码示例 `255.255.255.0`。仅直连机器人控制网络时，通常不需要给这个网卡填写默认网关和 DNS；按现场网络规划调整。不要把电脑设成 `.1` 或 `.2`。这里的 `/24` 是建议的本地布网方案，仍需与实际控制柜网络设置一致。

两个控制柜如果出厂都为同一 IP，应先隔离连接、通过匹配的官方软件逐一修改为不同地址，再接回同一网络。修改前记录原设置。

若选择保留现场其他 IP，必须同时检查：

- `task_skills/layout.py::PHYSICAL_NOVA_IP`；
- `dobot_control/robots/dobot.py::DobotRobot.__init__` 中硬编码的 `self.com_list`；
- `experiments/run_control.py` 中固件检查和型号查询使用的硬编码 IP。

只修改 `layout.py` 不够；原代码的夹爪字典和 HTTP 请求仍可能指向旧地址。首次准备建议在现场允许的前提下使用仓库默认网段，任何改动都要记录。

### 5.2 在 DobotStudio Pro 中配置两台控制柜

以下流程参考官方轻量版手册印刷页 21～25、DobotStudio Pro 2.8 手册。这里只使用同一台 Windows 电脑：先作为控制柜配置终端，再作为遥操主机，不需要另配一台 Ubuntu 电脑。

1. 若两个控制柜 IP 相同，先只把**待设置为右侧 `.2` 的控制柜**连接到电脑网络；保持另一台隔离。把电脑有线网卡设为未占用的 `192.168.5.10`、掩码 `255.255.255.0`。Windows 可运行 `ncpa.cpl`，右键正确的有线适配器 → 属性 → IPv4 → 属性，填写地址。
2. 打开已安装的 DobotStudio Pro，搜索或按控制柜当前实际 IP 手动添加设备，确认连接的是目标控制柜。不要同时接入两个地址相同的控制柜后靠搜索顺序判断左右。
3. 进入“设置 → 通讯设置 → IP 设置”，选手动设置，把该控制柜改成 `192.168.5.2`，应用后使用新地址重新连接。记录修改前的 IP。若现场已正确分配 `.1/.2`，跳过修改。
4. 恢复套装网络连接，确认左侧控制柜为 `.1`、右侧为 `.2`；在软件中分别添加、连接并记录两个控制柜型号和固件版本。不要把“软件版本 2.8”误当作“控制柜固件版本 2.8”。
5. 核对报警、安装姿态、实际工具及负载参数。官方原套装示例填写 0.5 kg，但这是示例装配的设置，当前装置应按实装末端负载核对，不在本步骤照抄后立即使能。
6. 核对第 7.2 节的初始姿态差异。**此时不要直接套用官方旧版初始角度，也不要点击“移动至初始点位”**。本仓库的 home 与该官方示例左右相反，需先确认实际物理映射和运动路径。
7. 对每台控制柜进入“设置 → 远程控制”，将模式设置为“TCP/IP 二次开发”，按对应固件界面要求应用。这里不是启动一个自主运动的“远程工程”，也不需要运行示教脚本。
8. 两台分别完成后，退出 DobotStudio 的设备控制连接，停止其中的运行工程。不要让它与 Python 遥操同时发运动指令。后续发生报警时，先停止遥操，再用 DobotStudio 查明并处置。

若菜单名称与上述不同，以实际匹配版本手册为准；第 1～3 步软件准备已完成，但本助手未核验控制柜界面，以上第 5 步设置仍需现场确认。若固件是 V4，先停止此 V3 仓库的启动流程，核实兼容的控制代码与软件版本。

### 5.3 检查网络服务

以下命令只检查连通性，不发机器人使能或运动指令。需在完成接线后由你执行：

```powershell
Test-Connection 192.168.5.1 -Count 2
Test-Connection 192.168.5.2 -Count 2

foreach ($robotIp in @('192.168.5.1', '192.168.5.2')) {
    foreach ($tcpPort in @(29999, 30003, 30004, 22000)) {
        Test-NetConnection -ComputerName $robotIp -Port $tcpPort -InformationLevel Detailed
    }
}
```

| TCP 端口 | 当前代码用途 |
| --- | --- |
| 29999 | Dashboard：使能、速度、关节/位姿、报警等 |
| 30003 | 运动命令，包括 ServoJ |
| 30004 | 控制柜状态反馈 |
| 22000 | HTTP 查询 `/settings/version`、`/properties/controllerType` |
| 6001 | 电脑内部 ZMQ 服务；它不是机器人控制柜端口 |

**验收：** 两臂所需 TCP 端口可达；Ping 可能被禁用，不能仅凭 Ping 成败下结论。TCP 通也不等于接口协议和固件兼容。若 VPN/其他网卡也使用 `192.168.5.0/24`，先排查路由冲突；无需为此关闭整个 Windows 防火墙。

## 6. 备份并修改本机硬件配置

**2026-10-22 已完成本机字段修改，下面的原值表用于追溯，不要重新覆盖回原值。** 修改前备份：`.local/backups/dobot_settings-before-preflight-20261022-161520.ini`。

| 当前配置段 | 实测端口/序列号 | 通讯证据与设置 |
| --- | --- | --- |
| `HAND_LEFT` | `COM6`，USB S/N `5B42009608` | IDs 1～8 全部响应；`baud_rate = 2000000` |
| `HAND_RIGHT` | `COM5`，USB S/N `5B42009490` | IDs 11～18 全部响应；`baud_rate = 2000000` |
| `GRIPPER_LEFT` | `COM3`，USB S/N `BG038N42A` | ID21，1000000 baud；读取位置 3051，物理侧待确认 |
| `GRIPPER_RIGHT` | `COM4`，USB S/N `BG038LL4A` | ID22，1000000 baud；读取位置 3048，物理侧待确认 |
| `CAMERA` | `top = 218622278417` | 用户确认在顶部；`enabled = top`、`fps = 15` |

主手在 1000000 baud 下无响应，在 2000000 baud 下各 8 个 ID 的 PING/位置读取全部成功。未向设备写入波特率，只改了电脑使用的连接参数。重新插拔若 COM 号变化，按 USB S/N 对应更新，不能按枚举顺序分左右。

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
New-Item -ItemType Directory -Force E:\Dobot\.local\backups | Out-Null
$backupStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
Copy-Item -LiteralPath scripts\dobot_config\dobot_settings.ini -Destination "E:\Dobot\.local\backups\dobot_settings-$backupStamp.ini"
notepad scripts\dobot_config\dobot_settings.ini
```

### 6.1 逐项核对 INI

| 配置项 | 仓库值 | 现场操作 |
| --- | --- | --- |
| `HAND_LEFT.port` | `/dev/ttyUSB0` | 当前 COM6；按 ID 确认，配置名称不等于实体侧 |
| `HAND_RIGHT.port` | `/dev/ttyUSB3` | 当前 COM5；本轮实体左主手事件对应此段 |
| `GRIPPER_LEFT.port` | `/dev/ttyUSB2` | 改成该配置段对应夹爪的实测 `COMn`，同时核对下方映射 |
| `GRIPPER_RIGHT.port` | `/dev/ttyUSB1` | 同上 |
| `HAND_LEFT.joint_ids / append_id` | `1,2,4,5,6,7 / 3` | 与主手标签/配置核对 |
| `HAND_RIGHT.joint_ids / append_id` | `11,12,14,15,16,17 / 13` | 与主手标签/配置核对 |
| 两主手 `baud_rate` | `1000000` | 按实际配置核对；识别脚本也尝试 `2000000` |
| 主手夹爪 `gripper_config` | 左 `8,204,174`；右 `18,203,173` | 格式为 ID、关闭角、打开角；角度是该装置标定值 |
| 两主手 `joint_offsets / joint_signs` | 已有数值 | 不保证适用于新设备；按第 7 步核对/标定 |
| 两主手 `using_sensor` | `0` | 按实际传感器确认，不能仅改为 1 就声称保护生效 |
| `GRIPPER_LEFT.id / pos` | `21 / 2048,3052` | 核对舵机 ID 与机械行程 |
| `GRIPPER_RIGHT.id / pos` | `22 / 2048,3052` | 同上 |
| `CAMERA.top` | `230322276936` | 改成顶部相机序列号 |
| `CAMERA.left` | `218622275674` | 改成左侧相机序列号 |
| `CAMERA.right` | `218622275344` | 改成右侧相机序列号 |
| `COMPUTER.passcode` | `<configure-locally>` | Windows 串口扫描跳过 sudo/chmod，不需要填 Windows 密码 |

相机序列号与 COM 号不是通用默认值。最好通过逐一插拔、查看设备管理器并贴标签确认端口，避免用猜测的 COM5/COM6 等覆盖配置。

### 6.2 特别核对左右映射

当前代码按配置通道使用以下映射；配置 LEFT/RIGHT 不代表操作者看到的实体左右：

| 人手输入 | 六轴运动目标 | 夹爪目标通道 |
| --- | --- | --- |
| HAND_LEFT / COM6 / hand=0 | 代码右 Nova，默认 `.2` | 左 Nova 通道，默认 `.1` 的夹爪 |
| HAND_RIGHT / COM5 / hand=1 | 代码左 Nova，默认 `.1` | 右 Nova 通道，默认 `.2` 的夹爪 |

六轴交换由 `master_to_nova_action()` 和 `master_to_nova_flags()` 实现；夹爪由 `master_to_nova_gripper_flags()`、`apply_same_side_gripper_targets()` 保留同侧通道。

但是**夹爪通道究竟连到哪只实体夹爪，必须追线确认**：

- 遥操 `DobotRobot`：`.1 → GRIPPER_LEFT`，`.2 → GRIPPER_RIGHT`，串口读自 INI。
- 原子任务 `task_skills/layout.py`：物理左绑定 `GRIPPER_RIGHT / COM8`，物理右绑定 `GRIPPER_LEFT / COM7`。

这两套配置的绑定并不相同，不能拿原子任务的 COM7/COM8 直接作为遥操配置答案。先完成本次遥操链路的端口和物理绑定验收，再考虑另一套入口。夹爪“0/255 表示开/关”的方向也要实测；README 中的 `0=夹紧、255=张开` 是原装置的原子任务约定，不能自动等同于你现场的夹爪安装方向。

### 6.3 自动找串口脚本是可选步骤

只有已经确认主手/夹爪型号、ID、机械状态且能够处理舵机力矩变化时，才考虑：

```powershell
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m scripts.1_find_port
```

它**不是只读扫描**：会构造主手/夹爪驱动、写入力矩/PID/PWM 等参数，并更新 INI。主手驱动构造时实际上使能力矩，不能只看附近的旧注释；夹爪驱动会写力矩限制。程序至少要求四个 USB 串口，额外未识别串口还可能触发末尾断言。不要把设备识别失败当成理由无限重跑。

**验收：** 重新执行 `tools/check_teleop_local.py --inventory`，四个配置串口和 `CAMERA.enabled` 中的相机都能在列表中找到。当前仅要求顶部一台相机；2026-10-22 已返回 0。仍须人工核对物理接线与通道，枚举通过不能代替这项确认。

## 7. 主手标定与从臂初始姿态

**2026-10-25 当前停点：** 本次更新的是下面的详细操作流程，尚未替现场完成标定或写入新偏置。现有 `joint_offsets`、`joint_signs`、`start_joints`、夹爪参数、home、Tool 1 和安全边界保持不变。此前“以仓库为准”只是采用现有数值，不等于实体零位与方向已经验证；按 7.1.1～7.1.10 将主手关节读数标定到对应从臂的关节坐标。

### 7.1 启动遥操前：以对应从臂的关节坐标标定主手

本节于 **2026-10-25** 扩充。按 **7.1.1 → 7.1.10** 顺序做完再运行 9.1/9.2。先做 COM5 对应左 Nova，再对 COM6 对应右 Nova 完整重复。两个主手的偏置不能互相复制。

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

依据：官方轻量版手册印刷页 28～29（本地 PDF 阅读器第 33～34 页）也是“独立解锁主手 → 与从手初始姿态一致 → 锁定并退出摆位脚本 → 保存同步基准 → 启动遥操”的顺序。本文命令与偏置保存方式以本地改造后的代码为准，不直接执行官方旧文件名 `2_gello_get_offset.py`。本地资料：[X-Trainer 轻量版手册](manuals/XTrainer-Light-V1.3-20240729-cn.pdf)、[DobotStudio Pro 手册](manuals/DobotStudio-Pro-2.8-CR-Nova-cn.pdf)。

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

4. 用文本编辑器打开 INI，确认两段 `absolute_mapping_verified = 0`。若以前已有一侧验收为 1，本次要改那一侧的标定时先改回 0；改动前已有上述备份。不要改相机、夹爪、home 或安全边界。

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
$poseProcessId = [int](Read-Host '91504')
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
6. 六轴记录建议填表：`J序号 / servo ID / 从臂前后角 / 主手前后raw / 推导sign / 方向确认方式`。当前两段 signs 都是 `1,1,-1,-1,-1,1`，它们只是待核验的已有值，不是本节结论。只有反复实测证明某轴符号不符，才在本侧 INI 修改对应项，并保持核验标志为 0。
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
| absolute_mapping_verified、验收人/日期 | 当前 0，待验收 | 当前 0，待验收 |

### 7.2 从臂

`task_skills/layout.py` 给出的初始关节角（度）为：

```text
物理左 Nova：[ 90, 0,  90, 0, -90, -90]
物理右 Nova：[-90, 0, -90, 0,  90,  90]
```

`robot_pose_init()` 会检查并可能把**两臂**插值移动到这些角度，未经现场核对不能直接采用。目标姿态安全也不代表从当前位置到目标的整条路径安全。

**官方原版示例与本 GitHub 提交的关键差异：**

| 对象 | 官方轻量版手册印刷页 24 | 本仓库 `layout.py` |
| --- | --- | --- |
| 左侧从手 / 默认 `.1` | `[-90, 0, -90, 0, 90, 90]` | `[90, 0, 90, 0, -90, -90]` |
| 右侧从手 / 默认 `.2` | `[90, 0, 90, 0, -90, -90]` | `[-90, 0, -90, 0, 90, 90]` |

本次以 GitHub 代码为准，不能先按官方表把实体臂移动到位，再假定仓库不会改变其姿态。两表相反可能对应不同装置摆放、物理左右定义或代码改造；本次没有实物证据去判断是哪一种。现场必须追线确认 IP→实体臂、安装方向和主手映射，再决定采用经核验的 home 配置；未确认前停在本步骤，不运行第 9.3 节自动归位。

还要核对 `DobotRobot.__init__` 的工具配置：它会写入 `SetTool(1, 0, 0, 197, 0, 0, 0)` 并选择 Tool 1，即以 197 mm 的 Z 偏置作为工具设置。确认实际工具长度、控制柜负载和安装姿态适合这套设置，才能进入真机阶段。原子任务使用的 Tool 0 配置不等于此处的 Tool 1。

## 8. 独立验证当前启用的顶部相机

先关闭 RealSense Viewer，再执行：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m scripts.5_camera_read
```

本地已修改该脚本：读取 `CAMERA.enabled/fps`，只打开顶部相机，不再调用 `hardware_reset()`，不连接机器人或主手。按 Q、Esc 或 Ctrl+C 结束，退出时释放相机。不要与 Viewer 或遥操同时占用相机。

本机当前配置：

2026-10-22 晚更新：正式 15 fps 配置已完成预热并连续采集 450 组彩色/深度帧（29.969 秒），结束后释放相机；详细记录见联调文档。9.1 服务仍正常运行时，只需重试 9.2，不要为相机问题归档仍有效的设备锁。

- 顶部 RealSense D405：`218622278417`，固件 `5.13.0.55`；
- Depth：`640×480`、`z16`、`15 fps`；Color：`640×480`、`bgr8`、`15 fps`；
- 当前 USB 协商为 `2.1`。30 fps 在早先短测通过，但 10 月 22 日晚复测在预热时出现 `Frame didn't arrive within 5000`，当前降为 15 fps；15 fps 对照测试已收到 180 组帧。确切硬件原因尚未确定，不能仅凭 USB 2.1 判定线缆损坏。

**早先 30 fps 验证记录（不代表持续稳定）：** 实际相机类完成预热后，连续采集 180 组彩色/深度帧，用时 6.000 秒；遥操实际相机线程另运行 5 秒，顶部裁剪、JPEG 编码和线程状态正常，结束后已释放相机。最后的 `--inventory` 返回 0。

可重复的有限采集检查（不弹窗、不动机器人）：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m scripts.5_camera_read --frames 180 --no-display
```

接手时用预览检查实际画面方向和取景。运行遥操后仅显示顶部画面，保持上游的顶部旋转/裁剪处理。以后需要三相机时，再填入实测左右序列号并改成 `enabled = top,left,right`，重新验证带宽与三路出流；只有三路全部启用才恢复三视角录制。

## 9. 首次真机启动：严格按两个终端顺序

**进入条件：** 第 1～8 步通过；现场人员确认工作区清空、两臂和夹爪无持物、急停有效、通道/工具/标定已核对；没有 HTTP live 网关、原子任务或另一份遥操占用设备。急停属于现场硬件处置，不能用停止 Python 进程代替。

### 9.0 本次交接后先做这几项

1. 先完成第 7.1 节映射核验。本次两侧核验标志为 0，长按将出现 `ALIGN_REJECTED_UNCALIBRATED`；不应直接绕过。保留第 9.2 的“禁止自动归位”方式。该标志只阻止长按对齐，9.1 的夹爪启动动作和 9.2 的主手初始化仍会操作硬件。
2. 在 DobotStudio 结束控制并退出连接，关闭相机 Viewer/预览程序，确认控制柜处于 TCP/IP 二次开发模式。DobotStudio 能手动移动不等于 Python 控制端口的运动链路已经验收。
3. 追线核对 `.1/.2` 与实体左右臂、ID21/ID22 夹爪实际位置。当前夹爪按仓库 ID 对应 COM3/COM4，物理侧未验证，不要通过带物动作试错。
4. 若现场证实两只实体夹爪与预期通道相反，停止两个 Python 进程、按第 11 节处置设备与锁后再改配置；按设备身份成套交换对应 `port + id + pos`，不能只交换 COM3/COM4 而保留不匹配的 ID。修改前备份，重做只读查询，再从第 9.1 开始。
5. 第 9.2 出现 `Safe stop before pose motion` 属于归位检查停点。只有核对 home、Tool 1 和整条运动路径后才进入第 9.3。用户此前已实现跟随；本轮修改后的实体映射与运动仍待现场复测。

### 9.1 终端 A：启动机器人服务

2026-10-23 修复：启动夹爪动作后位置读取的 `IndexError` 已增加协议回包长度校验和动作结果检查，34 项离线测试通过；速度和左后右顺序不变。见 [夹爪回包修复记录](manuals/GRIPPER_REPLY_FIX_20261023.zh-CN.md)。

**重新启动时注意：** 上一次服务即使已经退出，也会保留设备锁。若提示 `owned or quarantined`，先按第 11.2 节检查进程和连接、归档确认为已退出进程留下的锁；不要反复启动或改用另一个锁路径。2026-10-22 16:57 已处理 PID 21616 留下的旧锁，记录见联调文档。

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
$env:DOBOT_DEVICE_LOCK_PATH = 'E:/Dobot/.local/runtime/device-owner'
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B experiments\launch_nodes.py --hostname 127.0.0.1 --robot-port 6001
```

**这条命令会立即操作硬件，没有 YES 确认提示。** 构造两台 `DobotRobot` 时，会依次：

1. 建立控制柜连接并调用 `EnableRobot()`。
2. 设置 `SpeedFactor/AccJ/SpeedJ` 为 5。
3. 写入并选用前述 Tool 1。
4. **先配置左夹爪、后配置右夹爪**：左通道 `.1 → GRIPPER_LEFT → COM3 / ID21` 先执行 `move(0,10,1)`、`move(255,10,1)`；每个目标均读取位置确认到位（0～255 量程内容差 5），再停顿 0.5 秒。左夹爪完成整个往返后，才继续右通道 `.2 → GRIPPER_RIGHT → COM4 / ID22`。启动速度从 100 降为 10，底层速度值由 4096 降为 409；这不是实际线速度的测量值。每个目标最多等待 10 秒，超时或读取失败会中止后续启动，不继续下一夹爪。终端打印 `[GRIPPER START]`（通道、IP、COM、ID）、`[GRIPPER MOVE]` 和 `[GRIPPER DONE]`，可据此对照实体侧。左右名称仍是配置标签，尚不代表物理位置已确认。
5. 退出拖拽模式，启动状态反馈监控。

因此必须在运行前清空夹爪附近区域；没有启用姿态初始化开关也不会阻止这些启动动作。这些速度设置不构成对所有 ServoJ 行为的独立安全保证。

**验收：** 服务正常显示 `Starting robot server on port 6001` 及绑定信息；两台控制柜无异常、夹爪与实际绑定符合预期。保持该终端运行。

### 9.2 终端 B：先不授权自动归位

2026-10-23 按键修复：主手无效短回包不再形成不规则数组；按钮数据过期、读线程或按钮线程异常时停止客户端继续下发动作。首次有效按键状态建立后才报告按钮初始化成功。见 [按键与线程修复记录](manuals/BUTTON_MONITOR_FIX_20261023.zh-CN.md)。修改后须先退出旧终端 B 程序再重启；A 仍正常运行时无需重启 A 或清锁。

新开一个 PowerShell：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
$env:DOBOT_DEVICE_LOCK_PATH = 'E:/Dobot/.local/runtime/device-owner'
Remove-Item Env:DOBOT_ALLOW_SAFE_POSE_INIT -ErrorAction SilentlyContinue
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B experiments\run_control.py --hostname 127.0.0.1 --robot-port 6001 --show-img
```

客户端会初始化当前启用的顶部相机、两个主手，查询控制柜固件与型号，设置指示灯 DO 状态，然后打印 `[POSE CHECK]`。主手初始化可能使能主手力矩；指示灯输出使用 DO 1/2/3，应确认现场连接的是套装指示灯。单相机模式会提示三视角录制禁用，B 键不会写入三视角数据集。

若从臂姿态与仓库 home 不同，会出现 `Safe stop before pose motion`，这是阻止自动归位的预期行为。**这不是全系统只读模式**：终端 A 已使能从臂并驱动过夹爪，终端 B 也已初始化主手和 DO。

若姿态已极精确匹配（阈值 `1e-6 rad`），程序可以直接继续进入待操状态，不会强制报这个停止信息。无论哪种情况，尚未核对完成前不要长按 A。

如果客户端退出或抛异常，确认其进程及后台线程已经退出，必要时在该终端 `Ctrl+C`；不要同时再启动第二个客户端。

### 9.3 核对路径后，明确启用归位

只有现场人员检查了打印出的当前角度、目标角度、全路径空间、两臂相互关系，确认该装置允许这条初始化路径后，才执行：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
$env:DOBOT_DEVICE_LOCK_PATH = 'E:/Dobot/.local/runtime/device-owner'
$env:DOBOT_ALLOW_SAFE_POSE_INIT = '1'
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B experiments\run_control.py --hostname 127.0.0.1 --robot-port 6001 --show-img
```

此时初始化会向两臂发送插值运动。代码默认每步最大关节增量 `0.02 rad`，步间等待 `0.05 s`；实际 `DobotRobot.command_joint_state()` 调用 ServoJ 时传入 `0.03`。初始化工具中的 `_SAFE_POSE_INIT_SERVO_T_S` 目前只参与打印，并未传到底层 ServoJ，不能以它来认定实际控制周期。

不要在首次调通时擅自调大这些参数。代码中的几何保护、主手突变检查和速度限幅不能替代现场路径核验及实体安全措施。

**验收：** 两臂按预期完成初始化；终端出现 `robot init success....`、`button thread init success...` 和 `Ok, let's start`，当前启用的顶部相机正常。初始化成功不等于按钮映射已经验证。

## 10. 从单侧到双侧验证遥操

**当前行为（2026-10-23）：映射核验和启动路径采样通过后，长按开启慢速对齐，再按主手映射后的绝对关节角跟随。当前两侧核验标志为 0，需先完成第 7.1 节。** 对齐默认 `0.15 rad/s`（约 8.6°/秒），六轴实测及指令误差均在 `0.02 rad`（约 1.15°）内连续 3 个样本后切换正常跟随。正常关节速度上限保留 `4.0 rad/s`，实际仍受 TCP 及其他保护约束。最新以 [映射排查记录](manuals/MAPPING_AUDIT_FIX_20261023.zh-CN.md) 为准；[绝对跟随调整记录](manuals/ABSOLUTE_TELEOP_20261023.zh-CN.md) 和下面相对跟随记录仅作历史参考。

2026-10-23 更新：已修复长按后预热周期 `cycle_elapsed` 未赋值，以及预热结束后主手参考被每周期重置导致相对位移为零的问题。新增带非零主手位移、双侧真实运动学检查和停止/重启的模拟流程，相关回归共 60 项通过。见 [长按跟随修复与验证记录](manuals/SERVO_ENABLE_FIX_20261023.zh-CN.md)。重试前退出旧终端 B 程序，再执行 9.2；A 仍正常运行时无需重启 A 或清锁。

### 10.1 按钮行为以当前代码为准

| 操作 | 代码行为 |
| --- | --- |
| A 短按后释放，时长小于 0.5 秒 | 切换相应主手的锁定/解锁 |
| A 长按后释放，时长大于 1 秒 | 切换相应主手跟随状态；开启时自动解除该主手力矩锁定 |
| A 按住约 0.5～1 秒 | 不满足上述分支，可能无动作 |
| B 按下/释放 | 切换采集逻辑，受跟随状态和内部计数影响；以采集输出为准 |

不要同时按两只主手的 A 键进行初次验收；按钮线程共用部分计时变量，逐一验证更容易核对行为。

1. 手持并支撑好左主手，确认两臂及两只夹爪附近均安全。
2. **第 7.1 节核验通过后**，握稳实体左主手（本次日志对应 `HAND_RIGHT / COM5 / hand=1`），A 长按超过 1 秒后释放。仅当映射验收标志及初始目标路径采样均通过，才出现 `ALIGN_START`，由左 Nova 慢速对齐。若出现 `ALIGN_REJECTED_UNCALIBRATED` 或 `ALIGN_REJECTED_UNSAFE`，本次不开启对齐，并关闭双侧跟随、重新锁定主手；按日志回到第 7.1 节，不反复长按。
3. 对齐期间保持主手稳定，`ALIGN_DONE` 后再小幅逐轴操作，确认左 Nova 六轴与已验证的对应关系一致。运行中主手目标会更新，每周期仍检查空间和关节约束；启动采样预检不保证之后任意移动都安全，也不是完整碰撞检查。可再次长按停止，不需开启 9.3 的 home 自动归位开关。
4. 再次左 A 长按超过 1 秒后释放，确认 `SERVO_OFF`，主手重新锁定、从臂不再跟随。不要把短按解锁/锁定等同于从臂伺服已关闭。
5. 对另一主手按实测 COM 标签重复，`HAND_LEFT / COM6 / hand=0` 对应右 Nova 六轴；不要仅凭配置段的 LEFT/RIGHT 推断实体侧。夹爪映射单独核对，不由六轴映射推断。
6. 两侧分别通过后，再依次开启两侧跟随，检查双臂空间关系，完成小范围双臂遥操。

这里“相同位置”指经当前偏置、方向符号和左右交叉映射后的六轴关节角一致，不是两台机器在同一世界坐标点重合；限速或反馈滞后时仍有暂时跟随误差。标定错误不能靠开启绝对跟随自动修正。夹爪在对应主手的从臂对齐完成后恢复原有同侧通道映射。

如果只出现 `TCP_SPEED_LIMITED`，程序会缩小该周期运动步长并继续跟随，不需要重新长按。若出现 `SERVO_REJECTED` / `MASTER_DROP_LOCK` / 按钮失联错误，仍按保护停点处理。空间/关节拒绝日志现包含准确原因，不再把所有位姿拒绝都标成 `joint_safety`。

若一侧映射、方向或夹爪动作与预期不符，立即停止跟随，按需要使用实体急停，退出后核对接线和配置。不要在跟随运行中换串口、改符号或重标定。

`using_sensor_protection` 在 `run_control.py` 中默认是 `False`；INI 中的 `using_sensor` 只是主手驱动读数配置，不能据此声称传感器防跌落保护已开启。当前另外有软件主手突变保护，但两者不是同一机制。

### 10.2 可选：最后再验证采集

先完成纯遥操，再按 B 验证采集。当前计数逻辑从零开始，首次 B 释放可能只是切到停止分支，不能简单认定“第一次按 B 必然开始录制”。观察终端和文件变化。

仓库默认输出路径根据 `Path(__file__).parent.parent.parent`（脚本文件的三级父目录）计算，因此在当前克隆位置会写到：

```text
E:\Dobot\datasets\dataset_package_test\collect_data\<时间戳>\
```

这是仓库移入子目录后自然得到的**工作区根目录下的数据路径**，不用修改代码。`save_data_path` 和 `project_name` 在 `Args` 中没有类型标注，不在当前 `--help` 中，不能直接添加 `--save-data-path` 使用。本次不要求采集训练数据，先不要按 B 开启录制；即使不录制，客户端也会创建输出目录。

**当前单相机模式禁用本节的数据录制，B 键不生成三视角数据。** 以后启用并验证三相机后，记录才包含三路图像和观测/动作数据，写盘通过有界异步队列完成。查看 `CONTROL_LOOP_PERF` 的循环频率与丢帧统计，出现 `RECORD_QUEUE_FULL` 说明写盘跟不上。相机帧率（当前为 15 fps）不等于遥操控制频率。

数据转换脚本 `4_collect2train_data.py` 等仍带 Linux 路径和独立依赖，属于后续训练准备，不纳入首次遥操验收。

## 11. 停止、退出与设备锁恢复

### 11.1 正常停止顺序

1. 若正在采集，停止采集并确认文件写入结束。
2. 分别长按 A 关闭已开启的跟随，确认 `SERVO_OFF` 和实体机器人静止；支撑主手，确认锁定状态。
3. 在终端 B 按 `Ctrl+C`，确认客户端进程结束。
4. 在终端 A 按 `Ctrl+C`，确认服务进程结束。
5. 按现场控制器流程确认两臂停止，并在需要时禁用机器人。**退出 Python 不保证发送 DisableRobot，也不保证夹爪松开。**
6. 在终端 B 清除临时归位开关：

```powershell
Remove-Item Env:DOBOT_ALLOW_SAFE_POSE_INIT -ErrorAction SilentlyContinue
```

如果有非预期运动，应优先执行现场急停流程；不要等待网络请求或 Python 正常退出。

### 11.2 下次启动提示“owned or quarantined”

这版 `launch_nodes.py` 会主动保留设备锁，因为旧 ZMQ 链路没有经过验证的完整关停协议。即使正常结束，也可能保留：

```text
E:\Dobot\.local\runtime\device-owner\owner.json
E:\Dobot\.local\runtime\device-owner.scene-invalid
```

处理顺序：

1. 现场确认两臂、夹爪、物体状态安全，确认所有控制程序停止。
2. 查看锁记录：

```powershell
Get-Content -LiteralPath E:\Dobot\.local\runtime\device-owner\owner.json
```

3. 对照 `host`、`pid`、`purpose` 检查对应进程；PID 可能复用，不能只看“PID 存在/不存在”就判断设备所有权。
4. 确认确实为已退出进程留下的锁后，在资源管理器中将**这个确切的 `device-owner` 文件夹**改名为带时间戳的备份，例如 `device-owner.closed-20261020-153000`，保留记录。不要删除整个 `.local/runtime`，不要在服务仍活着时处理锁。
5. 保留旁边的 `device-owner.scene-invalid`。它记录遥操可能改变了场景；未来切换 HTTP live 网关时，必须另按网关文档完成现场复位。
6. 下次在所有控制终端继续使用同一个 `DOBOT_DEVICE_LOCK_PATH`，然后重新从第 9 步开始。

`task_skills.gateway.operator` 只记录现场复位，**不会清除 owner 锁**。也不要通过改成另一个锁路径绕过尚存的控制进程。`run_control.py` 本身不申请该锁，必须靠单客户端运行纪律避免多个客户端同时连接同一服务。

## 12. 常见问题，按现象排查

| 现象 | 排查方向 |
| --- | --- |
| `No module named serial/requests/...` | 确认使用 `.venv\Scripts\python.exe`；运行 lock 文件安装和 `pip check` |
| `No module named dobot_control` | 回到根目录；对于数字开头的 scripts 文件使用本文 `python -m scripts.1_find_port` 等形式 |
| `/dev/ttyUSB...` 无法打开 | Windows 上须填写实际 COM 号 |
| COM Access denied / 串口占用 | 退出串口调试工具、其他 Python 控制进程；检查设备管理器 |
| 只能识别主手，夹爪失败 | 核对夹爪供电、串口、ID 21/22、波特率；检查实体与配置段绑定 |
| 相机启动失败 / 不支持流配置 | 检查 CAMERA.enabled 中启用的 SN、fps、USB 带宽、Viewer 占用；当前 top / 15 fps；先单独运行第 8 步采集检查 |
| 不显示图片仍启动相机 | `--show-img` 只控制显示；按 CAMERA.enabled 初始化，本次仅 top，不要求侧相机 |
| 固件检查失败或卡住 | 检查 `22000`、V3 版本及 HTTP 响应格式；当前 HTTP 查询没有显式 timeout |
| `Waiting to connect` 长时间等待 | 先确认终端 A 启动成功、两端都为 6001；ZMQ 客户端接收也没有超时 |
| `Safe stop before pose motion` | 是归位保护；先评估打印姿态和路径，再按第 9.3 步选择是否授权 |
| `owned or quarantined` | 按第 11.2 节核实并处理保留锁，不要盲删 |
| A 没反应 | 核对短按/长按时间、主手端口与按键日志；避免两只 A 同时按 |
| 按实体左右观察与名称不一致 | 用配置段/COM/hand 索引追踪：COM5 hand=1 → 左 Nova，COM6 hand=0 → 右 Nova；夹爪独立核验 |
| `ALIGN_REJECTED_UNCALIBRATED` | 当前映射未登记验收；回第 7.1 节做只读参考姿态和方向核验，不直接改标志绕过 |
| `ALIGN_REJECTED_UNSAFE` | 初始目标或采样路径不满足原边界/关节约束；查日志首个拒绝点，停止重试，不扩大边界 |
| 反馈帧/报警错误 | 保存日志，核对控制器状态、协议与固件；不要屏蔽状态检查后重试 |
| 保护频繁触发、跟随卡顿 | 查标定、串口链路和 `CONTROL_LOOP_PERF`，保持保护阈值，先定位原因 |

主要日志位于仓库内：`E:\Dobot\dobot-atomic-skills\scripts\logs\teleop_runtime.log`、`E:\Dobot\dobot-atomic-skills\scripts\logs\log.txt`。需要更详细伺服跟踪时，可在终端 B 运行前设置 `$env:DOBOT_TELEOP_TRACE='1'`；它会增加日志量和 I/O，正常使用保持默认关闭。

## 13. 关键代码导航

以下位置均相对 `E:\Dobot\dobot-atomic-skills`，函数名用于避免行号随代码更新失效。在线核对可从[固定提交的源码目录](https://github.com/yanggoumao2/dobot-atomic-skills/tree/c402e9c1242e18eee87369accb140253bca495ac)进入。

| 路径 / 关键对象 | 作用与本次关注点 |
| --- | --- |
| `README.md` | 当前仓库范围与入口；重点是原子技能/网关，不能直接当遥操指南 |
| `requirements.txt` | 上游完整环境依赖，含训练、仿真相关包 |
| `docs/upstream/README.dobot.md` | X-Trainer 原始工程背景 |
| `experiments/launch_nodes.py::launch_robot_server` | 申请设备锁，创建两台真实机器人，启动 ZMQ 服务 |
| `experiments/run_control.py::main` | 三相机、双主手初始化，固件检查，姿态初始化，遥操与采集主循环 |
| `experiments/run_control.py::button_monitor_realtime` | A/B 按钮时序与锁定/跟随/采集状态 |
| `experiments/run_control.py::check_firmware_version` | V3 版本门槛、22000 HTTP 请求及硬编码 IP |
| `scripts/dobot_config/dobot_settings.ini` | 主手端口、ID、偏移、符号；夹爪端口/行程；相机序列号 |
| `scripts/manipulate_utils.py::load_ini_data_*` | 配置读取；路径固定为仓库内这份 INI |
| `scripts/manipulate_utils.py::robot_pose_init` | 默认阻止有差异的归位运动，环境变量控制显式启用 |
| `scripts/manipulate_utils.py::master_to_nova_*` | 六轴交叉映射与夹爪使能映射 |
| `scripts/manipulate_utils.py::apply_same_side_gripper_targets` | 夹爪目标恢复同侧通道 |
| `dobot_control/agents/dobot_agent.py` | 主手配置对象、关节/按键读取、力矩接口 |
| `dobot_control/agents/agent.py::BimanualAgent` | 双主手组合与读取 |
| `dobot_control/robots/dynamixel.py` | 主手原始角度的偏移、符号、夹爪归一化 |
| `dobot_control/dynamixel/driver.py` | 主手串口、Dynamixel 协议、力矩/PID、按钮读取线程 |
| `third_party/DynamixelSDK/python/src/dynamixel_sdk/` | 仓库内附带的 Dynamixel SDK |
| `dobot_control/robots/dobot.py::DobotRobot` | Nova 实机初始化、夹爪绑定、状态监控、ServoJ 与关节/位姿读取 |
| `dobot_control/robots/dobot_api.py` | 控制柜 Dashboard、运动和状态反馈 TCP 协议封装 |
| `dobot_control/robots/robot.py::BimanualRobot` | 双从臂对象、命令分发及观测 |
| `dobot_control/robots/robot_node.py` | 本机 ZMQ RPC；默认使用 pickle，保持监听 `127.0.0.1` |
| `dobot_control/env.py::RobotEnv` | `step/get_obs` 与机器人客户端的接口 |
| `dobot_control/teleop_safety.py` | 主手突变、位姿保护与预热阶段检查 |
| `dobot_control/gripper/dobot_gripper.py` | Feetech 夹爪读写、行程换算及错误报告 |
| `third_party/feetech/scservo_sdk/` | 仓库内附带的 Feetech SDK |
| `dobot_control/cameras/realsense_camera.py` | 相机流参数、翻转、RGB/Depth 读取 |
| `dobot_control/async_writer.py` | 有界异步写盘队列与丢样统计 |
| `scripts/format_obs.py` | 观测/动作数据保存辅助 |
| `scripts/function_util.py` | 串口枚举、日志、文件和计时辅助；Windows 与 Linux 权限逻辑不同 |
| `scripts/1_find_port.py` | 自动识别端口并写配置；有硬件参数写入副作用 |
| `scripts/2_get_offset.py` | 只读主手映射诊断；确认参考姿态后可输出偏置候选 JSON，不修改 INI 或电机 |
| `scripts/3_just_buttonA.py` | 独立主手摆位旧辅助，main 不连接 Nova，但构造主手会写力矩/PID；使用和退出限制见 7.1.5，不替代正式遥操入口 |
| `scripts/5_camera_read.py` | 三相机预览；会先复位发现的相机 |
| `scripts/4_collect2train_data.py`、`scripts/script_collect2train.py`、`scripts/6_dataset_count.py` | 后续数据转换与统计，需另核对路径和依赖 |
| `examples/example_dobot_robot.py`、`examples/example_gripper.py` 等 | 硬件示例；运行前读代码，不能视为无动作诊断 |
| `tests/test_teleop_regressions.py` | 双臂分发、夹爪通道、保护、异步写盘等离线回归 |
| `tests/test_gripper_diagnostics.py` | 夹爪通信与错误输出离线验证 |
| `task_skills/layout.py` | 默认 IP、两臂 home、主手参考角度、原子任务夹爪绑定 |
| `task_skills/device_lease.py` | 合作式设备所有权和场景失效标记 |
| `task_skills/run_task.py`、`task_skills/parser.py` | 中文模板解析与原子任务现场入口，和主手遥操是另一条链路 |
| `task_skills/dobot_adapter.py`、`task_skills/scene_config.json` | 原子任务的真机适配与场景标定 |
| `task_skills/atomic_graph.py`、`task_skills/atomic_executor.py` | 原子节点图和执行 |
| `task_skills/run_gateway.py`、`task_skills/gateway/` | HTTP 网关、状态、幂等、设备所有权、场景复位 |
| `task_skills/chemistry/`、`task_skills/run_beaker_*.py` | 化学实验原子技能及烧杯示例，不是本次遥操前置条件 |
| `docs/local-control.md` | 人工执行原子抓放任务的说明 |
| `docs/execution-api/README.md` | 上游执行接口、Mock/live 网关和恢复流程 |

遥操数据流：

```text
两主手 --USB串口--> DobotAgent/BimanualAgent
                         |
                         v
run_control.py --> 交叉六轴目标 / 同侧夹爪目标 / 保护检查
                         |
                         v
RobotEnv --> ZMQClientRobot --本机127.0.0.1:6001--> launch_nodes.py
                                                       |
                                                       v
                                                BimanualRobot
                                                /           \
                                      Nova TCP控制       夹爪USB串口

三台RealSense --USB3--> run_control.py --> 图像显示 / 异步数据保存
```

## 14. 最终验收记录

按实际进度填写。第 1～5 步由你确认完成，2026-10-22 第 6～8 步的读取与相机验证记录如下，机器人运动验收仍由现场完成：

| 项目 | 记录 |
| --- | --- |
| 第 1～3 步：方案、软件/环境准备、离线检查 | 2026-10-22 用户确认完成；本次未重复测试 |
| 照片所示从臂 | Nova 2 / DT-NV020-0 / S/N NV203-4616-0801-00；实体左右待绑定 |
| 另一台从臂型号及 S/N | 待核对另一台铭牌，不能用两张同 S/N 照片代替 |
| DobotStudio 实际版本、路径 | 软件准备已完成，具体值待登记 |
| 两控制柜型号及固件 | `.1/.2` HTTP 实测 Nova 2 / 3.5.8.5，源码解析为 3585，版本门槛通过 |
| 左/右从臂 IP、电脑网卡 IP | `.1/.2` 可达；实体侧与电脑网卡地址待现场登记 |
| 四路串口物理标签、COM、ID | HAND_LEFT COM6 / IDs1～8，实体侧待确认；HAND_RIGHT COM5 / IDs11～18，本次实体左主手按键日志对应此通道；均 2 Mbps。夹爪 COM3 / ID21、COM4 / ID22，实体侧待确认 |
| 当前相机型号、序列号、流配置 | 顶部 D405 / 218622278417，640×480 彩色+深度，当前 15 fps；30 fps 曾超时，侧相机禁用 |
| 主手标定文件及日期 | 按用户要求沿用仓库 INI 原值；未重算偏置、未现场验证标定精度 |
| HAND_LEFT / COM6 → 右 Nova 六轴 | 代码路由已核对，实体姿态映射待验收，verified=0 |
| HAND_RIGHT / COM5 → 左 Nova 六轴 | 用户已观察实体左主手带动左从臂；零位/方向未验收，verified=0 |
| 两主手夹爪输入 → 实体夹爪 | 待逐一确认 |
| 归位目标、整条路径、Tool 1 | 待确认 |
| 单侧开启、停止，再到双侧跟随 | 待验收 |
| 正常退出后的静止/禁用状态与设备锁 | 待验收 |
| 可选采集路径、图像、动作记录 | 待验收 |

只有现场完成上述核对与小范围实际跟随，才能标记“本机有线双臂遥操调通”；本次交付完成的是代码、软件环境和可执行准备流程。
