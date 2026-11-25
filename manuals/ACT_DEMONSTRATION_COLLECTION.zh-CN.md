# ACT 左臂两视角试管插入示教数据采集操作手册

可通过仓库根目录 `open_local_console.cmd` 打开本地网页，复制或执行本手册的 A/B 启动及数据检查指令，并使用夹爪和归位控制。详见 [本地控制台说明](../docs/local-console.zh-CN.md)。

更新：2026-11-11。适用于 `E:\Dobot\dobot-atomic-skills` 当前 Windows 主从遥操。

目标：杯子固定在桌面，实体左臂从统一准备姿态出发，抓取试管、对准窄口杯、插入到位、松开夹爪并移开左臂，试管稳定留在杯内。当前只启用顶部和实体左腕相机，右腕相机禁用；本机采集、检查与转换，训练和推理在其他机器完成。当前允许只开启左臂和左夹爪跟随，右臂由操作者保持稳定；双臂控制能力及14维记录保留，后续可加入右臂打开插槽盖子的配合动作。

## 1. 首次使用前

1. 固定桌面窄口杯，统一试管规格、初始放置位置、插入深度标记和左臂撤离位置。到位位置须能在松爪后支承试管。先验证人工遥操能够完成插入、释放及撤离，再正式采集。
2. 确认两臂准备姿态、杯子位置及左臂释放后的撤离路径。启动前移走夹爪开闭范围内的物品：服务 A 会执行原有夹爪开闭初始化。
3. 实机标准：**RIGHT 右从臂 .1 / 右夹爪 GRIPPER_RIGHT / COM4 / ID22；LEFT 左从臂 .2 / 左夹爪 GRIPPER_LEFT / COM3 / ID21**。14 维数据继续按 **`.1 六轴、.1 夹爪、.2 六轴、.2 夹爪`** 保存，即右侧在前、左侧在后。HAND_LEFT/RIGHT 是主手配置段，按原标定路由保留。
4. 现场已确认夹爪 **0=完全闭合、255=完全打开**。保存量为驱动位置/255，因此训练观测和动作均为 **0=闭合、1=打开**；数值方向不改变。分别空载开闭两个夹爪验证实际反馈。
5. 当前两路采集使用完整640×480、15 fps，`depth_enabled=false`，保存RGB JPEG、质量50，以及机器人状态/动作。顶部和左腕均旋转180°。顶部看全局，左腕看杯口对齐和释放。本轮增加有色标记并关闭深度流，建议使用新项目 `tube_insert_cup_2cam_rgb_marked_v2`。

本轮采集采用试管与插口有色标记、位置基本固定、偏移限制在预定几厘米区域内、相近抓取与插入姿态的方案。标记贴在非接触位置，避开插入口内缘、试管插入段和夹持接触面。标记应能在实际录制图像中帮助辨认试管方向、尖端位置和插口中心。每条录制过程中固定插口，按相近夹持高度、伸出长度、倾角、插入深度和撤离方向完成任务。不同条之间按预定位置组小幅变化，并在备注中记录位置组，训练与部署保留相同标记。

6. 保持现有标定、工作区和限速设置，按主操作手册完成启动前检查。完整控制细节与锁恢复见 [原遥操手册](../TELEOP_WINDOWS_SETUP.zh-CN.md) 第 9～11 节。

### 当前录制规格（与之前单相机一致）

| 项目 | 当前设置 |
|---|---|
| 启用视角与固定顺序 | top、left；right禁用 |
| RGB分辨率 | 640×480 |
| 相机 / 原始录制 / ACT导出频率 | 15 fps / 最高15 Hz / 15 Hz；控制循环沿用原频率 |
| 深度开关 | `depth_enabled=false`，仅开启RGB流 |
| 原始落盘 | topImg/leftImg下同编号JPEG（质量50）和observation下PKL；不单独保存深度 |
| 图像方向 | top旋转180°、实体左腕旋转180°、实体右腕不旋转 |
| 机器人数据 | 双臂14维，同一行观测/动作配对；右臂在前、左臂在后 |
| 当前USB连接 | 三台均为USB 3.2，采用已验证的单集线器直连电脑方式 |

录制使用独立15Hz节拍，从当前控制周期选取同一行观测/动作和最新有效图像，不再每个控制周期重复写图；卡顿后不追赶补写。ACT默认队列为64条（约4秒缓冲），样本仍有界、提交不阻塞控制，结束标记不占样本容量并在先前样本写完后落盘。若原B终端曾设DOBOT_RECORD_QUEUE_SIZE=8，请先移除该覆盖。

### 相机绑定与当前启用列表

相机使用 USB 序列号，COM 编号属于主手/夹爪。2026-11-10 已现场确认并绑定：top=218622278417（旋转180°）、left=409122273461（实体左腕，旋转180°）、right=218622277170（实体右腕，不旋转）；当前enabled=top,left，右腕编号保留备用。

以下枚举/单机定位步骤供更换设备时使用，日常无需重新绑定。先退出占相机的旧 B/Viewer，在仓库目录运行：

```powershell
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m scripts.5_camera_read --list-devices
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m scripts.5_camera_read --serial 替换为实际序列号
```

依次遮挡左右腕部镜头确认序列号。在 `scripts/dobot_config/dobot_settings.ini` 中填入 `left`（实体左腕）及 `right`（实体右腕），当前设 `enabled = top,left`；`[ACT_TASK]` 同时为 `camera_names=top,left`、`active_arms=left`，用 `left_flip/right_flip` 配置方向（true 为180°旋转）。保留 `fps=15`、`top_flip=true`。不要修改 HAND/GRIPPER 段。

原共享USB 2.1集线器出现三路超时。2026-11-10换集线器并单层直连电脑后，三台均识别为USB 3.2，RGB-only与RGB+深度均通过三轮重复启动和两分钟持续检查。2026-11-11按用户确认，将当前两路相机设为 `depth_enabled=false`，用于RGB-only ACT采集。预览与采集共用此配置，退出占用相机的旧进程后重新启动生效。

现有桌面五条精选来自旧配置，用于原配置链路试跑。本轮深度开关变化会改变配置哈希与 `camera_depth_enabled` 元数据，当前转换器会拒绝将这两种配置混为同一个导出数据集。有色标记也改变了图像外观，新采数据使用独立项目并导出新版本。

当前USB 3.2正式相机线程复测通过：两种模式各三轮，最后一轮两分钟约15fps；RGB+深度最大图像年龄约81ms、最大跨路接收时间差约67ms。详情见 [2026-11-10启动前检查](../docs/act-empty-recording-preflight-20261110.zh-CN.md)。

再做两路预览，检查杯口、试管和释放后的结果可见，支架/线缆不干涉动作。当前右臂保持统一准备姿态；后续需要右腕视角时再恢复right，并同步更新ACT_TASK相机列表、活动臂配置及数据版本。

相机独立预览（先退出占相机的旧 B，不需要启动 A）：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m scripts.5_camera_read
```

Q/Esc/Ctrl+C 退出。该入口只开相机，预览方向与采集相同。旧 JPG/MP4 裁掉的边缘没有保存，修改后重新录制才能得到全画幅。

采集工具所需的额外依赖只有 h5py，本机已安装。重建环境时运行：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -m pip install -r requirements-act-data.txt
```

## 2. 退出旧程序后启动新版

首次绑定修正按先A后B完整启动。此次相机流规格生效需先停止录制并关闭跟随、退出旧B及全部预览，再启动新B；若A已按当前绑定且带ACT反馈正常运行，可继续使用。若需重启A，按原手册核验锁所有者后再先A后B启动。A 将 .1 连接至 COM4/ID22、.2 连接至 COM3/ID21。首次启用 ACT 时两终端都设置反馈开关。
终端 A：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
$env:DOBOT_DEVICE_LOCK_PATH = 'E:/Dobot/.local/runtime/device-owner'
$env:DOBOT_ACT_RECORDING = '1'
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B experiments\launch_nodes.py --hostname 127.0.0.1 --robot-port 6001
```

该命令会使能控制柜，设置原有工具和速度参数，并依次初始化两侧夹爪。等到 `Starting robot server on port 6001`。遇到设备锁占用，按原手册 11.2 确认所有者后处理，不能换锁路径启动。

终端 B：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
$env:DOBOT_DEVICE_LOCK_PATH = 'E:/Dobot/.local/runtime/device-owner'
$env:DOBOT_ACT_RECORDING = '1'
Remove-Item Env:DOBOT_ALLOW_SAFE_POSE_INIT -ErrorAction SilentlyContinue
Remove-Item Env:DOBOT_RECORD_QUEUE_SIZE -ErrorAction SilentlyContinue
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B experiments\run_control.py --hostname 127.0.0.1 --robot-port 6001 --show-img --project-name tube_insert_cup_2cam_rgb_marked_v2
```

两个终端都要设置 `DOBOT_ACT_RECORDING=1`。B 检测不到 A 的真实反馈字段会报错，退出后按顺序重启 A、B；不要在旧 A 上继续采。

新版B在相机初始化后先创建两视角预览窗口，再初始化两只主手，避免首次开窗阻塞采样而触发250ms按钮过期。窗口出现时继续等待 `robot init success`、`button thread init success` 和 `Ok, let's start`，再按原流程操作主手。A正常运行且带ACT反馈时，此次修复只需重启B。详情见 [主手按钮过期修复](../docs/master-preview-startup-fix-20261110.zh-CN.md)。

观察 B 的相机、映射与跟随初始化信息，确认录制根目录为：

`E:\Dobot\datasets\tube_insert_cup_2cam_rgb_marked_v2\collect_data`

初始姿态匹配容差为每个关节0.1°，两臂所有关节均在容差内时跳过归位，不发送动作；超过容差仍需按原手册核对路径并显式授权归位。夹爪开度不参与此判定。

本轮使用新目录 `tube_insert_cup_2cam_rgb_marked_v2`。相机流配置更新后重启B，A正常运行且带ACT反馈时可继续使用。

## 3. 第一次先做空载录制验收

1. 不录制，按现有操作方法分别启停跟随，验证主从方向及夹爪控制。
2. A 键短按不足 0.5 秒切换主手锁定；长按超过 1 秒切换该侧跟随。本次只需左侧完成慢速对齐，观察 `ALIGN_DONE`，再开始录制。0.5～1 秒区间可能无动作。
3. 左侧跟随开启、对齐完成后，按任一 B 键并松开一次，观察 `RECORD_START` 和 episode 路径。两只 B 功能相同，共用一个录制状态：右 B 开始后，左 B 会停止；下一次任一 B 再开始新 episode。不是各自录各自的臂。以完整按下并松开为一次操作；同一轮检测到两只同时松开只切换一次，先后松开可能切换两次，因此每次只按一只 B。
4. 空载缓慢移动左臂，并完整开闭左夹爪，持续至少10秒。右臂保持稳定，无需开启其跟随。
5. 再完整按下并松开 B，观察 `RECORD_STOP`。不要在停止录制前关闭跟随。
6. 停录后等待缓冲排空（通常很快，磁盘停顿后可能需要数秒），确认该 episode 中出现 `episode.json`、`closed.json`、`observation`、`topImg` 和 `leftImg`。
7. 按下一节检查，确认质量报告 PASS，并检查 `telemetry.gripper_range` 左侧通道（第二项）有明显变化。没有变化时先核对动作和反馈，不正式采集。
8. 将空载记录标为 failure，备注“空载验收”，使其不进入成功训练集。

录制开启了约 10 Hz 后台夹爪反馈读取，读写事务共用串口锁；观测只读取缓存。现场要比较开关 ACT 模式前后的跟随响应及 `CONTROL_LOOP_PERF`。出现新增明显卡顿或反馈持续过期时先停止采集并反馈日志。

## 4. 每条插入示教

1. 在录制外固定杯子、复位试管和双臂，核对试管初始位置、深度标记和撤离位置。
2. 让两臂进入统一准备姿态，左侧跟随开启且已完成对齐。录制从抓取之前开始。
3. 按 B 开始，留一小段稳定起始画面。
4. 实体左臂抓取试管，移到杯口并对齐，缓慢插入到位。当前右臂由操作者保持稳定；不改变其跟随控制逻辑。不要把人工摆放、等待或复位混进示教。
5. 到位后松开左夹爪，移开左臂至统一避让位置；确认试管稳定留在杯内、杯子未移位，保持约1秒结束画面后再按 B 停止。必须录到松爪和撤离全过程。
6. 等待落盘完成，记录 episode 名称并进行检查、预览和标注。

失败也按 B 正常结束，标为 failure 并填写原因。机器人/相机/串口异常先按原手册处理；中断且没有完整收尾的记录会被检查器拒绝。不要手工创建 `closed.json` 或改时间数据强行通过。

先采 5 条试验数据，全部检查并交训练端试读，通过后再补至约 50 条成功记录。成功示教以稳定流程为主，可保留少量自然纠偏后成功的完整轨迹。失败另存，默认不进入第一版 ACT。

## 5. 检查、预览与标注

在终端 C 操作，替换下面的示例 episode 名。建议停止当前录制并等待落盘后执行，正式批量转换在退出 B 后执行。

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
$episode = 'E:\Dobot\datasets\tube_insert_cup_2cam_rgb_marked_v2\collect_data\替换为实际episode名称'
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m dobot_control.act_dataset check $episode
```

输出 `PASS` 表示技术检查通过，成功与否仍由任务结果标注决定。结果写入 episode 内的 `quality_report.json` 和同名 HTML。

检查器验证：schema 2、正常收尾、保存计数、14 维有限数值、夹爪范围、当前任务要求的左臂/左夹爪使能、发送完成、单调时间、图像完整、图像帧号、反馈时效和指令间断。新任务还检查当前任务视角顺序和跨路主机接收时间差（上限0.10秒）。质量报告列出图像年龄和时间差统计；这些是主机接收时间。当前状态/图像最大年龄 0.2 秒、夹爪反馈最大年龄 0.35 秒、相邻指令最大间断 0.2 秒。超过阈值会报告具体文件和原因。

生成回看视频：

```powershell
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m dobot_control.act_dataset preview $episode --output "$episode\preview.mp4"
Start-Process "$episode\preview.mp4"
```

视频按 top/left 拼成1280×480，采用时间戳重采样并叠加名称、时间、样本编号；不会把原始高速重复行直接当 15 fps 播放。输出文件存在时拒绝覆盖，需要重生成时选择新名称。

成功标注示例：

```powershell
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m dobot_control.act_dataset label $episode --status success --note '试管规格A，窄口杯固定；深度标记到位；左臂已松爪并移开，试管稳定；已核对top/left两视角及.1/.2实体侧'
```

失败标注示例：

```powershell
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m dobot_control.act_dataset label $episode --status failure --note '撤离时带出试管，未完成；试管规格A'
```

标签默认 `unreviewed`。只有 `success` 且质量检查通过的记录才能导出。失败与未标注数据保留原处。标签可通过同一命令重新设置，备注应记录试管/杯子规格、插入深度、释放和撤离结果、纠偏及现场通道核验。插入后仍夹持或撤离时带出试管不标成功。

## 6. 一次转换为 ACT 数据集

完成当天采集并退出 B 后：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m dobot_control.act_dataset convert 'E:\Dobot\datasets\tube_insert_cup_2cam_rgb_marked_v2\collect_data' --output 'E:\Dobot\datasets\tube_insert_cup_2cam_rgb_marked_v2\act_v1' --fps 15
```

输出目录必须是新目录，避免新旧 episode 和划分混在一起。再次导出使用 `act_v2`。每条候选记录都会给出质量与标签结果；技术合格但 failure/unreviewed 的记录不导出。没有可导出记录时命令以失败退出。

输出：

```text
act_v1/
  episode_0.hdf5
  episode_1.hdf5
  ...
  dataset_manifest.json
  splits.json
  quality_report.json
  quality_report.html
```

50 条成功记录默认生成 40 训练/10 验证，固定随机种子 0，按整条 episode 划分。少量试采也能导出；只有一条时没有验证集，不用于正式训练评估。不同 INI 哈希、相机列表或通道顺序不能混成同一个数据版本。

转换按统一 15 Hz 网格，选择网格时刻之前最近的完整样本，状态、图像和动作来自同一行；不使用未来帧，不对夹爪切换插值。图像转 RGB，HDF5 使用无损 LZF 压缩。保存的动作是通过检查后的目标，不等同于每一时刻机器人已经到位。速度在转换阶段按重采样后的实际状态差分计算。

## 7. 交给训练人员的内容

交付 `act_v1` 整个目录、以下代码及本手册；原始数据另外保留备份：

- `dobot_control/act_dataset.py`
- `dobot_control/act_recording.py`
- `scripts/act_training_adapter.py`
- 对应的包目录/`__init__.py`，或直接交付这次源码版本。

HDF5：`observations/images/{top,left}` 各为 `[T,480,640,3] uint8 RGB`；`observations/qpos [T,14]`；`observations/qvel [T,14]`；`action [T,14]`；`timestamp [T]`；`source_index [T]`。

训练端必须采用本项目的同一行观测/动作配对，取消原 ALOHA 加载器的真实数据 `start_ts-1` 偏移。使用提供的适配器可处理变长 episode 与动作 padding，并只用训练集计算归一化统计。训练端在仓库根目录运行：

```text
python -m scripts.act_training_adapter /path/to/act_v1
```

该命令在已安装 PyTorch 的训练机器上读一个 batch；本机不运行。接入 ACT 时使用 `from scripts.act_training_adapter import load_data`，调用 `load_data(dataset_dir, batch_size=8, chunk_size=30)`，返回 train_loader、val_loader、stats、is_sim。两相机 `camera_names=['top','left']`，图像张量为 `[B,2,3,480,640]`，state/action 14 维，chunk_size 与模型 num_queries 一致。保留训练端 ACTPolicy 原有图像归一化，不重复预处理。

训练端还需用一个 batch 调用其 ACTPolicy 完成前向及反向检查，保存统计量、划分、相机与动作约定。本次本机验收覆盖 HDF5 和 NumPy 加载，不包含 ACT 模型运行。

## 8. 正常退出、常见问题与回退

先按 B 停录并等待文件稳定，再分别长按 A 关闭跟随，确认实体静止、主手支撑并锁定；B Ctrl+C 后检查写盘是否正常结束，再 A Ctrl+C。按原手册完成控制柜与设备锁处理。进程退出不会自动保证机器人禁用或夹爪松开。

| 现象 | 处理 |
|---|---|
| B 提示 A 没有 ACT 反馈 | 检查两个终端环境变量并重启 A、B |
| gripper stale/invalid | 停止采集，检查 A 串口、真实开闭反馈与日志；不要用命令填反馈 |
| 图像过期或帧号不增长 | 检查相机/USB/进程占用，修复后重采 |
| required arm/gripper must be enabled | 本次检查左臂和左夹爪，确认对应跟随开启并完成对齐 |
| 未正常收尾/计数不符 | 检查是否异常退出、写盘故障或队列溢出；保留原数据并重采 |
| 有 RECORD_QUEUE_FULL | 本段会中止并保留aborted收尾；保留日志和数据，检查磁盘停顿及队列覆盖值，不作为成功数据使用 |
| 导出数量少于成功数量 | 看质量报告；成功标签不会覆盖技术错误 |
| 旧3段调试记录被拒绝 | 属正常行为；旧数据缺真实反馈与新时间字段，不能补造 |
| 反馈模式下出现新增跟随卡顿 | 停录、退出 B/A，在两个终端移除 ACT 环境变量并重新启动普通遥操；保留日志用于定位 |

普通遥操回退：两个终端执行 `Remove-Item Env:DOBOT_ACT_RECORDING -ErrorAction SilentlyContinue`，按原手册重启 A、B。普通记录不会被 ACT v2 转换器接收。

首次现场验收请记录：不录制与录制时跟随响应、夹爪通道对应、真实开度变化、B 第一次开始/第二次停止、质量报告、视频、一次完整插入、松爪、撤离结果。相机已完成USB 3.2三轮重复启动及两分钟连续出图检查；真实遥操、夹爪和空载录制验收由操作者按本手册完成。

## 近期操作补充

- 2026-11-10对齐后GetAngle空字符串解析修复在服务A侧：本次需要按原手册正常退出B/A、核验设备锁所有者后先A后B重启。仅重启B不会加载服务A的新收包逻辑。新版本按完整Dashboard回复收包，错误会显示机器人IP、查询和原始回复；见 [通信修复说明](../docs/dashboard-reply-fix-20261110.zh-CN.md)。

- 夹爪端点单独检查：先退出 A/B，双击仓库根目录 `open_gripper_panel.cmd`；选择左 COM3/ID21 或右 COM4/ID22，再点全开/全闭。完成后关闭窗口，再按第 2 节重启 A/B。
- 本次新记录带 `hardware_layout_version=2` 和修正的 gripper_ports；转换器按绑定版本隔离数据。历史 4 段调试录制仍保持 failure 与原回看视频，正式采集使用新目录。
- 报警后出现 RECOVERY_LIGHT_FAILED 表示指示灯更新失败，程序仍执行遥操复位。查 DobotStudio 报警；GetErrorID 第一组 -2 为碰撞检测触发。处理触发原因后恢复。
- `Safe stop before pose motion` 的显式归位入口见主手册 9.3，home 为右 .1 `[90,0,90,0,-90,-90]`、左 .2 `[-90,0,-90,0,90,90]`。

本轮写盘修复及60秒实测结果见 [两视角写盘修复](../docs/two-camera-recording-writer-fix-20261110.zh-CN.md)。
