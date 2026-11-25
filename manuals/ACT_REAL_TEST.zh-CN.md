# ACT真机测试平台与试验方案

11月17日16时按用户要求切回首次65条数据、20,000步run的best.pt，实际checkpoint为第18,500步。配套使用旧模型归一化，继续沿用当前top x≥371遮罩和现有测试控制流程。该旧模型原训练使用完整top画面。ACT72的独立入口为open_act72_local_model.cmd。

## 入口与运行位置

2026-11-17更新。笔记本仓库为 `E:\Dobot\dobot-atomic-skills`。模型在笔记本RTX5060 Laptop GPU上本地推理，笔记本网线连接机器人网络，相机通过USB连接。

控制平台使用 `E:\Dobot\.venv\Scripts\python.exe`，模型服务使用独立环境 `E:\Dobot\.venv-act-inference\Scripts\python.exe`。当前常用入口默认首次20,000步run的第18,500步best.pt。

平台入口为 `open_act_test.cmd`。默认shadow模式，打开top/left预览并显示预测时效，执行模式按R准备后在平台窗口内持续按住空格，才会向实体左臂下发动作。

程序包含三个模式。

| 模式 | 数据来源 | 作用 |
|---|---|---|
| replay | 已录制HDF5 | 检查笔记本本地推理，显示录制画面和预测误差 |
| shadow | 实时相机和A反馈 | 预览实时预测、检查图像方向及模型延迟 |
| execute | 实时相机和A反馈 | 显式按键准备与按住空格执行左臂 |

## 1. 启动笔记本本地模型服务

当前旧best模型后台服务已就绪，health显示step=18500，可直接复用。后台日志为 `.local\runtime\act65-best-model.stdout.log`。以下命令用于服务结束后的重新启动。

模型、归一化统计、固定ACT源码及ResNet缓存已放在项目目录

`E:\Dobot\dobot-atomic-skills\.local\models\act65_20k`

网线从训练主机切回机器人网络后，在PowerShell运行以下命令，或双击仓库里的open_act_local_model.cmd，保留该窗口。

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
.\open_act_local_model.cmd
```

模型服务先加载权重并进行GPU预热。看到 `ACT_INFERENCE_READY 127.0.0.1:8787 step=18500` 后继续。客户端通过笔记本回环地址调用模型，模型加载所需源码和缓存均在本地。服务从当前本地config读取top x≥371填黑参数。

切换到首次run的20,000步last模型时，先结束当前模型服务，再运行

```powershell
.\open_act_local_model.cmd last.pt
```

如果提示8787端口已占用，先关闭旧隧道窗口或旧模型服务，再启动本地服务。

本地best.pt与last.pt为从训练主机导出的推理checkpoint，保留完整策略权重、训练步数及模型参数。训练主机保留原始含优化器状态的checkpoint。normalization.json与splits.json随模型一起保存。

## 2. 先看离线回放

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
.\open_act_test.cmd --mode replay --episode E:\Dobot\datasets\tube_insert_cup_2cam_rgb_marked_v2\act72_final_20261117\episode_35.hdf5 --frames 203
```

显示两视角录制画面，终端输出逐帧请求耗时与左臂单步误差。ESC退出。需要只看终端时加 `--headless`。

## 3. 准备现场与终端A

杯位、试管、相机位置和准备姿态沿用成功示教。移除工作区内妨碍运动的物品，由一名现场人员掌握机器人实体急停。

先停止录制和主手跟随，退出终端B、旧相机预览及其他控制窗口。平台会独占两路相机。既有A必须使用本次代码重新启动，才能提供ACT控制会话仲裁。关闭A和归档旧设备锁沿用Nova2控制台的现有流程。

通过Nova2控制台启动A，或在独立PowerShell按以下命令启动。

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
$env:DOBOT_DEVICE_LOCK_PATH='E:\Dobot\.local\runtime\device-owner'
$env:DOBOT_ACT_RECORDING='1'
E:\Dobot\.venv\Scripts\python.exe -X utf8 experiments\launch_nodes.py --robot-port 6001
```

**A启动会使能双臂并初始化夹爪，现场先确认两臂和夹爪附近安全。** A保持运行，B保持退出。平台复用A，不再建立第二套机器人驱动连接。

2026-11-16串口更新：Windows的Feetech接收等待预算从20ms调整为50ms。初始化位置读取遇到接收超时时最多重试两次读取；夹爪动作写入失败后仍停止，不自动重复动作。ACT夹爪反馈线程会短暂等待正在进行的串口写入，获取总线失败后以10ms间隔再尝试，避免每次竞争都跳过整个100ms采样周期。客户端仍要求夹爪反馈在300ms内有效。

使用现有归位/遥操准备方式将机器人放到与示教一致的准备姿态，然后退出B和相机预览。平台本身从当前姿态开始，无自动归位动作。

实体RIGHT为192.168.5.1，夹爪COM4/ID22，向量0—6。实体LEFT为192.168.5.2，夹爪COM3/ID21，向量7—13。平台只允许后者执行。

## 4. 实时预测预览

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
.\open_act_test.cmd --mode shadow --seconds 60
```

窗口显示top和实体左腕视角、模型步数、预测年龄、当前状态。相机使用当前INI的640×480、15fps、RGB与180度旋转配置。画面应与采集时一致。

检查终端没有状态、夹爪反馈或相机失效错误，预测年龄保持在250ms以内。首次模型请求可能较慢，等待稳定后再开始执行。若预测年龄持续过大，先排查网线/隧道/主机负载，修复后再试。

ESC退出当前平台。实时模式的最长单次运行120秒，默认60秒。

## 5. 分阶段真机试验

### 第一步，左臂短时执行，夹爪写入关闭

准备空旷场景并保持夹爪处于需要的位置，运行

```powershell
.\open_act_test.cmd --mode execute --seconds 30
```

在平台窗口内按R，等新预测到达后持续按住空格，先执行1—2秒，松开观察。松开空格、切换到其他窗口或按P都会停止发送新的ServoJ。R重新准备，ESC退出。

观察运动方向和跟踪是否合理，再逐步延长按住时间。若显示FAULT，先看终端和日志中的原因，现场确认后按R重新准备。

### 第二步，加入左夹爪

```powershell
.\open_act_test.cmd --mode execute --allow-gripper --seconds 60
```

按R并持续按住空格，模型才控制左臂和左夹爪。实际夹爪反馈用于模型输入，夹爪输出按遥操相同范围限幅到0—1。夹爪松开可能导致试管掉落，开始前准备接物区域。

### 第三步，首轮10次完整任务

固定杯位和准备姿态。每次完成抓取、对准、插入、松爪、撤离后，观察试管是否稳定留在杯中。每次单独启动平台生成一个trial目录。

记录人工干预、异常停止，以及失败首次发生在grasp/align/insert/release/withdraw哪个阶段。10次试验后汇总完整成功次数和分阶段结果，作为首轮真机成功率。

## 运动规则与遥操统一（2026-11-17）

按用户要求，ACT直接复用当前遥操的AbsoluteTeleopPlanner及检查函数。默认对齐0.15rad/s，跟随3.2rad/s，单步上限0.35rad，规划器每次计算的时间上限0.05秒。环境变量DOBOT_ALIGN_SPEED_RAD_S、DOBOT_SERVOJ_MAX_SPEED_RAD_S、DOBOT_SERVOJ_MAX_STEP_RAD与遥操使用同一入口，实际值在ACT启动终端和窗口显示。

- 跟随使用六关节共同插值比例，与遥操一致。
- 使能时采样检查到预测目标的关节路径，按遥操容差0.02rad、连续3次稳定后完成对齐。夹爪在对齐完成后按预测值限幅到0—1。
- 复用limit_tcp_descent，只限制遥操现有的向下TCP速度1m/s，按现有实现缩步后复核。复用check_pose_protection工作区与向下速度检查、check_joint_safety的J3方向检查、servo_action_check的0.9rad指令差检查。
- 目标变化检查沿用遥操MasterRunawayGuard，默认0.25rad最小跳变阈值、12.5rad/s、连续2周期触发，读取相同环境变量。
- ACT原先独立的10°/秒关节限速、全方向50mm/秒TCP限制与缩放、3°跟踪误差拒绝、夹爪0.02死区及每秒3单位开合限制已移除。

模型输入有效性检查、预测时效、会话独占、R准备/按住空格执行继续用于现有ACT推理操作流程。执行范围仍由当前客户端的实体左臂通道决定。上述约束统一的是运动速度、位置与运动保护，实际控制频率取决于模型和RPC延迟。

使用时退出旧ACT测试窗口，重新运行open_act_test.cmd加载新版客户端。A及模型服务沿用现有接口。默认execute关闭夹爪写入，需要夹爪时使用--allow-gripper。

## 日志与结果标记

每轮目录位于 `E:\Dobot\act_real_tests\日期时间_随机后缀`，终端会打印TRIAL_OUTPUT。

- config.json记录模式、模型与启动参数。
- events.jsonl记录实测关节、模型预测、预测年龄、下发动作、按键状态及故障。
- 执行记录和故障记录的 `control` 字段保存本次使用的动作块索引、实际选用预测、实测关节、上一命令、控制间隔，以及 `teleop_tcp_descent_limited` 与 `aligned`，用于查看遥操下降限速和对齐状态。检查提前失败时仅保存已计算的字段。
- preview.avi记录双视角和界面信息，按约15fps采样保存。
- execute的session.json保存会话恢复信息。

试验结束后按实际情况标记结果。

```powershell
E:\Dobot\.venv\Scripts\python.exe -X utf8 tools\act_real_test.py --label-run E:\Dobot\act_real_tests\本轮目录 --result success --stage complete --note "试管稳定留在杯中"
E:\Dobot\.venv\Scripts\python.exe -X utf8 tools\act_real_test.py --label-run E:\Dobot\act_real_tests\本轮目录 --result failure --stage insert --note "杯口碰撞后停止"
```

若客户端异常退出，A仍保留ACT所有权，可在现场确认停止、原客户端PID退出后，用本轮session.json释放会话。

```powershell
E:\Dobot\.venv\Scripts\python.exe -X utf8 tools\act_real_test.py --release-session E:\Dobot\act_real_tests\本轮目录\session.json
```

客户端设备锁按现有设备锁处理方式核对owner.json的PID已退出后归档，目录为 `E:\Dobot\.local\runtime\act-client-owner`。正常退出会释放该锁。

## 历史交付验证记录（下列11月16日限速方案已由11月17日统一规则替代）

2026-11-16修复首次执行的命令TCP超速：加入关节增量统一缩放，保留50mm/秒最终保护及原有工作区检查。ACT和遥操相关离线测试57项通过，包含本次故障附近姿态的真实FK回归。使用修复时退出旧测试窗口，重新运行第5节第一步的命令；本次修改不涉及终端A或模型服务，无需为此重启它们。现场仍按1—2秒短时执行观察。

保护逻辑与遥操RPC兼容性的模拟测试共48项通过。平台已完成录制HDF5模型回放联调。现场相机、机械臂和夹爪执行效果由上述分阶段试验验收。

2026-11-14的远程方案曾测得100—130ms请求延迟，日志在 `E:\Dobot\act_real_tests\20261114_020113_80ccdc`。当前已改为笔记本本地模型服务。推理线程按15Hz上限提交，本地30Hz为控制循环目标值，实时反馈及ServoJ的实际周期需要在shadow和短时执行中观察。

2026-11-16在笔记本RTX5060上完成8帧本地回放，日志为 `E:\Dobot\act_real_tests\20261116_145959_c7ffbe`。服务启动时完成GPU预热，首次客户端请求约151ms，后续约58—86ms，多数为58—65ms。本次使用20000步last.pt，原有测试平台成功调用本地服务。现场操作时先等待模型READY，再运行shadow。

## 本地推理环境

独立环境位于 `E:\Dobot\.venv-act-inference`。使用Python3.11、PyTorch2.7.1+cu128及torchvision0.22.1+cu128，其余依赖记录在requirements-act-inference.txt。版本配对依据[PyTorch官方安装说明](https://pytorch.org/get-started/previous-versions/#v271)。

若以后需要重建环境，先用Python3.11创建上述venv，再运行

```powershell
E:\Dobot\.venv-act-inference\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
E:\Dobot\.venv-act-inference\Scripts\python.exe -m pip install -r E:\Dobot\dobot-atomic-skills\requirements-act-inference.txt
```
