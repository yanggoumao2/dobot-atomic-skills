# 五条示教 ACT 全链路试跑 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用现有五条示教完成服务器训练、Windows 本地推理、受保护的左臂闭环试跑，并分别记录软件链路与任务完成结果。

**Architecture:** 服务器沿用作者 ACTPolicy/DETR 模型，增加独立 Dobot 训练入口并复用当前数据适配器；模型包回传 Windows，在独立策略进程推理，通过现有机器人服务与保护链路执行。初期无网络实时推理依赖，遥操入口保留。

**Tech Stack:** Linux、NVIDIA GPU、PyTorch/torchvision、作者 ACT、HDF5；Windows、PyTorch CUDA、现有 RealSense/ZMQ/Dobot 控制栈。

**Spec:** 用户本次要求“先用这五条数据训一个 ACT 测试全链路能不能跑通”，并确认“服务器训练、Windows 控制”；任务与协议参照 `E:/Dobot/ACT_handoff_20261110.md`、当前采集手册和桌面 `act_v1/dataset_manifest.json`。本文件是方案；当前未实施下列代码或启动训练、机器人。

## Global Constraints

- 数据目录：`C:/Users/Lenovo/Desktop/ACT_试管插入_精选5条_20261110/act_v1`；五条、1887 帧，SHA256.json 验证后上传服务器英文路径。
- `camera_names=['top','left']`；RGB uint8 原图 480×640；15 Hz；网络输入 `[B,2,3,480,640]`。
- 14 维依次为实体右臂六关节＋右夹爪、实体左臂六关节＋左夹爪；rad、夹爪0闭1开；保持同一行观测/动作，禁止 start_ts-1 偏移。
- 原数据、splits.json 和既有遥操配置不修改；正式试跑训练使用 `[4,3,0,1]`，验证 `[2]`；统计量只由这四条训练数据计算。
- 本次策略运行只启用实体左臂及左夹爪（现有通道 flags=[0,1]）；14 维模型输出保留，右侧输出不下发。该限制仅属于本次策略运行，不改变双臂遥操能力。
- 现有标定、主从映射、工作空间和保护阈值不修改；设备运行锁沿用 `E:/Dobot/.local/runtime/device-owner`。
- 不提交推送、不合并 main；保留当前工作区修改。

## Review Focus

- 在线普通 joint_positions 的夹爪项可能是占位值：必须使用与录制相同的 dataset_joint_positions 和有效实测夹爪反馈，缺失则停。
- 在线颜色/翻转/预处理可能与训练不同：按录制图像流水线复现并检查，RGB、/255 和 ImageNet normalization 各执行一次。
- 单次 chunk=30 对应2秒：禁止把整段预测作为立即执行的轨迹；每66.7ms重新观察并更新动作，只执行当前选定目标。
- 策略与执行不同步：用观测序号/单调时钟拒绝过期、重复、乱序预测；超时不补发积压动作，进入现有停止路径。
- 14维与物理左右不同于部分历史变量命名：独立验证左侧下发映射、右侧禁止下发、弧度和夹爪范围。

---

### Task 1: 打通服务器训练入口与模型包接口

**Files:**
- Reuse: `scripts/act_training_adapter.py`、`dobot_control/act_dataset.py`、`dobot_control/act_recording.py`。
- Create: `scripts/train_act_dobot.py`、`configs/act_tube_smoke.json`、`tests/test_act_policy_contract.py`。
- External: 作者 ACT 独立 checkout，记录 commit；不覆盖本仓库源码。

**Interfaces:**
- Consumes: `load_data(directory, batch_size=2, chunk_size=30)`，返回 train/val loaders、stats、is_sim；作者 `ACTPolicy`。
- Produces: 模型包 `policy_best.ckpt`、`policy_last.ckpt`、`dataset_stats.npz`、`policy_config.json`、原始 `splits.json`、数据/代码版本及环境清单。stats为现有四个mean/std向量，float32、各14维。

- [ ] 校验上传后的全部 SHA-256；独立 Linux 环境安装与服务器 GPU/驱动匹配的 PyTorch/torchvision，固定解析后版本；确认 CUDA 张量运算成功。
- [ ] 固定 ACT commit，复用模型而新增 Dobot 入口；处理原构造函数内部 argparse，避免新入口参数被再次解析；不导入 ALOHA 环境、仿真环境或其任务配置。
- [ ] 参数初值：ResNet18 ImageNet 初始化、hidden_dim=256、dim_feedforward=2048、encoder=4、decoder=7、nheads=8、chunk_size=num_queries=30、lr=1e-4、lr_backbone=1e-5、KL权重100、batch_size=2、seed=0、num_workers=0；FP32先打通。以上是本次试跑初值，显存和耗时由100步实测决定。
- [ ] 运行契约测试：输入image `[2,2,3,480,640]`、qpos `[2,14]`、action `[2,30,14]`、mask `[2,30]` bool；loss有限，反传有有限梯度，optimizer.step后参数改变；保存/重载后同一输入的eval输出一致且shape `[2,30,14]`。验证末尾padding、四条训练统计与独立计算一致。

### Task 2: 分三段训练并保存可复核结果

**Files:**
- Create: `scripts/eval_act_offline.py`、`tests/test_act_offline_eval.py`。
- Output: 独立运行目录 `runs/act_tube/smoke`、`overfit_one`、`train4_val1`。

**Interfaces:**
- Consumes: Task 1 模型包与HDF5。
- Produces: 每次更新的loss日志、固定验证窗口的指标、动作曲线、最后/最佳checkpoint及恢复训练状态。

- [ ] smoke：真实4/1划分运行100次optimizer更新，记录每步耗时、峰值显存和loss，验证checkpoint恢复；不能以加载成功代替参数更新验收。
- [ ] overfit_one：独立诊断run仅选episode_3的固定8个窗口，训练200～500步；统计只来自该条，不能读取验证条。固定模型结构与学习率，检查训练L1下降及抓放夹爪变化是否被学到；该模型不作为4/1结果发布。
- [ ] train4_val1：用全部四条训练数据随机采窗口，先1000次更新，每100步固定验证；若训练仍明显改善且离线动作合理，再延至3000步。当前Dataset长度=episode数，batch=2时一轮只有2次更新，训练入口按step计数并循环DataLoader。
- [ ] 固定验证条episode_2上选起始/中段/结尾等固定窗口，推理时不给真实未来action，分别记录关节误差（度）、夹爪MAE和开合事件偏差；绘制左臂六关节＋夹爪的预测/目标曲线。若loss下降但动作近常数、夹爪未学到，先查时间采样/归一化/对齐，不进真机。

### Task 3: Windows 加载同一模型并测量离线推理

**Files:**
- Create: `dobot_control/act_policy_runtime.py`、`scripts/act_infer_offline.py`、`tests/test_act_policy_runtime.py`。
- Output: `runs/act_tube/windows_offline/`。

**Interfaces:**
- `PolicyRuntime.predict(qpos: np.ndarray, images: dict[str,np.ndarray]) -> np.ndarray`，输入实测14维和top/left RGB图像，输出反归一化的 `[30,14]` float32绝对目标。
- 消费Task 1模型包；返回结果须包含在调用侧绑定的观测序号、采集时间、预测完成时间。

- [ ] 新建Windows推理专用环境，不更改现有遥操venv；当前GPU为RTX5060 Laptop约8GB，先验证所选PyTorch wheel确实支持该GPU并可运行模型。
- [ ] 使用训练时相同结构/权重/统计量；按camera_names排序，图像/255一次，ImageNet normalization留在ACTPolicy内。保存并重放同一输入，对比Linux/Windows输出在浮点容差内一致。
- [ ] 测试14维顺序、反归一化、RGB和翻转、缺相机、NaN与越界夹爪拒绝路径；在线状态源必须匹配dataset_joint_positions，不能用占位夹爪。
- [ ] batch=1、warmup后连续预测，测量GPU同步后的耗时，以及含读图/预处理/状态读取的端到端p50/p95/max；目标15Hz、周期66.7ms。若不能稳定满足，先优化/调整推理部署并重测，不靠积压chunk维持假实时。

### Task 4: 连接实时观测与受保护执行器，先不运动

**Files:**
- Create: `experiments/run_act_policy.py`、`dobot_control/act_execution.py`、`tests/test_act_execution.py`。
- Reuse interfaces: `RobotEnv.get_obs/step`、当前相机流水线、机器人服务、设备运行锁和现有保护函数。避免直接import会初始化设备或线程的遥操脚本；需要共享时仅抽取纯保护逻辑，并针对受影响模块做一轮回归。

**Interfaces:**
- `prepare_target(predicted: np.ndarray, measured: np.ndarray, arm_flags: np.ndarray) -> np.ndarray`：校验有限14维和夹爪范围，保留右侧当前实测值，只允许左侧动作。
- 执行入口默认dry-run，只有显式execute模式才允许调用下发端；live-shadow只读取观测，不发送运动、使能或夹爪命令。机器人服务已有启动流程可能初始化，应单独按手册安排。

- [ ] 契约测试覆盖flags=[0,1]、右侧不下发、缺/过期反馈、错序/过期预测、越界、超时停止、退出清空队列；使用fake robot证明dry-run发送次数为0。
- [ ] 按当前采集链路取得top/left及实测14维；保持序列号、翻转、画幅、RGB及JPEG质量50处理一致，并保留原协议时效与双相机skew限制。
- [ ] 初版每15Hz重新推理，执行chunk第0个目标，不开启temporal aggregation；执行器用现有较高频率平滑跟踪该目标，保持现有工作空间、关节和速度保护。chunk后续帧留给离线诊断；后续若启用temporal aggregation另行对齐时间索引与验收。
- [ ] 真机旁live-shadow持续至少60秒：记录实测状态、预测、保护判定与端到端延迟，无策略运动下发；重放相同输入核对与离线模式一致。夹爪反馈间歇超龄是已知问题，触发即停并记原始时间戳，不放宽0.35s阈值。

### Task 5: 分段真机闭环验收

**Files:**
- Output: `runs/act_tube/hardware_trial_<timestamp>/`（日志、双视角视频、模型/数据哈希、结果表）。
- Update after execution: 本计划勾选结果和当前操作手册中的策略入口说明。

**Interfaces:**
- 消费Task 4通过的策略与保护执行器，保留人工急停/停止通道。
- 结果分开记录：链路可用、模型动作质量、任务是否完整完成。

- [ ] 核对最新硬件交接/标定；杯与试管、相机和起始姿态按示教布置。先审阅预测首目标与当前关节的差值，不自动跳转到数据首帧；复用已确认的准备姿态流程。
- [ ] 先在无碰撞空间执行2～3秒左臂小范围动作并停止；确认右臂无策略命令、夹爪方向和停止路径正确。
- [ ] 按抓取前接近→抓取→杯口上方→插入/松爪/撤离分段检查；所有片段在当前实时反馈下重新推理，不回放预录轨迹。初始执行速度由已有保护范围内的策略专用更低上限控制，具体值由现场允许空间与起始差值确定并记录；不能为追求速度改遥操默认值。
- [ ] 相同初始布局执行完整流程1～3次，记录各阶段完成、是否需人工接管、停止原因和延迟；单次设置明确超时并禁止自动重试。完整验收为当前观察→模型→保护→下发→新反馈持续闭环，并至少一次完成抓取、插入、松爪和撤离。

## 完成交付

- 一份可加载的模型包、环境与源码版本记录。
- 100步训练日志、单条过拟合诊断、4/1训练验证曲线和Windows离线预测曲线。
- 本地推理延迟、shadow日志和真机试跑记录。
- 五条数据适合先验证固定布置下的链路；后续需要通过增加示教覆盖来验证不同物体位置与起始姿态。

## 2026-11-11 复核与后续采集建议

本次复核依据为五条原始质量报告、五张双视角时序接触表，以及20261110151255158的210号顶部/左腕原始640×480图片。数据结构和训练接入满足当前适配器要求，五条均success且技术PASS。ACT以RGB图像、机器人状态和动作序列训练，频率、视角数量、动作维度由部署协议确定。当前15Hz、两视角、14维是本项目配置。qvel为差分估计，现有ACT适配器实际输入图像与qpos。

### 采集质量判断

- 五条共1887个原始样本，时长23.23～27.29秒。最大命令间隔约110ms，最大图像接收时间年龄约79ms，双相机最大时间差约62ms。上述时间采用主机时间，不能当作精确曝光时差。
- 20261110151224742、20261110151404205双相机skew的p95约60ms。前者顶部349样本中290个唯一frame ID，后者顶部367样本中308个唯一ID，表明存在重复使用相机帧的样本。现有检查允许这种情况，精细接近和插入阶段需控制动作速度并关注实际视觉更新。
- 原始顶部图中透明试管和杯体在浅色背景上对比弱。腕相机可见杯口圆环、夹爪与杯体，部分时刻试管边缘仍弱。JPEG质量50的图像应结合原始尺寸判断杯口和试管尖端可见性。
- 五条可用于先训练链路和固定布局试跑。继续采50条前，先提升关键边缘的可见性和采集一致性。训练结果以真机试验确定。

### 位置变化与操作规范

- 先定义部署范围。杯子固定的当前任务，优先固定杯子，仅让试管在预定抓取区域内变化。如果未来杯子会移动，则单独覆盖杯子移动，以及试管/杯子位置的独立组合。
- 可见且被示教覆盖的位置变化能训练视觉适应。只覆盖少数成对变化容易产生位置关联，应覆盖相同杯位/不同试管位和相同试管位/不同杯位的组合。
- 确保每次抓取高度、试管夹持长度和倾角接近，尤其关注尖端相对夹爪位置。保持插入深度、松爪和撤离方向的一致定义。
- 从操作者移出工作区域、场景复位后开始录制，完成松爪/撤离与短暂稳定观察后结束。减少准备、长时间等待和频繁无目的修正，保留清楚可学的正常纠偏。人工扶正物体、碰撞或反馈异常另作记录。
- 15Hz数据适合先做慢速接近和插入。增加条数不能补回短时间内发生而相机未观察到的接触变化。

### 透明物体处理

- 本数据集只向ACT提供RGB与关节状态。优先检查RGB中的杯口、试管尖端、夹爪接触位置，而非依靠当前未保存的深度。
- 建议先尝试哑光深色纯色背景和稳定漫射照明，再根据实际画面选择对比最清楚的颜色与视角，避免眩光、运动模糊与关键时刻遮挡。
- 若任务允许，可在杯外侧与试管非接触段增加有色标记，避开杯口内缘、插入段和夹持接触面，以保持实际几何与摩擦条件。采用标记后训练和部署保持一致。
- 若计划改变相机位置、图像质量或标记方案，先固定新方案再采主训练集。现有五条仍用于原配置的链路检查，混合训练前明确图像流水线与可见性差异。

### 五十条采集预算与效果验证

以下是建议预算，位置范围按部署需要和可见性现场确定。

- 35条覆盖主要工作区域，包括常见位置与少量边界位置；10条补充边缘、容易偏抓和正常小幅纠偏；另5条预留验证。固定杯位时将覆盖预算集中在试管位置与允许姿态。
- 已有五条可计入同一配置下的对应位置组。按整条轨迹划分，并记录位置组/采集批次，避免相邻重复示教造成过度乐观验证。
- 保存模型后固定权重和评价流程，重新摆放真实物体，先做20次独立闭环试验，逐条记录抓取、对准、插入、松爪撤离和人工接管。
- 若以80%～90%作为工程目标，20次中16～18次完整成功可作为初步结果，之后扩展至50～100次并覆盖部署位置。这个区间是目标与验收示例，当前五条数据尚无对应的实测成功率。
- 原论文50条示教的若干任务取得约80%～90%成功，其相机、控制频率、任务和数据分布与本项目不同。当前任务最需验证的是透明物体精确对准、夹持姿态变化以及部署时的时序与动作跟踪。

### 对初版方案的补充

- 原1～3次完整试跑保留为全链路检查，成功率判断采用上述独立试验。
- 初版每周期取chunk第0步便于核对接口，正式效果评估时应在相同测试条件下比较temporal aggregation，按15Hz时间索引聚合同一执行时刻的预测，避免引入错位或额外滞后。
- 1000～3000次更新作为初始预算，依据固定验证窗口、训练曲线和真机失效阶段决定后续训练量。

## 2026-11-11 用户确认的主采集方案

- 用户确认关闭深度流，给试管和插口增加有色标记，尽量固定位置、仅允许几厘米偏移，并采用相近姿态采集。该方案适合先验证固定工位、有限位置范围的完整任务。
- INI已设置 `depth_enabled=false`。静态配置加载检查通过，确认top/left、15fps、depth=False及左侧活动配置。相机相关现有模拟测试9项通过，实际相机启动效果需重新启动采集进程后核对。
- 标记避开插入口内缘、试管插入段和夹持接触面，保持插入几何。以录制的640×480实际图像核对标记、杯口和试管尖端的可见性。
- 几厘米是每条之间的初始位置变化范围。每条内部固定插口，插入阶段依据实时图像完成精细对准。建议先明确中心位置和允许区域，常见位置多采，区域边界少量补充。
- 抓取高度、试管伸出长度和倾角保持相近，完成抓取、插入、松爪、撤离并观察稳定结果。插口与试管位置变化按位置组记录。
- 建议新采集项目名 `tube_insert_cup_2cam_rgb_marked_v2`。现有五条旧数据有不同配置哈希与深度开关元数据，当前转换器会拒绝混合导出，新批次单独导出。
- 方案的可行性判断来自当前数据和预定范围，实际完成率由新标记配置下的独立闭环试验测量。

## 参考资料

- 当前采集手册：`manuals/ACT_DEMONSTRATION_COLLECTION.zh-CN.md` 第7节。
- 作者仓库：https://github.com/tonyzhaozh/act
- 模型入口：https://github.com/tonyzhaozh/act/blob/main/policy.py
- 原训练入口：https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py
- 构造函数：https://github.com/tonyzhaozh/act/blob/main/detr/main.py
- 外部ACT的固定commit在实际搭建时记录；本次调研读取2026-11-10 main。
- ACT论文：https://roboticsproceedings.org/rss19/p016.pdf
- 透明物体RGB可见性：https://www.roboticsproceedings.org/rss08/p35.pdf
- ClearGrasp透明物体深度问题：https://arxiv.org/abs/1910.02550
