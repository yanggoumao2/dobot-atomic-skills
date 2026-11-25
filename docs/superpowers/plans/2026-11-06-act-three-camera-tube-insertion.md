# ACT 三相机试管插入示教采集 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前拧瓶盖采集任务替换为桌面固定窄口杯、实体左臂插入试管后松爪并移开，每段示教完整记录顶部及实体左右腕部三个视角。

**Architecture:** 复用现有多相机线程、原始 JPEG/PKL 写盘器和 ACT HDF5 转换链路。替换任务元数据与示教协议，补齐相机绑定、三视角时效检查、回看和加载顺序；沿用现有双臂跟随、使能检查及 14 维数据契约。

**Tech Stack:** Windows、Python、pyrealsense2、OpenCV、NumPy、h5py、现有 pytest/unittest。

**Spec:** 本文“已确认需求与采集协议”；依据当前对话及本地 `d473eec` 源码。本文仅为实施方案，尚未修改运行代码或连接硬件。

## 已确认需求与采集协议

- 用户确认：杯子固定在桌面，当前步骤由实体左臂插试管，插入到位后松开夹爪并移开左臂。保留现有双臂跟随与录制逻辑，当前暂不用右臂执行任务动作，后续需要右臂配合打开插槽盖子；本轮不新增开盖动作或右臂禁用限制。两台新增相机与顶部同型号，分别安装在实体左右腕部，序列号未知。
- 实体左臂是 `.2`，实体右臂是 `.1`。现有 14 维顺序是右臂六轴和夹爪、左臂六轴和夹爪。沿用现有双臂、双夹爪使能检查，不新增按任务设定的使能掩码或单侧例外。当前只用左臂完成动作是示教流程约定，右臂在现有跟随机制下由操作者保持稳定。
- 任务标识建议 `test_tube_insert`，夹具标识 `narrow_mouth_cup`，协议版本 `1`，数据目录 `E:\Dobot\datasets\tube_insert_cup_3cam_v1`。以后换真实试管槽时更新夹具与数据版本。
- 第一版统一流程：录制外固定杯子和试管初始位置；左臂从准备姿态出发，抓取试管、移到杯口、调整姿态、插入到位、松开夹爪，再移开左臂至统一的避让位置。确认试管留在杯内且稳定，保持约 1 秒结束画面后再按 B 停录。录制必须包含松爪和撤离全过程。
- 插入深度由实际杯高、试管长度、夹持位置确定，记录毫米值和标记照片；现场选定能够在松爪后支承试管的到位位置。成功：插入到位、夹爪完全释放、左臂移开，试管留在杯内稳定且杯子未移位；失败备注区分偏口、卡住、未到位、松爪后倾倒/滑落、撤离时带出试管、杯子移位。保留少量自然纠偏后成功的完整轨迹。
- 顶部用于全局定位，左腕用于近距离杯口/试管对齐及松爪观察，右腕提供辅助斜视。当前右臂保持能看见操作区的统一准备姿态，这是操作约定，不是程序锁定。后续右臂开盖时其腕部相机可随臂移动，三视角链路无需改变。
- 后续加入右臂开盖时，更新任务协议版本和数据批次，以区分“插入并释放”和“开盖后插入并释放”的示教流程；双臂控制与 14 维记录结构直接沿用。
- 先做约 10 秒空载验证，再采 5 条完整任务试采，核对全部三视角与数据后再扩充。原先约 50 条成功记录可作为首批采集目标。

## Global Constraints

- “只改动这个任务，其他都不改”：不改其他原子技能、主从映射、标定、home、关节限速、工作区、夹爪端点及通用故障保护。
- 本轮交付方案，实施阶段只修改下方列出的采集相关文件；不启动训练、不新增自动插入动作。
- 保留任一 B 键开始/停止同一段录制的行为，以及现有双侧跟随、对齐、双臂/双夹爪使能检查；三相机全部开启。本轮不添加右臂关闭跟随、禁止右臂运动、单侧使能检查等逻辑。
- 继续使用原有 14 维 qpos/qvel/action、夹爪 0=闭合/1=打开、同一行观测与动作配对、按整段划分训练/验证。
- RGB 全画幅 640×480、初始 15 fps；保持顶部已确认的 180°旋转。腕部方向逐台预览确认，单独配置，不沿用“左不旋转、其他旋转”的隐含假设。
- 相机使用 USB 序列号，不使用 COM。现有 left/right 序列号是未核验历史值，不能作为本次实际设备绑定。
- 相机命名固定 `top,left,right`，分别指顶部、实体左腕、实体右腕，与历史机器人代码中的 left/right 命名分开。

## 当前源码结论

| 位置 | 已有能力或缺口 |
|---|---|
| `dobot_control/cameras/settings.py` | 已支持 top/left/right 子集、序列号去重与帧率校验 |
| `scripts/dobot_config/dobot_settings.ini` | 当前仅 enabled=top；侧相机编号未核验 |
| `experiments/run_control.py` | 已按配置启动多个线程、分别存图；task 写死 bottle_cap_open，旋转由名字推断 |
| `dobot_control/act_recording.py` | 锁内保存每路最新 JPEG/主机接收时间/发布计数；并非硬件同步曝光 |
| `dobot_control/act_dataset.py` | 多相机检查和 HDF5 循环已存在；现有双臂双夹爪使能检查继续保留，preview 写死 top，加载按字母排序 |
| `scripts/act_training_adapter.py` | 已能返回多视角张量，需验证顺序契约 |
| `dobot_control/cameras/realsense_camera.py` | 现有 get_device_ids 会 hardware_reset；新增枚举功能不能复用该复位路径 |

## Review Focus

1. 任务替换意外改变双臂行为或丢失右臂数据，以及在松爪撤离前提前结束示教：Task 1/3 回归双侧使能与双臂数据保留；Task 5 逐条验收释放、撤离后的稳定结果。
2. 拔插后左右腕部画面交换、重复序列号、方向错误：Task 2 测试绑定与配置；现场按遮挡镜头确认。
3. 一路缺失、旧图或不同步仍被当作完整示教：Task 3 检查每路年龄、时间差、帧进度与完整收尾。
4. 多视角加载按 left/right/top 排序，使模型输入与约定不同：Task 4 用三个不同颜色视角验证整个链路。
5. 三路写盘压力导致队列溢出/半段文件，或不同任务/机位数据混合导出：Task 3/4 测试并在 Task 5 现场测量。

## Task 1：替换任务元数据，保留双臂采集契约

**Files:** Create `dobot_control/act_task.py`; Modify `experiments/run_control.py`, `scripts/dobot_config/dobot_settings.ini`; Test `tests/test_act_task.py`。

**Interfaces:** `load_act_task(path) -> dict` 输出 `task,fixture,task_protocol_version,required_camera_names`。新增 `[ACT_TASK] task=test_tube_insert, fixture=narrow_mouth_cup`；新任务要求三视角，任务元数据不控制任何一侧跟随或使能。缺少新配置的旧环境仍解析为旧任务。

- [ ] 编写失败测试：新任务与夹具、协议版本及三视角要求保存正确；旧配置回退旧任务；未知任务/夹具显式报错。任务加载与录制元数据不得修改传入的双侧控制标志。
- [ ] 运行 `E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m pytest tests/test_act_task.py -q`，确认新增测试因缺实现失败。
- [ ] 实现上述配置加载；启动时读取一次，将配置冻结传入 `RecordingSample` 的新增可选 `metadata` 字段，由写盘器写入 episode.json，替换写死的任务字符串。不让写盘线程每行重新读取可变配置。
- [ ] 增加双臂数据保存回归：输入包含右臂变化的 14 维观测/动作，断言两侧数值和实际使能标志均原样保存；不把右侧填零、冻结或屏蔽。当前试采中右臂稳定由操作者保持，不修改 planner。
- [ ] 重跑本任务测试与 `tests/test_teleop_regressions.py`；通过后单独提交任务配置改动。

## Task 2：识别设备、绑定物理视角与方向

**Files:** Modify `dobot_control/cameras/settings.py`, `scripts/5_camera_read.py`, `experiments/run_control.py`, `scripts/dobot_config/dobot_settings.ini`; Test `tests/test_camera_recording_controls.py`。

**Interfaces:** `CameraSettings` 新增 `flips: dict`，缺项保持原默认行为。预览脚本增加 `--list-devices` 与 `--serial SERIAL`：前者只枚举型号、序列号和可获得的 USB 类型后退出；后者仅预览指定相机用于定位。正式预览和采集共用同一配置与方向。

- [ ] 编写失败测试：枚举不调用 hardware_reset 或任何机器人接口；三路绑定按 top/left/right 排列；重复/空序列号失败；逐路 flip 一致传给预览和采集；顶部默认 flip=True。
- [ ] 运行 `E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m pytest tests/test_camera_recording_controls.py -q`，确认新增测试失败。
- [ ] 直接用 `rs.context().query_devices()` 实现只读枚举，不调用旧 get_device_ids；在相机初始化前列出缺失设备的名称与序列号。
- [ ] 将 `flip=name != 'left'` 替换为配置值；预览叠加物理视角名称和序列号。使用 `top_flip/left_flip/right_flip` 配置项。
- [ ] 现场依次遮挡左右腕部镜头确认序列号和朝向，再填入 INI 并设 `enabled=top,left,right`。序列号未知时保留明确“待现场绑定”状态，不写假值宣称可采集。
- [ ] 重跑测试并提交。驱动现有彩色/深度启动方式先保留；深度流关闭或分辨率调整不作为本轮默认改动。

## Task 3：三视角质量检查与双臂行为回归

**Files:** Modify `dobot_control/act_recording.py`, `experiments/run_control.py`, `dobot_control/act_dataset.py`; Test `tests/test_act_task.py`, `tests/test_act_dataset.py`, `tests/test_camera_recording_controls.py`。

**Interfaces:** 新增 `validate_camera_packet(meta, frames, capture_time) -> None`，验证失败抛 ValueError，由录制分支处理。新任务三路顺序必须完全等于 `['top','left','right']`。不新增 validate_task_flags 或任务专用使能掩码；现有双臂、双夹爪使能检查直接沿用。

- [ ] 编写三相机失败测试：缺一路/空图、未来时间、过期、时间差超限均拒绝。增加回归：新旧任务均按现有双侧使能规则检查；两侧使能且右臂动作变化的数据可通过，右臂参与不被任务标签拒绝。
- [ ] 编写故障测试：录制中某一路冻结、相机线程失败、写盘异常、队列溢出及部分文件存在的 episode 不能被当成成功三视角记录。
- [ ] 运行对应测试，确认新增用例失败后实现共享检查。相机每路年龄上限保持 0.2 秒；三路接收时间差初始上限建议 0.10 秒，写入元数据 `max_camera_skew_s`，由试采统计评估；阈值调整形成新采集配置版本。
- [ ] B 键开始后、提交首条数据前验证三视角条件。运行中无效样本不进入队列；终止本段录制并标记 aborted，记录原因，保持原有控制/故障处理路径。不在控制循环里等待相机凑帧；录制检查不更改任一侧跟随状态。
- [ ] 保留双侧真实状态、夹爪反馈有效性和使能要求。记录每路唯一帧数、接收时间间隔、年龄和跨相机 skew 的 p50/p95/max；在相机读返回后立即取主机时间，再做颜色转换与编码。
- [ ] 帧号仍命名为发布计数，不声称是设备帧号。允许高速控制行复用较新的 15 fps 图像，依靠时效和逐路进度判定冻结；不要求每控制行每路图像都递增。
- [ ] episode.json 增加三路序列号、安装位置、逐路 flip、分辨率/fps、任务协议、时效阈值；逐行实际使能标志按原逻辑记录，不新增任务期望标志字段。保留 raw schema=2，提升 image_pipeline_version=3。闭段 aborted 必须持久化；异常未闭段继续由现有检查拒绝。
- [ ] 重跑相关测试，通过后提交录制与检查改动。跨路主机接收时间检查用于控制采样偏差，不将其标为硬件同步曝光。

## Task 4：三视角回看、导出和固定加载顺序

**Files:** Modify `dobot_control/act_dataset.py`, `scripts/act_training_adapter.py`; Test `tests/test_act_dataset.py`。

**Interfaces:** 保留 `load_act_window(path,start=0,chunk_size=30)` 返回形式，从 HDF5 的明确 `camera_names` 属性取顺序。新文件保存该属性和 manifest 顶层 camera_names；旧文件优先读 metadata_json 中 camera_names，确实没有信息时才使用原排序策略并提示。

- [ ] 编写三色图端到端测试：真实异步写盘器→三路 JPEG/PKL→检查→HDF5→load_act_window，断言顺序 top/left/right、RGB 值、图像形状 `(3,480,640,3)`，qpos/action 仍为 14 维且动作同一行对齐。
- [ ] 编写测试：缺图/空相机列表/重复相机名/逐行与元数据相机列表不一致失败；不同 task、fixture、protocol、serials、flips、时效阈值禁止混合导出；旧单相机数据仍按原契约可读。
- [ ] 编写 preview 测试：读取同一个 source_index 的三张图，按固定顺序拼为 1920×480，叠加名称、时间和编号；缺一路直接报错，不填黑图冒充有效视角；单相机回看尺寸仍为 640×480。
- [ ] 运行测试确认失败；实现以上改动。HDF5 继续分别保存 `observations/images/top`、`left`、`right`，每路 `[T,480,640,3]`；拼图只用于回看。
- [ ] 保留统一 15 Hz 因果选样、14 维统计、padding 和按 episode 划分逻辑。三视角训练适配器输出应为 `[B,3,3,480,640]`，验证加载即可，不启动训练。
- [ ] 重跑 `tests/test_act_dataset.py`，通过后提交数据链路改动。

## Task 5：操作手册与验收

**Files:** Modify `manuals/ACT_DEMONSTRATION_COLLECTION.zh-CN.md`；仅同步更新 `TELEOP_WINDOWS_SETUP.zh-CN.md` 中受影响的相机/记录说明；记录到新 `docs/act-three-camera-tube-insertion-20261106.zh-CN.md`。

- [ ] 将现行 ACT 手册的任务、数据目录、操作步骤、成功/失败示例改为左臂插试管、松爪、撤离。保留原“双侧跟随开启并对齐完成”的采集前提，以及原服务启动、标定、限速、按钮和退出操作。
- [ ] 明确当前先用左臂完成任务，右臂由操作者保持稳定；右腕相机持续采集。统一杯子固定位置、试管规格、初始摆放、深度标记、松爪位置、左臂撤离位置与机位版本。完整成功标注要求释放后试管稳定，左臂已移开；插入后仍夹持或撤离时带出试管不标成功。
- [ ] 运行离线回归：`E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m pytest tests/test_act_task.py tests/test_act_dataset.py tests/test_camera_recording_controls.py tests/test_teleop_regressions.py tests/test_gripper_diagnostics.py -q`。通过条件：新旧任务用例均通过，三色图无顺序/通道交换，遥操回归通过。
- [ ] 硬件验收第一步仅开三相机：确认序列号/名称/朝向、可见杯口与试管，持续预览至少 5 分钟并检查 USB 链路、各路有效帧率。不得通过抬高时间阈值掩盖持续卡帧。
- [ ] 在现场操作者控制下做约 10 秒空载录制：按原双侧跟随与对齐流程启动，左臂慢速移动并开闭夹爪，操作者保持右臂稳定，任一 B 开始/停止；检查三目录同编号图片、时间数组长度 3、原双侧使能检查、正常收尾和质量报告。
- [ ] 比较未录制/录制控制周期、队列丢弃和磁盘吞吐；三路增加 JPEG 写入量，当前实现按控制采样保存可能重复图。先测量再决定是否另开采集降频优化，不默认调整控制频率、队列大小或存储格式。
- [ ] 完成 5 条包含“插入到位→松开夹爪→左臂移开”的完整试采，逐条三视角回看、标注、导出、加载；确认录像覆盖末段松爪与撤离，撤离后试管稳定留在杯内。确认夹爪附近的试管边缘和杯口在当前 JPEG 质量下清晰。若需要提高 JPEG 质量，作为同一图像配置版本记录并重做吞吐检查。
- [ ] 现场检查腕部相机支架、线缆与杯子、桌面、另一只臂的间隙；插入目标保持在现有工作区内。未满足时调整摆放/安装，不扩大安全边界完成采集。
- [ ] 记录实际型号/序列号、运行版本、有效帧率/skew、磁盘和控制周期统计、样例路径与验收结果；提交文档。推送沿用 ACT_Train，不合并 main。

## 完成标准与待现场输入

完成标准：沿用现有双臂跟随与采集逻辑，完成当前由左臂执行的“插入、松爪、撤离”示教；每段都含同编号 top/left/right 图像，相机异常正确判定，导出和加载顺序固定；双侧控制及 14 维数据完整保留，现有遥操行为回归通过。后续右臂开槽盖可直接使用现有双臂控制能力。

实施前可完成全部离线功能与模拟测试。结束动作已确认是松爪并移开左臂；现场需补齐左右腕部序列号和方向、杯/试管尺寸、能够支承试管的插入深度及左臂撤离位置。这些值写入采集协议后再形成正式数据批次。硬件验收与离线测试分别记录。
