# NOVA2 实机对应关系修正与操作手册更新

日期：2026-11-06。依据：用户在当前对话中明确确认的实机对应关系。

| 实体侧别 | IP | 夹爪配置 | 串口 | ID | home J1～J6（度） |
| --- | --- | --- | --- | --- | --- |
| RIGHT 右从臂 | 192.168.5.1 | GRIPPER_RIGHT | COM4 | 22 | 90,0,90,0,-90,-90 |
| LEFT 左从臂 | 192.168.5.2 | GRIPPER_LEFT | COM3 | 21 | -90,0,-90,0,90,90 |

## 代码修正

- `task_skills/layout.py` 按实体侧统一 IP、夹爪段/串口和 home；home 数值跟随原 IP，不给原 IP 下发另一臂目标。
- `dobot_control/robots/dobot.py` 修正 .1→GRIPPER_RIGHT、.2→GRIPPER_LEFT；INI 的 ID、串口数值保留，补充已确认的从臂标签。
- `tools/gripper_panel.py` 左按钮→COM3/ID21，右按钮→COM4/ID22。
- 化学任务 `profile.json` 的两侧 IP/串口/ID 同步，移除当前配置中旧 COM7/COM8 绑定。
- 保持遥操 14 维向量 `.1 → .2` 的已标定顺序：`TELEOP_NOVA_SIDES=(right,left)`，服务构造与 `robot_pose_init` 按此取实体配置；主手偏置、符号、交叉路由、FK/工作区键和保护数值保留。
- 双臂 RPC 报警和工作区告警按 IP/向量槽显示实体侧别；固件检查显示 LEFT .2、RIGHT .1。
- ACT 新记录使用 `hardware_layout_version=2` 与正确的 gripper_ports；转换器按版本与绑定检查兼容性，禁止混合旧绑定导出。原始数据不改写。

`HAND_LEFT/HAND_RIGHT` 表示主手配置段，不能用从臂命名覆盖。现有六轴路由为 COM5/HAND_RIGHT→RIGHT .1、COM6/HAND_LEFT→LEFT .2；后续按用户现场反馈修正主手夹爪反向：夹爪与六轴使用同一 Nova 槽，HAND_LEFT→.2/COM3/ID21，HAND_RIGHT→.1/COM4/ID22；目标、使能、对齐门控同步修正。

## 手册更新来源与内容

核对近期对话“制定拧瓶盖ACT采集计划”“推送已调试代码到分支”和当前“代码报错修复”，并与工作树源码核对。主手册第 0/0.1 节集中汇总：

- ACT 真实夹爪反馈、动作发送结果、正常收尾与 aborted 记录，check/preview/label/convert 数据流程。
- 全 640×480 画幅、保留 180°旋转及共享 B 键切换行为。
- 0 闭合、255 打开；单次低速夹爪窗口入口与共享设备锁。
- ServoJ 故障时亮红灯失败的独立捕获，保证后续遥操复位继续执行。
- 显式归位授权、当前实机 home、完整 A/B 重启步骤。
- 已有 4 段调试录制的 failure 标签、回看视频和待补双夹爪空载验收。

主操作手册为 `TELEOP_WINDOWS_SETUP.zh-CN.md`；ACT 分手册同步修正。历史交接与烧杯实验记录保留原证据，在开头标明新标准。工作区根目录主手册副本同步更新。

## 生效与数据

停止录制及跟随 → 退出 B → 退出 A → 按主手册 11.2 核验旧锁 → 先 A 后 B 完整重启。新项目名使用 `bottle_cap_layout_v2`。空载分别核对两臂及 COM4/ID22、COM3/ID21，录制和回看后再采正式数据。

原子任务/化学任务的旧场景点位保存了此前采集时的侧别含义。当前已统一硬件地址，旧点位与路径须在新绑定下现场复核后再执行；本次不推算或重写示教坐标。

## 本轮验证

一轮相关离线回归：**142 passed，9.28 秒**；1 条现有 SDK 文档字符串转义 DeprecationWarning。没有启动真实机器人、串口或相机。

```powershell
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m pytest tests/test_teleop_regressions.py tests/test_gripper_panel.py tests/test_gripper_diagnostics.py tests/test_camera_recording_controls.py tests/test_act_dataset.py tests/test_atomic_runner_unittest.py tests/test_task_skills.py::test_physical_nova_home_is_explicitly_separate_from_master_home tests/test_chemistry_skills.py::ProfileTests tests/test_chemistry_skills.py::RegistryTests tests/test_chemistry_skills.py::HomingOperationTests -q
```

覆盖真实驱动构造路径的模拟串口选择、服务与 home 的 IP 顺序、实体报警名称、窗口左右选择、ACT 元数据和旧绑定混合拒绝，以及现有遥操/录制/夹爪/原子运行/化学资源与归位回归。差异格式检查通过；没有扩展到全仓测试。

当前工作区已有其他对话留下的未提交改动，本次保留并在其上修正；尚未提交或推送。
