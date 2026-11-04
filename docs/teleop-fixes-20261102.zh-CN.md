# 2026-11-02 遥操三项修复与验证

## 修改

1. 允许活动范围朝操作者方向平移约 150 mm。按当前装置模型，先采用两臂基坐标负 Y 方向：Y 从 `[-750, -160]` 变为 `[-900, -310] mm`。X、最低 Z=40 mm、主从偏置、home、工具和关节约束未改。对齐预检和运行时保护共用 `dobot_control/teleop_safety.py` 的边界。此修改只移动允许范围，不自动移动机器人。
2. 默认正常跟随关节速度上限由 4.0 降为 3.2 rad/s（80%）；慢速对齐仍为 0.15 rad/s。若另行设置 `DOBOT_SERVOJ_MAX_SPEED_RAD_S`，仍以该显式设置为准。
3. 修复录制：单相机配置原本被三相机门控禁用；原 B 键计数使第一次释放进入停止分支。现在已启用跟随后，任一 B 键完整按下并释放一次开始，再按一次停止。未开启跟随时输出 `record_ignored / servo_off`。停止全部跟随会清除录制状态。

录制保存已启用相机的图像及同周期观测/动作，不生成未启用视角。当前顶部相机输出 `topImg/*.jpg` 和 `observation/*.pkl`；观测增加 `_teleop_camera_names` 字段。每次开始生成独立 episode，终端输出 `RECORD_START` 和保存路径；写盘失败输出 `RECORD_WRITER_FAILURE`。已有三视角目录名称及观测/动作格式保留。

生效方式：停止跟随、退出终端 B 的客户端，按原手册重启 B。当前默认数据根目录仍为 `E:\Dobot\datasets\dataset_package_test\collect_data`。

## 验证

下列命令记录的是清理前执行的测试。本地调试新增的脚本随后按用户要求备份并移出仓库，不再作为当前可运行命令。

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m unittest tests.test_workspace_shift tests.test_recording tests.test_camera_selection tests.test_button_monitor_safety tests.test_absolute_teleop tests.test_servo_enable_cycle tests.test_teleop_regressions
```

结果：71 项通过，`git diff --check` 通过。只运行与三项修改相关的测试。

- 新旧范围重叠区、新增允许区、移除区、边界及越界点、X/Z 保持、非活动臂不误拦截；对齐预检与运行时保护一致。
- 双臂默认速度及不同指令间隔、慢速对齐、启停/重启、故障停机、真实 FK 下的模拟主循环。
- 左/右 B 键第一次启动、停止与再次启动；无跟随时反馈；启动时已按住的 B 松开不误触发。
- 单视角与三视角通过异步队列真实写盘，读回 JPEG 和 pickle 核对图像、观测、动作及相机名；缺失帧错误传播。测试使用合成图像、模拟设备，不连接实体相机或机器人。

## 代码库清理与发布前检查

清理前在上述测试之外一并执行已有主手标定、夹爪回包、夹爪启动和夹爪诊断测试，共 **90 项通过**。随后移除 9 份本地调试新增测试脚本：`test_absolute_teleop.py`、`test_button_monitor_safety.py`、`test_camera_selection.py`、`test_feetech_reply_validation.py`、`test_gripper_startup.py`、`test_master_calibration.py`、`test_servo_enable_cycle.py`、`test_recording.py`、`test_workspace_shift.py`。删除前已备份至本机仓库外的 `E:\Dobot\.local\backups\teleop-passed-tests-20261102-134132-718.zip`，不上传该本机备份。

保留上游正式回归测试、SDK 示例和 `tools/check_teleop_local.py` 环境检查工具。仓库文件 SHA-256 去重检查未发现非空完全重复文件；内容相同的必要空初始化文件和 SDK 标记文件保留。本轮清理没有进一步修改运行代码。

## 现场边界

用户授权先按模型判断方向实施。负 Y=朝操作者仍是未实测的安装方向假设，照片不能确认基坐标轴；现场应以 DobotStudio 的基坐标读数确认方向，再验收允许范围。离线通过不代表蓝框范围或真机录制已获现场验收。本轮没有使能、运动机器人或操作硬件。
