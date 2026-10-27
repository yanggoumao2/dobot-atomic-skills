# Windows 遥操当前版本（2026-10-27）

用户于 2026-10-27 确认当前本机版本已经调试成功。本分支保存当前源码、现场 INI、Windows 遥操依赖及离线测试。发布过程中没有改动控制代码、标定值或安全边界，没有连接或操作机器人。

目标分支：[HYJn14/dobot-atomic-skills / windows_teleop](https://github.com/HYJn14/dobot-atomic-skills/tree/windows_teleop)。原上游基线为 `c402e9c1242e18eee87369accb140253bca495ac`，同时保留目标分支已有的 `0ef3d5a8c08fbf9afe698b706bc5dc83995b1107` 执行结果上传样例提交。

## 文档入口

- [Windows 部署与操作手册](../TELEOP_WINDOWS_SETUP.zh-CN.md)
- [历史交接文档](../manuals/HANDOFF_NEXT_CHAT.zh-CN.md)

两份文档来自仓库外的 `E:\Dobot` 工作区，正文记录截至 2026-10-25 的历史调试过程。关于“尚未调通”“verified=0”“未写入新偏置”的旧状态，以本页的用户确认和本次提交的 INI 为准；未补录逐项现场验收数据。

文档中的命令按原工作区布局编写：仓库位于 `E:\Dobot\dobot-atomic-skills`，虚拟环境位于 `E:\Dobot\.venv`。其他工作区资料（官方 PDF、设备照片、采样记录、工作区级辅助脚本）未随本次代码提交上传，正文指向这些资料的链接需在原工作区查看。

## 提交中的配置

`scripts/dobot_config/dobot_settings.ini` 保留用户当前本机值：

| 配置 | 当前值 |
| --- | --- |
| HAND_LEFT | COM6，2000000 baud，absolute_mapping_verified=1 |
| HAND_RIGHT | COM5，2000000 baud，absolute_mapping_verified=1 |
| GRIPPER_LEFT / GRIPPER_RIGHT | COM3 / ID21；COM4 / ID22 |
| 相机 | 仅 top，序列号 218622278417，15 fps |

两侧新偏置原样保留在 INI 中。这些值对应当前装置，迁移到其他装置需按手册独立核对。

## 本次发布验证

2026-10-27：以下 81 项离线测试全部通过。测试中的超时、夹爪动作和越界提示来自模拟接口；本次没有进行新的真机验收，现场调试成功状态来自用户确认。

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m unittest tests.test_master_calibration tests.test_absolute_teleop tests.test_servo_enable_cycle tests.test_button_monitor_safety tests.test_camera_selection tests.test_feetech_reply_validation tests.test_gripper_startup tests.test_gripper_diagnostics tests.test_teleop_regressions
```

依赖见 `requirements-teleop-windows.txt` 和精确版本快照 `requirements-teleop-windows.lock.txt`。虚拟环境、缓存和新增的运行日志不纳入提交。
