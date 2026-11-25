> 2026-11-06：当前实机与近期操作汇总已更新到 [主手册](../TELEOP_WINDOWS_SETUP.zh-CN.md) 第 0/0.1 节：右 .1 / COM4 / ID22，左 .2 / COM3 / ID21；ACT、全画幅、B 键、夹爪窗口、异常恢复一并收录。

# Windows 遥操当前版本（2026-11-05）

2026-11-05：用户确认最新版代码已跑通。已定位实际录制根目录 `E:\Dobot\datasets\dataset_package_test\collect_data`，三段 episode 分别为 `20261102164247264`（3046 对）、`20261102164425905`（465 对）、`20261102171323465`（985 对）；各段顶部 JPG 与观测 PKL 同名配对且非空。本次发布前 34 项相关离线回归通过。录制数据留在本机，不随源码上传。用户确认不替代未登记的逐项现场测量记录。

2026-11-02 更新：工作区改为 .1 X+150 mm、.2 X-150 mm，Y 恢复 [-750,-160] mm，并修正 .1 的 J6 标定参考角、默认跟随限速 3.2 rad/s、支持单相机录制，见[三项修复与验证](teleop-fixes-20261102.zh-CN.md)。已通过的本地调试测试脚本按用户要求移出仓库，保留上游回归测试和运行环境检查工具。最新方向与腕部依据见[本地修正记录](workspace-x-wrist-correction-20261102.zh-CN.md)。以下 10 月记录供追溯。

用户于 2026-10-27 确认当时本机版本已经调试成功。本分支保存源码、现场 INI 和 Windows 遥操依赖。10 月发布过程中没有改动控制代码、标定值或安全边界，没有连接或操作机器人。

目标分支：[HYJn14/dobot-atomic-skills / windows_teleop](https://github.com/HYJn14/dobot-atomic-skills/tree/windows_teleop)。原上游基线为 `c402e9c1242e18eee87369accb140253bca495ac`，同时保留目标分支已有的 `0ef3d5a8c08fbf9afe698b706bc5dc83995b1107` 执行结果上传样例提交。

## 文档入口

- [Windows 部署与操作手册](../TELEOP_WINDOWS_SETUP.zh-CN.md)
- [历史交接文档](../manuals/HANDOFF_NEXT_CHAT.zh-CN.md)

操作手册已于 2026-11-02 对照源码 `a97d247` 更新，作为当前启动、跟随、录制和退出的操作依据；工作区根目录副本同步。交接文档保留历史正文，旧的“尚未调通”“verified=0”“未写入新偏置”等状态不能覆盖最新手册与当前 INI；本次未补造逐项现场验收数据。

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

## 2026-10-27 发布验证（历史记录）

2026-10-27：以下 81 项离线测试全部通过。测试中的超时、夹爪动作和越界提示来自模拟接口；本次没有进行新的真机验收，现场调试成功状态来自用户确认。

以下是清理前的历史命令，其中本地调试测试脚本已于 2026-11-02 移出仓库，不再作为当前运行命令。

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m unittest tests.test_master_calibration tests.test_absolute_teleop tests.test_servo_enable_cycle tests.test_button_monitor_safety tests.test_camera_selection tests.test_feetech_reply_validation tests.test_gripper_startup tests.test_gripper_diagnostics tests.test_teleop_regressions
```

依赖见 `requirements-teleop-windows.txt` 和精确版本快照 `requirements-teleop-windows.lock.txt`。虚拟环境、缓存和新增的运行日志不纳入提交。
