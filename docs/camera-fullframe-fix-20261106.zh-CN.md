# 2026-11-06 相机全画幅修复与 B 键核验

用户根据现场照片确认：默认180°旋转正确，只需恢复完整视野；夹爪0为闭合、255为完全打开。

## 最终修改

- `experiments/run_control.py` 移除顶部 `[150:420,220:480]` 裁切和拉伸，原生640×480 RGB只转BGR后供显示/JPEG使用。保留顶部 `flip=True`，预览脚本与相机驱动旋转行为保持原样。
- 新记录元数据为 `top_crop=null`、`top_flip=true`、`image_pipeline_version=2`，注明夹爪归一化后0闭合、1打开。命令和反馈数值不反向处理。
- ACT转换器新增画幅/方向/图像流程版本一致性检查，拒绝旧裁切与新全画幅混成一个数据集。
- 更新ACT操作手册：共享B键切换、全画幅、原旋转、夹爪端点及新数据集名 `bottle_cap_fullframe_v1`。
- 已有4段调试数据及MP4保持原样；被裁掉的像素无法从旧文件恢复。

## B键行为

两个B控制同一个全局录制状态，均在完整按下并松开后触发。右B开始、左B停止，下一次任一B开始新episode。单臂跟随开启时原有逻辑也允许录制，但ACT完整任务检查要求双臂/双夹爪使能。无跟随时B被忽略。

同一轮检测到双B松开只切换一次；先后松开跨轮则会切换两次。操作时每次用一只即可。

## 验证

命令：

```powershell
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m pytest tests/test_camera_recording_controls.py tests/test_act_dataset.py tests/test_teleop_regressions.py tests/test_gripper_diagnostics.py -q
```

结果：**68 passed in 6.23s**。8项相机/B键测试、22项ACT测试、34项遥操回归、4项夹爪诊断回归。日志：`E:\Dobot\.local\camera-fullframe-tests.log`。

验证完整图像四角经过JPEG仍保留，原180°旋转与RGB/BGR通道顺序正确，独立预览/采集都传 `flip=True`；核验跨主手B切换、同时释放、无跟随忽略及启动时已按下的按钮不误触发。另验证元信息和混合画幅拒绝逻辑。

`git -c core.whitespace=cr-at-eol diff --check` 通过。本轮未启动真实相机或机器人。

## 生效方法

当前ACT反馈服务A正常时，先停止录制与双侧跟随，退出旧B。可先运行只开相机的 `python -m scripts.5_camera_read`，确认全幅与方向，退出预览后按ACT手册重启B。A无需为此次画幅修改重启。录制使用新项目名 `bottle_cap_fullframe_v1`。
