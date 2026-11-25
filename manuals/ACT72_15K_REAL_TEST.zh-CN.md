# ACT72续训与真机测试交接

11月17日16时后，常用open_act_local_model.cmd已按用户要求切回首次20,000步run的best。本文为ACT72准备记录，重新使用ACT72时结束当前模型服务，再运行open_act72_local_model.cmd。其默认checkpoint为第14,250步best。

2026年11月17日，已从10,000步last.pt恢复并完成追加5,000步训练，耗时约21分钟。使用62条训练、10条验证，top x≥371填黑。本次按用户要求选用best.pt，对应第14,250步，最佳验证总损失0.119662。

## 后台训练与产物

主机run为 `/home/yxy/data/projects/dobot-act/runs/act72_seed0_resume10k_to15k_20261117_142812`。

日志为 `/home/yxy/data/projects/dobot-act/reports/act72_resume_15k_20261117_142812.log`。

best/last先验离线评估及模型导出均已完成。推理权重、配置、归一化、划分及manifest已导出为run目录中的 `act72_15k_inference.zip`。best按包含前10,000步的验证总损失纪录选取，last为第15,000步。

## 下载模型

本次已下载并解压模型包，best.pt和配套配置、归一化均在本地。以下命令供后续重新下载时使用。

训练及导出完成后，笔记本保持与主机连通，在PowerShell运行以下命令。SCP按提示输入主机登录密码。

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
& .\.local\training\Fetch-ACT72-15k.ps1
```

新模型目录为 `E:\Dobot\dobot-atomic-skills\.local\models\act72_15k`，本地固定ACT源码和ResNet缓存已备好。配置和归一化将随模型下载，实时模型服务按config中的top_mask_x对top输入填黑，与训练一致。

## 模型启动与回放

下载完成后，结束当前本地模型服务，运行以下命令。

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
.\open_act_local_model.cmd
```

常用入口open_act_local_model.cmd和独立入口open_act72_local_model.cmd均默认使用best.pt。等待 `ACT_INFERENCE_READY 127.0.0.1:8787 step=14250`。需要比较last时使用 `.\open_act_local_model.cmd last.pt`。

另开终端，使用本次验证episode回放。

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
.\open_act_test.cmd --mode replay --episode E:\Dobot\datasets\tube_insert_cup_2cam_rgb_marked_v2\act72_final_20261117\episode_35.hdf5 --frames 203
```

## 现场测试

按[现有真机手册](E:/Dobot/dobot-atomic-skills/manuals/ACT_REAL_TEST.zh-CN.md)准备A、相机和示教起始姿态，退出B。使用新模型服务后，客户端入口沿用现有命令。

```powershell
.\open_act_test.cmd --mode shadow --seconds 60
.\open_act_test.cmd --mode execute --seconds 30
.\open_act_test.cmd --mode execute --allow-gripper --seconds 60
```

逐次运行。shadow先看实时预测，execute按R准备并持续按住空格执行，松开即停止发送新动作。首次短时观察运动方向和跟踪，再进行带夹爪任务。

本次沿用已与遥操统一的对齐、速度、单步、位置及运动保护参数。现场重点记录夹到试管、移到杯口、松爪入杯三个阶段，另记录夹爪反复开合、试管下端碰杯口、完整成功次数及完成时间。每轮日志保存在 `E:\Dobot\act_real_tests`。

## 本地准备完成记录

11月17日已完成模型下载、CUDA预热、health接口和验证episode 35整段203帧回放。health确认best.pt、第14,250步、top/left RGB输入。回放左臂关节MAE为0.783°，单次请求延迟中位数72.0毫秒、P95为100.8毫秒，首次请求311.8毫秒。

回放日志位于 `E:\Dobot\act_real_tests\20261117_151538_95e2a3`。机器可读交接记录为[local_readiness.json](E:/Dobot/dobot-atomic-skills/.local/models/act72_15k/local_readiness.json)。

本地模型服务已在后台运行，地址为 `http://127.0.0.1:8787`，监听进程PID为104980，启动器PID为93436。准备连接机械臂时可直接复用该服务。模型输出日志位于 `E:\Dobot\dobot-atomic-skills\.local\runtime\act72-best-model.stdout.log`。

后续换模型或重启服务时，在PowerShell确认当前8787监听进程为ACT模型服务后，结束该服务，再运行open_act_local_model.cmd。首次现场步骤为接入机器人网络、启动A、shadow观察，再按现有手册分阶段execute。本轮完成至机械臂连接前的本地准备。
