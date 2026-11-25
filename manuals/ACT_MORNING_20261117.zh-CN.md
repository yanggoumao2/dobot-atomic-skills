# 11月17日采集、转换和 ACT 重训操作单

11月17日最新安排为全部72条，62训练/10验证，最新命令见 [72条训练交接](ACT_TRAIN_72_20261117.zh-CN.md)。

## 当前进度

本轮目标为50条训练、10条验证。11月16日全部46条新记录均已审核，包含17:30之前5条。38条可训练成功示教，6条任务失败，2条技术失败，待确认0。当前阶段数据集为 `E:\Dobot\datasets\tube_insert_cup_2cam_rgb_marked_v2\act60_stage_20261116`，28训练/10验证。还需22条合格成功示教。若人工复核改标签，以重新打包的统计为准。

完整审核在 `E:\Dobot\datasets\tube_insert_cup_2cam_rgb_marked_v2\review_evening_20261116\index.html`。失败原始记录保留，JSON采用现有工具的`failure`值。审核采用每条12组双视角关键帧、夹爪观测/命令，个别片段追加原分辨率末段画面。这是人工视觉抽帧审核。

## 早上采集

保持杯位、试管规格、相机位置及准备姿态与本批一致。ACT速度及位置保护已按11月17日用户要求统一为遥操规则，详见ACT_REAL_TEST.zh-CN.md。先完成1条试采回看，再补足22条成功记录。抓到试管后先抬升，使试管最低端高于杯口并留出余量，再水平移送，到杯中心上方后下降和松爪。余量可先按20—30mm准备，由现场实物、倾斜和路径确定。此处是采集建议，尚未通过标定获得自动执行坐标。

抓取后保持约0.3—0.5秒确认夹持，插入到位后及时松开，打开后保留约0.5秒再撤离，末段录约1秒。避免对准后长时间等待、低空横移和多次开合。抓空、碰杯后硬推、杯外释放均标失败。成功判定关注夹住试管、移到杯口、松爪后留杯。

退出ACT测试窗口和相机预览。A若正常运行且带ACT反馈，可复用。重新启动A前先确认双臂及夹爪附近安全，现有入口会初始化双臂。

终端A使用PowerShell。

```powershell
Set-Location E:/Dobot/dobot-atomic-skills
$env:DOBOT_DEVICE_LOCK_PATH='E:/Dobot/.local/runtime/device-owner'
$env:DOBOT_ACT_RECORDING='1'
E:/Dobot/.venv/Scripts/python.exe -X utf8 -B experiments/launch_nodes.py --hostname 127.0.0.1 --robot-port 6001
```

终端B继续使用本批目录，采集保存完整原图。

```powershell
Set-Location E:/Dobot/dobot-atomic-skills
$env:DOBOT_DEVICE_LOCK_PATH='E:/Dobot/.local/runtime/device-owner'
$env:DOBOT_ACT_RECORDING='1'
Remove-Item Env:DOBOT_ALLOW_SAFE_POSE_INIT -ErrorAction SilentlyContinue
Remove-Item Env:DOBOT_RECORD_QUEUE_SIZE -ErrorAction SilentlyContinue
E:/Dobot/.venv/Scripts/python.exe -X utf8 -B experiments/run_control.py --hostname 127.0.0.1 --robot-port 6001 --show-img --project-name tube_insert_cup_2cam_rgb_marked_v2
```

等待`Ok, let's start`。左主手按原流程解锁并长按A开启跟随，看到`ALIGN_DONE`后按B起录。结束按B停录，等落盘再关闭跟随。若提示姿态不匹配，按既有归位流程处理。`GetAngle -1`时查看左臂控制柜模式和报警，再恢复。不要反复切换跟随尝试穿过故障。

## 新采数据标注

停止B后，对新增记录生成预览并回看。替换实际episode路径。

```powershell
$episode='E:/Dobot/datasets/tube_insert_cup_2cam_rgb_marked_v2/collect_data/替换为episode'
E:/Dobot/.venv/Scripts/python.exe -X utf8 -B -m dobot_control.act_dataset check $episode
E:/Dobot/.venv/Scripts/python.exe -X utf8 -B -m dobot_control.act_dataset preview $episode --output "$episode/preview.mp4"
Start-Process "$episode/preview.mp4"
```

看清三阶段后选择一条标注命令。

```powershell
E:/Dobot/.venv/Scripts/python.exe -X utf8 -B -m dobot_control.act_dataset label $episode --status success --note '抓取成功；抬高后移送；杯口到位；松爪后留杯'
```

```powershell
E:/Dobot/.venv/Scripts/python.exe -X utf8 -B -m dobot_control.act_dataset label $episode --status failure --note '填写实际失败阶段和原因'
```

无法判断的记录保留unreviewed并提供episode编号进行复核。新增示教需要实际审核后纳入，打包入口读取现有success标签。

## 转换及冻结50/10

在笔记本仓库根目录执行。输出使用全新目录。若已存在，修改输出版本名并同步后面的传输路径。

```powershell
E:/Dobot/.venv/Scripts/python.exe -X utf8 -B tools/prepare_act60.py --raw E:/Dobot/datasets/tube_insert_cup_2cam_rgb_marked_v2/collect_data --since 20261116 --cache E:/Dobot/datasets/tube_insert_cup_2cam_rgb_marked_v2/act_new_unreviewed_20261116 --output E:/Dobot/datasets/tube_insert_cup_2cam_rgb_marked_v2/act60_final_20261117 --top-mask-x 371
```

工具复用今天已经转换的全幅HDF5，明天新增记录调用现有转换器。只取按来源顺序排列的前60条合格success，独立文件保存，seed0留出10条验证。多余success保留在原始目录。统计在preparation_summary.json，只有train=50、validation=10时达到本轮目标。当前阶段版本28/10用于准备，正式训练前以最终60条重新冻结划分。

输出含HDF5、source映射、划分、训练集归一化、training_support和training_scripts。原始采集保留全幅，顶部图导出时x>=371填黑，左腕完整。371是按用户红线比例选择的竖直边界候选，正式补采前看预览确认关键路径有余量；如调整边界，使用全幅cache重新导出新版本。

训练config携带image_preprocess，部署模型服务按同一边界处理top。replay对已遮罩图像重复填同一区域结果一致。该固定mask有望减少右侧背景干扰，具体收益待本轮训练与任务评估。

## 采集完成后连接训练主机

笔记本PowerShell执行，沿用已有SSH配置。主机地址及环境来自已完成的上一轮训练，本次准备尚未重新登录主机。

```powershell
scp -r E:/Dobot/datasets/tube_insert_cup_2cam_rgb_marked_v2/act60_final_20261117 yxy@192.168.5.100:/home/yxy/data/projects/dobot-act/datasets/
scp E:/Dobot/dobot-atomic-skills/tools/train_act60.sh yxy@192.168.5.100:/home/yxy/data/projects/dobot-act/training/train_act60.sh
ssh yxy@192.168.5.100
```

主机Bash执行。

```bash
bash /home/yxy/data/projects/dobot-act/training/train_act60.sh /home/yxy/data/projects/dobot-act/datasets/act60_final_20261117
```

建议在主机已有tmux会话或本地终端执行。脚本核对50/10划分，先20步全链路试跑，再独立初始化正式10000步，最后评估best/last。训练GPU0、FP32、batch8、chunk30、ResNet18、lr1e-5、KL10、每250步验证、每1000步保存。数据和优化从本轮开始，ResNet沿用ImageNet初始化。

正式输出在runs/act60_seed0_时间_10k。保留完整run目录及日志。随后依据验证曲线和夹爪时序决定是否续至累计20000步。

## 模型回传与部署

从最终run取回best.pt、last.pt、config.json、normalization.json、splits.json、result.json至新的本地模型目录，并从现有act65_20k复制act源码目录与cache目录。参照旧操作手册第9节的复制命令，将目录名替换为本轮模型名。

新模型必须使用本次更新的`tools/act_inference_server.py`，它读取config中的遮罩参数。显式启动新run目录，先READY，再replay、shadow和短时左臂执行，最后评估完整任务。执行夹爪时需要`--allow-gripper`。

## 本次代码准备与验证

新增`tools/prepare_act60.py`和`tools/train_act60.sh`。训练入口修正了smoke按旧48条编号分组的逻辑，并按真实训练集条数记录normalization文案。模型服务增加config控制的top遮罩。采集、机器人驱动、关节及TCP限速、工作区和夹爪控制逻辑沿用当前实现。

Python修改已通过语法检查，打包入口--help可用，当前38条阶段数据集执行了实际打包。正式GPU训练、明日现场采集和新模型闭环试验需要在数据补齐后执行。

本机Git Bash语法检查未执行成功，启动Bash时遇到Windows命名对象访问拒绝。训练主机首次执行前可运行 bash -n /home/yxy/data/projects/dobot-act/training/train_act60.sh 检查Shell语法。Python语法检查与阶段数据实际打包已完成。
