# ACT 新一轮采集与重训操作手册

11月17日最新安排为全部72条，62训练/10验证，最新命令见 [72条训练交接](ACT_TRAIN_72_20261117.zh-CN.md)。

编写日期：2026-11-16。适用目录：`E:\Dobot\dobot-atomic-skills`。

2026-11-16晚用户将首轮目标调整为50条训练、10条验证，且今天此前5条计入。本轮最新执行入口见 [11月17日操作单](ACT_MORNING_20261117.zh-CN.md)。本文中100条目标属于调整前方案。

本轮目标：重新采集左臂“抓取试管 → 对准杯口 → 插入 → 松爪 → 撤离”的完整示教，用新批次训练 ACT，再进行离线评估和真机任务对比。

本文给出下一轮的操作流程；新数据采集和训练尚未执行。以当前仓库采集、训练、评估及本地推理入口为依据。

## 1. 本轮安排

本轮确定采集 **100 条新的完整成功示教**，全部来自本轮采集，旧数据不参与训练或验证。先采 5 条回看流程，这 5 条符合要求即可计入最终 100 条；50 条仅作为采集中途进度节点，正式训练在凑齐 100 条后开始。数量按完整成功且技术合格的 episode 计，失败记录另计。

新训练不从旧 ACT checkpoint 恢复，ResNet18 继续使用原有 ImageNet 预训练初始化。100 条新数据按整条 episode 划分为 80 条训练、20 条验证，归一化统计仅由这 80 条计算。旧数据和旧权重留存归档，不作为本轮模型或遮罩效果对比的输入。

顶部相机拟采用红线右侧固定遮罩，左腕画面保持完整。原始采集保存完整图像，具体处理方案见第 3.4 节。**当前仓库尚未实现该遮罩；下方原有转换、训练和推理命令目前仍处理全幅图像。遮罩版训练前需要接入第 3.4 节规定的共同预处理。**

| 阶段 | 做什么 | 进入下一步的依据 |
|---|---|---|
| 场景准备 | 固定杯位、相机、标记、准备姿态和完成标准 | 人工遥操能稳定完成全过程 |
| 试采 | 先采 5 条完整成功示教 | 两视角回看合格，转换器能正常导出 |
| 正式采集 | 完成 100 条新的成功示教 | 成功标签与实际结果一致，阶段覆盖完整 |
| 数据冻结 | 导出新版本，生成训练统计与配套源码 | 使用固定的整集划分和该训练集统计 |
| 重训 | 20 步试跑，再从头训练 10,000 步 | 能加载、反向传播、保存和重载模型 |
| 模型选择 | 评估 best/last，必要时续到累计 20,000 步 | 看动作误差、夹爪时序和高误差片段 |
| 真机对比 | replay → shadow → 短时执行 → 完整任务 | 记录完整成功率、人工干预和失败阶段 |

评估遮罩收益时，建议用这同一批新数据训练“全幅 top”和“右侧遮罩 top”两组，固定同一 80/20 划分、初始化种子、训练步数及其他参数；仅改变顶部图像预处理。这样才能将遮罩带来的变化与重新采集数据带来的变化分开。

## 2. 本轮目录与终端

本手册统一使用以下名称。日期表示本轮批次，跨天续采且场景不变时可继续使用同一名称。

| 用途 | 路径或名称 |
|---|---|
| 新采集项目 | `tube_insert_cup_2cam_rgb_marked_v3_20261116` |
| 原始数据 | `E:\Dobot\datasets\tube_insert_cup_2cam_rgb_marked_v3_20261116\collect_data` |
| 5 条试采导出 | 同项目下 `act_pilot_v1` |
| 正式导出 | 同项目下 `act_new_v1_20261116` |
| 主机数据 | `/home/yxy/data/projects/dobot-act/datasets/act_new_v1_20261116` |
| 主机训练输出 | `/home/yxy/data/projects/dobot-act/runs/act_new_…` |
| 笔记本新模型 | `E:\Dobot\dobot-atomic-skills\.local\models\act_new_v1_20261116` |
| 真机试验记录 | `E:\Dobot\act_real_tests` 下每次独立 trial |

采集时使用三个 PowerShell 窗口：A 为机器人服务，B 为遥操与录制，C 为数据回看、标注和导出。训练在 Ubuntu 主机的 Bash 终端执行。下面每个代码块均标注运行环境；`替换为…` 为需要填写的实际值。

## 3. 采集前固定好什么

### 3.1 场景和示教要求

- 杯子固定，统一试管规格、颜色标记、插入深度标记、夹持高度和试管伸出长度。标记避开夹持面、插入段及杯口接触面。
- 相机支架、焦点附近光照和背景保持稳定，top 能看见整体任务，left 能看清试管尖端、杯口及释放结果。采集和真机部署沿用同一画面方向与标记。
- 左臂从统一准备姿态出发，右臂保持固定准备姿态。准备姿态、复位和人工摆放在录制外完成。
- 先学稳定的完整轨迹：抓取、移送、对准、插入、松爪、撤离。减少无目的停顿、反复试探和突然加速；保留少量平顺、有效的小幅纠偏。
- 试管稳定留在杯内、左夹爪已松开、左臂已撤离、杯子未移位，才算完整成功。仅插入但仍夹着不算完成。

建议 100 条的覆盖比例：约 60 条标准位置，约 30 条预先标记的小幅位置变化，约 10 条自然小纠偏后的成功。偏移范围以人工稳定操作区为准，先从小范围开始，不同时大幅改变位置、朝向和夹持高度。每批 10 条左右休息并回看，防止后半批示教质量下降。

### 3.2 保持当前数据规格

| 项目 | 本轮沿用 |
|---|---|
| 相机及顺序 | `top,left`，右腕关闭 |
| 序列号 | top `218622278417`；left `409122273461` |
| 图像 | 640×480、RGB、15 fps；两视角均旋转 180°；关闭深度 |
| 原始保存 | JPEG 质量 50、同编号 observation PKL |
| 状态与动作 | 14 维，右侧 7 维在前、左侧 7 维在后 |
| 关节单位 | rad |
| 夹爪单位 | 0 闭、1 开；驱动原始值为 0 闭、255 开 |
| 实体左臂 | `192.168.5.2`，COM3 / ID21，向量索引 7—13 |
| 实体右臂 | `192.168.5.1`，COM4 / ID22，向量索引 0—6 |
| 训练动作块 | 30 步，15 Hz 下约 2 秒 |

配置文件为 `scripts\dobot_config\dobot_settings.ini`；当前 `camera_names=top,left`、`active_arms=left`。这一配置规定任务所需通道，**A 启动仍会使能双臂并初始化两侧夹爪，B 仍使用两只主手**。如果现场要求右臂及右夹爪完全不参与初始化，先落实单臂启动方式，再执行下面的 A/B 启动命令。

本轮不改主手标定、端口映射、工作区或限速。相机位置、图像处理或任务规格改变时另建批次；不要中途改 INI 后继续凑同一个正式数据版本。

### 3.3 相机预览

先退出旧 B、ACT 测试窗口和占用相机的 Viewer。在笔记本 PowerShell 执行：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m scripts.5_camera_read
```

确认两个画面方向正确，杯口、试管和撤离结果可见；退出预览后再启动 B。场景照片和本轮准备姿态说明保存在本轮项目目录，便于部署时复原。

### 3.4 顶部相机右侧遮罩方案与可行性

**判断：可行，值得作为本轮优先尝试的图像预处理，有望减少右侧背景变化造成的干扰。** 用户提供的图中，任务用的左侧夹爪、试管和绿色杯子均在红线左边；右边主要是另一机械臂、桌面物件、电脑和人员。去掉这些画面内容可以减少模型利用无关视觉线索的机会。实际收益取决于完整动作是否始终处于保留区域，以及右侧背景原本对策略影响多大。

红线位于画面宽度约 56%—58% 处，略向右倾斜，不是严格的 50% 中线。按当前 640 像素宽图像换算，边界约在 x=362—371；这是从截图比例估计的候选范围。应在旋转后的原始 top 图上确认最终边界，不直接把截图坐标当相机坐标。可先考虑 x≈371 的竖直边界，或按确认后的红线做固定多边形遮罩。

实施约定：

1. **原始数据保留全幅。** 不覆盖原始 JPEG，不需要重新采两批；同一 100 条数据可用于全幅与遮罩对比。
2. **仅处理 top。** 保持 640×480 尺寸和左侧像素坐标不变，将边界右侧替换为固定 RGB 常量。首版建议黑色 `(0,0,0)`，不模糊整张图，也不裁切后拉伸。左腕图不处理。
3. **顺序统一。** 图像按现有方向旋转、转换为 RGB 后，在 `/255` 与 ImageNet 归一化前应用相同遮罩。若现有读取阶段已完成旋转，不再重复旋转。
4. **所有模型入口一致。** 训练、离线评估、replay、shadow、execute 均采用同一处理函数和同一遮罩参数。推荐数据读取时处理训练/评估图像，模型服务处理 replay/实时请求；模型预览同时展示实际输入，方便确认边界。
5. **随模型保存配置。** 保存启用状态、处理视角 top、参考尺寸、边界坐标、填充值和预处理版本。全幅基线关闭遮罩，遮罩模型加载自己的配置。
6. **先看 5 条完整试采。** 从抓取前到撤离结束，杯口、试管尖端、夹持区域及关键运动路径应处于保留区域，并有覆盖计划位置变化的余量。如接近或跨过边界，将边界右移或改成保留任务区域的多边形，再采后续数据。

预计收益主要是抗右侧人员走动、物件变化和另一机械臂外观干扰。固定遮罩不会让杯口像素变多，保持原尺寸也不会明显减少网络计算量；精细插入表现仍受示教质量、腕部可见性和动作一致性影响。截图左上方仍有人和桌面背景，右侧遮罩只消除所覆盖区域的干扰。

相关研究 [MaDi](https://arxiv.org/abs/2312.15339) 在视觉强化学习及 UR5 实验中展示了抑制视觉干扰对泛化的帮助；这为方向提供依据。本项目固定遮罩对 ACT 的具体收益，由同一新数据上的对照训练和完整任务结果测量。

## 4. 启动 A/B 并试采

启动前确认两臂、两侧夹爪周围安全，现场人员掌握实体急停；退出旧控制进程，设备锁按原遥操手册处理。不能通过换锁路径绕过占用。归位沿用现场确认过的方式，B 不额外开启自动归位授权。

**终端 A，笔记本 PowerShell：**

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
$env:DOBOT_DEVICE_LOCK_PATH = 'E:/Dobot/.local/runtime/device-owner'
$env:DOBOT_ACT_RECORDING = '1'
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B experiments\launch_nodes.py --hostname 127.0.0.1 --robot-port 6001
```

等待 `Starting robot server on port 6001`。

**终端 B，笔记本 PowerShell：**

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
$env:DOBOT_DEVICE_LOCK_PATH = 'E:/Dobot/.local/runtime/device-owner'
$env:DOBOT_ACT_RECORDING = '1'
Remove-Item Env:DOBOT_ALLOW_SAFE_POSE_INIT -ErrorAction SilentlyContinue
Remove-Item Env:DOBOT_RECORD_QUEUE_SIZE -ErrorAction SilentlyContinue
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B experiments\run_control.py --hostname 127.0.0.1 --robot-port 6001 --show-img --project-name tube_insert_cup_2cam_rgb_marked_v3_20261116
```

窗口出现后，继续等待 `robot init success`、`button thread init success` 和 `Ok, let's start`。按原流程完成左侧对齐，看到 `ALIGN_DONE` 后采集。

| 按钮 | 操作 |
|---|---|
| A 短按，小于 0.5 秒 | 切换该主手锁定 |
| A 长按，大于 1 秒 | 切换该侧跟随 |
| 任一 B 完整按下并松开一次 | 开始或结束一条录制；两只 B 共用同一录制状态 |

每次只按一只 B，并以终端 `RECORD_START` / `RECORD_STOP` 为准。正式动作只开启左侧跟随。

先录一条约 10 秒空载记录，包含左臂缓慢移动和左夹爪完整开闭，正常停录；按第 6 节回看并将其标为 failure，备注“空载试采”。确认反馈与图像正常后采 5 条完整任务。

## 5. 每条示教的操作顺序

1. 在录制外放好试管，确认杯位、标记、准备姿态和夹爪状态。
2. 左侧跟随开启且对齐完成，按 B 开始；确认 `RECORD_START` 和新 episode 路径。
3. 留一小段稳定起始画面，开始抓取试管；避免长时间静止。
4. 平顺移动至杯口，对准后缓慢插入到深度标记。
5. 松开夹爪，沿统一方向撤离，观察试管稳定留在杯内。
6. 保留约 1 秒结果画面，按 B 停录；等待写盘收尾后再关闭跟随和复位。
7. 记下 episode 名、位置组、成功与否、失败阶段及是否有纠偏。每批回看并标注。

失败也正常停录、记录原因。失败和未审数据保留，默认不进入训练，不自动删除。异常中止的记录按实际情况保留；不要补写 `closed.json` 或修改时间戳来凑成功数据。

推荐备注格式：`组=A标准位；试管=A；深度=标记到位；释放=完成；撤离=完成；纠偏=无；结果=稳定留杯`。记录人工干预的示教时写清干预发生在哪一段；人工直接接触物体辅助完成的记录不列为自主完整示教。

## 6. 回看、标注与导出

以下在笔记本终端 C 执行。数据质量检查用于判断示教能否进入训练；文件保存和传输不另做哈希或校验和检查。

### 6.1 回看与标注单条

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
$episode = 'E:\Dobot\datasets\tube_insert_cup_2cam_rgb_marked_v3_20261116\collect_data\替换为实际episode名称'
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m dobot_control.act_dataset check $episode
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m dobot_control.act_dataset preview $episode --output "$episode\preview.mp4"
Start-Process "$episode\preview.mp4"
```

已有 `preview.mp4` 时直接打开，重生成需用新文件名。回看 top/left 是否一致、夹持和插入是否可见、是否录全释放和撤离。

成功与失败命令二选一：

```powershell
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m dobot_control.act_dataset label $episode --status success --note '组=A标准位；深度标记到位；已松爪撤离；试管稳定留杯；纠偏=无'
```

```powershell
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m dobot_control.act_dataset label $episode --status failure --note '失败阶段=withdraw；撤离带出试管'
```

`PASS` 表示时间、图像、反馈、完整收尾等技术条件通过；`success` 由实际任务结果决定。转换器要求两者同时满足。

### 6.2 5 条试采先导出

退出 B 后执行，A 按现场停机流程处理：

```powershell
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m dobot_control.act_dataset convert 'E:\Dobot\datasets\tube_insert_cup_2cam_rgb_marked_v3_20261116\collect_data' --output 'E:\Dobot\datasets\tube_insert_cup_2cam_rgb_marked_v3_20261116\act_pilot_v1' --fps 15
```

试采符合要求后，再启动 B 继续采集同一项目。无需将试采复制一遍；正式导出会重新收集原始目录中的全部合格成功记录。

### 6.3 正式导出并冻结

达到本轮成功数量后，完成标注、退出 B，再执行：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m dobot_control.act_dataset convert 'E:\Dobot\datasets\tube_insert_cup_2cam_rgb_marked_v3_20261116\collect_data' --output 'E:\Dobot\datasets\tube_insert_cup_2cam_rgb_marked_v3_20261116\act_new_v1_20261116' --fps 15
```

输出包含 `episode_*.hdf5`、`dataset_manifest.json`、`splits.json` 和质量报告。本轮 100 条按整条 episode、种子 0 划分为 80 条训练/20 条验证。全幅与遮罩实验使用完全相同的划分。模型训练开始后固定此划分；新增数据另导出 `v2`，不向已冻结版本追加文件。导出失败留下的目录也不作为正式训练输入，处理原因后使用新目录重新导出。

### 6.4 补齐训练统计和数据读取源码

当前普通转换命令不生成 `train_normalization_stats.json` 和 `training_support`，训练入口需要这两项。正式导出后，在仓库根目录执行一次以下 PowerShell 块：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
@'
import json
import shutil
from pathlib import Path
from scripts.act_training_adapter import training_stats

dataset = Path('E:/Dobot/datasets/tube_insert_cup_2cam_rgb_marked_v3_20261116/act_new_v1_20261116')
split = json.loads((dataset / 'splits.json').read_text(encoding='utf-8'))
stats = training_stats(dataset, split['train'])
(dataset / 'train_normalization_stats.json').write_text(
    json.dumps({k: v.tolist() for k, v in stats.items()}, indent=2), encoding='utf-8')
support = dataset / 'training_support' / 'dobot_control'
support.mkdir(parents=True, exist_ok=True)
for name in ['__init__.py', 'act_dataset.py', 'act_recording.py']:
    shutil.copy2(Path('dobot_control') / name, support / name)
scripts = dataset / 'training_scripts'
scripts.mkdir(exist_ok=True)
for name in ['train_act_host.py', 'evaluate_act_host.py', 'act_training_adapter.py']:
    shutil.copy2(Path('scripts') / name, scripts / name)
shutil.copy2('scripts/dobot_config/dobot_settings.ini', dataset / 'collection_settings.ini')
(dataset / 'batch_notes.json').write_text(json.dumps({
    'batch': 'tube_insert_cup_2cam_rgb_marked_v3_20261116',
    'train_episodes': len(split['train']),
    'validation_episodes': len(split['validation']),
    'normalization': 'computed from this split.train only',
    'action_alignment': 'same_row_no_shift'
}, indent=2), encoding='utf-8')
'@ | E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -
```

归一化只使用新训练集。状态和动作按同一行配对，沿用本项目读取器；不加入原 ALOHA 的 `start_ts-1` 偏移。当前训练脚本的 config 文案仍写着“55 training episodes only”，实际统计来自这里生成的文件；真实条数以本轮 `splits.json`、`batch_notes.json` 和训练 `result.json` 为准。

## 7. 传到训练主机

沿用已配置的 `yxy@192.168.5.100` 和 `/home/yxy/data/projects/dobot-act`。下述为本轮执行命令，本次编写手册未登录主机。

笔记本 PowerShell：

```powershell
scp -r 'E:\Dobot\datasets\tube_insert_cup_2cam_rgb_marked_v3_20261116\act_new_v1_20261116' yxy@192.168.5.100:/home/yxy/data/projects/dobot-act/datasets/
ssh yxy@192.168.5.100
```

主机目标使用新的数据版本目录。若已有同名正式版本，换新版本名并同步后续路径，避免合并到旧版本。

进入主机 Bash 后设置本轮变量：

```bash
source /home/yxy/data/projects/dobot-act/activate.sh
dataset=/home/yxy/data/projects/dobot-act/datasets/act_new_v1_20261116
trainer="$dataset/training_scripts/train_act_host.py"
evaluator="$dataset/training_scripts/evaluate_act_host.py"
```

## 8. 新数据重新训练

### 8.1 20 步试跑

当前 `--smoke` 按旧 65 条数据的 `<48` 和 `>=48` 分组选样；新数据不使用该选项。直接用本轮完整划分跑 20 步：

```bash
smoke_run="$ACT_ROOT/runs/act_new_trial20_$(date +%Y%m%d_%H%M%S)"
python -u "$trainer" --dataset "$dataset" --run "$smoke_run" \
  --steps 20 --batch-size 8 --workers 4 --validate-every 10 --save-every 20
```

完成时脚本输出 `passed: true`，推理形状为 `[1,30,14]`，并生成 `result.json`、best/last checkpoint。失败时按报错处理数据、环境或显存问题，再使用新的 run 目录重跑。正式训练不恢复这次试跑模型。

### 8.2 正式从头训练 10,000 步

在训练主机本地终端或可保持运行的 tmux 会话执行；新会话先重新执行第 7 节激活和变量设置。

```bash
set -o pipefail
run="$ACT_ROOT/runs/act_new_seed0_b8_10k_$(date +%Y%m%d_%H%M%S)"
python -u "$trainer" --dataset "$dataset" --run "$run" \
  --steps 10000 --batch-size 8 --workers 4 \
  --validate-every 250 --save-every 1000 \
  2>&1 | tee "$ACT_ROOT/reports/$(basename "$run").log"
```

本轮保持原基线：GPU0、FP32、双视角 ResNet18、chunk30、hidden512、FFN3200、4 层 encoder、7 层 decoder、8 heads；batch8，AdamW 学习率和 backbone 学习率均 1e-5，weight decay 1e-4，KL 权重 10，梯度范数上限 1。验证每条验证 episode 取 5 个固定位置。

训练目录包含 `config.json`、`splits.json`、`normalization.json`、`metrics.jsonl`、`best.pt`、`last.pt`、定期 checkpoint 和完成后的 `result.json`。best 按验证总 loss 选择。旧 65 条训练约 0.24 秒/步，新批次总耗时还受验证集规模和磁盘读取影响，以本轮日志为准。

### 8.3 评估，再决定是否续训

保持同一 Bash 会话中的 `$run`，或将其设为实际训练目录：

```bash
python "$evaluator" --run "$run" --output "$run/prior_evaluation"
```

评估同时读取 best 和 last。查看 `evaluation.json` 中左臂单步/30 步关节 MAE、P95、左夹爪误差和开合一致率，以及高误差 episode 曲线。字段名 `fixed_50_window_actions` 沿用旧名称，实际窗口数读取 `protocol.fixed_windows`，由当前验证集决定。

验证仍有改善、轨迹未出现持续偏移时，可从本轮 last 续至累计 20,000 步：

```bash
run10="$run"
run="$ACT_ROOT/runs/act_new_seed0_b8_20k_$(date +%Y%m%d_%H%M%S)"
python -u "$trainer" --dataset "$dataset" --run "$run" \
  --resume "$run10/last.pt" --steps 20000 --batch-size 8 --workers 4 \
  --validate-every 250 --save-every 1000 \
  2>&1 | tee "$ACT_ROOT/reports/$(basename "$run").log"
```

`--steps 20000` 指累计训练到第 20,000 步。保持同一数据版本、划分和统计。保留 10k 目录以便比较。

当前续训脚本继承历史最优 loss，但只有再次改善时才在新 run 写 `best.pt`。续训后若新目录没有 best，把 10k 目录的 best 复制到新目录，再用上面的评估命令评估：

```bash
if [ ! -f "$run/best.pt" ]; then
  cp "$run10/best.pt" "$run/best.pt"
fi
python "$evaluator" --run "$run" --output "$run/prior_evaluation"
```

如果误差集中在抓取、杯口对齐或释放阶段，回到相应示教片段分析并补采；不要仅凭训练 loss 下降无限延长训练。

## 9. 将新模型放到笔记本

以下以最终选定的 run 为准。笔记本 PowerShell 中将 `$remoteRun` 替换为实际完整路径：

```powershell
$remoteRun = '/home/yxy/data/projects/dobot-act/runs/替换为实际run目录名'
$newModel = 'E:\Dobot\dobot-atomic-skills\.local\models\act_new_v1_20261116'
New-Item -ItemType Directory -Path $newModel
foreach ($name in @('best.pt','last.pt','config.json','normalization.json','splits.json','result.json')) {
    scp "yxy@192.168.5.100:${remoteRun}/$name" "$newModel\$name"
}
$oldBundle = 'E:\Dobot\dobot-atomic-skills\.local\models\act65_20k'
Copy-Item -LiteralPath "$oldBundle\act" -Destination "$newModel\act" -Recurse
Copy-Item -LiteralPath "$oldBundle\cache" -Destination "$newModel\cache" -Recurse
Copy-Item -LiteralPath "$oldBundle\act-commit.txt" -Destination "$newModel\act-commit.txt"
```

本轮模型结构和上游 ACT 版本沿用现有基线，因此复用本地固定 ACT 源码和 ResNet 缓存。checkpoint、配置和归一化必须来自同一新 run。上面直接复制完整训练 checkpoint，保留主机文件，不需要额外转换或哈希校验。

关闭旧模型服务或占用 8787 的旧隧道，在笔记本新 PowerShell 窗口启动：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
$env:PYTHONNOUSERSITE = '1'
$env:CUDA_VISIBLE_DEVICES = '0'
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
E:\Dobot\.venv-act-inference\Scripts\python.exe -X utf8 tools\act_inference_server.py --run 'E:\Dobot\dobot-atomic-skills\.local\models\act_new_v1_20261116' --checkpoint last.pt
```

等待 `ACT_INFERENCE_READY`。切换 best 时先 Ctrl+C 停止服务，再把参数改为 `--checkpoint best.pt`。现有 `open_act_local_model.cmd` 固定指向旧 `act65_20k`；测试本轮模型请用上面的显式路径命令。

可在另一 PowerShell 窗口读取当前服务身份：

```powershell
Invoke-RestMethod http://127.0.0.1:8787/health
```

记录返回的 run、checkpoint 和 step，作为本次试验所用模型。

## 10. 新模型试验顺序

### 10.1 离线回放

从新数据 `splits.json` 的 validation 列表选一条编号，替换下面路径；此步骤不需要 A/B：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
.\open_act_test.cmd --mode replay --episode 'E:\Dobot\datasets\tube_insert_cup_2cam_rgb_marked_v3_20261116\act_new_v1_20261116\episode_替换为验证集编号.hdf5' --frames 60
```

### 10.2 实时 shadow

恢复与新示教一致的现场，按第 4 节启动 A。通过既有遥操方式准备好机器人后退出 B 和其他相机预览，再执行：

```powershell
.\open_act_test.cmd --mode shadow --seconds 60
```

观察画面方向、反馈有效性及预测时效。预测年龄应稳定低于 250ms；持续 WAITING 或 FAULT 时先处理具体原因。

### 10.3 短时执行，再加入夹爪

第一步保持空旷场景，关闭模型夹爪写入：

```powershell
.\open_act_test.cmd --mode execute --seconds 30
```

在平台窗口按 R 准备，持续按空格先执行 1—2 秒，松开观察方向和运动。确认后再逐步延长。第二步启用左夹爪：

```powershell
.\open_act_test.cmd --mode execute --allow-gripper --seconds 60
```

当前执行只向左臂输出；左夹爪需上述额外参数。沿用当前 10°/秒关节限速、50mm/秒指尖 FK 速度限制、工作区及反馈保护。松开空格、按 P 或窗口失焦停止新指令；紧急情况使用实体急停。平台不自动归位。

### 10.4 完整任务与遮罩效果对比

新模型先做 10 次完整任务，每次单独启动平台，固定杯位、物体和准备姿态。成功要求完成抓取、插入、松爪、撤离且试管稳定留杯；中途人工辅助单独记录，不计作无干预成功。

具备稳定执行条件后，若进行遮罩效果对比，用同一新 100 条训练出的全幅模型和遮罩模型各执行相同的 10 组初始条件，尽量交替测试。分别观察标准场景，以及右侧背景正常变化时的任务结果；先以完整成功率、人工干预次数和失败阶段判断，再结合离线动作误差。人员和物件变化应在工作区外进行。记录：

| 试验 | 模型 run/checkpoint/step | 位置组 | 完整成功 | 人工干预 | 首次失败阶段 | 耗时 | trial 目录 |
|---|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |  |

失败阶段使用 grasp / align / insert / release / withdraw；反馈或通信异常单独写入备注。每次试验结束后标记结果，以下两条二选一：

```powershell
E:\Dobot\.venv\Scripts\python.exe -X utf8 tools\act_real_test.py --label-run 'E:\Dobot\act_real_tests\替换为本轮目录' --result success --stage complete --note '新模型；组=A；无人工干预；试管稳定留杯'
```

```powershell
E:\Dobot\.venv\Scripts\python.exe -X utf8 tools\act_real_test.py --label-run 'E:\Dobot\act_real_tests\替换为本轮目录' --result failure --stage insert --note '新模型；组=A；插入阶段碰杯后停止'
```

首轮汇总完整成功次数/总次数、人工干预次数、各失败阶段次数。若同一阶段反复失败，下一轮补采围绕该阶段的前后连续过程，仍保留完整任务示教。

## 11. 收工和问题处理

采集结束：B 停录并等待收尾 → 关闭跟随、支撑并锁定主手 → 退出 B → 按现场流程退出 A、处理控制柜和设备锁。ACT 测试结束：松开空格并退出测试窗口，关闭模型服务，按现场流程处理 A。程序退出不会自动保证机器人禁用或夹爪松开。

| 现象 | 处理 |
|---|---|
| B 看不到真实 ACT 反馈 | 确认 A/B 均设置 `DOBOT_ACT_RECORDING=1`，按顺序重启 |
| 按 B 无明确起停事件 | 以终端状态为准，完整按下松开一次；不要连续乱按 |
| 夹爪反馈过期、串口错误 | 停止采集或执行，保存日志，排查串口占用及实际反馈 |
| 相机过期、画面冻结 | 退出占用进程，排查 USB 和相机；恢复后重采受影响记录 |
| `RECORD_QUEUE_FULL` 或 aborted | 保留异常记录，不列入成功训练集，处理写盘停顿 |
| 导出提示配置不兼容 | 按场景/配置分批导出，不改元数据强行合并 |
| 导出数量少于预期 | 看成功标签及质量报告，补标注或重采失败片段 |
| 训练缺少统计或 `dobot_control` | 完成第 6.4 节后传输完整数据目录 |
| 训练 run / 评估 output 已存在 | 使用新输出名称，保留旧实验结果 |
| 新模型仍显示旧步数/目录 | 停止旧服务，用第 9 节显式路径重新启动 |
| ACT 会话占用、客户端异常退出 | 按真机手册用对应 trial 的 session.json 恢复；先确认原客户端已退出、机器人停止 |

本轮最终保留：原始记录及标签、冻结 HDF5 数据版本与训练统计、场景说明、训练 run 与日志、离线评估结果、新模型目录、真机试验日志和对比表。文件操作直接进行，不附加传输哈希或重复校验。

## 12. 现有资料与实现入口

- [原采集手册](ACT_DEMONSTRATION_COLLECTION.zh-CN.md)：相机、按钮、数据规范和采集故障细节。
- [真机测试手册](ACT_REAL_TEST.zh-CN.md)：2026-11-16 本地推理、控制保护、会话恢复和日志说明。
- [完整遥操手册](../../TELEOP_WINDOWS_SETUP.zh-CN.md)：标定、归位、设备锁及现场启动细节。
- [上一轮 20k 评估](../docs/act65-20k-evaluation-20261114.zh-CN.md)：旧模型离线基线。
- [数据转换器](../dobot_control/act_dataset.py)、[训练适配器](../scripts/act_training_adapter.py)、[训练入口](../scripts/train_act_host.py)、[评估入口](../scripts/evaluate_act_host.py)、[模型服务](../tools/act_inference_server.py)：本手册命令和数据接口依据。
