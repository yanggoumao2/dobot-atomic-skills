# ACT 采集改造与验证记录

日期：2026-11-06。基础提交：`2a3a4df`。修改保留在当前工作树，未提交、未推送。

## 修改结果

- 增加显式 `DOBOT_ACT_RECORDING=1` 模式；A 以后台缓存采集真实夹爪位置，B 写入 ACT schema 2 原始记录。普通遥操继续使用现有控制状态、映射、夹爪目标保持、工作区及限速。
- 为保护现有控制逻辑，`joint_positions` 在控制端沿用原有语义，新增 `dataset_joint_positions` 保存实际六轴与夹爪反馈；记录器复制真实数据，控制端后续回填夹爪命令不会污染副本。
- 夹爪读写串口事务共用可重入锁；每约 0.1 秒尝试读取，观测 RPC 只读缓存。记录反馈时间和有效性，读失败不以命令替代真实反馈。反馈模式的真实串口竞争开销需现场验收。
- 修正原始速度占位问题：不再返回位置作为速度，原始速度/末端位姿标为无效；导出速度由实际状态有限差分计算。
- 相机图像、主机接收时间和帧号在锁内发布成一致快照；明确时间是主机收帧时间而非硬件曝光时间。
- 保存动作前状态、请求目标、最终目标、双臂与夹爪使能、发送完成时间。只在 `env.step` 无异常返回后提交样本；发送异常标记 aborted，转换器拒绝该 episode。这里的发送完成不是控制柜到位确认。
- 录制正常收尾产生 `closed.json`；检查保存计数与关闭标记。标签默认 unreviewed，操作者回看后标 success/failure。
- 增加 `python -m dobot_control.act_dataset`：check、preview、label、convert。提供 JSON/HTML 质量报告，MP4 回看视频，因果重采样、RGB HDF5、整段训练/验证划分和数据清单。
- 提供训练服务器加载适配器，支持变长 episode、动作 padding mask、同一行观测/动作配对、仅训练集归一化。取消原 ALOHA 的真实数据前移一帧处理。
- 原始 JPG/PKL 保留；拒绝覆盖导出目录/文件，不再删除不配对原始文件。
- 新增 `--project-name`、`--save-data-path` 参数，默认值保留原设置。
- 删除无调用方的旧三相机转换脚本 `4_collect2train_data.py`、`script_collect2train.py`，以及未调用、方向取反的 `_get_gripper_pos`。历史 `version.txt` 中的旧脚本名作为历史记录保留。

## 测试结果

| 测试 | 结果 | 证据 |
|---|---|---|
| ACT 数据/遥测/CLI 功能测试 | 21 通过 | `tests/test_act_dataset.py` |
| 原遥操回归 | 34 通过 | `tests/test_teleop_regressions.py` |
| 夹爪诊断回归 | 4 通过 | `tests/test_gripper_diagnostics.py`；测试夹爪 fixture 补串口锁 |
| 专项合计 | **59 通过** | `E:\Dobot\.local\act-targeted-tests.log` |
| 全仓 pytest 最终轮 | **317 通过、8 失败** | `E:\Dobot\.local\act-pytest-full-final.log` |
| 修改前基线 task_skills | 5 通过、相同 7 项失败 | `E:\Dobot\.local\act-baseline-tests.log` |
| 修改前基线计时测试 | 6 次中 5 通过、1 失败 | `E:\Dobot\.local\act-baseline-timing.log` |
| 命令行独立功能测试 | check / label / preview / convert 全部 exit 0 | `E:\Dobot\.local\act-functional-20261106\functional_result.json` |
| CLI 参数检查 | run_control --help 显示新参数 | `E:\Dobot\.local\act-client-help.txt` |
| 差异格式检查 | 通过 | `git -c core.whitespace=cr-at-eol diff --check` |

专项覆盖：真实异步写盘器 → 单相机 JPEG/PKL → 检查 → 无损压缩 HDF5 → NumPy 训练窗口；验证 RGB 顺序、14 维字段、15 Hz 时间、同一行动作、变长 padding 与 mask、训练统计排除验证数据。拒绝过期/未来时间、无效反馈、未使能通道、未正常关闭、发送失败、缺图、NaN、覆盖已有输出；验证写盘异常向调用方传播和反馈缓存失败语义。原遥操回归继续覆盖主从路由、单臂/夹爪独立下发、故障保护、主手旧包/过期状态、工作区、腕部修正和队列非阻塞。

独立功能测试以合成图像和遥测生成 6 条记录，经 CLI 标注 1 条 failure 后导出 5 条，划分 4 训练/1 验证。输出图像窗口 `(1,480,640,3)`、qpos `(14,)`、action `(30,14)`、mask `(30,)`。模拟数据放在 `.local` 下，不混入真实训练数据。

全仓失败均在本次未修改的模块：7 项为 `tests/test_task_skills.py` 的目标引用、动作解析、模拟状态/反馈接口预期不一致；在 `git archive HEAD` 解包的基线中复现相同 7 项。另 1 项为化学 DAG 模拟 dwell 测试，断言等待至少 0.4 秒，但本机粗粒度计时偶发观测约 0.391 秒；基线重复执行也复现。没有为本次采集改造调整这些模块或放宽它们的测试。

首次在受限环境跑 unittest 时，本地 HTTP 测试因 loopback 套接字限制失败，且环境未安装 pytest。安装 pytest 并在允许本地套接字的执行环境下完成上表全仓检查。另安装 h5py 3.12.1 供转换，无训练/推理框架安装。

复跑专项：

```powershell
Set-Location E:\Dobot\dobot-atomic-skills
E:\Dobot\.venv\Scripts\python.exe -X utf8 -B -m pytest tests/test_act_dataset.py tests/test_teleop_regressions.py tests/test_gripper_diagnostics.py -q
```

## 现场待验收

开发期间没有连接或运动真实机器人，没有运行 ACT 模型。现场按 [ACT 示教采集操作手册](../manuals/ACT_DEMONSTRATION_COLLECTION.zh-CN.md) 依次做普通遥操、空载录制、真实开闭反馈、B 启停和落盘检查，再尝试试剂瓶任务。检查真实串口反馈模式的控制周期与响应、IP/实体侧、夹爪开闭方向及视野遮挡。

成功完成试采 5 条并通过检查后交训练服务器试读。ACT 模型 batch 前向/反向在训练机器完成；本机已经验证 HDF5 读取与 NumPy 适配核心。瓶型、初始松紧程度和成功操作时长在现场试做后固定。
