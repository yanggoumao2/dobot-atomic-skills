# ACT 65条数据交付与训练方案

2026-11-13。数据传输及20步全链路测试已完成，正式长训练待启动。

## 数据位置与划分

主机为 yxy@192.168.5.100，数据集位于

`/home/yxy/data/projects/dobot-act/datasets/act_65_v1_20261113`

本地统一版本位于 `E:\Dobot\datasets\tube_insert_cup_2cam_rgb_marked_v2\act_65_v1_20261113`。合并原48条与新增17条，统一编号0—64，来源映射保存在 dataset_manifest.json。HDF5总计12,685,834,207字节，约12.69GB，共16,323帧。

- 训练集55条，13,825帧。
- 验证集10条，2,498帧。
- 用随机种子0按采集批次分层抽样，验证集包含旧批7条、新批3条。这是本次为覆盖两批数据采用的划分方式。
- 验证集ID为2、16、24、25、26、31、32、53、57、63，完整划分在 splits.json。
- train_normalization_stats.json 仅使用新的55条训练集重新计算。
- 15Hz，top、left双视角，480×640 RGB，状态/动作14维。右臂6关节和夹爪在前，左臂在后，关节rad，夹爪0闭1开。
- 状态与动作使用同行配对，动作尾部padding保留mask。原始48条版本的划分和归一化统计已由统一版本替代。

## 全链路测试结果

测试目录为 `/home/yxy/data/projects/dobot-act/runs/smoke65_20261113_b8`。用4条训练episode覆盖两批数据，2条验证episode，每条验证取0%、25%、50%、75%与最后一帧，训练20步。

| 项目 | 实测 |
|---|---|
| GPU | GPU0，Quadro RTX8000，48GiB |
| 模型参数量 | 83.89M |
| batch size | 8 |
| 稳态单步耗时 | 0.234秒，去掉前3步预热 |
| PyTorch显存分配峰值 | 4.23GiB |
| 测试耗时 | 20.91秒，含验证、保存及重载 |
| 训练loss | 第1步82.13，第20步10.90 |
| 验证loss | 第10步28.73，第20步12.97 |
| checkpoint重载 | 重建模型后预测与保存前完全一致 |
| 推理输出 | [1,30,14] |

loss均有限，完成数据加载、前向、反向、优化器更新、验证、best/last/定期checkpoint保存和模型重载。上述loss来自短测试的4/2子集，正式训练采用完整55/10划分。吞吐来自小子集缓存读取，正式训练的磁盘读取和验证开销会影响总耗时。

## 首轮训练配置

| 参数 | 方案 |
|---|---|
| 设备与精度 | GPU0单卡，FP32 |
| 图像 | 双视角原分辨率480×640，RGB/255，ACTPolicy内一次ImageNet归一化 |
| Backbone | ImageNet预训练ResNet18 |
| ACT | hidden512，feedforward3200，encoder4层，decoder7层，8 heads |
| 动作块 | chunk30，15Hz下约2秒 |
| batch / workers | 8 / 4 |
| 优化器 | AdamW，lr与backbone lr均1e-5，weight decay1e-4 |
| 损失 | 上游ACT masked L1 + 10×KL |
| 梯度裁剪 | norm1.0，非有限梯度立即报错 |
| 首轮规模 | 10,000 optimizer steps，约80,000个采样窗口 |
| 验证 | 每250步，10条episode各5个固定位置，固定潜变量采样随机状态 |
| 保存 | best.pt、last.pt、每1,000步定期checkpoint |
| 随机种子 | 0 |

batch8已有实际运行依据，速度满足本数据规模，首轮沿用此配置。按0.234秒/步计算，10,000步训练计算约39分钟，包含全量验证、保存和读取，预留45—60分钟。完成首轮后看固定验证集重建误差、KL与曲线趋势，再决定是否延长到20,000—30,000步。

## 主机启动方式

训练入口为 `/home/yxy/data/projects/dobot-act/training/train_act_host.py`。启动脚本将使用完整55/10数据，不带 --smoke。

```bash
bash /home/yxy/data/projects/dobot-act/training/train_act65.sh
```

脚本按时间生成新的run目录，stdout日志保存在reports。可通过第二终端查看日志。长时间运行建议在主机终端或已有tmux会话里启动。

每次运行保存config.json、splits.json、normalization.json、metrics.jsonl与checkpoint，checkpoint包含模型、优化器、步数和随机状态。主机测试结果在 result.json，正式入口同样记录吞吐、显存及模型重载结果。

本轮新增训练入口已通过上述20步真实数据测试。未进行正式10,000步训练，也没有真机策略测试。
