# 左臂两视角采集与写盘修复

2026-11-10，ACT_Train。本轮禁用right相机，仅启用top/left；保持640×480、15fps、RGB+深度和JPEG质量50。任务配置camera_names=top,left、active_arms=left，协议版本2。允许左臂和左夹爪单侧使能，保留右臂控制能力及右侧在前的14维状态/动作。

## 写盘问题及改动

此前按约50Hz控制循环提交三路图像，重复图像增加写盘负载，8条队列溢出后中止；结束标记也走同一满队列，导致3段没有closed.json。

1. ACT录制独立限频至最高15Hz，不睡眠或降低控制频率，不在卡顿后突发补写。观测、目标动作与相机包仍来自同一个控制周期。
2. 只对样本限制队列容量；结束标记不占样本槽位，与样本共用FIFO，保证排在已提交样本之后且不会因样本队列满而丢失。写盘异常仍传播，不伪造正常收尾。
3. 首次两视角负载测量发现写盘中位约7ms、p95约19ms，但最长停顿2.35秒：15Hz下8条缓冲仍不足。ACT默认缓冲改为64条，约4秒；普通遥操默认仍为8，支持原环境变量覆盖。持续溢出时仍中止并标记aborted。
4. 新目录tube_insert_cup_2cam_v1，旧三视角中断段保持原样，不标为成功或修改原始样本。

## 验证

一轮tests/test_recording_pipeline.py、tests/test_act_task.py、tests/test_act_dataset.py、tests/test_camera_recording_controls.py、tests/test_teleop_regressions.py：89 passed，21.10秒（随后针对默认64条缓冲新增用例，并局部复核recording_pipeline与teleop_regressions，45 passed、2.16秒）。覆盖15Hz节拍、卡顿不补写、满样本队列仍可靠有序收尾、异常工厂释放槽位、两视角仅左侧使能导出/加载、旧三视角/双侧契约及遥操回归。

实际top/left RGB+深度相机 + 真实JPEG/PKL写盘器在E盘15Hz连续60秒：900组样本，top/left各900张JPEG，900份PKL，dropped=0，closed=true、aborted=false。写盘耗时p50/p95/max为6.97/10.79/1976.28ms，缓冲吸收了近2秒停顿。目录：E:\Dobot\.local\recording-io-check-20261110-143011，报告report.json。

这是磁盘负载诊断，观测数据为明确标记的占位字段，不是机器人示教，未连接或驱动机器人。实际关节与夹爪反馈验收由操作者下次空载测试完成。

## 接手

正常退出旧B及相机预览，A正常运行且带ACT反馈时可保持。B终端清除旧DOBOT_RECORD_QUEUE_SIZE覆盖，使用新项目tube_insert_cup_2cam_v1启动。只需开启实体左臂对应主手跟随并完成ALIGN_DONE，无需启用右臂；按B录制至少10秒，期间缓慢移动左臂并完整开闭左夹爪，再按B停止，等待缓冲排空和closed.json生成后check。合格空载段标为failure，备注空载验收。
