# ACT TCP Speed Limiter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将首步 TCP 超速转为受限的小步动作，并保留最终保护。
**Architecture:** 在现有关节限幅之后，按共同系数缩放左臂六关节相对上一命令的增量；通过现有 FK 按速度比例缩小并复核（最多3轮），再检查最终速度和关节界限。执行客户端保存候选命令、时间间隔和缩放诊断。
**Tech Stack:** Python、NumPy、pytest、既有 Nova 2 FK。
**Spec:** 用户本会话已确认的修复方案；manuals/ACT_REAL_TEST.zh-CN.md。

## Global Constraints
- 末端硬阈值保持 0.05 m/s，关节限速保持 10°/s。
- 仅缩放下标 7:13，不修改右臂和夹爪命令。
- 保留工作区、预测时效、跟踪和会话保护；本次不操作硬件。
- 用户已授权按方案修复，本会话直接实施，不提交其他已有工作。

## Review Focus
- 限速后的两个指尖都满足阈值：真实 FK 回归测试。
- 低速和零步长输入保持原值：不变性测试。
- 右臂、夹爪、输入数组保持不变：隔离测试。
- 非有限 FK 或时间间隔拒绝执行：无效输入测试。
- 缩放不能破坏实测关节步长界限：跟踪滞后测试。

## Task 1: 限速、诊断与操作说明
**Files:** 修改 dobot_control/act_runtime.py、tools/act_real_test.py、tests/test_act_real_test.py、manuals/ACT_REAL_TEST.zh-CN.md。
**Interfaces:** limit_left_tcp_speed(command, previous, qpos, dt, fk, diagnostics=None) 返回最终命令；fk 沿用 calculate_vel_pos 签名，diagnostics 保存缩放前后速度、系数与命令。
- [x] 添加真实 FK 超速复现、低速、隔离、无效输入及跟踪滞后测试，运行确认缺少函数时失败。
- [x] 实现二分缩放与最终关节界限检查，接入下发前检查并记录诊断。
- [x] 运行 tests/test_act_real_test.py 与 tests/test_teleop_regressions.py：57 passed。
- [x] 更新操作说明及测试结果；不创建提交。

## 后续完整流程检查（用户另行授权）
- [x] 修复发送期限起点，发送前复核数据时效，期限同时受原预测和反馈的失效时间限制。
- [x] 推理期间不排队旧观测；暂时缺少新鲜预测时停止发送并等待，持续3秒无新鲜预测时故障停发。
- [x] 真实ZMQ、模拟硬件的CLI执行测试，覆盖5ms/120ms模型延迟、过期预测、执行、松开、暂停、恢复、退出、会话释放和连续重启。
- [x] A启动仅针对夹爪ReadPos接收超时重试读取两次，不重复动作写入；持续超时和设备错误仍退出。
- [x] 真实20k模型60帧HDF5回放通过；真实相机和现有A反馈各60秒shadow两轮，无通信故障。取消排队后预测年龄中位数222ms降到177ms，超过250ms从113/494降到1/426。
- [x] ACT/遥操/夹爪诊断/录制相关测试86项与夹爪面板9项分别通过。面板曾有一次Tk初始化环境错误，单独复测9项通过。
- [x] A重启、双夹爪初始化、真实左臂执行、松开停发、退出后shadow重启已检查。最终不带夹爪执行20261116_155750_458f38记录282条动作、零故障、末端计算速度最大49.50mm/s；松开后18秒无新命令。shadow重启20261116_160100_9e7a82零故障。
- [x] 真机反馈滞后导致TCP缩放与关节步长约束冲突时，停止发送并等待跟上，3秒未跟上时故障。发送前刚跨过250ms的预测也丢弃等待，不发送。
- [x] 加入左夹爪后修复后台反馈线程因写入占用总线而跳过采样的调度问题。20261116_161420_6e607a记录483条动作时夹爪反馈最大年龄126ms，首次保护时双夹爪反馈年龄29/39ms。
- [x] 用户现场明确要求暂时关闭8°预测目标偏差拒绝条件，保留偏差日志、10°/s关节限速、50mm/s末端限制、3°跟踪误差及工作区保护。
- [x] 右夹爪启动写入与左夹爪连续写入先后遇到接收超时，Windows串口接收预算20ms调整为50ms。只读诊断两侧各80次读取均成功，常见5—6ms，最慢15ms；25ms迟到包的模拟回归验证20ms失败、50ms成功。
- [x] 用户自行恢复场景并重启A后，50ms接收预算下左臂＋左夹爪执行20261116_162456_d277bf记录1934条动作、零故障，最大命令末端速度49.935mm/s、夹爪反馈最大年龄131.6ms、右臂实测关节零变化。退出后锁及ACT会话释放，20秒shadow重启20261116_162847_96be00零故障。
- [x] 最终相关模拟回归103项通过，真机记录和原因汇总保存至E:\Dobot\act_real_tests\20261116_真机流程检查报告.md。
