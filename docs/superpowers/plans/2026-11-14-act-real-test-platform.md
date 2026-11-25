# ACT真机测试平台 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 在笔记本仓库交付ACT离线回放、实时预测及受控真机执行平台，配套试验方案和操作说明。
**Architecture:** 笔记本复用A服务、相机配置和FK保护，主机用现有隔离环境与20k模型提供回环HTTP推理，经SSH隧道连接。模型输出使用每次新观测的第一个动作，控制端限速插值。独占ACT会话阻止A服务接受其他客户端的运动写入。
**Tech Stack:** Python3.11/3.10、OpenCV、ZMQ、NumPy、主机PyTorch、标准库HTTP。
**Spec:** 用户2026-11-14真机测试平台与方案请求，现有ACT数据契约及现场遥操手册。

## Global Constraints
- 本轮开发、离线测试与主机模型回放，不启动A/B、不打开真实相机或下发机器人运动。
- RIGHT .1占0—6，LEFT .2占7—13，仅左臂执行，默认夹爪写入关闭。
- 复用现有工作区与FK，默认shadow，execute须用户按R后持续按住空格。
- 超时、失效状态、错误形状、越界、过大预测跳变均停止新命令并锁定本轮，R重新准备。
- 普通保存及传输直接执行，不附加文件哈希校验。

## Review Focus
- A原有RPC调用保持兼容，ACT持有会话时拒绝其他控制客户端写入。
- 相机RGB/旋转、真实夹爪反馈、双视角时间差和状态时效与采集契约一致。
- 失去按键、窗口焦点或模型连接时停止新ServoJ，软件停止与现场物理急停分别说明。
- 始终丢弃右臂预测，恢复按键后使用新观测预测。
- 主机服务仅回环监听，模型服务与真实设备控制分离。

## Tasks
- [x] 新增纯计算命令保护与ACT会话，接入A服务RPC仲裁。
- [x] 新增主机推理服务及笔记本三模式平台、试验日志和结果标记。
- [x] 新增启动入口、主机启动脚本、真机试验方案及操作说明。
- [x] 针对运动隔离、超时、映射及保护边界运行必要模拟测试。
- [x] 主机模型服务加笔记本离线HDF5回放完成一次无硬件集成运行，交付结果。

用户已要求直接构建，本会话直接实施。

交付记录：48项定向测试通过，12帧笔记本到主机回放通过。实际稳态推理8—10Hz，本地控制目标30Hz，按动作块时间偏移执行。现场硬件试验按 manuals/ACT_REAL_TEST.zh-CN.md 操作。

