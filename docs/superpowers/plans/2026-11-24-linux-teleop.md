# Linux teleoperation Implementation Plan

**Goal:** Windows/Linux 共享遥操、示教采集和训练源码，Linux 硬件地址留空。
**Architecture:** 保留 A/B、SDK、安全门控与数据格式；统一配置和平台辅助层；移除无关任务编排。
**Tech Stack:** Python 3.10、NumPy、OpenCV、PySerial、ZeroMQ、RealSense、ACT/PyTorch。
**Spec:** 本次用户请求：迁移至原 ACT Ubuntu 主机，从接线交接，模拟 Linux 测试一次。

## Global Constraints
保留原有未提交修复。硬件连接参数由现场填写。保留 Windows 操作能力。不接硬件运行测试。

## Review Focus
空配置在连接前拒绝；左右槽顺序固定；串口别名正确解析；采集元数据反映实际配置；Linux 进程与锁恢复不调用 Windows API。

## Tasks
- [x] 配置：dobot_control/config.py，config/linux.example.ini，统一所有入口；测试空配置/重复地址/元数据。
- [x] 平台：迁出 layout/device_lease；取消 sudo 密码和 chmod；Linux 控制台、恢复、启动脚本。
- [x] 精简：删除 task_skills 的编排/网关/化学实验及其专属测试和文档；保留共用控制模块。
- [x] 交接：操作手册、项目说明、变更清单与一次模拟测试报告。

交付：manuals/TELEOP_LINUX_SETUP.zh-CN.md、docs/PROJECT_GUIDE.zh-CN.md；一次模拟运行 145 tests OK。
