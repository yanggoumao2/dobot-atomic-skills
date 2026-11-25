# ACT 主机隔离环境与训练流程 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 在 Ubuntu 主机准备独立 ACT 环境，确定存储和训练流程，完成后停止于数据传输之前。
**Architecture:** 全部新增内容放 /home/yxy/data/projects/dobot-act；独立 Conda 前缀、下载缓存、模型缓存、数据和输出目录。默认只使用 GPU 0。
**Tech Stack:** Python 3.10.13、PyTorch 2.5.1/cu121、torchvision 0.20.1、NumPy 1.26.4、h5py、OpenCV headless、ACT。
**Spec:** 用户 2026-11-13 本次请求；数据契约参见 2026-11-13-act-48-wired-host-training.md。

## 本轮范围更新（用户最终确认）
本轮仅完成环境和必要验证。下面后续数据划分、随机种子、模型参数、batch、步数和验证频率均为可调整候选，等待全量转换结果后再定。

## Global Constraints
- 本次配置环境和规划流程；不传示教数据、不训练、不测速。
- 保留已有环境、驱动、网络配置和其他项目。
- 新环境不读取用户级 site-packages；缓存和临时文件在独立数据盘。
- 只运行 pip check、核心依赖/ACTPolicy 导入和 GPU 0 一次小矩阵前反向验证。
- 用户已授权直接按上述方案配置，本次直接执行；不启用子代理。

## Review Focus
- 系统盘剩余 67 GB，所有大文件落独立数据盘（约 2.6 TB 可用）。
- 主机 DNS 无法解析软件源：使用临时、仅回环监听的 SSH 代理，结束关闭。
- ACT 内部有顶层 util/models 导入：会话环境包含源码根和 detr，源码固定提交。
- 本地 adapter 验证集随机取起点，后续训练入口须改为固定窗口。
- 无数据阶段不宣称实际 batch 显存或任务训练已验证。

## Task 1：隔离环境（本次）
- [x] 检查主机系统、显卡、环境和目录。
- [x] 创建 /home/yxy/data/projects/dobot-act/{env,cache,tmp,src,datasets,runs,reports}。
- [x] 使用 Conda 本地缓存创建独立 Python；安装锁定 torch/torchvision 及训练导入依赖。
- [x] 获取 ACT 官方提交归档源码，记录提交；生成 activate.sh，限定 GPU 0 和用户包隔离。
- [x] 核心依赖/ACTPolicy 导入、pip check、小型 CUDA 计算；记录 freeze 和验收报告。
- [x] 关闭临时网络通道，报告结果，不传数据。

## 后续流程方向（本轮不执行）
1. 全部示教转换完成后，确定最终版本名称和 episode 总数，再安排传输和校验。现有48条版本属于历史中间产物。
2. 按最终总数以整条 episode 为单位划分训练/验证集，保存划分，归一化统计只由最终训练集计算；数量、比例和随机种子届时确定。
3. 对接项目 training_support 和 ACTPolicy，核对新增数据契约，验证集采用可复现的固定采样。
4. 默认 GPU 0 单卡起步；真实数据到位后决定模型参数、batch、训练步数、验证频率及checkpoint策略。
5. 必要的真实batch验证随首次训练准备进行；本轮不运行模型训练、数据加载测试或性能测速。



