# ACT 主机环境配置进展（2026-11-13）

## 结果
环境配置完成，停在数据传输之前。用户已确认最终数据不止48条；等待全部转换后再按实际总数划分。训练参数本轮不固定。

## 主机与存储
- SSH：yxy@192.168.5.100；主机 yxy-HP-Z8-G4-Workstation；Ubuntu20.04.6。
- CPU：2×Xeon Gold6240，72逻辑线程；内存376GiB。
- GPU：2×Quadro RTX8000，每张48GiB；驱动570.133.07。默认仅GPU0可见。
- 数据盘：/dev/sdb1，ext4，挂载/home/yxy/data，检查时约2.6TB可用。
- 项目根目录：/home/yxy/data/projects/dobot-act。
- 独立环境：env/；数据预留：datasets/；训练输出预留：runs/；源码：src/act/；缓存：cache/；日志与版本：reports/。
- 环境占用约5.4GiB，缓存约2.9GiB。datasets与runs均为空。

## 已安装
- Python3.10.13；PyTorch2.5.1+cu121；torchvision0.20.1+cu121；CUDA运行库12.1。
- NumPy1.26.4、h5py3.12.1、opencv-python-headless4.10.0.84、einops0.8.0、IPython8.30.0、matplotlib3.9.2、PyYAML6.0.2、packaging24.2。
- PyTorch/torchvision采用[官方匹配版本](https://pytorch.org/get-started/previous-versions/#v251)。
- ACT源码来自[官方仓库指定提交](https://github.com/tonyzhaozh/act/tree/742c753c0d4a5d87076c8f69e5628c79a8cc5488)，提交742c753c0d4a5d87076c8f69e5628c79a8cc5488。
- Git克隆遇TLS断连，最终使用该提交官方ZIP归档；源码目录不含.git。提交及归档保留可追溯。
- ResNet18预训练权重已缓存，下载过程验证SHA256前缀f37072fd。

## 隔离与使用
不修改系统Python、已有项目环境或显卡驱动。独立Conda前缀，禁用用户级site-packages；显式设置ACT源码路径，缓存及临时文件在数据盘。激活仅影响当前shell。

在主机执行：
```bash
source /home/yxy/data/projects/dobot-act/activate.sh
```

该脚本默认CUDA_VISIBLE_DEVICES=0。主机访问外网本次经网线SSH通道转发到笔记本Wi-Fi；不修改主机默认网关或DNS。安装结束关闭临时代理和SSH转发，后续下载需再启用联网通道。

## 必要验收（已完成）
- pip check：No broken requirements found。
- 核心库和ACTPolicy：导入成功。
- GPU0：识别Quadro RTX8000，计算能力7.5；一次16×16 CUDA矩阵计算及反向成功。
- 用户级site-packages关闭；使用独立环境Python。
- 未传示教数据，未运行ACT模型训练、真实数据加载、benchmark或机器人操作。

主机证据：
- reports/environment-check.json
- reports/pip-check.txt
- reports/requirements.freeze.txt
- reports/conda-explicit.txt
- reports/act-commit.txt
- reports/install.sh、fetch_source.py、verify_environment.py及各安装日志

## 后续方向
1. 等全部示教转换完成，确定最终版本、有效episode总数与数据契约。
2. 届时再安排传输、整条episode划分及训练集归一化统计。
3. 对接训练入口，默认GPU0单卡起步；依据真实数据再确定模型参数、batch、训练预算、验证频率和checkpoint策略。
本轮不实现这些后续步骤。

