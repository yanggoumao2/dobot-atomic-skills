# Linux Docker 完整流程测试与发布（2026-11-25）

## 已交付

- [Linux 分支](https://github.com/HYJn14/dobot-atomic-skills/tree/linux)
- 镜像源码提交：01fafb32453e3840ed3f6b31b1864db2eaddfa03。
- [成功的构建与测试](https://github.com/HYJn14/dobot-atomic-skills/actions/runs/45564900283)
- [Docker 镜像下载](https://github.com/HYJn14/dobot-atomic-skills/actions/runs/45564900283/artifacts/14999059165)：约234 MB，外层 ZIP，内含 dobot-teleop-linux-amd64.tar.gz，artifact 保留到2026-12-25。
- [容器测试日志](https://github.com/HYJn14/dobot-atomic-skills/actions/runs/45564900283/artifacts/14999661660)

## 测试结果

Windows本地：167项通过（17.98秒），完整离线流程通过。Linux镜像：Ubuntu22.04/Python3.10，160项通过、7项Windows专属用例跳过（4.99秒）；随后完整离线流程通过。7项为3项PowerShell测试和4项Windows前台键盘门控的ACT live CLI用例。

完整流程实际执行：临时Linux配置检查 → A/B本机ZeroMQ命令/状态往返 → HTTP控制台接口与9张命令卡/Bash语法 → 两条合成示范异步写盘 → 质量检查 → MP4回看 → success标注 → HDF5导出 → 仅训练集归一化 → ACT图像/状态/动作窗口加载。

浏览器预览核对：Bash文字、Linux数据路径、/dev夹爪串口、episode更新后的检查与视频入口；截图保存在 E:/Dobot/deliverables/linux-console-preview-20261125.png。预览服务禁用全部硬件动作。

测试使用合成图像、状态和机器人替身。GPU训练与实体硬件接线运行按操作手册接续。

## 本轮修复

- 控制台从后端读取shell和数据根目录，移除Windows路径及PowerShell固定文字；串口占位改为读取配置。
- Linux命令卡保留具体操作说明与确认提示。
- 补充完整流程测试、Dockerfile、Compose现场设备/显示/共享锁配置和部署手册。
- 初次CI镜像构建成功，但xvfb-run启动后无输出直至超时；改为显式Xvfb启动、15秒就绪检查和有限时的测试子进程后通过。
- Windows专属CLI用例限定平台，Linux通信和数据链路由独立完整流程覆盖。

## 使用

下载并解压外层ZIP后：

```bash
docker load -i dobot-teleop-linux-amd64.tar.gz
docker run --rm dobot-teleop:linux
```

默认运行离线测试。现场操作见 [Docker部署手册](../../manuals/DOCKER_LINUX.zh-CN.md)，从接线开始见 [Linux操作手册](../../manuals/TELEOP_LINUX_SETUP.zh-CN.md)。
