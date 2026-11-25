# 三相机预览首次显示触发主手按钮过期修复

2026-11-10，ACT_Train。

## 现场故障及复现

启动已通过robot init success、按钮初始化和归位检查，进入主循环后立即报告LEFT master（配置HAND_LEFT/COM6）按钮缓存超过250ms。此处LEFT是配置主手名。

仅发送现有关节SyncRead和AA55AA按钮查询、绕过主手构造中的力矩/PID写入，检查COM5/COM6：单独按钮查询、混合关节/按钮查询均无非法回包。加入三相机及BimanualAgent读取后，在首次cv2.imshow时复现同类故障（本次表现为RIGHT）：窗口首次显示耗时0.424秒，两个串口有效回包间隔同时达到约0.313秒，触发250ms保护；没有坏回包。

证据指向Windows HighGUI首次窗口初始化阻塞同进程Python采样线程。LEFT或RIGHT谁先报错取决于线程执行顺序。

## 修复

在experiments/run_control.py中增加initialize_camera_preview，在相机启动后、两只DobotAgent构造之前创建并首次绘制同一个预览窗口。主循环复用show_camera_preview绘制该窗口；退出时注册destroyAllWindows清理。

未改250ms按钮/关节时效保护、主从映射、标定、归位、限速、串口协议及失联停控逻辑。

## 验证

- 一轮针对性与回归：tests/test_camera_recording_controls.py、tests/test_teleop_regressions.py，52 passed，1.86秒；既有dobot_api.py弃用警告保留。
- 使用实际三相机RGB+深度、COM5/COM6只读采样与BimanualAgent复测：先初始化窗口，再启动主手采样；主循环首次显示约0.0073秒，连续10秒预览及按钮/关节缓存读取无异常，有效按钮包各1397次、非法包0、主循环阶段最大采样间隔32ms。
- 未启动机器人控制客户端或发送运动、力矩、PID命令。既有服务A保持运行。
- 原复现报告：E:\Dobot\.local\camera-binding\master-buttons-gui-report.json。
- 修复后报告：E:\Dobot\.local\camera-binding\master-buttons-gui-fixed-report.json。

## 接手

测试进程退出释放相机和COM5/COM6后，保留正常运行且启用ACT反馈的A，仅重启B。新版B会先显示三视角窗口，再初始化主手；双侧ALIGN_DONE后按原手册做10秒空载录制及质量检查。
