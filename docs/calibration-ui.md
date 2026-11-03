# 相机到基座标定 UI

入口：`python -m task_skills.run_calibration_ui`。这是仓库内的本地标定工作流，不是执行网关的一部分；不会修改现有示教位姿、`scene_config.json`、chemistry profile 或 device-owner 状态。

## 环境与启动

支持 Python 3.11/3.12，标定运行时只固定两个依赖：NumPy 1.26.4 与 OpenCV 4.8.1.78。

Windows PowerShell（创建隔离环境并启动）：

```powershell
.\Start-Calibration.ps1
```

等价手工命令：

```powershell
py -3.12 -m venv .venv-calibration
.\.venv-calibration\Scripts\python.exe -m pip install -r requirements-calibration.txt
.\.venv-calibration\Scripts\python.exe -X utf8 -B -m task_skills.run_calibration_ui --open-browser
```

Linux/macOS：

```bash
./start-calibration.sh --open-browser
```

默认地址为 `http://127.0.0.1:8765/`，数据目录为被 Git 忽略的 `calibration-data/`。启动本身不会打开相机、机械臂、串口或夹爪。非回环监听必须显式追加 `--allow-remote-bind`；界面没有公网身份认证，正常现场应使用回环或 SSH 端口转发。

## 七步流程

1. 检查依赖、创建或恢复任务，确认相机 ID、分辨率、11×8 内角点、格长与阈值。
2. 上传/采集内参图，分配 train/val；最低 10/3，界面显示角点覆盖、缩略图、每图重投影误差与总体 RMS。此阶段移动棋盘，后续固定。
3. 上传/采集固定板图，检测并确认 88 个内角点的 normal/reverse 编号。确认后编号版本不可改；重新检测生成新版本并使下游失效。
4. 左右臂分别编辑 robot_id、base_frame、TCP 名、mm/m 与示教点，支持 CSV 导入导出。点击编号图选择角点并手工填写 XYZ；可选 `GetPose()` 必须先勾选示教器基座、尖端 TCP、单位三项确认。
5. 左右臂独立后台拟合，显示阶段、持久日志、PnP、训练/保留点误差、逐点残差、坐标轴图与 4×4 变换。退出码 0/1/2 分别为通过/运行错误/计算完成但超阈值；失败证据保留但不可下游使用。
6. 在原始图像上点击像素并按渲染比例换算回原图坐标；输入有符号 `board_z_m`，显示左右臂 m/mm。深度模式默认禁用，只有显式保存同一光学-Z 注册确认后才可用；默认不外推。
7. 左右臂分别输入或导入至少 3 个独立参考点，说明测量方法并确认独立性。展示 RMS、最大误差和逐点误差；独立验收与拟合 held-out 明确分开。仅当前哈希、拟合和独立验收都有效的臂可导出 ZIP。

## 有效性与安全边界

- 服务端在查询、导出和集成加载时重新检查输入哈希、产物 SHA256、单位、robot_id、base_frame、TCP、阈值和 synthetic 标记，不只依赖按钮禁用。
- 修改内参图/划分、相机几何、棋盘参数/编号、左右臂元数据、示教点或相关阈值会原子地使下游拟合和验收失效。运行中编辑会令完成结果标记为 stale，不能复用旧通过状态。
- 会话写入使用原子替换；写请求有界并发，图像/JSON 有大小限制，会话 ID 与产物路径均做格式及目录边界校验。
- 静态资源仅来自 `task_skills/calibration/web/`，无 CDN。写请求要求同源、本机 Host、CSRF 令牌；跨源预检被拒绝。
- 机器人只读模块只发送 `GetPose()`，不实例化运动 SDK、驱动、串口或夹爪代码。物理相机采集和 GetPose 都只在操作者显式点击后发生。
- `synthetic_only` 会话贯穿全部产物，不能被提升为真实验收。演示参考值只存在于模拟任务中。

## 感知侧集成

使用只读 API，不自动激活场景变换：

```python
from task_skills.calibration.integration import load_validated_calibration

cal = load_validated_calibration(
    "calibration-data",
    session_id="cal-...",
    arm="left",
    expected_camera_id="overhead_camera",
    expected_robot_id="NOVA2_LEFT",
    expected_base_frame="base",
    expected_tcp_name="tip",
    expected_unit="mm",
)
xyz_m = cal.point_base_m(640.0, 480.0, board_z_m=0.0)
snapshot = cal.snapshot()  # 包含当前哈希、指标、范围与 synthetic_only
```

`CalibrationTransform` 在每次属性访问和点查询前重新验证会话；任何输入或产物变化都会抛出 `CalibrationError`，调用方不得缓存并绕过验证。真实任务默认拒绝 synthetic 数据。

## 验证

```powershell
py -3.12 -m venv .venv-verification
.\.venv-verification\Scripts\python.exe -m pip install -r requirements-calibration.txt playwright==1.49.1
.\.venv-verification\Scripts\python.exe -X utf8 -B -m unittest tests.test_calibration_geometry tests.test_calibration_behavior -v
.\.venv-verification\Scripts\python.exe -X utf8 -B -m tests.browser_calibration_ui --output C:/temp/calibration-browser
```

浏览器测试默认使用本机 Chrome，仅对模拟任务操作，不打开相机或机器人。真机相机、TCP 精度和机械臂网络协议仍需现场按实现边界单独验收；离线演示通过不构成硬件验收。

算法来源、原文件与迁移后文件的 SHA256 见 `docs/calibration-source-provenance.json`。
