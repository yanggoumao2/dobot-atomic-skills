# 结果包与本地回环演示

这里的结果接收器只实现仓库内定义的 **`provisional-3.3-mock/v1` 临时联调契约**，用于本机验证结果包、HTTP ACK、回执查询和幂等行为。它不是合作方 3.3 正式接口，也不代表已连接或验证过真实接收端；“3.3”在这里指合作方闭环联调范围，不是 Dobot 厂商版本号。

## 一条命令跑完整演示

在仓库根目录运行：

```powershell
python -B -m task_skills.results_demo --output-dir .\outputs\dobot-3.3-mock
```

演示会在回环地址启动临时 mock 接收器和 `mock` 模式网关；将 `docs/execution-api/examples/runtime-chemistry-beaker-pour.json` 的副本提交到真实 HTTP 路由 `POST /v1/execute-graph`，只缩短副本中的等待时长，然后轮询执行状态并验证网关的 `ResultsDelivery` 回执。它还会生成九个明确标为 synthetic 的结果包，覆盖运动超时、控制器报警、断连、夹爪失败、少倒/多倒、溢洒、人工取消和缺少证据；这些案例包含结构化阶段、命令、反馈、事件和适用时的合成测量值。程序成功但测量失败的案例会保留这两个不同结论。

`--output-dir` 下会写入 `cases/<案例名>/bundle.json` 与 `receipt.json`、网关提交用的 `gateway-package.json`、实际 mock 网关执行的 `gateway-receipt.json`、`manifest.json` 和 `demo_report.json`，接收器持久化数据保存在 `receiver-data/`。报告会明确标出接收器并非合作方 3.3 正式接收端、所有演示数据均未连接硬件，并记录重复提交复用回执及同 ID 不同内容返回 HTTP 409 的检查结果。演示创建的服务都在本次进程内启动，并在结束或出错时关闭。

## 分开运行接收器和 mock 网关

需要手动观察网关时，可在两个终端分别启动本地接收器与网关。接收器默认只绑定 `127.0.0.1`，并把 mock 回执写入给定目录：

```powershell
python -B -m task_skills.results.receiver --host 127.0.0.1 --port 8333 --data-dir .\gateway-data\results-mock
```

第二个终端启动 `mock` 网关，并配置独立的结果投递地址：

```powershell
python -B -m task_skills.run_gateway --mode mock --host 127.0.0.1 --port 8080 --data-dir .\gateway-data\mock-demo --results-url http://127.0.0.1:8333
```

可以用既有执行客户端向该网关提交 mock 图：

```powershell
python -B -m task_skills.gateway.demo_client --package docs/execution-api/examples/runtime-chemistry-beaker-pour.json --url http://127.0.0.1:8080
```

接收器用 `GET http://127.0.0.1:8333/mock-3.3/v1/experiments` 列出本地 mock 回执，再用 `GET .../experiments/<bundle_id>` 读取一份回执。使用 `ResultsDelivery` 或演示程序只表示客户端按此仓库的临时 mock 契约发送数据；不能据此确认合作方 3.3 正式接口、真实设备结果或真实媒体。案例中的测量和事件均标记为 synthetic，演示不会生成或引用不存在的照片、视频或其他媒体。
