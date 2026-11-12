# Dobot 外部原子技能执行网关

正式接入与联调文档：[手册 v1.1（PDF）](Dobot_原子技能执行接口与联调手册_v1.1.pdf)；[完整正文](Dobot_原子技能执行接口与联调手册_v1.1.md)。手册 v1.1 只覆盖 `embodied.*`；`dobot.*` 化学原子技能组以本文件与 `GET /v1/catalog` 为准，待手册下次修订并入。

现在链路为：**上游生成执行图 → 本项目接收/拉取图 → 校验与资源占用 → 执行已注册技能 → 状态查询与事件回传**。新入口不调用自然语言解析器。旧 `run_task` 保留为现场人工入口，与网关共享设备所有权锁。

基于对方《设备能力、原子技能库与动态执行图接口规范》`1.0-draft` 实现。外部包使用 `schema_version=2.1 / graph_version=0.1`；本次实现的是明确声明的 `dobot-dag/1` 子集，不能宣称完整支持所有 2.1 动态图。早期 `skill-dag/1.0` 设计稿仅供历史参考。

## 本地联调

在项目根目录运行，不需要新安装依赖：

```powershell
python -B -m task_skills.run_gateway --mode mock --port 8080 --data-dir ./gateway-data/mock-demo
```

另一个终端模拟上游直接下发：

```powershell
python -B -m task_skills.gateway.demo_client --output ./gateway-data/demo-request.json
```

或者使用两阶段协议（使用另一份新 mock 数据目录，或先完成下面的 mock 状态复位）：

```powershell
python -B -m task_skills.gateway.demo_client --two-phase
```

示例执行七节点 A/B 抓取、A 等待、双臂放置、B 再抓取并返回的图。mock 双臂技能可以并行，`repick-B` 必须等待 `place-A` 和 `place-B` 都成功。程序输出 `execution_id`、节点状态和事件数量。示例客户端明确拒绝真机网关。

示例结束后 A 位于 `target:target_A`、B 回到 `object:B`。数据库会保存这个逻辑状态，不能直接用新幂等键重放最初的抓取图。后续上游图需引用当前位置；若要再次演示初始图，先关闭网关，再对 **mock 数据目录**执行：

```powershell
python -B -m task_skills.gateway.operator --mode mock --data-dir ./gateway-data/mock-demo --confirm RESET_VERIFIED
```

该命令保留执行、幂等及事件历史，只重置逻辑场景。模拟复位不会清除真机的场景失效标记。

## 控制面接口

| 方法与路径 | 请求/行为 |
|---|---|
| `GET /health` | 模式、目录版本、执行占用及隔离状态 |
| `GET /v1/catalog`（别名 `/v1/capabilities`） | 本机设备、技能、固定物体/位置绑定和执行子集 |
| `POST /v1/execute-graph` | 对方规范 7.2 的直接下发入口。请求体可以直接是 `RuntimeExecutionPackage`，也可以是 `{"graph", "resume_from_node_id": null, "execution_id": null, "request_id"}`；带 `Idempotency-Key`（优先）或 `request_id` 时按规范 9.2 去重，不带时每次新建执行；202 返回 `execution_id/status/graph_id/accepted_at` |
| `POST /v1/execute-graph/idempotent` | 幂等单次提交；使用 `Idempotency-Key` 或 `request_id` 去重 |
| `POST /device-gateway/v1/execution-plans/validate` | `{"package": RuntimeExecutionPackage,"reserve_resources":true}`，返回绑定包的 `validation_token`；默认预留 60 秒 |
| `POST /device-gateway/v1/executions` | `{"package": RuntimeExecutionPackage,"validation_token":"..."}`，正式执行 |
| `GET /v1/executions/{id}` | 总状态、节点状态、错误、物体逻辑状态与证据等级 |
| `GET /v1/executions/{id}/events?after=0&limit=100` | 按序列分页，最大 limit=1000 |
| `POST /v1/executions/{id}/cancel` | 请求体 `{}`；发起停止，可继续查询终态 |
| `GET /v1/delivery-status` | 回传待投递、已确认、错误与是否配置上游 |

查询/事件/取消也支持 `/device-gateway/v1/executions/{id}` 前缀。JSON 请求必须 `Content-Type: application/json`，不支持 chunked，最大 2 MiB。`/v1/execute-graph/idempotent`、两阶段预检和正式提交必须带 `Idempotency-Key`，或请求体 `request_id`；两者都有时必须相同。同一请求键+同一原始包返回原会话，不重复动作；换包报 409。每个动作还必须带唯一 `skill_invocation.idempotency_key`，已使用的节点键不能交给另一个会话。

**2026-11-12 起按对方《设备能力、原子技能库与动态执行图接口规范》修订：** `POST /v1/execute-graph` 接受规范 7.2 的信封，`resume_from_node_id`/`execution_id` 只能为 null（不支持续跑，非 null 返回 400）。带 `Idempotency-Key` 或 `request_id` 时（请求头优先，两者不同不报错），同一键重试返回原执行、不重复动作；新键会新建执行，节点键由服务端按请求键派生，所以同一编译图换新 `request_id` 再下发仍会执行。以下为不带键时的行为（原描述）。事件回传遇到 409 视为对方已按 `event_id` 去重收到。

`POST /v1/execute-graph` 不带键时每次接收有效请求都创建新执行，不需要 `Idempotency-Key`。它接受三种等价写法：请求体就是运行包本身（联调最省事，直接 POST 序列化好的图），或包在 `{"graph": ...}`、`{"package": ...}` 里（后两者只能有一个，混用返回 400）；不使用上游 `run_id` 拉图，也不接受预检 token、恢复或模式字段。若提供 `request_id`，必须是有效 ID，但不会用于去重。网关在包的深拷贝上生成唯一 dispatch key 和每个动作新的 `skill_invocation.idempotency_key`，同时保留 `package_id`、包内 `run_id` 和 `node_id`。因此自动重试或重复点击会创建另一份执行；依然受执行占用、预留、隔离、场景/物体状态及设备锁检查约束，不会取消当前任务或复位场景。需要网络重试保持幂等的自动化客户端应使用 `/v1/execute-graph/idempotent`；两阶段接口行为不变。

live 图执行完成后，网关为每个参与臂发送 `arm_finalizing` 和 `arm_finalized` 事件；归位失败或取消时发送 `arm_finalization_failed` 或 `arm_finalization_cancelled`。臂标识位于事件的 `payload.arm`，归位状态和反馈也在 payload 中。这些执行级事件不对应图节点，不包含 `node_id`。收尾归位在动之前先做一次只读探测：关节已在标定 home 容差内且 `RobotMode` 为 5 的臂记为 `skipped`、不发任何指令（图里已自行归位的包因此不会多动一次），只有确实不在 home 的臂才进入完整会话。live 网关把带令牌的 POST 本身视为现场授权，收尾归位不再依赖交互式确认；此前这条缺失会让一个全部成功的包在收尾报 `FINAL_HOME_FAILED`，并把 rig 置为隔离。

## 当前可执行的图

基础形状见 [runtime-package.profile.schema.json](runtime-package.profile.schema.json)，完整模板见 [runtime-package.example.json](runtime-package.example.json)。运行包可省略 `catalog_revision`；如提供，入参值可为任意有限 JSON 元数据，网关会原样保留并纳入完整包指纹，不与本机目录修订值比较。`GET /v1/catalog` 中网关生成的 `catalog_revision` 仍表示当前目录，仅供查看或由上游自行记录。样例值只作任意元数据示例。

| 项目 | 当前合同 |
|---|---|
| 节点 | `action`、`wait`；初始 `status=pending`；1–100 个节点 |
| 技能 | 一个包只用一族：`embodied.pick@1.0.0`/`embodied.place@1.0.0`，或 `dobot.*@1.0.0` 化学原子组（见下）；混用返回 422 `INVALID_GRAPH` |
| 设备 | `dobot-nova-left` → A，`dobot-nova-right` → B |
| 参数 | pick: `object_id/source_ref`；place: `object_id/destination_ref`，不接受额外动作参数 |
| 位置 | A: `object:A`、`target:target_A`；B: `object:B`、`target:target_B`，由本机示教场景解析，无任意坐标 |
| 边 | `{source,target,edge_type:"success"}`；多条入边是 AND，所有依赖成功才可启动 |
| 等待 | `duration_s` 0–60 秒，且不超过节点 timeout；无需技能调用 |
| 完成条件 | action: `motion_sequence_completed`；wait: `wait_elapsed`，可省略使用默认 |
| 超时 | 每技能总时间 `timeout_s` 0.01–900 秒，默认 180 秒，包含该技能内部步骤与资源等待 |
| 失败 | 立即停止下游并保留证据；接受 `on_fail` 的外部决策元数据，但不会本地重试/修改图/启动恢复节点 |
| 资源 | 单 rig 一个执行；同臂动作必须显式依赖并串行执行；mock/live 独立跨臂动作可并行，最多左右臂各一个 |

拒绝环、未知技能、未知条件、额外 preconditions/abort_conditions、非 success 边、`condition/observe/parallel/human_approval/recovery` 节点及断点续跑。并行通过多入口 DAG 表达。拒绝不能实现的安全条件，而不是忽略它。目录修订值不作为运行包兼容性门槛；场景 stamp 的内部复位检查仍独立生效。所有图必须让已抓取物体最终释放；执行成功后保存命令层逻辑位置，现场传感器尚不能独立验证物体是否实际到位。

失败后的恢复、重新感知和派生图由上游决策。当前 fresh-run 子集不提供恢复节点/断点动作；真机失败后需现场复位再执行新图。双方仍须确认正式 JSON Schema、完整 typed-edge/wait/条件结构、取消合同及事件 ACK。

### 化学原子技能组 `dobot.*`

第二个技能族，与 `embodied.*` **共用同一信封**（`schema_version=2.1` / `graph_version=0.1`）、同一批路由、同一套事件与幂等规则；不新增第二个 HTTP 入口。技能 ID 是化学原子 ID 加 `dobot.` 前缀，同一份 `GET /v1/catalog` 的 `atomic_skills` 里发布。

| 技能 ID（`1.0.0`） | 最小动作 | 参数 |
|---|---|---|
| `dobot.arm.move_joint` | 关节运动到一个教示位姿 | `target_ref`（必填）、可选 `source_ref`、`speed`（默认 20，1–100）、`grip_requirement`（`any`/`open`/`clamped`）、`tilt_limit_deg`（0–10） |
| `dobot.arm.move_linear` | 从教示起点直线到教示终点 | `source_ref`、`target_ref` 均必填，其余同上 |
| `dobot.arm.move_to_home` | 回到本机标定初始关节位姿 | 可选 `source_ref`、`speed`（默认 20）；**不接受目标位姿参数**，home 只有一个来源 |
| `dobot.gripper.set_position` | 夹爪一次开度动作 | `position`（必填，0–255）、`speed`（默认 10）、`allow_release`（默认 `false`）、可选 `at_ref` |
| `dobot.arm.rotate_about_pivot` | 绕杯心一次连续倾转 | `source_ref` 必须为 `right.beaker.work`、`tilt_deg`（默认 80，(0,90]）、`offset_mm`（3 个数，缺省取 `profile.pour.offset_mm`）、`segments`（默认 16，2–128）、`speed`（默认 15）、`cp`（默认 100） |

| 项目 | 化学族的合同 |
|---|---|
| 设备 | `dobot-nova-left`/`dobot-nova-right`，与 `embodied.*` 相同；`idempotency_key` 仍全局唯一 |
| 位姿引用 | 只接受 `profile.json` 里属于该臂的教示位姿；未知位姿、别的手臂的位姿、白名单外的技能名都返回 422 |
| 参数校验 | 由 `task_skills.chemistry.registry.normalize_parameters` 单独校验（网关不重写默认值/范围），缺省值按上表补齐后原样落到节点 |
| 等待 | `control.hold` **不是技能**：用信封自带的 `wait` 节点（`duration_s` 0–60，且 ≤ `timeout_s`）表达停留与错峰 |
| 对象状态 | `dobot.*` 移动的是工具而不是场景对象，因此**不参与** `embodied.pick/place` 的 `held`/`location` 状态机，`modeled_object_state` 保持 A/B 原位 |
| 完成条件 | action: `motion_sequence_completed`；wait: `wait_elapsed`，可省略使用默认 |
| live 开始闸门 | **不是**"两臂必须已在示教 home"。化学驱动持自己的会话（设备租约 → `ClearError` → `EnableRobot` → 速度 → `Tool0`/`User0`），live 前逐节点按拓扑序对"从一次真机读数推演出的逐臂状态"做只读预检，预检失败即拒绝整包；跑完仍由网关为每个参与臂发 `arm_finalizing`/`arm_finalized` 并发归位 |
| 独占 | 化学包由化学 `LiveDriver` **独占**机械臂会话，网关只转发、不另开连接；同一 `DOBOT_DEVICE_LOCK_PATH` 的旧 CLI/ZMQ 进程必须先停 |

可提交示例：[examples/runtime-chemistry-beaker-pour.json](examples/runtime-chemistry-beaker-pour.json)（20 节点 / 24 边的双臂"右臂烧杯倒进左臂离心管"，含去程 8 s、回程 5 s 两次错峰与两臂同时复位）。它由 `examples/chemistry/right_beaker_pour_into_left_tube.json` 经 `task_skills.gateway.chemistry.to_runtime_package` 机械生成，测试逐字节钉住两者一致——对手工改过的 JSON，校验器可能接受但会与库的行为脱节，因此请以生成路径为准。

mock 联调：

```powershell
python -B -m task_skills.run_gateway --mode mock --port 8080 --data-dir ./gateway-data/mock-chem
python -B -m task_skills.gateway.demo_client --package docs/execution-api/examples/runtime-chemistry-beaker-pour.json
```

真机：先按下一节做现场复位，再 `--mode live --allow-live` 起同一网关；化学包在该模式下不再要求"两臂在示教 home"，但仍要求现场无旧控制进程持锁。

## 接上游服务

```powershell
$env:DOBOT_COMPILER_TOKEN = '<从现场凭据配置注入>'
python -B -m task_skills.run_gateway --mode mock --compiler-url https://compiler.example.com --data-dir ./gateway-data/partner-mock
```

上游 base URL 只由服务端启动配置，请求体不能指定回调地址。网关启动后顺序投递 `POST /v1/integrations/catalog/validate`、`POST /v1/integrations/catalog/snapshots`，并向 `/v1/integrations/device-state-events` 投递动态状态。发送事件到 `POST /v1/executions/{execution_id}/events`，保持原外部节点 ID、技能版本、`event_id`、单会话递增 `sequence`。

除直接传完整 `graph/package`，配置上游后还支持请求 `{"run_id":"..."}`：网关 GET `/v1/integrations/execution-plans/{run_id}`，接受完整包或 `{"package":包}`，再执行同一校验流程。这是本机扩展，不能替代对方 execute 入口要求的完整 graph。

事件与状态事务落盘，后台按流顺序投递。断网/404（上游尚未登记 execution_id）/协议解析失败均保留队列；重试使用相同 event_id/sequence。当前 ACK 约定：2xx 的 JSON object（204 空体视为 `{}`），如有 `accepted` 必须为布尔 true，带错误 `code` 不算确认。服务重启会继续投递；未完成动作会标为 interrupted，**不会自动重放物理动作**。目录验证返回阻塞诊断后不会发布该目录，待上游修正。没配置上游时仍保存事件，可查询，不会声称已送达。

## 真机启用与现场恢复

本次仅验证 mock 和 fake 控制器；没有向真实机械臂或对方服务发请求。live 仅允许 Windows 环境，并须显式 `--mode live --allow-live` 和 `DOBOT_GATEWAY_TOKEN`。默认端口只监听本机；非 loopback 需要令牌与 TLS，实验室明文例外须明确指定 `--allow-insecure-http`。这是 HTTP bearer/TLS 联调实现，尚未实现生产 mTLS/OAuth2。

live 初次执行前：确认两臂在示教 home、已使能/空闲/无报警，两夹爪张开，A/B 实际放回初始示教位置；停止旧 CLI/ZMQ/gateway 进程。先对将要使用的 live 目录记录现场复位，再启动：

```powershell
$env:DOBOT_DEVICE_LOCK_PATH = 'C:/现场共享路径/dobot-device-owner'
python -B -m task_skills.gateway.operator --mode live --data-dir ./gateway-data/live --confirm RESET_VERIFIED
$env:DOBOT_GATEWAY_TOKEN = '<现场注入令牌>'
python -B -m task_skills.run_gateway --mode live --allow-live --data-dir ./gateway-data/live
```

所有代码副本与旧控制入口必须使用同一 `DOBOT_DEVICE_LOCK_PATH`；未升级的旧进程不能被此协作锁约束。网关会检查 controller 模式/home，将速度倍率、关节/直线速度及加速度参数设为 20%；成功结束时所有参与臂并行归 home，并等待全部归位完成后才报告成功；已在标定 home 的臂会被跳过（`skipped`，不发指令），不会为了凑动作再发一次关节移动。不会自动清报警、使能或在启动时移动机械臂。内部场景 stamp 绑定唯一 registry、复位 epoch 与场景修订，另一个数据目录的复位会使旧目录失效；数据库固定 mock/live 模式，不能混用。

取消/失败停止双臂、夹爪原位 hold 并检查稳定反馈；停止后命令 gate 禁止追加动作，全部执行线程退出后再次验证停止。停止未确认或真机执行未完整结束会隔离 rig，不再接新执行。ZMQ 老入口退出没有可靠停止验证，因此保留 owner 锁。不要自动回原点、松爪或清除这个锁。

恢复顺序：关闭控制进程并现场确认机器人/夹爪已静止 → 人工处理未完成任务并恢复初始场景 → 核实 `owner.json` 对应进程已退出后由现场管理员清除该 owner 目录 → 对同一 live 数据目录运行上述 operator 命令 → 重启网关。operator 不删 owner 锁、不连接设备、不删执行历史。未知进程/未知持物状态必须先现场处理。

证据等级为 `command_trace`：控制器位姿与夹爪反馈可证明指令执行过程，不能独立证明抓住/放准物体。跨臂并行不代替真机碰撞检测、PLC 或硬件急停。

## 离线验证

```powershell
python -B -m unittest discover -s tests -p test_execution_gateway.py -v
python -B -m unittest discover -s tests -p test_chemistry_skills.py -q
python -B -m unittest discover -s tests -p test_atomic_runner_unittest.py -q
python -B -m unittest discover -s tests -p test_teleop_regressions.py -q
```

覆盖真实 loopback HTTP、模拟上游目录/拉图/事件回传、幂等并发、两阶段 token、取消/超时/失败、重启、逻辑物体状态、协作锁、模式隔离、场景 stamp，以及 `dobot.*` 技能族的目录发布、族隔离（混用 422）、位姿白名单、参数缺省补齐、live 化学前置闸门与 20 节点图 mock 端到端。真机部署和物理成功仍需现场联调。
