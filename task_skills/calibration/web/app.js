"use strict";

const STEP_META = [
  null,
  { eyebrow: "STEP 01 / 07", title: "环境、依赖与任务准备", description: "确认隔离环境与固定几何，创建真实任务或明确标记的模拟任务。" },
  { eyebrow: "STEP 02 / 07", title: "相机内参与多视图检查", description: "移动棋盘覆盖画面，分配 train/val，检查逐图残差、角点覆盖与整体 RMS。" },
  { eyebrow: "STEP 03 / 07", title: "固定棋盘与不可变编号", description: "固定棋盘，检测 88 个内角点，人工确认 normal/reverse 并冻结编号版本。" },
  { eyebrow: "STEP 04 / 07", title: "左右臂 TCP 角点示教", description: "左右臂独立记录元数据与示教点；点击编号图选点，单位始终显式标注。" },
  { eyebrow: "STEP 05 / 07", title: "左右臂外参拟合", description: "每臂独立运行平面刚体 + PnP 拟合，查看实时日志、残差图、坐标轴与 4×4 变换。" },
  { eyebrow: "STEP 06 / 07", title: "像素到左右臂基座查询", description: "在原始分辨率图像上点击，按有符号板高度映射到左右臂基座，正确换算显示像素。" },
  { eyebrow: "STEP 07 / 07", title: "独立验收与报告导出", description: "输入独立测量参考值，分别验收左右臂，并导出带哈希、单位、版本和范围说明的报告包。" }
];

const state = {
  bootstrap: null,
  csrf: null,
  sessionId: null,
  sessionGeneration: 0,
  sessionRequestGeneration: 0,
  sessionListRequestGeneration: 0,
  sessionDependencyFingerprint: null,
  session: null,
  step: Number(localStorage.getItem("calibration-step") || 1),
  activeArm: "left",
  teachSelections: { left: null, right: null },
  teachDrafts: { left: [], right: [] },
  verifyDrafts: {
    left: { method: "", independence_confirmed: false, rows: [] },
    right: { method: "", independence_confirmed: false, rows: [] }
  },
  jobs: {},
  logCache: {},
  pollTimer: null,
  cameraTimer: null,
  cameraErrorShown: false,
  busyCount: 0,
  queryResult: null,
  queryPixel: null,
  querySequence: 0,
  exportRecord: null,
  dirtyTeach: { left: false, right: false },
  dirtyVerify: { left: false, right: false },
  draftRevision: {
    teach: { left: 0, right: 0 },
    verify: { left: 0, right: 0 }
  },
  pendingDraftResets: { teach: {}, verify: {} }
};

const byId = (id) => document.getElementById(id);
const armsCn = { left: "左臂", right: "右臂" };

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = String(value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key === "style") Object.assign(node.style, value);
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === "checked" || key === "disabled" || key === "readOnly" || key === "multiple") node[key] = Boolean(value);
    else if (key === "value") node.value = value;
    else node.setAttribute(key, String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function replace(container, ...children) {
  if (!container) return;
  container.replaceChildren(...children.flat(Infinity).filter((item) => item !== null && item !== undefined));
}

function clearNode(container) {
  if (container) container.replaceChildren();
}

function setText(node, value) {
  if (node) node.textContent = value === null || value === undefined ? "" : String(value);
}

function badge(text, kind = "neutral") {
  return el("span", { class: `badge ${kind}`, text });
}

function metric(label, value, note = "", kind = "") {
  const card = el("div", { class: `metric-card ${kind}` },
    el("span", { text: label }),
    el("strong", { text: value }),
    note ? el("small", { text: note }) : null
  );
  return card;
}

function fmt(value, digits = 3) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return number.toFixed(digits);
}

function fmtTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-CN", { hour12: false });
}

function pct(value, digits = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "—";
}

function assetUrl(relative) {
  if (!relative || !state.sessionId) return "";
  const encoded = String(relative).split("/").map((part) => encodeURIComponent(part)).join("/");
  return `/api/sessions/${encodeURIComponent(state.sessionId)}/assets/${encoded}`;
}

function apiUrl(path) {
  return `/api/sessions/${encodeURIComponent(state.sessionId)}${path}`;
}

function sessionContext() {
  if (!state.sessionId) return null;
  return { sessionId: state.sessionId, generation: state.sessionGeneration };
}

function isCurrentSession(context) {
  return Boolean(context) && context.sessionId === state.sessionId
    && context.generation === state.sessionGeneration;
}

function assertCurrentSession(context) {
  if (!isCurrentSession(context)) {
    throw new ApiError("当前任务已切换，本次操作已取消。", "SESSION_CHANGED", 409);
  }
  return context;
}

async function sessionApi(context, path, options = {}) {
  assertCurrentSession(context);
  const result = await api(`/api/sessions/${encodeURIComponent(context.sessionId)}${path}`, options);
  assertCurrentSession(context);
  return result;
}

function parseRequiredNumber(value, label, code = "INVALID_NUMBER") {
  if (value === null || value === undefined) {
    throw new ApiError(`${label} 不能为空`, code);
  }
  const text = String(value).trim();
  if (text === "") throw new ApiError(`${label} 不能为空`, code);
  const number = Number(text);
  if (!Number.isFinite(number)) {
    throw new ApiError(`${label} 必须是有限数字`, code);
  }
  return number;
}

function parseRequiredInteger(value, label, code = "INVALID_INTEGER") {
  if (value === null || value === undefined || String(value).trim() === "") {
    throw new ApiError(`${label} 不能为空`, code);
  }
  const number = Number(String(value).trim());
  if (!Number.isInteger(number)) throw new ApiError(`${label} 必须是整数`, code);
  return number;
}

function markDraftDirty(domain, arm) {
  state.draftRevision[domain][arm] += 1;
  (domain === "teach" ? state.dirtyTeach : state.dirtyVerify)[arm] = true;
}

function hasDirtyDrafts() {
  return ["left", "right"].some((arm) => state.dirtyTeach[arm] || state.dirtyVerify[arm]);
}

function invalidateQueryResult({ clearPixel = false } = {}) {
  state.querySequence += 1;
  state.queryResult = null;
  if (clearPixel) state.queryPixel = null;
  if (byId("queryResults")) renderQueryResults();
}

function recordProjection(record, fields) {
  if (!record || typeof record !== "object") return null;
  return Object.fromEntries(fields.map((field) => [field, record[field] ?? null]));
}

function sessionDependencySnapshot(session, arms = ["left", "right"]) {
  const ui = session?.ui || {};
  const gates = ui.gates || {};
  const board = session?.board || {};
  const confirmation = board.confirmation;
  const intrinsics = session?.intrinsics || {};
  const result = intrinsics.result;
  const armSnapshot = {};
  for (const arm of arms) {
    const armState = session?.arms?.[arm] || {};
    const verification = armState.verification || {};
    armSnapshot[arm] = {
      metadata: armState.metadata ?? null,
      metadata_hash: armState.metadata_hash ?? null,
      points: armState.points ?? null,
      points_hash: armState.points_hash ?? null,
      fit: recordProjection(armState.fit, [
        "input_hash", "artifact_sha256", "status", "exit_code",
        "validation_passed", "usable", "stale", "stale_reason"
      ]),
      fit_gate: gates.arms?.[arm]
        ? { ok: gates.arms[arm].fit_ok, reason: gates.arms[arm].fit_reason }
        : { ok: false, reason: "" },
      verification_draft: {
        rows: verification.rows ?? [],
        method: verification.method ?? "",
        independence_confirmed: Boolean(verification.independence_confirmed)
      },
      verification: recordProjection(verification.result, [
        "input_hash", "artifact_sha256", "status", "exit_code",
        "validation_passed", "stale", "stale_reason"
      ]),
      verification_gate: gates.arms?.[arm]
        ? { ok: gates.arms[arm].verification_ok, reason: gates.arms[arm].verification_reason }
        : { ok: false, reason: "" }
    };
  }
  return {
    synthetic_only: Boolean(session?.synthetic_only),
    settings: {
      camera_id: session?.settings?.camera_id ?? null,
      image_size_wh: session?.settings?.image_size_wh ?? null,
      board: session?.settings?.board ?? null,
      thresholds: session?.settings?.thresholds ?? null
    },
    intrinsics: {
      images: (intrinsics.images || []).map((image) => ({
        id: image.id ?? null, split: image.split ?? null,
        original_path: image.original_path ?? null, sha256: image.sha256 ?? null
      })),
      result: recordProjection(result, [
        "input_hash", "artifact_sha256", "status", "exit_code",
        "validation_passed", "stale", "stale_reason"
      ]),
      gate: gates.intrinsics
        ? { ok: gates.intrinsics.ok, reason: gates.intrinsics.reason }
        : { ok: false, reason: "" }
    },
    board: {
      revision: board.revision ?? null,
      state: board.state ?? null,
      source_path: board.source_path ?? null,
      source_sha256: board.source_sha256 ?? null,
      confirmation: recordProjection(confirmation, [
        "input_hash", "observation_sha256", "corner_order",
        "immutable_revision", "fixed_confirmed", "stale", "stale_reason"
      ]),
      gate: gates.board
        ? { ok: gates.board.ok, reason: gates.board.reason }
        : { ok: false, reason: "" }
    },
    depth_registration: recordProjection(session?.depth_registration, [
      "confirmed", "registration_note", "intrinsics_input_hash"
    ]),
    depth_gate: gates.depth
      ? { ok: gates.depth.ok, reason: gates.depth.reason }
      : { ok: false, reason: "" },
    arms: armSnapshot
  };
}

function stableStringify(value) {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function sessionDependencyFingerprint(session, arms = ["left", "right"]) {
  return stableStringify(sessionDependencySnapshot(session, arms));
}

function exportMatchesSession(record, session) {
  if (!record?.dependencies || !session) return false;
  const arms = Array.isArray(record.arms) && record.arms.length ? record.arms : ["left", "right"];
  return stableStringify(record.dependencies) === sessionDependencyFingerprint(session, arms);
}

function acceptSessionSnapshot(session) {
  const nextFingerprint = sessionDependencyFingerprint(session);
  const dependenciesChanged = Boolean(state.sessionDependencyFingerprint
    && state.sessionDependencyFingerprint !== nextFingerprint);
  if (dependenciesChanged) {
    invalidateQueryResult();
    state.exportRecord = null;
  }
  state.session = session;
  state.sessionDependencyFingerprint = nextFingerprint;
  if (dependenciesChanged && byId("exportResult")) renderExportPanel();
}

function queueDraftResets(domain, arms, context) {
  for (const item of arms || []) {
    const arm = typeof item === "string" ? item : item.arm;
    const revision = typeof item === "string" ? undefined : item.revision;
    if (!arm || !["left", "right"].includes(arm)) continue;
    state.pendingDraftResets[domain][arm] = {
      sessionId: context.sessionId,
      generation: context.generation,
      revision: revision ?? state.draftRevision[domain][arm]
    };
  }
}

function applyPendingDraftResets(context) {
  for (const domain of ["teach", "verify"]) {
    for (const arm of ["left", "right"]) {
      const pending = state.pendingDraftResets[domain][arm];
      if (!pending) continue;
      if (pending.sessionId !== context.sessionId || pending.generation !== context.generation) {
        delete state.pendingDraftResets[domain][arm];
        continue;
      }
      if (state.draftRevision[domain][arm] === pending.revision) {
        if (domain === "teach") resetTeachDraft(arm);
        else resetVerifyDraft(arm);
      }
      delete state.pendingDraftResets[domain][arm];
    }
  }
}

function imageContentType(file) {
  if (file.type === "image/png" || file.type === "image/jpeg" || file.type === "image/jpg") return file.type;
  const name = (file.name || "").toLowerCase();
  if (name.endsWith(".png")) return "image/png";
  if (name.endsWith(".jpg") || name.endsWith(".jpeg")) return "image/jpeg";
  return "";
}

class ApiError extends Error {
  constructor(message, code = "REQUEST_FAILED", status = 0, payload = null) {
    super(message);
    this.code = code;
    this.status = status;
    this.payload = payload;
  }
}

async function api(path, options = {}) {
  const method = options.method || "GET";
  const headers = { ...(options.headers || {}) };
  const request = { method, headers, credentials: "same-origin" };
  if (method !== "GET" && state.csrf) headers["X-Calibration-Token"] = state.csrf;
  if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
    request.body = JSON.stringify(options.json);
  }
  if (options.body !== undefined) request.body = options.body;
  const response = await fetch(path, request);
  const contentType = response.headers.get("Content-Type") || "";
  let payload = null;
  if (contentType.includes("application/json")) {
    payload = await response.json().catch(() => null);
  } else {
    payload = await response.text().catch(() => "");
  }
  if (!response.ok) {
    const message = payload && typeof payload === "object" && payload.message
      ? payload.message : `请求失败（HTTP ${response.status}）`;
    const code = payload && typeof payload === "object" && payload.code ? payload.code : "REQUEST_FAILED";
    throw new ApiError(message, code, response.status, payload);
  }
  return payload;
}

function showToast(message, kind = "info", timeout = 4300) {
  const region = byId("toastRegion");
  if (!region) return;
  const toast = el("div", { class: `toast ${kind === "error" ? "error" : kind === "success" ? "success" : ""}`, text: message });
  region.appendChild(toast);
  window.setTimeout(() => toast.remove(), timeout);
}

function showError(error) {
  const message = error instanceof ApiError
    ? `${error.message}${error.code ? ` [${error.code}]` : ""}`
    : "操作失败；请查看本地终端日志。";
  showToast(message, "error", 6500);
}

function setBusy(label) {
  if (label) {
    state.busyCount += 1;
    byId("busyText").textContent = label;
    byId("busyBar").classList.remove("hidden");
  } else {
    state.busyCount = Math.max(0, state.busyCount - 1);
    if (state.busyCount === 0) byId("busyBar").classList.add("hidden");
  }
}

async function action(label, fn) {
  setBusy(label);
  try {
    return await fn();
  } catch (error) {
    if (error instanceof ApiError && error.code === "SESSION_CHANGED") return undefined;
    showError(error);
    return undefined;
  } finally {
    setBusy(null);
  }
}

function activeJobs(session = state.session) {
  return (session?.jobs || []).filter((job) => ["queued", "running"].includes(job.status));
}

function latestJob(kind, arm = null) {
  const jobs = state.session?.jobs || [];
  for (let index = jobs.length - 1; index >= 0; index -= 1) {
    const job = jobs[index];
    if (job.kind === kind && (arm === null || job.arm === arm)) return job;
  }
  return null;
}

function resetTeachDraft(arm) {
  state.draftRevision.teach[arm] += 1;
  if (!state.session) {
    state.teachDrafts[arm] = [];
    state.dirtyTeach[arm] = false;
    return;
  }
  state.teachDrafts[arm] = (state.session.arms[arm].points || []).map((point) => ({
    ...point, x: String(point.x), y: String(point.y), z: String(point.z)
  }));
  state.dirtyTeach[arm] = false;
}

function resetVerifyDraft(arm) {
  state.draftRevision.verify[arm] += 1;
  if (!state.session) {
    state.verifyDrafts[arm] = { method: "", independence_confirmed: false, rows: [] };
    state.dirtyVerify[arm] = false;
    return;
  }
  const verification = state.session.arms[arm].verification;
  state.verifyDrafts[arm] = {
    method: verification.method || "",
    independence_confirmed: Boolean(verification.independence_confirmed),
    rows: (verification.rows || []).map((row) => Object.fromEntries(
      ["u_px", "v_px", "board_z_m", "reference_x_m", "reference_y_m", "reference_z_m"]
        .map((key) => [key, String(row[key])])
    ))
  };
  state.dirtyVerify[arm] = false;
}

function resetDrafts({ preserveDirty = false, forceArms = [] } = {}) {
  for (const arm of ["left", "right"]) {
    const force = forceArms.includes(arm);
    if (force || !preserveDirty || !state.dirtyTeach[arm]) resetTeachDraft(arm);
    if (force || !preserveDirty || !state.dirtyVerify[arm]) resetVerifyDraft(arm);
  }
}

function gate(arm, name) {
  if (!state.session) return { ok: false, reason: "请先创建或恢复任务" };
  if (name === "arm") return state.session.ui.gates.arms[arm];
  return state.session.ui.gates[name] || { ok: false, reason: "前置条件未知" };
}

function stepCompletion() {
  const session = state.session;
  if (!session) return [false, false, false, false, false, false, false];
  const armReady = (arm) => session.ui.gates.arms[arm].train_count >= 6 && session.ui.gates.arms[arm].val_count >= 3
    && session.arms[arm].metadata.robot_id && session.arms[arm].metadata.base_frame && session.arms[arm].metadata.tcp_name;
  return [
    true,
    session.ui.gates.intrinsics.ok,
    session.ui.gates.board.ok,
    armReady("left") && armReady("right"),
    session.ui.gates.arms.left.fit_ok && session.ui.gates.arms.right.fit_ok,
    session.ui.gates.arms.left.fit_ok && session.ui.gates.arms.right.fit_ok,
    session.ui.gates.arms.left.verification_ok && session.ui.gates.arms.right.verification_ok
  ];
}

function renderGlobal() {
  const session = state.session;
  setText(byId("sessionBadge"), session ? session.display_name : "未选择任务");
  byId("sessionBadge").className = `badge ${session ? "cyan" : "neutral"}`;
  byId("syntheticBadge").classList.toggle("hidden", !session?.synthetic_only);
  const deps = state.bootstrap?.dependencies;
  setText(byId("serverBadge"), deps?.ok ? "本机服务正常" : "依赖异常");
  byId("serverBadge").className = `badge ${deps?.ok ? "success" : "danger"}`;
  const camera = state.bootstrap?.camera || {};
  setText(byId("cameraBadge"), camera.open ? `相机 ${camera.actual_wh?.join("×") || "已连接"}` : "相机未打开");
  byId("cameraBadge").className = `badge ${camera.open ? "success" : "neutral"}`;
  byId("emptySessionBanner").classList.toggle("hidden", Boolean(session));
}

function renderSessionList() {
  const select = byId("sessionSelect");
  if (!select) return;
  const sessions = state.bootstrap?.sessions || [];
  const first = el("option", { value: "", text: "新建或选择任务" });
  select.replaceChildren(first);
  for (const session of sessions) {
    const label = `${session.display_name} · ${session.mode === "synthetic" ? "模拟" : "真实"} · ${fmtTime(session.updated_utc)}`;
    select.appendChild(el("option", { value: session.session_id, text: label }));
  }
  if (state.sessionId) {
    if (!sessions.some((item) => item.session_id === state.sessionId)) {
      select.appendChild(el("option", { value: state.sessionId, text: state.session?.display_name || state.sessionId }));
    }
    select.value = state.sessionId;
  }
}

function renderStepNav() {
  const completion = stepCompletion();
  const running = activeJobs();
  const runningKinds = new Set(running.map((job) => job.kind));
  document.querySelectorAll(".step-button").forEach((button) => {
    const step = Number(button.dataset.step);
    button.classList.toggle("is-active", step === state.step);
    button.classList.remove("state-complete", "state-blocked", "state-running");
    if (completion[step - 1]) button.classList.add("state-complete");
    if (step === 2 && runningKinds.has("intrinsics")) button.classList.add("state-running");
    if (step === 5 && runningKinds.has("fit")) button.classList.add("state-running");
    if (step === 7 && runningKinds.has("verify")) button.classList.add("state-running");
    if (step === 1 && runningKinds.has("demo")) button.classList.add("state-running");
    if (state.session && step > 1 && !completion[step - 1] && !button.classList.contains("state-running")) button.classList.add("state-blocked");
    button.setAttribute("aria-current", step === state.step ? "step" : "false");
  });
  byId("prevStep").disabled = state.step <= 1;
  byId("nextStep").disabled = state.step >= 7;
}

function renderStageVisibility() {
  const meta = STEP_META[state.step] || STEP_META[1];
  setText(byId("stageEyebrow"), meta.eyebrow);
  setText(byId("stageTitle"), meta.title);
  setText(byId("stageDescription"), meta.description);
  document.querySelectorAll("[data-stage]").forEach((section) => {
    section.classList.toggle("hidden", Number(section.dataset.stage) !== state.step);
  });
  renderStepNav();
}

function renderDependencies() {
  const deps = state.bootstrap?.dependencies || {};
  const grid = byId("dependencyGrid");
  replace(grid,
    dependencyItem("Python", deps.python || "缺失", deps.python_supported ? "推荐 3.11/3.12" : "当前解释器不在推荐范围"),
    dependencyItem("NumPy", deps.numpy_runtime || "缺失", `固定 ${deps.pinned_numpy}`),
    dependencyItem("OpenCV", deps.opencv_runtime || "缺失", `固定发行包 ${deps.pinned_opencv_python}`),
    dependencyItem("固定依赖匹配", deps.pinned_match ? "是" : "否", deps.pinned_match ? "隔离环境符合基线" : "需复核 venv 与版本"),
    dependencyItem("VideoCapture API", deps.videoio_available ? "可用" : "缺失", "启动时未打开设备"),
    dependencyItem("硬件连接", "0", "导入/启动不连接相机、机械臂或串口")
  );
  const overall = byId("dependencyOverall");
  setText(overall, deps.ok ? (deps.pinned_match ? "依赖符合" : "依赖可运行") : "依赖异常");
  overall.className = `badge ${deps.ok ? (deps.pinned_match ? "success" : "warning") : "danger"}`;
}

function dependencyItem(label, value, note) {
  return el("div", { class: "dependency-item" }, el("strong", { text: label }), el("span", { text: value }), el("small", { class: "microcopy", text: note }));
}

function fillSettingsForm() {
  const panel = byId("settingsPanel");
  if (!state.session) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");
  const form = byId("settingsForm");
  const settings = state.session.settings;
  const values = {
    camera_id: settings.camera_id,
    width: settings.image_size_wh[0],
    height: settings.image_size_wh[1],
    cols: settings.board.cols,
    rows: settings.board.rows,
    square_m: settings.board.square_m,
    intrinsics_max_rms_px: settings.thresholds.intrinsics_max_rms_px,
    fit_max_pnp_rms_px: settings.thresholds.fit_max_pnp_rms_px,
    fit_max_error_mm: settings.thresholds.fit_max_error_mm,
    verify_max_error_mm: settings.thresholds.verify_max_error_mm
  };
  for (const [name, value] of Object.entries(values)) {
    if (form.elements[name]) form.elements[name].value = String(value);
  }
  setText(byId("settingsRevision"), `revision ${state.session.revision} · ${state.session.mode}`);
}

function syncCreateMode() {
  const form = byId("createSessionForm");
  const synthetic = form.elements.mode.value === "synthetic";
  for (const name of ["width", "height", "cols", "rows", "square_m"]) {
    // Keep fixed demo geometry in FormData; disabled controls are omitted and
    // would otherwise submit as null/0.
    form.elements[name].readOnly = synthetic;
  }
  if (synthetic) {
    form.elements.width.value = "1280";
    form.elements.height.value = "960";
    form.elements.cols.value = "11";
    form.elements.rows.value = "8";
    form.elements.square_m.value = "0.03";
  }
}

function numericForm(form, names) {
  const result = {};
  for (const name of names) result[name] = Number(form.elements[name].value);
  return result;
}

async function loadSession(sessionId, { reset = true, preserveStep = true,
  allowDiscardDirty = false } = {}) {
  if (!sessionId) return false;
  const switching = state.sessionId !== sessionId;
  if (switching && state.session && hasDirtyDrafts() && !allowDiscardDirty
      && !window.confirm("当前任务有未保存的示教或独立测量草稿。切换任务会丢弃这些草稿，是否继续？")) {
    renderSessionList();
    return false;
  }
  const context = { sessionId, generation: state.sessionGeneration + 1 };
  state.sessionGeneration = context.generation;
  const requestGeneration = ++state.sessionRequestGeneration;
  state.sessionId = sessionId;
  if (switching) {
    state.session = null;
    state.sessionDependencyFingerprint = null;
    state.pendingDraftResets = { teach: {}, verify: {} };
    invalidateQueryResult({ clearPixel: true });
    state.exportRecord = null;
    resetDrafts({ preserveDirty: false });
    renderAll();
  }
  localStorage.setItem("calibration-session", sessionId);
  const url = new URL(window.location.href);
  url.searchParams.set("session", sessionId);
  history.replaceState(null, "", url);
  const data = await api(`/api/sessions/${encodeURIComponent(sessionId)}`);
  if (!isCurrentSession(context) || requestGeneration !== state.sessionRequestGeneration) return false;
  acceptSessionSnapshot(data.session);
  await refreshJobLogs(context, requestGeneration);
  if (!isCurrentSession(context) || requestGeneration !== state.sessionRequestGeneration) return false;
  invalidateQueryResult({ clearPixel: true });
  state.exportRecord = null;
  if (reset) resetDrafts({ preserveDirty: false });
  if (!preserveStep) state.step = 1;
  renderAll();
  schedulePoll();
  return true;
}

async function refreshSession(reset = false, options = {}) {
  const context = options.context || sessionContext();
  if (!context || !isCurrentSession(context)) return false;
  queueDraftResets("teach", options.resetTeachArms, context);
  queueDraftResets("verify", options.resetVerifyArms, context);
  const requestGeneration = ++state.sessionRequestGeneration;
  const data = await api(`/api/sessions/${encodeURIComponent(context.sessionId)}`);
  if (!isCurrentSession(context) || requestGeneration !== state.sessionRequestGeneration) return false;
  acceptSessionSnapshot(data.session);
  await refreshJobLogs(context, requestGeneration);
  if (!isCurrentSession(context) || requestGeneration !== state.sessionRequestGeneration) return false;
  if (reset) resetDrafts({ preserveDirty: options.preserveDirty !== false });
  applyPendingDraftResets(context);
  renderAll();
  schedulePoll();
  return true;
}

function renderAll() {
  renderGlobal();
  renderSessionList();
  fillSettingsForm();
  renderDependencies();
  renderStepNav();
  renderStageVisibility();
  renderIntrinsics();
  renderBoard();
  renderTeach();
  renderFits();
  renderQuery();
  renderVerification();
  renderRail();
  updateCameraControls();
}

function option(value, label, selected = false) {
  const node = el("option", { value, text: label });
  node.selected = Boolean(selected);
  return node;
}

function orderedBoardCorners() {
  const board = state.session?.board;
  if (!board?.raw_corners_uv?.length) return [];
  const chosen = board.confirmation && !board.confirmation.stale
    ? board.confirmation.corner_order
    : document.querySelector('input[name="cornerOrder"]:checked')?.value || "normal";
  return chosen === "reverse" ? [...board.raw_corners_uv].reverse() : [...board.raw_corners_uv];
}

function renderJobPanel(container, job, emptyLabel) {
  if (!container) return;
  if (!job) {
    container.classList.add("hidden");
    replace(container);
    return;
  }
  container.classList.remove("hidden");
  const statusKind = job.status === "passed" ? "success"
    : job.status === "failed_threshold" ? "warning"
      : job.status === "error" || job.status === "invalidated" ? "danger"
        : job.status === "running" ? "cyan" : "neutral";
  const progress = Math.round(Number(job.progress || 0) * 100);
  const cache = state.logCache[job.id];
  const logText = cache?.text || (job.status === "queued" ? "等待后台线程…" : "日志加载中…");
  replace(container,
    el("div", { class: "job-head" },
      el("strong", { text: `${job.kind}${job.arm ? ` · ${armsCn[job.arm]}` : ""}` }),
      el("div", { class: "inline-actions" }, badge(job.status, statusKind), badge(job.exit_code === null ? "exit —" : `exit ${job.exit_code}`, job.exit_code === 0 ? "success" : job.exit_code === 2 ? "warning" : job.exit_code === 1 ? "danger" : "neutral"))
    ),
    el("div", { class: "microcopy", text: `${job.phase || "—"} · ${progress}% · ${fmtTime(job.heartbeat_utc || job.created_utc)}` }),
    el("div", { class: "progress-track" }, el("i", { style: { width: `${Math.max(2, progress)}%` } })),
    job.error ? el("div", { class: "job-error", text: job.error }) : null,
    el("pre", { class: "job-log", text: logText || emptyLabel || "无日志" })
  );
}

function renderIntrinsics() {
  const session = state.session;
  const gallery = byId("intrinsicsGallery");
  const runButton = byId("runIntrinsics");
  if (!session) {
    replace(gallery, el("div", { class: "empty-media", text: "请先创建或恢复任务。" }));
    byId("intrinsicsChart").className = "chart-box empty-media";
    byId("intrinsicsChart").textContent = "运行后显示每张图重投影 RMS。";
    replace(byId("intrinsicsMetrics"));
    replace(byId("intrinsicsJob"));
    runButton.disabled = true;
    setText(byId("intrinsicsGateReason"), "请先创建任务");
    setText(byId("intrinsicsCountBadge"), "0 / 0");
    setText(byId("intrinsicsStateBadge"), "未运行");
    byId("intrinsicsStateBadge").className = "badge neutral";
    return;
  }
  const images = session.intrinsics.images || [];
  const trainCount = images.filter((image) => image.split === "train").length;
  const valCount = images.filter((image) => image.split === "val").length;
  setText(byId("intrinsicsCountBadge"), `${trainCount} train / ${valCount} val`);
  byId("intrinsicsCountBadge").className = `badge ${trainCount >= 10 && valCount >= 3 ? "success" : "warning"}`;
  setText(byId("galleryCount"), `${images.length} 张`);

  const result = session.intrinsics.result;
  const currentIntr = gate("intrinsics", "intrinsics");
  const job = latestJob("intrinsics");
  let stateLabel = "未运行";
  let stateKind = "neutral";
  if (job && ["queued", "running"].includes(job.status)) { stateLabel = "运行中"; stateKind = "cyan"; }
  else if (currentIntr.ok) { stateLabel = "已通过"; stateKind = "success"; }
  else if (result) { stateLabel = result.stale ? "已失效" : result.exit_code === 2 ? "超阈值" : "不可用"; stateKind = result.stale || result.exit_code === 1 ? "danger" : "warning"; }
  setText(byId("intrinsicsStateBadge"), stateLabel);
  byId("intrinsicsStateBadge").className = `badge ${stateKind}`;

  const metrics = [];
  if (result?.summary) {
    metrics.push(metric("训练 RMS", `${fmt(result.summary.train_rms_px, 4)} px`, "OpenCV 内参拟合"));
    metrics.push(metric("最大单图 RMS", `${fmt(result.summary.max_view_rms_px, 4)} px`, `阈值 ${fmt(result.summary.threshold_px, 3)} px`));
    metrics.push(metric("val 最大 RMS", `${fmt(result.summary.val_max_view_rms_px, 4)} px`, "仅检查，不参与拟合"));
    metrics.push(metric("图像数量", `${result.summary.train_count} / ${result.summary.val_count}`, "train / val"));
  }
  replace(byId("intrinsicsMetrics"), metrics);

  const canRun = trainCount >= 10 && valCount >= 3 && !activeJobs().some((item) => item.kind === "intrinsics" && ["queued", "running"].includes(item.status));
  runButton.disabled = !canRun;
  runButton.title = canRun ? "启动后台内参计算" : "至少需要 10 train / 3 val，且不能有同类任务运行";
  setText(byId("intrinsicsGateReason"), canRun ? "后台任务会持续写入阶段与日志。" : "至少 10 train / 3 val；等待同类任务结束。");

  renderJobPanel(byId("intrinsicsJob"), job);
  renderIntrinsicsChart(result);
  renderIntrinsicsGallery(images);
}

// Gallery selection survives re-renders; ids that left the gallery are dropped.
const gallerySelection = new Set();
let galleryAnchor = null;

function batchImages(label, body, confirmText) {
  if (confirmText && !window.confirm(confirmText)) return;
  action(label, async () => {
    const context = sessionContext();
    const result = await sessionApi(context, "/intrinsics/images/batch", { method: "POST", json: body });
    gallerySelection.clear();
    galleryAnchor = null;
    if (!await refreshSession(true, { context })) return;
    showToast(`${label}：${result.count} 张` + (result.count ? "，旧内参结果已失效。" : "（无变化）"), "success");
  });
}

function renderGalleryToolbar(images) {
  const toolbar = byId("galleryToolbar");
  if (!images.length) { replace(toolbar); return; }
  const ids = [...gallerySelection];
  const n = ids.length;
  const button = (text, kind, disabled, onclick) => el("button", { type: "button", class: `button ${kind}`, text, disabled, onclick });
  replace(toolbar,
    el("span", { class: "microcopy", text: `已选 ${n} / ${images.length}（Shift+点击可连选）` }),
    button("全选", "ghost", n === images.length, () => { images.forEach((image) => gallerySelection.add(image.id)); renderIntrinsicsGallery(images); }),
    button("取消选择", "ghost", n === 0, () => { gallerySelection.clear(); galleryAnchor = null; renderIntrinsicsGallery(images); }),
    button("所选设为 train", "secondary", n === 0, () => batchImages("批量设为 train", { action: "split", split: "train", image_ids: ids })),
    button("所选设为 val", "secondary", n === 0, () => batchImages("批量设为 val", { action: "split", split: "val", image_ids: ids })),
    button(`删除所选 (${n})`, "secondary", n === 0, () => batchImages("批量删除", { action: "delete", image_ids: ids },
      `删除所选 ${n} 张图像？该操作会使内参及下游结果失效。`)),
    button("清空全部", "ghost", false, () => batchImages("清空图库", { action: "delete", all: true },
      `清空全部 ${images.length} 张内参图像？该操作会使内参及下游结果失效。`))
  );
}

function renderIntrinsicsGallery(images) {
  const gallery = byId("intrinsicsGallery");
  const present = new Set(images.map((image) => image.id));
  for (const id of [...gallerySelection]) if (!present.has(id)) gallerySelection.delete(id);
  renderGalleryToolbar(images);
  if (!images.length) {
    replace(gallery, el("div", { class: "empty-media", text: "尚无内参图像。上传多张不同位姿的棋盘图，或显式打开本机相机采集。" }));
    return;
  }
  const cards = images.map((image, index) => {
    const selectBox = el("input", { type: "checkbox", checked: gallerySelection.has(image.id), "aria-label": `选择 ${image.display_name}`,
      onclick: (event) => {
        const on = event.target.checked;
        if (event.shiftKey && galleryAnchor !== null) {
          const [lo, hi] = [Math.min(galleryAnchor, index), Math.max(galleryAnchor, index)];
          images.slice(lo, hi + 1).forEach((item) => on ? gallerySelection.add(item.id) : gallerySelection.delete(item.id));
        } else if (on) gallerySelection.add(image.id);
        else gallerySelection.delete(image.id);
        galleryAnchor = index;
        renderIntrinsicsGallery(images);
      } });
    const preview = assetUrl(image.residual_overlay_path || image.overlay_path || image.thumbnail_path || image.original_path);
    const splitSelect = el("select", { class: "mini-select", "aria-label": `划分 ${image.display_name}` },
      option("train", "train", image.split === "train"),
      option("val", "val", image.split === "val")
    );
    splitSelect.addEventListener("change", () => action("更新数据划分", async () => {
      const context = sessionContext();
      await sessionApi(context, `/intrinsics/images/${encodeURIComponent(image.id)}`, { method: "PATCH", json: { split: splitSelect.value } });
      if (!await refreshSession(true, { context })) return;
      showToast("数据划分已更新，旧内参结果已失效。", "success");
    }));
    const coverage = image.coverage || {};
    return el("article", { class: `image-card${gallerySelection.has(image.id) ? " selected" : ""}` },
      el("img", { src: preview, alt: `角点叠加：${image.display_name}`, loading: "lazy" }),
      el("div", { class: "image-card-body" },
        el("div", { class: "image-card-title" }, el("label", { class: "image-select" }, selectBox,
          el("span", { title: image.display_name, text: image.display_name })), splitSelect),
        el("div", { class: "mini-metrics" },
          el("span", {}, "覆盖 ", el("strong", { text: pct(coverage.grid_bins_fraction) })),
          el("span", {}, "残差 ", el("strong", { text: image.reprojection_rms_px === null || image.reprojection_rms_px === undefined ? "—" : `${fmt(image.reprojection_rms_px, 3)} px` }))
        ),
        el("div", { class: "coverage-bar", title: `角点 ${coverage.corner_count || 0}；bbox ${pct(coverage.bbox_area_fraction)}` }, el("i", { style: { width: `${Math.max(1, Number(coverage.grid_bins_fraction || 0) * 100)}%` } })),
        image.detected ? null : el("div", { class: "microcopy", text: `未识别：${image.detection_error || "完整棋盘不可见"}` }),
        el("div", { class: "card-actions" },
          el("a", { class: "link", href: assetUrl(image.original_path), target: "_blank", rel: "noopener", text: "原图" }),
          el("a", { class: "link", href: assetUrl(image.overlay_path), target: "_blank", rel: "noopener", text: "角点叠加" }),
          el("button", { class: "icon-button", type: "button", text: "删除", onclick: () => {
            if (!window.confirm(`删除 ${image.display_name}？该操作会使内参及下游结果失效。`)) return;
            action("删除图像", async () => {
              const context = sessionContext();
              await sessionApi(context, `/intrinsics/images/${encodeURIComponent(image.id)}`, { method: "DELETE" });
              if (!await refreshSession(true, { context })) return;
              showToast("图像已删除，相关结果已失效。", "success");
            });
          } })
        )
      )
    );
  });
  replace(gallery, cards);
}

function renderIntrinsicsChart(result) {
  const container = byId("intrinsicsChart");
  if (!result?.views?.length) {
    container.className = "chart-box empty-media";
    setText(container, "运行后显示每张图重投影 RMS。");
    return;
  }
  container.className = "chart-box";
  const threshold = Number(result.summary?.threshold_px || state.session.settings.thresholds.intrinsics_max_rms_px);
  const maxValue = Math.max(threshold * 1.25, ...result.views.map((view) => Number(view.reprojection_rms_px) || 0), 0.05);
  const bars = result.views.map((view) => el("div", {
    class: `bar ${view.split === "val" ? "val" : ""}`,
    style: { height: `${Math.max(2, (Number(view.reprojection_rms_px) / maxValue) * 100)}%` },
    "data-label": `${view.display_name}: ${fmt(view.reprojection_rms_px, 3)} px`,
    title: `${view.display_name} · ${view.split} · ${fmt(view.reprojection_rms_px, 3)} px`
  }));
  const thresholdLine = el("div", { class: "threshold-line", style: { top: `${Math.max(2, (1 - threshold / maxValue) * 100)}%` } }, el("span", { text: `limit ${fmt(threshold, 2)} px` }));
  const chart = el("div", { class: "bar-chart" }, bars, thresholdLine);
  replace(container, chart,
    el("div", { class: "chart-legend" },
      el("span", {}, el("i", { class: "legend-dot" }), "train"),
      el("span", {}, el("i", { class: "legend-dot val" }), "val"),
      el("span", { text: result.stale ? "结果已失效，仅作证据" : "" })
    )
  );
}

function renderBoard() {
  const session = state.session;
  const original = byId("boardOriginal");
  const numbered = byId("boardNumbered");
  const confirmButton = byId("confirmBoard");
  const detectButton = byId("detectBoard");
  if (!session) {
    original.classList.add("hidden");
    numbered.classList.add("hidden");
    setText(byId("boardStateBadge"), "未上传");
    byId("boardStateBadge").className = "badge neutral";
    setText(byId("boardRevisionBadge"), "");
    setText(byId("boardConfirmation"), "请先创建或恢复任务。");
    byId("boardConfirmation").className = "result-box empty-media";
    confirmButton.disabled = true;
    detectButton.disabled = true;
    return;
  }
  const board = session.board;
  const confirmation = board.confirmation;
  const boardGate = gate("board", "board");
  let label = "未上传";
  let kind = "neutral";
  if (board.state === "uploaded") label = "待检测";
  else if (board.state === "detected") { label = "待确认"; kind = "warning"; }
  else if (boardGate.ok) { label = "已冻结"; kind = "success"; }
  else if (board.state === "stale" || confirmation?.stale) { label = "已失效"; kind = "danger"; }
  setText(byId("boardStateBadge"), label);
  byId("boardStateBadge").className = `badge ${kind}`;
  setText(byId("boardRevisionBadge"), board.revision ? `revision ${board.revision}` : "");

  if (board.source_path) {
    original.src = assetUrl(board.source_path);
    original.classList.remove("hidden");
  } else {
    original.removeAttribute("src");
    original.classList.add("hidden");
  }
  const selectedOrder = confirmation && !confirmation.stale
    ? confirmation.corner_order
    : document.querySelector('input[name="cornerOrder"]:checked')?.value || "normal";
  const previewPath = board.previews?.[selectedOrder];
  if (previewPath) {
    numbered.src = assetUrl(previewPath);
    numbered.classList.remove("hidden");
  } else {
    numbered.removeAttribute("src");
    numbered.classList.add("hidden");
  }
  document.querySelectorAll('input[name="cornerOrder"]').forEach((radio) => {
    radio.checked = radio.value === selectedOrder;
    radio.disabled = Boolean(confirmation && !confirmation.stale);
  });
  detectButton.disabled = !board.source_path || activeJobs().some((job) => job.kind === "demo");
  const canConfirm = board.state === "detected" && byId("boardFixedConfirm").checked && !activeJobs().some((job) => job.kind === "demo");
  confirmButton.disabled = !canConfirm;
  confirmButton.title = canConfirm ? "冻结当前编号版本" : "需要先检测、勾选固定确认，并等待演示任务结束";
  if (confirmation && !confirmation.stale) {
    byId("boardConfirmation").className = "result-box good";
    replace(byId("boardConfirmation"),
      el("strong", { text: `编号已冻结：${confirmation.corner_order}` }),
      el("div", { class: "microcopy", text: `immutable revision ${confirmation.immutable_revision} · ${fmtTime(confirmation.confirmed_utc)}` }),
      el("div", { class: "microcopy", text: "重新检测会生成新版本，并使示教、拟合和验收全部失效。" })
    );
  } else if (confirmation?.stale) {
    byId("boardConfirmation").className = "result-box bad";
    setText(byId("boardConfirmation"), `旧编号已失效：${confirmation.stale_reason || "上游输入改变"}`);
  } else if (board.raw_corners_uv?.length) {
    byId("boardConfirmation").className = "result-box";
    setText(byId("boardConfirmation"), `检测到 ${board.raw_corners_uv.length} 个内角点。请核对编号图并确认 normal/reverse。`);
  } else {
    byId("boardConfirmation").className = "result-box empty-media";
    setText(byId("boardConfirmation"), "尚未检测。");
  }
}

function renderTeach() {
  const container = byId("teachArms");
  document.querySelectorAll(".arm-tab").forEach((tab) => {
    const active = tab.dataset.arm === state.activeArm;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  });
  if (!state.session) {
    replace(container, el("div", { class: "empty-media", text: "请先创建或恢复任务。" }));
    return;
  }
  if (!state.teachDrafts[state.activeArm]) state.teachDrafts[state.activeArm] = [];
  replace(container, buildTeachPanel(state.activeArm));
}

function buildTeachPanel(arm) {
  const session = state.session;
  const armState = session.arms[arm];
  const metadata = armState.metadata;
  const points = state.teachDrafts[arm];
  const selectedId = state.teachSelections[arm];
  const selectedPoint = points.find((point) => Number(point.corner_id) === Number(selectedId));
  const board = session.board;
  const order = board.confirmation?.corner_order || "normal";
  const previewPath = board.previews?.[order];
  const corners = orderedBoardCorners();
  const unit = metadata.unit;

  const markers = corners.map((corner, index) => {
    const point = points.find((item) => Number(item.corner_id) === index);
    const classes = ["corner-marker"];
    if (point?.split === "train") classes.push("is-train");
    if (point?.split === "val") classes.push("is-val");
    if (selectedId === index) classes.push("is-selected");
    return el("button", {
      type: "button",
      class: classes.join(" "),
      style: {
        left: `${Number(corner[0]) / session.settings.image_size_wh[0] * 100}%`,
        top: `${Number(corner[1]) / session.settings.image_size_wh[1] * 100}%`
      },
      title: `角点 ${index}${point ? ` · ${point.split}` : ""}`,
      "aria-label": `选择角点 ${index}${point ? `，当前 ${point.split}` : ""}`,
      onclick: () => {
        syncTeachInputs(arm);
        state.teachSelections[arm] = index;
        renderTeach();
      }
    });
  });

  const boardImg = previewPath
    ? el("img", { src: assetUrl(previewPath), alt: `固定棋盘编号图 ${order}` })
    : el("div", { class: "empty-media", text: "需要先在步骤 3 检测并确认固定棋盘。" });
  const boardWrap = el("div", { class: "teach-board" }, boardImg, markers);

  const countTrain = points.filter((point) => point.split === "train").length;
  const countVal = points.filter((point) => point.split === "val").length;
  const columns = session.settings.board.cols;
  const rows = session.settings.board.rows;
  const coverage = teachCoverage(points, columns, rows);
  const coverageSummary = el("div", { class: "coverage-summary" },
    metric("train / val", `${countTrain} / ${countVal}`, "最低 6 / 3"),
    metric("x 覆盖", pct(coverage.x), "板 x 方向跨度"),
    metric("y 覆盖", pct(coverage.y), "板 y 方向跨度")
  );

  const metadataForm = el("form", { class: "form-grid compact" },
    el("label", { class: "field" }, "robot_id", el("input", { name: "robot_id", required: true, maxlength: 128, value: metadata.robot_id })),
    el("label", { class: "field" }, "base_frame", el("input", { name: "base_frame", required: true, maxlength: 128, value: metadata.base_frame })),
    el("label", { class: "field" }, "TCP 名称", el("input", { name: "tcp_name", required: true, maxlength: 128, value: metadata.tcp_name })),
    el("label", { class: "field" }, "单位", el("select", { name: "unit" }, option("mm", "mm（毫米）", unit === "mm"), option("m", "m（米）", unit === "m"))),
    el("label", { class: "field" }, "控制器 IP / 主机名", el("input", { name: "controller_host", maxlength: 253, value: metadata.controller_host || "", placeholder: "可选；仅显式 GetPose 使用" })),
    el("label", { class: "field" }, "控制器端口", el("input", { name: "controller_port", type: "number", min: 1, max: 65535, value: metadata.controller_port || 29999 })),
    el("div", { class: "span-2 inline-actions wrap" },
      el("button", { type: "submit", class: "button secondary", text: "保存机械臂元数据" }),
      el("span", { class: "unit-chip", text: `当前单位：${unit}` }),
      el("span", { class: "microcopy", text: "切换单位不会自动换算已有示教数值。" })
    )
  );
  metadataForm.addEventListener("submit", (event) => {
    event.preventDefault();
    action(`${armsCn[arm]}元数据保存`, async () => {
      const context = sessionContext();
      const payload = Object.fromEntries(new FormData(metadataForm).entries());
      payload.controller_port = Number(payload.controller_port);
      await sessionApi(context, `/arms/${arm}/metadata`, { method: "PUT", json: payload });
      if (!await refreshSession(true, { context })) return;
      showToast(`${armsCn[arm]}元数据已保存，旧拟合与验收已失效。`, "success");
    });
  });

  const splitSelect = el("select", {},
    option("train", "train", !selectedPoint || selectedPoint.split === "train"),
    option("val", "val", selectedPoint?.split === "val")
  );
  const cornerInput = el("input", { type: "number", min: 0, max: columns * rows - 1, step: 1, value: selectedId ?? "", required: true });
  const xInput = el("input", { type: "number", step: "any", value: selectedPoint?.x ?? "", required: true, "aria-label": `X（${unit}）` });
  const yInput = el("input", { type: "number", step: "any", value: selectedPoint?.y ?? "", required: true, "aria-label": `Y（${unit}）` });
  const zInput = el("input", { type: "number", step: "any", value: selectedPoint?.z ?? "", required: true, "aria-label": `Z（${unit}）` });
  const addPointButton = el("button", { type: "button", class: "button primary", text: selectedPoint ? "更新该点" : "添加示教点" });
  addPointButton.addEventListener("click", () => {
    syncTeachInputs(arm);
    const cornerText = String(cornerInput.value).trim();
    const cornerNumber = cornerText === "" ? null : Number(cornerText);
    const point = {
      corner_id: cornerInput.value,
      split: splitSelect.value,
      x: xInput.value,
      y: yInput.value,
      z: zInput.value
    };
    const next = state.teachDrafts[arm].filter((item) => cornerNumber === null || Number(item.corner_id) !== cornerNumber);
    next.push(point);
    action("保存示教点", async () => {
      await saveTeachPoints(arm, next);
      state.teachSelections[arm] = point.corner_id;
      showToast(`已保存角点 ${point.corner_id}（${unit}）。`, "success");
    });
  });

  cornerInput.addEventListener("input", () => {
    const point = points.find((item) => Number(item.corner_id) === Number(cornerInput.value));
    if (point) {
      splitSelect.value = point.split;
      xInput.value = point.x;
      yInput.value = point.y;
      zInput.value = point.z;
    }
  });

  const selectionForm = el("div", { class: "teach-form" },
    el("div", { class: "section-label" }, el("h4", { text: `选点与手动 XYZ（${unit}）` }), el("span", { class: "unit-chip", text: unit })),
    el("div", { class: "teach-row" },
      el("label", { class: "field" }, "角点 ID", cornerInput),
      el("label", { class: "field" }, "划分", splitSelect),
      el("label", { class: "field" }, `X (${unit})`, xInput),
      el("label", { class: "field" }, `Y (${unit})`, yInput),
      el("label", { class: "field" }, `Z (${unit})`, zInput),
      addPointButton
    ),
    el("p", { class: "help", text: "点击编号图上的圆点选择角点。编号关系：id = row×列数 + col；0→1 为板 x，0→11 为板 y（默认 11×8）。" })
  );

  const poseResult = el("div", { class: "getpose-result", text: "GetPose 不会自动保存；读取后还需点击“添加/更新该点”。" });
  const confirmBase = el("input", { type: "checkbox" });
  const confirmTcp = el("input", { type: "checkbox" });
  const confirmUnit = el("input", { type: "checkbox" });
  const poseButton = el("button", { type: "button", class: "button secondary", text: "只读 GetPose()", disabled: true });
  const hostValue = metadata.controller_host || "";
  const updatePoseButton = () => {
    poseButton.disabled = !hostValue || !confirmBase.checked || !confirmTcp.checked || !confirmUnit.checked || selectedId === null;
    poseButton.title = poseButton.disabled
      ? "需要控制器地址、已选角点，并确认示教器基座/TCP/单位"
      : "只发送 GetPose()，不运动、不使能、不切坐标系";
  };
  [confirmBase, confirmTcp, confirmUnit].forEach((checkbox) => checkbox.addEventListener("change", updatePoseButton));
  updatePoseButton();
  poseButton.addEventListener("click", () => action(`${armsCn[arm]}读取 GetPose`, async () => {
    const context = sessionContext();
    if (selectedId === null) throw new ApiError("请先选择角点", "CORNER_REQUIRED");
    const result = await sessionApi(context, `/arms/${arm}/get-pose`, {
      method: "POST",
      json: {
        confirm_base: confirmBase.checked,
        confirm_tcp: confirmTcp.checked,
        confirm_unit: confirmUnit.checked
      }
    });
    xInput.value = String(result.xyz[0]);
    yInput.value = String(result.xyz[1]);
    zInput.value = String(result.xyz[2]);
    setText(poseResult, `机器人 ${result.robot_id} · ${result.base_frame} · TCP ${result.tcp_name}\nXYZ (${result.unit}) = ${result.xyz.map((value) => fmt(value, 4)).join(", ")}\nRPY raw = ${result.rpy_raw.map((value) => fmt(value, 4)).join(", ")}\n未自动保存。`);
    showToast("已读取当前 TCP；请核对后显式保存示教点。", "success");
  }));

  const getPoseBox = el("div", { class: "getpose-box" },
    el("h4", { text: "可选：显式只读 GetPose()" }),
    el("p", { class: "microcopy", text: "仅在操作者确认示教器基座、尖端 TCP 与返回单位后启用。模块只发送 GetPose()；不会实例化运动 SDK、驱动、串口或夹爪代码。" }),
    el("label", { class: "check" }, confirmBase, "已核对示教器基座坐标"),
    el("label", { class: "check" }, confirmTcp, "已核对当前尖端 TCP"),
    el("label", { class: "check" }, confirmUnit, `已核对 GetPose 返回单位：${unit}`),
    el("div", { class: "inline-actions" }, poseButton),
    poseResult
  );

  const table = buildTeachTable(arm, points, unit);
  const saveAll = el("button", { type: "button", class: "button secondary", text: "保存表格全部编辑" });
  saveAll.addEventListener("click", () => {
    syncTeachInputs(arm);
    action("保存示教表", async () => {
      await saveTeachPoints(arm, state.teachDrafts[arm]);
      showToast(`${armsCn[arm]}示教表已保存。`, "success");
    });
  });
  const importInput = el("input", { type: "file", accept: ".csv,text/csv", "aria-label": "导入示教 CSV" });
  importInput.addEventListener("change", () => {
    const file = importInput.files[0];
    if (!file) return;
    if (state.dirtyTeach[arm] && !window.confirm("当前手臂有未保存的示教草稿。导入 CSV 会替换服务器示教点，是否继续？")) {
      importInput.value = "";
      return;
    }
    action("导入示教 CSV", async () => {
      const context = sessionContext();
      syncTeachInputs(arm);
      const revision = state.draftRevision.teach[arm];
      const text = await file.text();
      await sessionApi(context, `/arms/${arm}/teach/import`, { method: "POST", json: { csv: text } });
      if (!await refreshSession(false, { context, resetTeachArms: [{ arm, revision }] })) return;
      showToast("示教 CSV 已导入，旧下游结果已失效。", "success");
    });
    importInput.value = "";
  });
  const tableTools = el("div", { class: "inline-actions wrap" },
    saveAll,
    el("label", { class: "button ghost file-button" }, "导入 CSV", importInput),
    el("a", { class: "button ghost", href: apiUrl(`/arms/${arm}/teach/export`), text: "下载 CSV" })
  );

  const panel = el("article", { class: "panel teach-panel", "data-arm": arm },
    el("div", { class: "panel-heading" },
      el("div", {}, el("p", { class: "eyebrow", text: `${arm.toUpperCase()} ARM · REVISION ${armState.revision}` }), el("h3", { text: `${armsCn[arm]}独立示教` })),
      el("div", { class: "inline-actions" }, badge(`${countTrain} train`, countTrain >= 6 ? "success" : "warning"), badge(`${countVal} val`, countVal >= 3 ? "success" : "warning"), badge(unit, "warning"))
    ),
    el("div", { class: "teach-layout" },
      el("div", { class: "panel-soft-card" },
        el("div", { class: "panel-heading" }, el("div", {}, el("p", { class: "eyebrow", text: "CLICKABLE CORNERS" }), el("h3", { text: "编号图与已示教覆盖" }))),
        boardWrap,
        el("div", { class: "board-legend" },
          el("span", {}, el("i", { class: "legend-dot" }), "train"),
          el("span", {}, el("i", { class: "legend-dot val" }), "val"),
          el("span", { text: "圆点按钮可键盘聚焦；点击后选择角点。" })
        ),
        coverageSummary
      ),
      el("div", { class: "teach-form" },
        el("div", { class: "panel-heading" }, el("div", {}, el("p", { class: "eyebrow", text: "ARM METADATA" }), el("h3", { text: "机械臂与 TCP 元数据" }))),
        metadataForm,
        selectionForm,
        getPoseBox,
        el("div", { class: "section-label" }, el("h4", { text: "示教点表" }), el("span", { class: "microcopy", text: "内联编辑后点击保存" })),
        tableTools,
        table,
        el("p", { class: "help", text: "重复角点、NaN/Inf、共线或狭窄覆盖会被服务器拒绝。左右臂数据完全独立；CSV 中的 unit/robot_id/base_frame/tcp_name 必须与当前臂一致。" })
      )
    )
  );
  return panel;
}

function teachCoverage(points, columns, rows) {
  if (!points.length) return { x: 0, y: 0 };
  const cols = points.map((point) => Number(point.corner_id) % columns);
  const rowsIndex = points.map((point) => Math.floor(Number(point.corner_id) / columns));
  return {
    x: (Math.max(...cols) - Math.min(...cols)) / Math.max(1, columns - 1),
    y: (Math.max(...rowsIndex) - Math.min(...rowsIndex)) / Math.max(1, rows - 1)
  };
}

function buildTeachTable(arm, points, unit) {
  const wrap = el("div", { class: "table-wrap" });
  if (!points.length) {
    wrap.appendChild(el("div", { class: "empty-media", text: "尚无示教点。点击编号图或手动输入。" }));
    return wrap;
  }
  const table = el("table", { class: "data", "data-role": "points-table" },
    el("thead", {}, el("tr", {},
      el("th", { text: "ID" }), el("th", { text: "split" }),
      el("th", { text: `X (${unit})` }), el("th", { text: `Y (${unit})` }), el("th", { text: `Z (${unit})` }), el("th", { text: "" })
    ))
  );
  const body = el("tbody");
  points.forEach((point, index) => {
    const splitSelect = el("select", {}, option("train", "train", point.split === "train"), option("val", "val", point.split === "val"));
    const x = el("input", { type: "number", step: "any", value: point.x, "aria-label": `角点 ${point.corner_id} X` });
    const y = el("input", { type: "number", step: "any", value: point.y, "aria-label": `角点 ${point.corner_id} Y` });
    const z = el("input", { type: "number", step: "any", value: point.z, "aria-label": `角点 ${point.corner_id} Z` });
    splitSelect.addEventListener("change", () => { state.teachDrafts[arm][index].split = splitSelect.value; markDraftDirty("teach", arm); });
    x.addEventListener("input", () => { state.teachDrafts[arm][index].x = x.value; markDraftDirty("teach", arm); });
    y.addEventListener("input", () => { state.teachDrafts[arm][index].y = y.value; markDraftDirty("teach", arm); });
    z.addEventListener("input", () => { state.teachDrafts[arm][index].z = z.value; markDraftDirty("teach", arm); });
    const remove = el("button", { type: "button", class: "icon-button", text: "删除" });
    remove.addEventListener("click", () => {
      if (!window.confirm(`删除角点 ${point.corner_id}？`)) return;
      syncTeachInputs(arm);
      const next = state.teachDrafts[arm].filter((item) => Number(item.corner_id) !== Number(point.corner_id));
      action("删除示教点", async () => {
        await saveTeachPoints(arm, next);
        showToast(`角点 ${point.corner_id} 已删除。`, "success");
      });
    });
    const tr = el("tr", { dataset: { index: String(index) } },
      el("td", { class: "mono", text: String(point.corner_id) }), el("td", {}, splitSelect),
      el("td", {}, x), el("td", {}, y), el("td", {}, z), el("td", {}, remove)
    );
    if (Number(point.corner_id) === Number(state.teachSelections[arm])) tr.style.background = "#fff6e9";
    body.appendChild(tr);
  });
  table.appendChild(body);
  wrap.appendChild(table);
  return wrap;
}

function syncTeachInputs(arm) {
  const panel = document.querySelector(`.teach-panel[data-arm="${arm}"]`);
  if (!panel) return;
  const table = panel.querySelector('[data-role="points-table"]');
  if (!table) return;
  table.querySelectorAll("tbody tr").forEach((row) => {
    const index = Number(row.dataset.index);
    const draft = state.teachDrafts[arm][index];
    if (!draft) return;
    const inputs = row.querySelectorAll("input");
    if (inputs[0]) draft.x = inputs[0].value;
    if (inputs[1]) draft.y = inputs[1].value;
    if (inputs[2]) draft.z = inputs[2].value;
    const split = row.querySelector("select");
    if (split) draft.split = split.value;
  });
}

function normalizeTeachPoints(arm, points) {
  const output = [];
  const seen = new Set();
  const unit = state.session.arms[arm].metadata.unit;
  const total = state.session.settings.board.cols * state.session.settings.board.rows;
  for (const point of points) {
    const corner = parseRequiredInteger(point.corner_id, "角点 ID", "INVALID_CORNER_ID");
    const x = parseRequiredNumber(point.x, `角点 ${corner} 的 X（${unit}）`);
    const y = parseRequiredNumber(point.y, `角点 ${corner} 的 Y（${unit}）`);
    const z = parseRequiredNumber(point.z, `角点 ${corner} 的 Z（${unit}）`);
    if (corner < 0 || corner >= total) throw new ApiError(`角点 ID 必须在 0–${total - 1}`, "INVALID_CORNER_ID");
    if (seen.has(corner)) throw new ApiError(`角点 ${corner} 重复`, "DUPLICATE_CORNER_ID");
    if (!["train", "val"].includes(point.split)) throw new ApiError("划分只能是 train/val", "INVALID_SPLIT");
    seen.add(corner);
    output.push({ corner_id: corner, split: point.split, x, y, z });
  }
  return output.sort((a, b) => a.corner_id - b.corner_id);
}

async function saveTeachPoints(arm, points) {
  const context = sessionContext();
  assertCurrentSession(context);
  syncTeachInputs(arm);
  const normalized = normalizeTeachPoints(arm, points);
  const submittedDraft = normalized.map((point) => ({
    ...point, x: String(point.x), y: String(point.y), z: String(point.z)
  }));
  if (stableStringify(submittedDraft) !== stableStringify(state.teachDrafts[arm])) {
    state.teachDrafts[arm] = submittedDraft;
    markDraftDirty("teach", arm);
  }
  const revision = state.draftRevision.teach[arm];
  await sessionApi(context, `/arms/${arm}/points`, { method: "PUT", json: { points: normalized } });
  return refreshSession(false, { context, resetTeachArms: [{ arm, revision }] });
}

function renderFits() {
  const container = byId("fitPanels");
  if (!state.session) {
    replace(container, el("div", { class: "empty-media", text: "请先创建或恢复任务。" }));
    return;
  }
  replace(container, ["left", "right"].map((arm) => buildFitPanel(arm)));
}

function fitPrerequisites(arm) {
  const session = state.session;
  const armInfo = session.ui.gates.arms[arm];
  const intr = session.ui.gates.intrinsics;
  const board = session.ui.gates.board;
  const metadata = session.arms[arm].metadata;
  const active = activeJobs().some((job) => job.kind === "fit" && job.arm === arm);
  const reasons = [];
  if (!intr.ok) reasons.push(`内参：${intr.reason}`);
  if (!board.ok) reasons.push(`固定棋盘：${board.reason}`);
  if (armInfo.train_count < 6 || armInfo.val_count < 3) reasons.push(`示教点 ${armInfo.train_count}/${armInfo.val_count}，最低 6/3`);
  if (!metadata.robot_id || !metadata.base_frame || !metadata.tcp_name) reasons.push("机械臂元数据不完整");
  if (active) reasons.push("拟合任务正在运行");
  return { ok: reasons.length === 0, reasons, active };
}

function buildFitPanel(arm) {
  const armState = state.session.arms[arm];
  const fit = armState.fit;
  const job = latestJob("fit", arm);
  const prereq = fitPrerequisites(arm);
  const gateInfo = state.session.ui.gates.arms[arm];
  const active = job && ["queued", "running"].includes(job.status);
  const runButton = el("button", { type: "button", class: "button primary", text: `运行${armsCn[arm]}拟合`, disabled: !prereq.ok });
  runButton.title = prereq.ok ? "后台执行真实刚体/PnP 拟合" : prereq.reasons.join("；");
  runButton.addEventListener("click", () => action(`${armsCn[arm]}拟合`, async () => {
    const context = sessionContext();
    await sessionApi(context, `/arms/${arm}/fit`, { method: "POST", json: {} });
    if (!await refreshSession(false, { context })) return;
    showToast(`${armsCn[arm]}拟合任务已排队。`, "success");
  }));

  const statusText = active ? `${job.phase} ${Math.round((job.progress || 0) * 100)}%`
    : gateInfo.fit_ok ? "当前可用"
      : fit ? (fit.stale ? "已失效，仅证据" : fit.exit_code === 2 ? "超阈值，不可用" : "不可用")
        : "尚未拟合";
  const statusKind = active ? "cyan" : gateInfo.fit_ok ? "success" : fit ? "danger" : "neutral";

  const metrics = [];
  if (fit?.summary) {
    metrics.push(metric("PnP RMS", `${fmt(fit.summary.pnp_rms_px, 4)} px`, `上限 ${fmt(fit.summary.thresholds?.max_pnp_rms_px, 3)} px`));
    metrics.push(metric("最大点误差", `${fmt(fit.summary.max_error_mm, 3)} mm`, `上限 ${fmt(fit.summary.thresholds?.max_error_mm, 3)} mm`));
    metrics.push(metric("train 最大", `${fmt(fit.summary.train_max_error_mm, 3)} mm`, `${fit.summary.train_count} 个训练点`));
    metrics.push(metric("held-out val 最大", `${fmt(fit.summary.held_out_max_error_mm, 3)} mm`, `${fit.summary.val_count} 个保留点（非独立验收）`));
  }

  let evidence = null;
  if (fit) {
    const evidenceImages = [];
    if (fit.axes_path) evidenceImages.push(el("figure", { class: "evidence-image" }, el("img", { src: assetUrl(fit.axes_path), alt: `${armsCn[arm]}板坐标轴`, loading: "lazy" }), el("p", { text: "板坐标轴叠加（PnP 观测）" })));
    if (fit.chart_path) evidenceImages.push(el("figure", { class: "evidence-image" }, el("img", { src: assetUrl(fit.chart_path), alt: `${armsCn[arm]}逐点残差`, loading: "lazy" }), el("p", { text: "逐点残差：蓝色 train / 橙色 held-out val" })));
    const matrix = fit.summary?.T_base_camera ? matrixElement(fit.summary.T_base_camera, `T_base_camera (${armsCn[arm]})`) : null;
    evidence = el("div", {},
      el("div", { class: "status-line" },
        badge(`status ${fit.status}`, fit.exit_code === 0 && !fit.stale ? "success" : fit.exit_code === 2 ? "warning" : "danger"),
        badge(`exit ${fit.exit_code}`, fit.exit_code === 0 ? "success" : fit.exit_code === 2 ? "warning" : "danger"),
        badge(fit.stale ? "STALE / 不可用" : fit.validation_passed ? "validation_passed" : "validation_failed", fit.stale || !fit.validation_passed ? "danger" : "success"),
        badge(fit.synthetic_only ? "synthetic_only" : "real", fit.synthetic_only ? "warning" : "real")
      ),
      el("div", { class: "fit-metrics" }, metrics),
      gateInfo.fit_ok ? null : el("div", { class: "notice danger", text: `当前不可用：${gateInfo.fit_reason}` }),
      el("div", { class: "fit-evidence" }, evidenceImages),
      matrix,
      fitResidualTable(fit)
    );
  }

  const counts = `${gateInfo.train_count} train / ${gateInfo.val_count} val`;
  return el("article", { class: "panel fit-panel" },
    el("div", { class: "panel-heading" },
      el("div", {}, el("p", { class: "eyebrow", text: `${arm.toUpperCase()} ARM · ${armState.metadata.robot_id}` }), el("h3", { text: `${armsCn[arm]}平面拟合与外参` })),
      badge(statusText, statusKind)
    ),
    el("div", { class: "inline-actions wrap" }, runButton, badge(counts, gateInfo.train_count >= 6 && gateInfo.val_count >= 3 ? "success" : "warning"), badge(armState.metadata.unit, "warning")),
    el("p", { class: "help", text: "训练点求 B=T_base_board 与 PnP C=T_camera_board，再由 X=B·C⁻¹ 得到 camera→base。val 点只检查，不参与求解。" }),
    renderJobPanelHost(job),
    evidence || el("div", { class: "empty-media", text: prereq.ok ? "准备就绪，可运行拟合。" : `暂不可运行：${prereq.reasons.join("；")}` })
  );
}

function renderJobPanelHost(job) {
  const container = el("div", { class: "job-panel hidden" });
  queueMicrotask(() => renderJobPanel(container, job));
  return container;
}

function matrixElement(matrix, caption) {
  if (!Array.isArray(matrix) || matrix.length !== 4) return null;
  const table = el("table", { class: "matrix" });
  for (const row of matrix) {
    const tr = el("tr");
    for (const value of row) tr.appendChild(el("td", { text: fmt(value, 6) }));
    table.appendChild(tr);
  }
  return el("div", { class: "matrix-wrap" }, el("div", { class: "section-label" }, el("h4", { text: caption }), badge("4×4 · metres", "neutral")), table);
}

function fitResidualTable(fit) {
  if (!fit.points?.length) return null;
  const table = el("table", { class: "data" },
    el("thead", {}, el("tr", {},
      el("th", { text: "ID" }), el("th", { text: "split" }), el("th", { text: "最大 mm" }),
      el("th", { text: "刚体 mm" }), el("th", { text: "像素-平面 mm" }), el("th", { text: "预测 base mm" })
    ))
  );
  const body = el("tbody");
  for (const point of fit.points) {
    body.appendChild(el("tr", {},
      el("td", { class: "mono", text: String(point.corner_id) }),
      el("td", {}, badge(point.split, point.split === "val" ? "warning" : "cyan")),
      el("td", { class: "mono", text: fmt(point.max_error_mm, 3) }),
      el("td", { class: "mono", text: fmt(point.rigid_error_mm, 3) }),
      el("td", { class: "mono", text: fmt(point.pixel_plane_error_mm, 3) }),
      el("td", { class: "mono", text: point.predicted_base_mm.map((value) => fmt(value, 2)).join(", ") })
    ));
  }
  table.appendChild(body);
  return el("div", {},
    el("div", { class: "section-label" }, el("h4", { text: "逐点残差" }), badge("train / held-out val 分开", "neutral")),
    el("div", { class: "table-wrap" }, table)
  );
}

function renderQuery() {
  const wrap = byId("queryImageWrap");
  const session = state.session;
  if (!session) {
    replace(wrap, el("div", { class: "empty-media", text: "请先创建或恢复任务。" }));
    wrap.className = "query-image-wrap empty-media";
    setText(byId("queryPixelBadge"), "未选择");
    byId("queryPixelBadge").className = "badge neutral";
    replace(byId("queryResults"), el("div", { class: "empty-media", text: "选择像素后查询。" }));
    byId("depthModeRadio").disabled = true;
    return;
  }
  const source = session.board?.source_path;
  if (!source) {
    replace(wrap, el("div", { class: "empty-media", text: "固定棋盘确认后显示原图。" }));
    wrap.className = "query-image-wrap empty-media";
  } else {
    wrap.className = "query-image-wrap";
    const img = el("img", { src: assetUrl(source), alt: "原始固定棋盘图像，点击查询像素" });
    img.addEventListener("click", (event) => {
      const rect = img.getBoundingClientRect();
      const u = (event.clientX - rect.left) * (img.naturalWidth / rect.width);
      const v = (event.clientY - rect.top) * (img.naturalHeight / rect.height);
      if (!Number.isFinite(u) || !Number.isFinite(v) || u < 0 || v < 0 || u >= img.naturalWidth || v >= img.naturalHeight) return;
      invalidateQueryResult();
      state.queryPixel = { u, v };
      byId("queryU").value = String(Number(u.toFixed(2)));
      byId("queryV").value = String(Number(v.toFixed(2)));
      setText(byId("queryPixelBadge"), `u ${u.toFixed(1)} · v ${v.toFixed(1)}`);
      byId("queryPixelBadge").className = "badge cyan";
      renderQueryMarker(wrap, img, u, v, session.settings.image_size_wh);
      renderQueryResults();
    });
    wrap.replaceChildren(img);
    if (state.queryPixel) {
      byId("queryU").value = String(Number(state.queryPixel.u.toFixed(2)));
      byId("queryV").value = String(Number(state.queryPixel.v.toFixed(2)));
      setText(byId("queryPixelBadge"), `u ${state.queryPixel.u.toFixed(1)} · v ${state.queryPixel.v.toFixed(1)}`);
      byId("queryPixelBadge").className = "badge cyan";
      renderQueryMarker(wrap, img, state.queryPixel.u, state.queryPixel.v, session.settings.image_size_wh);
    } else {
      setText(byId("queryPixelBadge"), "未选择");
      byId("queryPixelBadge").className = "badge neutral";
    }
  }
  const depth = session.ui.gates.depth;
  setText(byId("depthBadge"), depth.ok ? "深度模式已注册" : "深度禁用");
  byId("depthBadge").className = `badge ${depth.ok ? "success" : "warning"}`;
  byId("depthModeRadio").disabled = !depth.ok;
  byId("depthModeRadio").title = depth.ok ? "仅在确认光学-Z 注册后可查询" : depth.reason;
  const selectedMode = document.querySelector('input[name="queryMode"]:checked')?.value || "board";
  if (selectedMode === "depth" && !depth.ok) {
    document.querySelector('input[name="queryMode"][value="board"]').checked = true;
    syncQueryModeInputs();
  }
  const registration = session.depth_registration || {};
  const note = byId("depthNote");
  if (document.activeElement !== note) note.value = registration.registration_note || "";
  byId("depthConfirm").checked = Boolean(registration.confirmed);
  renderQueryResults();
}

function renderQueryMarker(wrap, img, u, v, size) {
  wrap.querySelectorAll(".query-marker").forEach((marker) => marker.remove());
  const marker = el("div", {
    class: "query-marker",
    style: { left: `${u / size[0] * 100}%`, top: `${v / size[1] * 100}%` },
    title: `原始像素 ${u.toFixed(2)}, ${v.toFixed(2)}`
  });
  wrap.appendChild(marker);
}

function syncQueryModeInputs() {
  const mode = document.querySelector('input[name="queryMode"]:checked')?.value || "board";
  byId("queryBoardZ").disabled = mode !== "board";
  byId("queryDepthZ").disabled = mode !== "depth";
}

function queryInputIdentity() {
  return stableStringify({
    u: byId("queryU")?.value ?? "",
    v: byId("queryV")?.value ?? "",
    mode: document.querySelector('input[name="queryMode"]:checked')?.value || "board",
    board_z_m: byId("queryBoardZ")?.value ?? "",
    depth_z_m: byId("queryDepthZ")?.value ?? "",
    allow_extrapolation: Boolean(byId("allowExtrapolation")?.checked)
  });
}

async function runPixelQuery(context, body) {
  assertCurrentSession(context);
  invalidateQueryResult();
  const sequence = state.querySequence;
  const dependencyFingerprint = state.sessionDependencyFingerprint
    || sessionDependencyFingerprint(state.session);
  const submittedInputIdentity = queryInputIdentity();
  const result = await sessionApi(context, "/query", { method: "POST", json: body });
  if (!isCurrentSession(context) || sequence !== state.querySequence
      || dependencyFingerprint !== sessionDependencyFingerprint(state.session)
      || submittedInputIdentity !== queryInputIdentity()) return false;
  state.queryResult = result;
  renderQuery();
  return true;
}

function renderQueryResults() {
  const container = byId("queryResults");
  if (!state.queryResult) {
    container.className = "query-results empty-media";
    setText(container, "选择像素后查询。");
    return;
  }
  container.className = "query-results";
  const result = state.queryResult;
  const items = [el("div", { class: "microcopy", text: `原始图像 ${result.image_size_wh[0]}×${result.image_size_wh[1]} · u=${fmt(result.u_px, 2)} · v=${fmt(result.v_px, 2)} · ${result.depth_z_m !== null ? `depth=${fmt(result.depth_z_m, 4)} m` : `board_z=${fmt(result.board_z_m, 4)} m`}` })];
  for (const arm of ["left", "right"]) {
    const value = result.arms[arm];
    if (!value.ok) {
      items.push(el("div", { class: "arm-result" },
        el("h4", {}, armsCn[arm], badge(value.code || "不可用", "danger")),
        el("div", { class: "microcopy", text: value.error || "该臂拟合不可用" })
      ));
      continue;
    }
    items.push(el("div", { class: "arm-result" },
      el("h4", {}, armsCn[arm], el("span", {}, badge(result.synthetic_only ? "synthetic_only" : "real", result.synthetic_only ? "warning" : "real"), badge(value.mode, "cyan"))),
      el("div", { class: "coord-mono", text: `base(m):  ${value.point_base_m.map((item) => fmt(item, 6)).join(", ")}\nbase(mm): ${value.point_base_mm.map((item) => fmt(item, 3)).join(", ")}\nboard(m): ${value.point_board_m.map((item) => fmt(item, 6)).join(", ")}` }),
      el("div", { class: "microcopy", text: `${value.robot_id} · ${value.base_frame} · TCP ${value.tcp_name} · 棋盘内矩形：${value.inside_board_inner_rectangle ? "是" : "否"}` }),
      value.extrapolated ? el("div", { class: "notice warning", text: "本次结果来自显式外推；必须另行独立验收。" }) : null
    ));
  }
  replace(container, items);
}

function renderVerification() {
  const container = byId("verifyPanels");
  if (!state.session) {
    replace(container, el("div", { class: "empty-media", text: "请先创建或恢复任务。" }));
    renderExportPanel();
    return;
  }
  replace(container, ["left", "right"].map((arm) => buildVerifyPanel(arm)));
  renderExportPanel();
}

function buildVerifyPanel(arm) {
  const session = state.session;
  const armState = session.arms[arm];
  const draft = state.verifyDrafts[arm];
  const fitOk = session.ui.gates.arms[arm].fit_ok;
  const result = armState.verification.result;
  const job = latestJob("verify", arm);
  const active = job && ["queued", "running"].includes(job.status);
  const methodInput = el("input", { type: "text", maxlength: 500, value: draft.method || "", placeholder: "例如：激光跟踪仪/独立标定针尖/外部测量臂" });
  const independence = el("input", { type: "checkbox", checked: Boolean(draft.independence_confirmed) });
  const runButton = el("button", { type: "button", class: "button primary", text: `运行${armsCn[arm]}独立验收` });
  const hint = el("span", { class: "microcopy" });
  const table = buildVerifyTable(arm);

  const updateReady = (markDirty = false) => {
    draft.method = methodInput.value;
    draft.independence_confirmed = independence.checked;
    if (markDirty) markDraftDirty("verify", arm);
    const ready = fitOk && draft.rows.length >= 3 && draft.method.trim() && draft.independence_confirmed && !active;
    runButton.disabled = !ready;
    runButton.title = !fitOk ? `拟合不可用：${session.ui.gates.arms[arm].fit_reason}`
      : draft.rows.length < 3 ? "至少需要 3 个独立测量点"
        : !draft.method.trim() ? "请填写独立测量方法"
          : !draft.independence_confirmed ? "必须确认参考点独立于拟合数据"
            : active ? "验收任务运行中" : "保存草稿并启动后台验收";
    setText(hint, runButton.disabled ? runButton.title : "参考值不会从拟合变换自动回填。");
  };
  methodInput.addEventListener("input", () => updateReady(true));
  independence.addEventListener("change", () => updateReady(true));

  const addRow = el("button", { type: "button", class: "button secondary", text: "添加一行" });
  addRow.addEventListener("click", () => {
    syncVerifyInputs(arm);
    const u = state.queryPixel?.u ?? "";
    const v = state.queryPixel?.v ?? "";
    const z = Number.isFinite(Number(byId("queryBoardZ").value)) ? byId("queryBoardZ").value : "0";
    draft.rows.push({ u_px: u === "" ? "" : String(Number(u.toFixed(2))), v_px: v === "" ? "" : String(Number(v.toFixed(2))), board_z_m: z, reference_x_m: "", reference_y_m: "", reference_z_m: "" });
    markDraftDirty("verify", arm);
    renderVerification();
  });
  const saveDraft = el("button", { type: "button", class: "button secondary", text: "保存验收草稿" });
  saveDraft.addEventListener("click", () => action("保存独立验收草稿", async () => {
    if (!await saveVerificationDraft(arm)) return;
    showToast(`${armsCn[arm]}独立验收草稿已保存。`, "success");
  }));
  runButton.addEventListener("click", () => action(`${armsCn[arm]}独立验收`, async () => {
    if (!await saveVerificationDraft(arm, { runJob: true })) return;
    showToast(`${armsCn[arm]}独立验收任务已排队。`, "success");
  }));
  const importInput = el("input", { type: "file", accept: ".csv,text/csv", "aria-label": "导入独立验收 CSV" });
  importInput.addEventListener("change", () => {
    const file = importInput.files[0];
    if (!file) return;
    if (state.dirtyVerify[arm] && !window.confirm("当前手臂有未保存的独立测量草稿。导入 CSV 会替换服务器测量点，是否继续？")) {
      importInput.value = "";
      return;
    }
    action("导入独立验收 CSV", async () => {
      const context = sessionContext();
      syncVerifyInputs(arm);
      const revision = state.draftRevision.verify[arm];
      const csv = await file.text();
      await sessionApi(context, `/arms/${arm}/verification/import`, { method: "POST", json: { csv } });
      if (!await refreshSession(false, { context, resetVerifyArms: [{ arm, revision }] })) return;
      showToast("独立验收 CSV 已导入。", "success");
    });
    importInput.value = "";
  });

  const tools = el("div", { class: "inline-actions wrap" },
    addRow, saveDraft, runButton,
    el("label", { class: "button ghost file-button" }, "导入 CSV", importInput),
    el("a", { class: "button ghost", href: apiUrl(`/arms/${arm}/verification/export`), text: "下载 CSV" }),
    hint
  );

  const statusText = active ? `${job.phase} ${Math.round((job.progress || 0) * 100)}%`
    : session.ui.gates.arms[arm].verification_ok ? "独立验收通过"
      : result ? (result.stale ? "已失效" : result.exit_code === 2 ? "超阈值" : "不可用") : "未验收";
  const statusKind = active ? "cyan" : session.ui.gates.arms[arm].verification_ok ? "success" : result ? "danger" : "neutral";
  const resultNode = result ? buildVerifyResult(arm, result) : el("div", { class: "empty-media", text: "尚未运行独立验收。参考 XYZ 必须由操作者独立测量。" });

  queueMicrotask(updateReady);
  return el("article", { class: "panel verify-panel", "data-arm": arm },
    el("div", { class: "panel-heading" },
      el("div", {}, el("p", { class: "eyebrow", text: `${arm.toUpperCase()} ARM · INDEPENDENT HOLDOUT` }), el("h3", { text: `${armsCn[arm]}独立验收` })),
      badge(statusText, statusKind)
    ),
    el("div", { class: "notice info" }, el("strong", { text: "独立测量参考值（米）" }), ` · 最低 3 点 · 不与拟合 held-out 混用`),
    fitOk ? null : el("div", { class: "notice danger", text: `拟合不可用：${session.ui.gates.arms[arm].fit_reason}` }),
    el("div", { class: "form-grid compact" },
      el("label", { class: "field span-2" }, "测量方法与独立性说明", methodInput),
      el("label", { class: "check span-2" }, independence, "我确认这些参考 XYZ 由独立方法测量，未从本次相机变换或拟合残差回填。")
    ),
    tools,
    table,
    renderJobPanelHost(job),
    resultNode
  );
}

function buildVerifyTable(arm) {
  const draft = state.verifyDrafts[arm];
  const wrap = el("div", { class: "table-wrap" });
  if (!draft.rows.length) {
    wrap.appendChild(el("div", { class: "empty-media", text: "尚无验收点。点击“添加一行”，或在步骤 6 选定像素后添加。" }));
    return wrap;
  }
  const table = el("table", { class: "data", "data-role": "verify-table" },
    el("thead", {}, el("tr", {},
      el("th", { text: "u px" }), el("th", { text: "v px" }), el("th", { text: "board z m" }),
      el("th", { text: "ref X m" }), el("th", { text: "ref Y m" }), el("th", { text: "ref Z m" }), el("th", { text: "" })
    ))
  );
  const body = el("tbody");
  draft.rows.forEach((row, index) => {
    const cells = ["u_px", "v_px", "board_z_m", "reference_x_m", "reference_y_m", "reference_z_m"].map((key) => {
      const input = el("input", { type: "number", step: "any", value: row[key] ?? "", "aria-label": `${key} 第 ${index + 1} 行` });
      input.addEventListener("input", () => { draft.rows[index][key] = input.value; markDraftDirty("verify", arm); });
      return el("td", {}, input);
    });
    const remove = el("button", { type: "button", class: "icon-button", text: "删除" });
    remove.addEventListener("click", () => {
      syncVerifyInputs(arm);
      draft.rows.splice(index, 1);
      markDraftDirty("verify", arm);
      renderVerification();
    });
    body.appendChild(el("tr", {}, cells, el("td", {}, remove)));
  });
  table.appendChild(body);
  wrap.appendChild(table);
  return wrap;
}

function syncVerifyInputs(arm) {
  const panel = document.querySelector(`.verify-panel[data-arm="${arm}"]`);
  const table = panel?.querySelector('[data-role="verify-table"]');
  if (!table) return;
  const rows = state.verifyDrafts[arm].rows;
  table.querySelectorAll("tbody tr").forEach((tr, index) => {
    if (!rows[index]) return;
    tr.querySelectorAll("input").forEach((input, column) => {
      const key = ["u_px", "v_px", "board_z_m", "reference_x_m", "reference_y_m", "reference_z_m"][column];
      if (key) rows[index][key] = input.value;
    });
  });
}

function normalizeVerifyDraft(arm) {
  syncVerifyInputs(arm);
  const draft = state.verifyDrafts[arm];
  const rows = draft.rows.map((row, index) => {
    const numeric = {};
    for (const key of ["u_px", "v_px", "board_z_m", "reference_x_m", "reference_y_m", "reference_z_m"]) {
      numeric[key] = parseRequiredNumber(row[key], `第 ${index + 1} 行 ${key}`, "INVALID_VERIFY_ROW");
    }
    return numeric;
  });
  const method = String(draft.method || "").trim();
  return { rows, method, independence_confirmed: Boolean(draft.independence_confirmed) };
}

async function saveVerificationDraft(arm, { runJob = false } = {}) {
  const context = sessionContext();
  const payload = normalizeVerifyDraft(arm);
  const revision = state.draftRevision.verify[arm];
  if (runJob) {
    if (payload.rows.length < 3) throw new ApiError("至少需要 3 个独立测量点", "VERIFY_POINTS_REQUIRED");
    if (!payload.method || !payload.independence_confirmed) {
      throw new ApiError("请填写方法并确认独立性", "VERIFY_CONFIRMATION_REQUIRED");
    }
  }
  await sessionApi(context, `/arms/${arm}/verification`, { method: "PUT", json: payload });
  if (runJob) await sessionApi(context, `/arms/${arm}/verify`, { method: "POST", json: {} });
  return refreshSession(false, { context, resetVerifyArms: [{ arm, revision }] });
}

function buildVerifyResult(arm, result) {
  const metrics = result.summary ? [
    metric("RMS", `${fmt(result.summary.rms_mm, 3)} mm`, "独立参考点"),
    metric("最大误差", `${fmt(result.summary.max_mm, 3)} mm`, `上限 ${fmt(result.summary.threshold_mm, 3)} mm`),
    metric("点数", String(result.summary.point_count || result.points?.length || 0), "最低 3"),
    metric("状态", result.validation_passed ? "通过" : "未通过", result.stale ? "已失效，仅证据" : "当前哈希")
  ] : [];
  const body = el("tbody");
  for (const point of result.points || []) {
    body.appendChild(el("tr", {},
      el("td", { class: "mono", text: `${fmt(point.u_px, 1)}, ${fmt(point.v_px, 1)}` }),
      el("td", { class: "mono", text: fmt(point.board_z_m, 4) }),
      el("td", { class: "mono", text: [point.reference_x_m, point.reference_y_m, point.reference_z_m].map((value) => fmt(value, 4)).join(", ") }),
      el("td", { class: "mono", text: (point.predicted_base_m || []).map((value) => fmt(value, 4)).join(", ") }),
      el("td", { class: "mono", text: fmt(point.error_mm, 3) }),
      el("td", {}, badge(point.inside_board_inner_rectangle ? "inside" : "outside", point.inside_board_inner_rectangle ? "neutral" : "warning"))
    ));
  }
  const table = el("table", { class: "data" },
    el("thead", {}, el("tr", {},
      el("th", { text: "像素" }), el("th", { text: "board z m" }), el("th", { text: "参考 base m" }),
      el("th", { text: "预测 base m" }), el("th", { text: "误差 mm" }), el("th", { text: "范围" })
    )),
    body
  );
  return el("div", {},
    el("div", { class: "status-line" },
      badge(`status ${result.status}`, result.exit_code === 0 && !result.stale ? "success" : result.exit_code === 2 ? "warning" : "danger"),
      badge(`exit ${result.exit_code}`, result.exit_code === 0 ? "success" : result.exit_code === 2 ? "warning" : "danger"),
      badge(result.validation_passed ? "independent_pass" : "independent_fail", result.validation_passed ? "success" : "danger"),
      badge(result.synthetic_only ? "synthetic_only" : "real", result.synthetic_only ? "warning" : "real")
    ),
    el("div", { class: "fit-metrics" }, metrics),
    result.stale ? el("div", { class: "notice danger", text: `该结果已失效：${result.stale_reason || "输入改变"}` }) : null,
    el("div", { class: "section-label" }, el("h4", { text: "独立验收点" }), badge("reference ≠ fit held-out", "neutral")),
    el("div", { class: "table-wrap" }, table),
    result.chart_path ? el("div", { class: "evidence-image", style: { marginTop: "10px" } }, el("img", { src: assetUrl(result.chart_path), alt: "独立验收残差图", loading: "lazy" }), el("p", { text: "独立验收逐点误差（非拟合 held-out）" })) : null
  );
}

function renderExportPanel() {
  const button = byId("createExport");
  const badgeNode = byId("exportStateBadge");
  if (!state.session) {
    button.disabled = true;
    setText(badgeNode, "未生成");
    badgeNode.className = "badge neutral";
    replace(byId("exportResult"), el("div", { class: "empty-media", text: "尚未生成报告包。" }));
    return;
  }
  const candidate = state.exportRecord || state.session.exports?.at(-1) || null;
  const latest = candidate && exportMatchesSession(candidate, state.session) ? candidate : null;
  setText(badgeNode, latest ? "已生成" : "未生成");
  badgeNode.className = `badge ${latest ? "success" : "neutral"}`;
  updateExportButtonState();
  if (!latest) {
    replace(byId("exportResult"), el("div", { class: "empty-media", text: "尚未生成报告包。左右臂独立验收通过后可分别或一起导出。" }));
    return;
  }
  const link = assetUrl(latest.bundle_path);
  replace(byId("exportResult"),
    el("div", { class: "result-box good" },
      el("strong", { text: `报告包：${latest.filename}` }),
      el("div", { class: "microcopy", text: `arms: ${(latest.arms || []).join(", ")} · size ${(Number(latest.size_bytes || 0) / 1024).toFixed(1)} KiB · ${fmtTime(latest.created_utc)}` }),
      el("div", { class: "coord-mono", text: `sha256: ${latest.sha256}` }),
      latest.synthetic_only ? el("div", { class: "notice warning", text: "synthetic_only：只能用于离线演示，不能作为真机验收。" }) : null,
      el("a", { class: "button primary", href: link, download: latest.filename, text: "下载 Zip 报告包" })
    )
  );
}

function updateExportButtonState() {
  const button = byId("createExport");
  if (!button) return;
  if (!state.session) { button.disabled = true; return; }
  const selected = [];
  if (byId("exportLeft").checked) selected.push("left");
  if (byId("exportRight").checked) selected.push("right");
  const blocked = selected.filter((arm) => !state.session.ui.gates.arms[arm].verification_ok);
  button.disabled = selected.length === 0 || blocked.length > 0;
  button.title = selected.length === 0 ? "至少选择一只手臂"
    : blocked.length ? `${blocked.map((arm) => armsCn[arm]).join("、")}独立验收未通过或已失效`
      : "生成带哈希与范围说明的报告包";
}

function renderRail() {
  const gateList = byId("gateList");
  const metricCards = byId("metricCards");
  const artifactList = byId("artifactList");
  const session = state.session;
  if (!session) {
    replace(gateList, el("li", { class: "gate-item bad" }, el("i", { class: "gate-icon" }), el("div", {}, el("strong", { text: "任务" }), el("span", { text: "尚未创建或恢复" }))));
    replace(metricCards, el("div", { class: "empty-media", text: "尚无指标。" }));
    replace(artifactList, el("div", { class: "empty-media", text: "尚无产物。" }));
    renderEvents();
    return;
  }
  const gates = [];
  const addGate = (label, ok, reason, warning = false) => gates.push(el("li", { class: `gate-item ${ok ? "ok" : warning ? "warn" : "bad"}` },
    el("i", { class: "gate-icon" }),
    el("div", {}, el("strong", { text: label }), el("span", { text: reason }))
  ));
  addGate("任务会话", true, `${session.display_name} · revision ${session.revision}`);
  addGate("相机内参", session.ui.gates.intrinsics.ok, session.ui.gates.intrinsics.ok ? "哈希匹配且达到阈值" : session.ui.gates.intrinsics.reason);
  addGate("固定棋盘", session.ui.gates.board.ok, session.ui.gates.board.ok ? `编号 ${session.board.confirmation?.corner_order} · 已冻结` : session.ui.gates.board.reason);
  for (const arm of ["left", "right"]) {
    const info = session.ui.gates.arms[arm];
    addGate(`${armsCn[arm]}拟合`, info.fit_ok, info.fit_ok ? "当前可用" : info.fit_reason);
    addGate(`${armsCn[arm]}独立验收`, info.verification_ok, info.verification_ok ? "当前通过" : info.verification_reason);
  }
  addGate("光学-Z 深度", session.ui.gates.depth.ok, session.ui.gates.depth.ok ? "已显式注册" : session.ui.gates.depth.reason, true);
  replace(gateList, gates);

  const metrics = [];
  const intr = session.intrinsics.result;
  if (intr?.summary) {
    metrics.push(metric("内参训练 RMS", `${fmt(intr.summary.train_rms_px, 4)} px`, `最大单图 ${fmt(intr.summary.max_view_rms_px, 3)} px`));
  }
  if (session.board.confirmation) {
    metrics.push(metric("棋盘角点", `${session.board.raw_corners_uv?.length || 0}`, `${session.board.confirmation.corner_order} · rev ${session.board.confirmation.immutable_revision}`));
  }
  for (const arm of ["left", "right"]) {
    const fit = session.arms[arm].fit;
    if (fit?.summary) metrics.push(metric(`${armsCn[arm]} PnP / max`, `${fmt(fit.summary.pnp_rms_px, 3)} px`, `${fmt(fit.summary.max_error_mm, 3)} mm · ${fit.stale ? "已失效" : fit.exit_code === 0 ? "可用" : "不可用"}`));
    const verify = session.arms[arm].verification.result;
    if (verify?.summary) metrics.push(metric(`${armsCn[arm]} 独立验收`, `${fmt(verify.summary.max_mm, 3)} mm`, `RMS ${fmt(verify.summary.rms_mm, 3)} mm · ${verify.stale ? "已失效" : verify.validation_passed ? "通过" : "未通过"}`));
  }
  replace(metricCards, metrics.length ? metrics : el("div", { class: "empty-media", text: "尚无计算结果。" }));

  const artifacts = [];
  const addArtifact = (label, relative) => {
    if (!relative) return;
    artifacts.push(el("a", { class: "artifact-link", href: assetUrl(relative), target: "_blank", rel: "noopener", text: label }));
  };
  if (intr?.artifact_path) addArtifact("内参 intrinsics.json", intr.artifact_path);
  if (session.board.source_path) addArtifact("固定棋盘原图", session.board.source_path);
  const boardOrder = session.board.confirmation?.corner_order;
  if (boardOrder && session.board.previews?.[boardOrder]) addArtifact("固定棋盘编号图", session.board.previews[boardOrder]);
  for (const arm of ["left", "right"]) {
    const fit = session.arms[arm].fit;
    if (fit?.artifact_path) addArtifact(`${armsCn[arm]} extrinsics.json`, fit.artifact_path);
    if (fit?.axes_path) addArtifact(`${armsCn[arm]} 坐标轴图`, fit.axes_path);
    if (fit?.chart_path) addArtifact(`${armsCn[arm]} 拟合残差图`, fit.chart_path);
    const verify = session.arms[arm].verification.result;
    if (verify?.artifact_path) addArtifact(`${armsCn[arm]} validation.json`, verify.artifact_path);
  }
  for (const exportItem of (session.exports || []).slice(-2)) addArtifact(`报告包 ${exportItem.filename}`, exportItem.bundle_path);
  replace(artifactList, artifacts.length ? artifacts : el("div", { class: "empty-media", text: "尚无产物。" }));
  renderEvents();
}

function renderEvents() {
  const container = byId("eventLog");
  const events = state.session?.events || [];
  setText(byId("eventCount"), String(events.length));
  if (!events.length) {
    replace(container, el("div", { class: "empty-media", text: "尚无事件。" }));
    return;
  }
  const items = events.slice(-30).reverse().map((event) => el("div", { class: `event-item ${event.level || ""}` },
    el("span", { class: "event-time", text: fmtTime(event.time) }),
    document.createTextNode(event.message || "")
  ));
  replace(container, items);
}

function jobSignature() {
  return (state.session?.jobs || []).map((job) => `${job.id}:${job.status}`).join("|");
}

async function refreshJobLogs(context = sessionContext(), requestGeneration = state.sessionRequestGeneration) {
  if (!isCurrentSession(context) || !state.session
      || requestGeneration !== state.sessionRequestGeneration) return;
  const jobs = state.session.jobs || [];
  const relevant = jobs.filter((job, index) => {
    if (["queued", "running"].includes(job.status)) return true;
    return index >= jobs.length - 4;
  });
  for (const job of relevant) {
    const cache = state.logCache[job.id] || { offset: 0, text: "" };
    try {
      const data = await api(`/api/sessions/${encodeURIComponent(context.sessionId)}/jobs/${encodeURIComponent(job.id)}/log?offset=${cache.offset}`);
      if (!isCurrentSession(context) || requestGeneration !== state.sessionRequestGeneration) return;
      if (data.log) cache.text += data.log;
      cache.offset = data.next_offset || data.log?.length || cache.offset;
      state.logCache[job.id] = cache;
    } catch (_) {
      // Keep the last readable log; the job panel still exposes status and errors.
    }
  }
}

function schedulePoll() {
  if (state.pollTimer) window.clearTimeout(state.pollTimer);
  state.pollTimer = null;
  if (!state.sessionId) return;
  if (!activeJobs().length) return;
  state.pollTimer = window.setTimeout(pollSession, 1300);
}

async function pollSession() {
  state.pollTimer = null;
  const context = sessionContext();
  if (!context || !state.session) return;
  const requestGeneration = ++state.sessionRequestGeneration;
  const before = jobSignature();
  const beforeActive = activeJobs().length;
  try {
    const data = await api(`/api/sessions/${encodeURIComponent(context.sessionId)}`);
    if (!isCurrentSession(context) || requestGeneration !== state.sessionRequestGeneration) return;
    acceptSessionSnapshot(data.session);
    await refreshJobLogs(context, requestGeneration);
    if (!isCurrentSession(context) || requestGeneration !== state.sessionRequestGeneration) return;
    applyPendingDraftResets(context);
    const after = jobSignature();
    const afterActive = activeJobs().length;
    const terminalTransition = before !== after && beforeActive > 0 && afterActive < beforeActive;
    if (terminalTransition && state.session.jobs.some((job) => job.kind === "demo")) {
      // Synthetic demo jobs populate teach/verification data entirely in the
      // background; refresh clean drafts when that data first becomes
      // available, while preserving any operator edits already in progress.
      resetDrafts({ preserveDirty: true });
    }
    const editing = document.activeElement && ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName);
    if (editing && !terminalTransition) {
      renderGlobal();
      renderStepNav();
      renderRail();
    } else {
      renderAll();
    }
  } catch (error) {
    if (isCurrentSession(context)) showError(error);
  } finally {
    if (isCurrentSession(context) && requestGeneration === state.sessionRequestGeneration) schedulePoll();
  }
}

function updateCameraControls() {
  const camera = state.bootstrap?.camera || {};
  const open = Boolean(camera.open);
  const hasSession = Boolean(state.session);
  byId("cameraStart").disabled = !hasSession || open;
  byId("cameraCaptureTrain").disabled = !hasSession || !open;
  byId("cameraCaptureVal").disabled = !hasSession || !open;
  byId("cameraStop").disabled = !open;
  byId("boardCameraCapture").disabled = !hasSession || !open;
  byId("cameraPreview").classList.toggle("hidden", !open);
  byId("cameraPlaceholder").classList.toggle("hidden", open);
  if (open) {
    byId("cameraPreview").src = `/api/camera/frame?t=${Date.now()}`;
    startCameraLoop();
  } else {
    stopCameraLoop();
  }
}

function startCameraLoop() {
  if (state.cameraTimer) return;
  state.cameraErrorShown = false;
  state.cameraTimer = window.setInterval(() => {
    if (!state.bootstrap?.camera?.open) return;
    byId("cameraPreview").src = `/api/camera/frame?t=${Date.now()}`;
  }, 450);
}

function stopCameraLoop() {
  if (state.cameraTimer) window.clearInterval(state.cameraTimer);
  state.cameraTimer = null;
}

async function stopCamera(reason = "相机已停止并释放。") {
  try {
    await api("/api/camera/stop", { method: "POST", json: {} });
    state.bootstrap.camera = { open: false };
    updateCameraControls();
    renderGlobal();
    showToast(reason, "success");
  } catch (error) {
    if (!(error instanceof ApiError && error.code === "CAMERA_NOT_RUNNING")) showError(error);
  }
}

async function refreshSessionListFromServer(context = sessionContext()) {
  const requestGeneration = ++state.sessionListRequestGeneration;
  const data = await api("/api/sessions");
  if (requestGeneration !== state.sessionListRequestGeneration
      || (context && !isCurrentSession(context))) return false;
  state.bootstrap.sessions = data.sessions || [];
  renderSessionList();
  return true;
}

function showStep(step) {
  step = Math.max(1, Math.min(7, Number(step) || 1));
  syncTeachInputs("left");
  syncTeachInputs("right");
  syncVerifyInputs("left");
  syncVerifyInputs("right");
  state.step = step;
  localStorage.setItem("calibration-step", String(step));
  renderStageVisibility();
  if (step === 2 && state.bootstrap?.camera?.open) startCameraLoop();
  else stopCameraLoop();
  byId("workspace").focus({ preventScroll: true });
}

async function boot() {
  try {
    setBusy("加载本机标定服务…");
    const bootstrap = await api("/api/bootstrap");
    state.bootstrap = bootstrap;
    state.csrf = bootstrap.csrf_token;
    renderDependencies();
    renderSessionList();
    renderGlobal();
    const params = new URLSearchParams(window.location.search);
    const sessionId = params.get("session") || localStorage.getItem("calibration-session");
    if (sessionId) {
      await loadSession(sessionId, { reset: true, preserveStep: true });
    } else {
      state.sessionGeneration += 1;
      state.sessionId = null;
      state.session = null;
      state.step = 1;
      resetDrafts();
      renderAll();
    }
  } catch (error) {
    showError(error);
    byId("serverBadge").textContent = "服务不可用";
    byId("serverBadge").className = "badge danger";
  } finally {
    setBusy(null);
  }
}

function wireEvents() {
  document.querySelectorAll(".step-button").forEach((button) => {
    button.addEventListener("click", () => showStep(button.dataset.step));
  });
  byId("prevStep").addEventListener("click", () => showStep(state.step - 1));
  byId("nextStep").addEventListener("click", () => showStep(state.step + 1));
  byId("sessionSelect").addEventListener("change", (event) => {
    if (!event.target.value) {
      if (state.session && hasDirtyDrafts()
          && !window.confirm("当前任务有未保存的示教或独立测量草稿。取消选择会丢弃这些草稿，是否继续？")) {
        renderSessionList();
        return;
      }
      state.sessionGeneration += 1;
      state.sessionRequestGeneration += 1;
      state.sessionId = null;
      state.session = null;
      state.sessionDependencyFingerprint = null;
      state.pendingDraftResets = { teach: {}, verify: {} };
      invalidateQueryResult({ clearPixel: true });
      state.exportRecord = null;
      resetDrafts({ preserveDirty: false });
      localStorage.removeItem("calibration-session");
      const url = new URL(window.location.href);
      url.searchParams.delete("session");
      history.replaceState(null, "", url);
      renderAll();
      return;
    }
    action("恢复任务", async () => {
      const loaded = await loadSession(event.target.value, { reset: true, preserveStep: true });
      if (!loaded) return;
      showToast("已恢复任务与持久化进度。", "success");
    });
  });

  const createForm = byId("createSessionForm");
  createForm.elements.mode.addEventListener("change", syncCreateMode);
  syncCreateMode();
  createForm.addEventListener("submit", (event) => {
    event.preventDefault();
    action("创建标定任务", async () => {
      const dirtyAtStart = hasDirtyDrafts();
      if (dirtyAtStart && !window.confirm("当前任务有未保存的示教或独立测量草稿。创建新任务后切换会丢弃这些草稿，是否继续？")) return;
      const priorSessionId = state.sessionId;
      const priorGeneration = state.sessionGeneration;
      const confirmedDraftRevision = stableStringify(state.draftRevision);
      const form = new FormData(createForm);
      const mode = form.get("mode");
      const body = {
        display_name: form.get("display_name"),
        mode,
        camera_id: form.get("camera_id"),
        width: Number(form.get("width")),
        height: Number(form.get("height")),
        board: { cols: Number(form.get("cols")), rows: Number(form.get("rows")), square_m: Number(form.get("square_m")) },
        thresholds: {
          intrinsics_max_rms_px: Number(form.get("intrinsics_max_rms_px")),
          fit_max_pnp_rms_px: Number(form.get("fit_max_pnp_rms_px")),
          fit_max_error_mm: Number(form.get("fit_max_error_mm")),
          verify_max_error_mm: Number(form.get("verify_max_error_mm"))
        }
      };
      const data = await api("/api/sessions", { method: "POST", json: body });
      if (state.sessionId !== priorSessionId || state.sessionGeneration !== priorGeneration) return;
      const context = sessionContext();
      await refreshSessionListFromServer(context);
      if (state.sessionId !== priorSessionId || state.sessionGeneration !== priorGeneration) return;
      await loadSession(data.session.session_id, {
        reset: true, preserveStep: false,
        allowDiscardDirty: dirtyAtStart && confirmedDraftRevision === stableStringify(state.draftRevision)
      });
      showToast(mode === "synthetic" ? "模拟任务已创建，后台正在生成 synthetic_only 数据。" : "真实任务已创建。", "success");
      if (mode === "synthetic") {
        showStep(2);
        schedulePoll();
      }
    });
  });

  byId("settingsForm").addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.target;
    action("保存任务设置", async () => {
      const context = sessionContext();
      await sessionApi(context, "/settings", {
        method: "PATCH",
        json: {
          camera_id: form.elements.camera_id.value,
          width: Number(form.elements.width.value),
          height: Number(form.elements.height.value),
          board: { cols: Number(form.elements.cols.value), rows: Number(form.elements.rows.value), square_m: Number(form.elements.square_m.value) },
          thresholds: {
            intrinsics_max_rms_px: Number(form.elements.intrinsics_max_rms_px.value),
            fit_max_pnp_rms_px: Number(form.elements.fit_max_pnp_rms_px.value),
            fit_max_error_mm: Number(form.elements.fit_max_error_mm.value),
            verify_max_error_mm: Number(form.elements.verify_max_error_mm.value)
          }
        }
      });
      if (!await refreshSession(true, { context })) return;
      showToast("任务设置已保存，相关下游结果已按依赖失效。", "success");
    });
  });

  byId("intrinsicsFiles").addEventListener("change", (event) => {
    const files = [...event.target.files];
    if (!files.length) return;
    const split = byId("uploadSplit").value;
    action("上传内参图像", async () => {
      const context = sessionContext();
      const errors = [];
      for (let index = 0; index < files.length; index += 1) {
        const file = files[index];
        const contentType = imageContentType(file);
        if (!contentType) { errors.push(`${file.name}: 不是 PNG/JPEG`); continue; }
        try {
          const query = `?split=${encodeURIComponent(split)}&filename=${encodeURIComponent(file.name)}`;
          await sessionApi(context, `/intrinsics/images${query}`, {
            method: "POST",
            headers: { "Content-Type": contentType },
            body: file
          });
        } catch (error) {
          errors.push(`${file.name}: ${error.message}`);
        }
      }
      if (!await refreshSession(false, { context })) return;
      if (errors.length) showToast(`${files.length - errors.length}/${files.length} 张上传成功；${errors.join("；")}`, "error", 8000);
      else showToast(`已上传 ${files.length} 张图像。`, "success");
    });
    event.target.value = "";
  });

  byId("runIntrinsics").addEventListener("click", () => action("启动内参任务", async () => {
    const context = sessionContext();
    await sessionApi(context, "/intrinsics/run", { method: "POST", json: {} });
    if (!await refreshSession(false, { context })) return;
    showToast("内参后台任务已排队。", "success");
  }));

  byId("cameraStart").addEventListener("click", () => action("打开本机相机", async () => {
    const context = sessionContext();
    assertCurrentSession(context);
    const device = Number(byId("cameraDevice").value);
    const data = await api("/api/camera/start", { method: "POST", json: { session_id: context.sessionId, device } });
    if (!isCurrentSession(context)) return;
    state.bootstrap.camera = data.camera;
    updateCameraControls();
    renderGlobal();
    showToast("相机预览已打开；停止时会释放句柄。", "success");
  }));
  byId("cameraStop").addEventListener("click", () => action("停止相机", async () => stopCamera()));
  byId("cameraCaptureTrain").addEventListener("click", () => captureCamera("train"));
  byId("cameraCaptureVal").addEventListener("click", () => captureCamera("val"));
  byId("boardCameraCapture").addEventListener("click", () => captureBoardCamera());
  byId("cameraPreview").addEventListener("error", () => {
    if (state.bootstrap?.camera?.open && !state.cameraErrorShown) {
      state.cameraErrorShown = true;
      showToast("相机预览读帧失败，正在释放句柄。", "error");
      stopCamera("相机读帧失败，句柄已释放。");
    }
  });

  byId("boardFile").addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const contentType = imageContentType(file);
    if (!contentType) { showToast("固定棋盘只接受 PNG/JPEG。", "error"); return; }
    action("上传固定棋盘", async () => {
      const context = sessionContext();
      await sessionApi(context, `/board/image?filename=${encodeURIComponent(file.name)}`, {
        method: "POST", headers: { "Content-Type": contentType }, body: file
      });
      if (!await refreshSession(false, { context })) return;
      showToast("固定棋盘已上传，请执行检测。", "success");
    });
    event.target.value = "";
  });
  byId("detectBoard").addEventListener("click", () => action("检测固定棋盘", async () => {
    const context = sessionContext();
    if (state.session?.board?.confirmation && !state.session.board.confirmation.stale) {
      if (!window.confirm("重新检测会产生新的不可变版本，并使示教、拟合和独立验收全部失效。继续？")) return;
    }
    await sessionApi(context, "/board/detect", { method: "POST", json: {} });
    if (!await refreshSession(false, { context })) return;
    showToast("固定棋盘检测完成，请核对编号图。", "success");
  }));
  byId("confirmBoard").addEventListener("click", () => action("冻结固定棋盘编号", async () => {
    const context = sessionContext();
    const order = document.querySelector('input[name="cornerOrder"]:checked')?.value || "normal";
    await sessionApi(context, "/board/confirm", { method: "POST", json: { corner_order: order, fixed_confirmed: byId("boardFixedConfirm").checked } });
    if (!await refreshSession(false, { context })) return;
    showToast("固定棋盘编号已冻结，下游旧结果已失效。", "success");
  }));
  byId("boardFixedConfirm").addEventListener("change", updateBoardConfirmButton);
  document.querySelectorAll('input[name="cornerOrder"]').forEach((radio) => radio.addEventListener("change", renderBoard));

  document.querySelectorAll(".arm-tab").forEach((tab) => tab.addEventListener("click", () => {
    syncTeachInputs(state.activeArm);
    state.activeArm = tab.dataset.arm;
    renderTeach();
  }));

  document.querySelectorAll('input[name="queryMode"]').forEach((radio) => radio.addEventListener("change", () => {
    syncQueryModeInputs();
    invalidateQueryResult();
  }));
  for (const id of ["queryU", "queryV", "queryBoardZ", "queryDepthZ"]) {
    byId(id).addEventListener("input", () => invalidateQueryResult());
  }
  byId("allowExtrapolation").addEventListener("change", () => invalidateQueryResult());
  byId("runQuery").addEventListener("click", () => action("查询像素坐标", async () => {
    const context = sessionContext();
    const u = parseRequiredNumber(byId("queryU").value, "像素 u", "PIXEL_REQUIRED");
    const v = parseRequiredNumber(byId("queryV").value, "像素 v", "PIXEL_REQUIRED");
    const mode = document.querySelector('input[name="queryMode"]:checked')?.value || "board";
    const body = { u_px: u, v_px: v, allow_extrapolation: byId("allowExtrapolation").checked };
    if (mode === "board") body.board_z_m = parseRequiredNumber(byId("queryBoardZ").value, "board_z_m（米）");
    else body.depth_z_m = parseRequiredNumber(byId("queryDepthZ").value, "depth_z_m（米）");
    assertCurrentSession(context);
    state.queryPixel = { u, v };
    await runPixelQuery(context, body);
  }));
  byId("saveDepthConfirmation").addEventListener("click", () => action("保存深度注册确认", async () => {
    const context = sessionContext();
    await sessionApi(context, "/depth-confirmation", {
      method: "POST",
      json: { confirmed: byId("depthConfirm").checked, registration_note: byId("depthNote").value }
    });
    if (!await refreshSession(false, { context })) return;
    showToast(byId("depthConfirm").checked ? "深度模式已绑定当前内参哈希。" : "深度模式已禁用。", "success");
  }));
  byId("disableDepth").addEventListener("click", () => action("禁用深度模式", async () => {
    const context = sessionContext();
    await sessionApi(context, "/depth-confirmation", { method: "POST", json: { confirmed: false, registration_note: byId("depthNote").value } });
    if (!await refreshSession(false, { context })) return;
    showToast("深度模式已禁用。", "success");
  }));

  byId("createExport").addEventListener("click", () => action("生成报告包", async () => {
    const context = sessionContext();
    const arms = [];
    if (byId("exportLeft").checked) arms.push("left");
    if (byId("exportRight").checked) arms.push("right");
    if (!arms.length) throw new ApiError("请至少选择一只手臂", "EXPORT_ARMS_REQUIRED");
    const capturedDependencies = sessionDependencyFingerprint(state.session, arms);
    const data = await sessionApi(context, "/exports", { method: "POST", json: { arms } });
    if (!isCurrentSession(context)
        || capturedDependencies !== sessionDependencyFingerprint(state.session, arms)
        || !exportMatchesSession(data.export, state.session)) return;
    state.exportRecord = data.export;
    renderExportPanel();
    await refreshSessionListFromServer(context);
    showToast("报告包已生成。", "success");
  }));
  byId("exportLeft").addEventListener("change", updateExportButtonState);
  byId("exportRight").addEventListener("change", updateExportButtonState);

  window.addEventListener("beforeunload", () => {
    if (state.bootstrap?.camera?.open && state.csrf) {
      fetch("/api/camera/stop", {
        method: "POST", keepalive: true,
        headers: { "Content-Type": "application/json", "X-Calibration-Token": state.csrf },
        body: "{}"
      }).catch(() => {});
    }
  });
}

function updateBoardConfirmButton() {
  const board = state.session?.board;
  const button = byId("confirmBoard");
  if (!button) return;
  button.disabled = !board || board.state !== "detected" || !byId("boardFixedConfirm").checked
    || activeJobs().some((job) => job.kind === "demo");
}

async function captureCamera(split) {
  await action(`采集 ${split} 图像`, async () => {
    const context = sessionContext();
    assertCurrentSession(context);
    const data = await api("/api/camera/capture", {
      method: "POST",
      json: { session_id: context.sessionId, target: "intrinsics", split, filename: `camera_${split}_${Date.now()}.png` }
    });
    if (!await refreshSession(false, { context })) return;
    showToast(`已采集 ${split}：${data.image.display_name}`, "success");
  });
}

async function captureBoardCamera() {
  await action("采集固定棋盘", async () => {
    const context = sessionContext();
    assertCurrentSession(context);
    await api("/api/camera/capture", {
      method: "POST",
      json: { session_id: context.sessionId, target: "board", filename: `fixed_board_${Date.now()}.png` }
    });
    if (!await refreshSession(false, { context })) return;
    showToast("固定棋盘图像已采集，请执行检测。", "success");
  });
}

document.addEventListener("DOMContentLoaded", () => {
  wireEvents();
  syncQueryModeInputs();
  boot();
});
