"use strict";

// Deterministic JavaScript regressions for calibration wizard state handling.
// This intentionally runs app.js in a VM instead of relying on response order
// from a live server: reordered GETs and fault injection must be reproducible.

const assert = require("assert/strict");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const APP = fs.readFileSync(
  path.join(__dirname, "..", "task_skills", "calibration", "web", "app.js"),
  "utf8"
);

function session(id, { running = false, leftX = 1, rightX = 2 } = {}) {
  const arm = (armName, x) => ({
    metadata: { unit: "mm" },
    points: [{ corner_id: 0, split: "train", x, y: 10, z: 20 }],
    verification: { method: "", independence_confirmed: false, rows: [] }
  });
  return {
    session_id: id,
    display_name: id,
    mode: "real",
    synthetic_only: false,
    settings: {
      board: { cols: 11, rows: 8, square_m: 0.03 },
      image_size_wh: [1280, 960],
      thresholds: {
        intrinsics_max_rms_px: 1,
        fit_max_pnp_rms_px: 1,
        fit_max_error_mm: 2,
        verify_max_error_mm: 2
      }
    },
    arms: { left: arm("left", leftX), right: arm("right", rightX) },
    jobs: running ? [{ id: "job-running", kind: "fit", arm: "left", status: "running" }] : [],
    exports: [],
    events: []
  };
}

function makeRuntime() {
  const pending = [];
  const calls = [];
  const queryResults = { className: "query-results", textContent: "stale XYZ" };
  const context = {
    console,
    URL,
    window: {
      location: { href: "http://127.0.0.1:8765/" },
      setTimeout: (fn) => setTimeout(fn, 0),
      clearTimeout: (timer) => clearTimeout(timer),
      setInterval: (fn) => setInterval(fn, 100000),
      clearInterval: (timer) => clearInterval(timer),
      confirm: () => context.__confirmResult
    },
    document: {
      addEventListener: () => {},
      querySelector: () => null,
      getElementById: (id) => id === "queryResults" ? queryResults : null,
      createElement: () => { throw new Error("DOM rendering is not used in this harness"); },
      activeElement: null
    },
    history: { replaceState: () => {} },
    localStorage: {
      values: {},
      getItem(key) { return this.values[key] ?? null; },
      setItem(key, value) { this.values[key] = String(value); },
      removeItem(key) { delete this.values[key]; }
    },
    fetch: () => { throw new Error("fetch must be mocked through api()"); },
    __confirmResult: true,
    __pending: pending,
    __calls: calls
  };
  vm.createContext(context);
  vm.runInContext(APP, context, { filename: "task_skills/calibration/web/app.js" });
  vm.runInContext(`
    renderAll = () => {};
    schedulePoll = () => {};
    refreshJobLogs = async () => {};
    api = (requestPath, options = {}) => {
      __calls.push({ path: requestPath, options });
      return new Promise((resolve, reject) => __pending.push({ path: requestPath, resolve, reject }));
    };
  `, context);
  const run = (source) => vm.runInContext(source, context, { filename: "verification.js" });
  const resolve = (requestPath, value) => {
    const index = pending.findIndex((entry) => entry.path === requestPath);
    assert.notEqual(index, -1, `no pending request for ${requestPath}; pending=${pending.map((p) => p.path)}`);
    const [entry] = pending.splice(index, 1);
    entry.resolve(value);
  };
  const resolveAt = (index, value) => {
    assert.notEqual(index, -1, `no pending request at index ${index}`);
    const [entry] = pending.splice(index, 1);
    entry.resolve(value);
  };
  const reject = (requestPath, error) => {
    const index = pending.findIndex((entry) => entry.path === requestPath);
    assert.notEqual(index, -1, `no pending request for ${requestPath}`);
    const [entry] = pending.splice(index, 1);
    entry.reject(error);
  };
  return { context, pending, calls, queryResults, run, resolve, resolveAt, reject };
}

async function waitForPending(runtime, predicate) {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const found = runtime.pending.findIndex(predicate);
    if (found !== -1) return found;
    await new Promise((resolveTick) => setTimeout(resolveTick, 0));
  }
  assert.fail(`expected request not found; pending=${runtime.pending.map((entry) => entry.path)}`);
}

async function testReorderedLoadDiscardsOldSession() {
  const runtime = makeRuntime();
  const { run, resolve } = runtime;
  const a = session("cal-a");
  const b = session("cal-b");
  const loadA = run("loadSession('cal-a', { reset: true, preserveStep: true })");
  const loadB = run("loadSession('cal-b', { reset: true, preserveStep: true })");
  resolve("/api/sessions/cal-b", { session: b });
  await loadB;
  resolve("/api/sessions/cal-a", { session: a });
  await loadA;
  assert.equal(run("state.sessionId"), "cal-b");
  assert.equal(run("state.session.session_id"), "cal-b");
  assert.equal(run("state.sessionGeneration"), 2);
}

async function testReorderedRefreshAndPollDiscardOldResponses() {
  {
    const runtime = makeRuntime();
    const { run, resolve } = runtime;
    const b = session("cal-b");
    const c = session("cal-c");
    const loadB = run("loadSession('cal-b', { reset: true })");
    resolve("/api/sessions/cal-b", { session: b });
    await loadB;
    const refresh = run("refreshSession(true)");
    const loadC = run("loadSession('cal-c', { reset: true })");
    resolve("/api/sessions/cal-c", { session: c });
    await loadC;
    resolve("/api/sessions/cal-b", { session: { ...b, display_name: "stale-B" } });
    await refresh;
    assert.equal(run("state.sessionId"), "cal-c");
    assert.equal(run("state.session.session_id"), "cal-c");
  }
  {
    const runtime = makeRuntime();
    const { run, resolve } = runtime;
    const b = session("cal-b", { running: true });
    const c = session("cal-c");
    const loadB = run("loadSession('cal-b', { reset: true })");
    resolve("/api/sessions/cal-b", { session: b });
    await loadB;
    const poll = run("pollSession()");
    const loadC = run("loadSession('cal-c', { reset: true })");
    resolve("/api/sessions/cal-c", { session: c });
    await loadC;
    resolve("/api/sessions/cal-b", { session: { ...b, display_name: "stale-poll-B" } });
    await poll;
    assert.equal(run("state.sessionId"), "cal-c");
    assert.equal(run("state.session.session_id"), "cal-c");
  }
}

async function testMutationIsBoundToCapturedSession() {
  const runtime = makeRuntime();
  const { run, resolve, calls } = runtime;
  const a = session("cal-a");
  const b = session("cal-b");
  const loadA = run("loadSession('cal-a', { reset: true })");
  resolve("/api/sessions/cal-a", { session: a });
  await loadA;
  run("__capturedContext = sessionContext()");
  const loadB = run("loadSession('cal-b', { reset: true })");
  resolve("/api/sessions/cal-b", { session: b });
  await loadB;
  const before = calls.length;
  await assert.rejects(
    run("sessionApi(__capturedContext, '/arms/left/points', { method: 'PUT', json: { points: [] } })"),
    (error) => error.code === "SESSION_CHANGED"
  );
  assert.equal(calls.length, before, "stale mutation must not reach fetch/api");
}

async function testLateMutationResponseCannotContinueAfterSessionSwitch() {
  const runtime = makeRuntime();
  const { run, resolve, pending, context } = runtime;
  const loadA = run("loadSession('cal-a', { reset: true })");
  resolve("/api/sessions/cal-a", { session: session("cal-a") });
  await loadA;
  const contextA = run("sessionContext()");
  context.__contextA = contextA;
  const mutation = run("sessionApi(__contextA, '/arms/left/points', { method: 'PUT', json: { points: [] } })");
  await waitForPending(runtime, (entry) => entry.path.endsWith("/arms/left/points"));
  const loadB = run("loadSession('cal-b', { reset: true })");
  resolve("/api/sessions/cal-b", { session: session("cal-b") });
  await loadB;
  const mutationPath = pending.find((entry) => entry.path.endsWith("/arms/left/points")).path;
  resolve(mutationPath, {});
  await assert.rejects(mutation, (error) => error.code === "SESSION_CHANGED");
  assert.equal(run("state.session.session_id"), "cal-b");
}

async function testBlankMeasurementsAreRejectedButExplicitZeroIsKept() {
  const runtime = makeRuntime();
  const { run } = runtime;
  run(`state.session = ${JSON.stringify(session("cal-a"))}; state.sessionId = "cal-a"; resetDrafts();`);
  assert.throws(
    () => run("normalizeTeachPoints('left', [{ corner_id: 0, split: 'train', x: '', y: 10, z: 20 }])"),
    (error) => error.code === "INVALID_NUMBER"
  );
  assert.throws(
    () => run("normalizeTeachPoints('left', [{ corner_id: 0, split: 'train', x: 1, y: 10, z: null }])"),
    (error) => error.code === "INVALID_NUMBER"
  );
  assert.throws(
    () => run("normalizeTeachPoints('left', [{ corner_id: 0, split: 'train', y: 10, z: 20 }])"),
    (error) => error.code === "INVALID_NUMBER"
  );
  assert.throws(
    () => run("normalizeTeachPoints('left', [{ corner_id: ' ', split: 'train', x: 1, y: 10, z: 20 }])"),
    (error) => error.code === "INVALID_CORNER_ID"
  );
  const zeroTeach = run("normalizeTeachPoints('left', [{ corner_id: 0, split: 'train', x: '0', y: 10, z: 20 }])");
  assert.equal(zeroTeach[0].x, 0);

  run(`state.verifyDrafts.left = {
    method: "独立测量", independence_confirmed: true,
    rows: [{ u_px: 640, v_px: 480, board_z_m: 0, reference_x_m: "  ", reference_y_m: 1, reference_z_m: 2 }]
  }`);
  assert.throws(
    () => run("normalizeVerifyDraft('left')"),
    (error) => error.code === "INVALID_VERIFY_ROW"
  );
  run(`state.verifyDrafts.left.rows[0].reference_x_m = "0"`);
  const zeroVerify = run("normalizeVerifyDraft('left')");
  assert.equal(zeroVerify.rows[0].reference_x_m, 0);
  run(`state.verifyDrafts.left.rows[0].reference_y_m = null`);
  assert.throws(
    () => run("normalizeVerifyDraft('left')"),
    (error) => error.code === "INVALID_VERIFY_ROW"
  );
  run(`delete state.verifyDrafts.left.rows[0].reference_z_m`);
  assert.throws(
    () => run("normalizeVerifyDraft('left')"),
    (error) => error.code === "INVALID_VERIFY_ROW"
  );
}

async function testSavingOneArmPreservesOtherDirtyArmDrafts() {
  const runtime = makeRuntime();
  const { run, resolve, calls } = runtime;
  const server = session("cal-a", { leftX: 999, rightX: 2 });
  const load = run("loadSession('cal-a', { reset: true })");
  resolve("/api/sessions/cal-a", { session: server });
  await load;

  run(`state.teachDrafts.right[0].x = "unsaved-right";
       state.dirtyTeach.right = true;
       state.verifyDrafts.right.method = "right unsaved method";
       state.verifyDrafts.right.rows = [{ u_px: "640", v_px: "480", board_z_m: "0", reference_x_m: "0.1", reference_y_m: "0.2", reference_z_m: "0.3" }];
       state.dirtyVerify.right = true;
       state.teachDrafts.left[0].x = "11";`);

  const saving = run("saveTeachPoints('left', state.teachDrafts.left)");
  // PUT points, then refresh GET.
  for (let attempt = 0; attempt < 8; attempt += 1) {
    if (runtime.pending.some((entry) => entry.path.endsWith("/arms/left/points"))) break;
    await new Promise((resolveTick) => setTimeout(resolveTick, 0));
  }
  resolve("/api/sessions/cal-a/arms/left/points", {});
  for (let attempt = 0; attempt < 8; attempt += 1) {
    if (runtime.pending.some((entry) => entry.path === "/api/sessions/cal-a")) break;
    await new Promise((resolveTick) => setTimeout(resolveTick, 0));
  }
  resolve("/api/sessions/cal-a", { session: server });
  await saving;

  assert.equal(run("state.teachDrafts.right[0].x"), "unsaved-right");
  assert.equal(run("state.verifyDrafts.right.method"), "right unsaved method");
  assert.equal(run("state.teachDrafts.right[0].x"), "unsaved-right");
  assert.equal(run("state.dirtyTeach.right"), true);
  assert.equal(run("state.dirtyVerify.right"), true);
  assert.equal(run("state.teachDrafts.left[0].x"), "999");
  assert.equal(run("state.dirtyTeach.left"), false);
  assert.equal(calls.filter((call) => call.path.endsWith("/arms/right/points")).length, 0);

  const reverse = makeRuntime();
  const { run: runReverse, resolve: resolveReverse } = reverse;
  const reverseLoad = runReverse("loadSession('cal-a', { reset: true })");
  resolveReverse("/api/sessions/cal-a", { session: session("cal-a") });
  await reverseLoad;
  runReverse(`state.teachDrafts.left[0].x = "unsaved-left";
             state.dirtyTeach.left = true;
             state.verifyDrafts.left.method = "left unsaved independent method";
             state.dirtyVerify.left = true;
             state.teachDrafts.right[0].x = "23";
             markDraftDirty("teach", "right");`);
  const reverseSaving = runReverse("saveTeachPoints('right', state.teachDrafts.right)");
  const reversePut = await waitForPending(reverse, (entry) => entry.path.endsWith("/arms/right/points"));
  reverse.resolveAt(reversePut, {});
  const reverseGet = await waitForPending(reverse, (entry) => entry.path === "/api/sessions/cal-a");
  reverse.resolveAt(reverseGet, { session: session("cal-a", { rightX: 23 }) });
  await reverseSaving;

  assert.equal(runReverse("state.teachDrafts.left[0].x"), "unsaved-left");
  assert.equal(runReverse("state.verifyDrafts.left.method"), "left unsaved independent method");
  assert.equal(runReverse("state.dirtyTeach.left"), true);
  assert.equal(runReverse("state.dirtyVerify.left"), true);
  assert.equal(runReverse("state.teachDrafts.right[0].x"), "23");
  assert.equal(runReverse("state.dirtyTeach.right"), false);
}

async function testSameArmTeachEditsDuringPutAndRefreshStayDirty() {
  const runtime = makeRuntime();
  const { run, resolve, resolveAt } = runtime;
  const initial = session("cal-a", { leftX: 1 });
  const load = run("loadSession('cal-a', { reset: true })");
  resolve("/api/sessions/cal-a", { session: initial });
  await load;
  run(`state.teachDrafts.left[0].x = "11"; markDraftDirty("teach", "left");`);
  const saving = run("saveTeachPoints('left', state.teachDrafts.left)");
  const putIndex = await waitForPending(runtime, (entry) => entry.path.endsWith("/arms/left/points"));
  resolveAt(putIndex, {});
  const getIndex = await waitForPending(runtime, (entry) => entry.path === "/api/sessions/cal-a");
  run(`state.teachDrafts.left[0].x = "later-teach-edit"; markDraftDirty("teach", "left");`);
  resolveAt(getIndex, { session: session("cal-a", { leftX: 11 }) });
  await saving;
  assert.equal(run("state.teachDrafts.left[0].x"), "later-teach-edit");
  assert.equal(run("state.dirtyTeach.left"), true);
  assert.equal(run("state.session.arms.left.points[0].x"), 11);
}

async function testSameArmVerificationEditsDuringPutAndRefreshStayDirty() {
  const runtime = makeRuntime();
  const { run, resolveAt } = runtime;
  const load = run("loadSession('cal-a', { reset: true })");
  runtime.resolve("/api/sessions/cal-a", { session: session("cal-a") });
  await load;
  run(`state.verifyDrafts.left = {
    method: "submitted method", independence_confirmed: true,
    rows: [
      { u_px: "640", v_px: "480", board_z_m: "0", reference_x_m: "0.1", reference_y_m: "0.2", reference_z_m: "0.3" },
      { u_px: "600", v_px: "450", board_z_m: "0", reference_x_m: "0.4", reference_y_m: "0.5", reference_z_m: "0.6" },
      { u_px: "700", v_px: "500", board_z_m: "0", reference_x_m: "0.7", reference_y_m: "0.8", reference_z_m: "0.9" }
    ]
  }; markDraftDirty("verify", "left");`);
  const saving = run("saveVerificationDraft('left')");
  const putIndex = await waitForPending(runtime, (entry) => entry.path.endsWith("/arms/left/verification"));
  resolveAt(putIndex, {});
  const getIndex = await waitForPending(runtime, (entry) => entry.path === "/api/sessions/cal-a");
  run(`state.verifyDrafts.left.method = "later verification edit"; markDraftDirty("verify", "left");`);
  resolveAt(getIndex, { session: session("cal-a") });
  await saving;
  assert.equal(run("state.verifyDrafts.left.method"), "later verification edit");
  assert.equal(run("state.dirtyVerify.left"), true);
}

async function testQueryRequestsAreBoundToSessionSequenceAndDependencies() {
  {
    const runtime = makeRuntime();
    const { run, resolveAt } = runtime;
    run(`state.session = ${JSON.stringify(session("cal-a"))};
         state.sessionId = "cal-a"; state.sessionGeneration = 1;
         state.sessionDependencyFingerprint = sessionDependencyFingerprint(state.session);
         renderQuery = () => {};`);
    const first = run(`runPixelQuery(sessionContext(), { u_px: 1, v_px: 2, board_z_m: 0 })`);
    await waitForPending(runtime, (entry) => entry.path.endsWith("/query"));
    const second = run(`runPixelQuery(sessionContext(), { u_px: 3, v_px: 4, board_z_m: 0 })`);
    await waitForPending(runtime, (entry) => entry.path.endsWith("/query"));
    const queryEntries = runtime.pending.map((entry, index) => ({ entry, index }))
      .filter(({ entry }) => entry.path.endsWith("/query"));
    resolveAt(queryEntries[1].index, { value: "newest" });
    assert.equal(await second, true);
    resolveAt(queryEntries[0].index, { value: "oldest" });
    assert.equal(await first, false);
    assert.equal(run("state.queryResult.value"), "newest");

    const pendingQuery = run(`runPixelQuery(sessionContext(), { u_px: 5, v_px: 6, board_z_m: 0 })`);
    const pendingIndex = await waitForPending(runtime, (entry) => entry.path.endsWith("/query"));
    run("invalidateQueryResult()");
    resolveAt(pendingIndex, { value: "superseded by input edit" });
    assert.equal(await pendingQuery, false);
    assert.equal(run("state.queryResult"), null);
  }

  {
    const runtime = makeRuntime();
    const { run, resolve, resolveAt, pending } = runtime;
    const loadA = run("loadSession('cal-a', { reset: true })");
    resolve("/api/sessions/cal-a", { session: session("cal-a") });
    await loadA;
    run("state.queryResult = { value: 'A result' }");
    const staleQuery = run(`runPixelQuery(sessionContext(), { u_px: 1, v_px: 2, board_z_m: 0 })`);
    await waitForPending(runtime, (entry) => entry.path.endsWith("/query"));
    const loadB = run("loadSession('cal-b', { reset: true })");
    resolve("/api/sessions/cal-b", { session: session("cal-b") });
    await loadB;
    const queryIndex = pending.findIndex((entry) => entry.path.endsWith("/query"));
    resolveAt(queryIndex, { value: "stale A" });
    await assert.rejects(staleQuery, (error) => error.code === "SESSION_CHANGED");
    assert.equal(run("state.session.session_id"), "cal-b");
    assert.equal(run("state.queryResult"), null);
  }
}

async function testDependencyRefreshAndPollClearQueryAndExportCaches() {
  for (const accept of ["refreshSession(false)", "pollSession()"] ) {
    const runtime = makeRuntime();
    const { run, resolveAt } = runtime;
    runtime.queryResults.textContent = "stale XYZ";
    const activeInput = { tagName: "INPUT", value: "unsaved input" };
    runtime.context.document.activeElement = activeInput;
    const loaded = session("cal-a", { running: accept === "pollSession()" });
    run(`state.sessionId = "cal-a"; state.sessionGeneration = 1;
         state.session = ${JSON.stringify(loaded)};
         state.sessionDependencyFingerprint = sessionDependencyFingerprint(state.session);
         state.queryResult = { value: "stale XYZ" }; state.exportRecord = { filename: "stale.zip" };
         state.dirtyTeach.left = true; state.teachDrafts.left = [{ x: "unsaved draft" }];`);
    const pending = run(accept);
    const requestIndex = await waitForPending(runtime, (entry) => entry.path === "/api/sessions/cal-a");
    const changed = session("cal-a", { running: accept === "pollSession()" });
    changed.arms.left.points[0].x += 1;
    resolveAt(requestIndex, { session: changed });
    await pending;
    assert.equal(run("state.queryResult"), null, accept);
    assert.equal(run("state.exportRecord"), null, accept);
    assert.equal(runtime.queryResults.textContent, "选择像素后查询。", accept);
    assert.equal(runtime.context.document.activeElement, activeInput, accept);
    assert.equal(run("state.dirtyTeach.left"), true, accept);
    assert.equal(run("state.teachDrafts.left[0].x"), "unsaved draft", accept);
  }
}

async function testInputInvalidationClearsVisibleQueryResult() {
  const runtime = makeRuntime();
  runtime.queryResults.textContent = "stale XYZ";
  runtime.run("state.queryResult = { value: 'stale XYZ' }; invalidateQueryResult()");
  assert.equal(runtime.run("state.queryResult"), null);
  assert.equal(runtime.queryResults.textContent, "选择像素后查询。");
}

async function testDirtySessionSwitchRequiresDeliberateDiscard() {
  const runtime = makeRuntime();
  const { run, resolve } = runtime;
  const load = run("loadSession('cal-a', { reset: true })");
  resolve("/api/sessions/cal-a", { session: session("cal-a") });
  await load;
  run("state.dirtyTeach.left = true");
  runtime.context.__confirmResult = false;
  const before = runtime.calls.length;
  assert.equal(await run("loadSession('cal-b', { reset: true })"), false);
  assert.equal(runtime.calls.length, before);
  assert.equal(run("state.sessionId"), "cal-a");
  assert.equal(run("state.session.session_id"), "cal-a");
}

async function main() {
  const tests = [
    ["reordered load responses", testReorderedLoadDiscardsOldSession],
    ["reordered refresh/poll responses", testReorderedRefreshAndPollDiscardOldResponses],
    ["captured-session mutation guard", testMutationIsBoundToCapturedSession],
    ["late mutation response session guard", testLateMutationResponseCannotContinueAfterSessionSwitch],
    ["blank measurements", testBlankMeasurementsAreRejectedButExplicitZeroIsKept],
    ["dirty draft preservation", testSavingOneArmPreservesOtherDirtyArmDrafts],
    ["same-arm teach draft preservation", testSameArmTeachEditsDuringPutAndRefreshStayDirty],
    ["same-arm verification draft preservation", testSameArmVerificationEditsDuringPutAndRefreshStayDirty],
    ["query sequence/session binding", testQueryRequestsAreBoundToSessionSequenceAndDependencies],
    ["query/export dependency invalidation", testDependencyRefreshAndPollClearQueryAndExportCaches],
    ["visible query invalidation on input", testInputInvalidationClearsVisibleQueryResult],
    ["dirty session switch policy", testDirtySessionSwitchRequiresDeliberateDiscard]
  ];
  const passed = [];
  for (const [name, test] of tests) {
    await test();
    passed.push(name);
  }
  process.stdout.write(JSON.stringify({ passed, count: passed.length }, null, 2) + "\n");
}

main().catch((error) => {
  process.stderr.write((error && error.stack) ? error.stack + "\n" : String(error) + "\n");
  process.exitCode = 1;
});
