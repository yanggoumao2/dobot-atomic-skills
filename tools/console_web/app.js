let token = '', selectedSide = '', busy = false, commandList = [], applied = {}, runtime = {};
const $ = id => document.getElementById(id);
function toast(message) {
  $('toast').textContent = message;
  $('toast').style.display = 'block';
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => $('toast').style.display = 'none', 3500);
}
async function api(path, data) {
  const response = await fetch(path, data === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json', 'X-Console-Token': token},
    body: JSON.stringify(data)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || '请求失败');
  return result;
}
function confirmAction(title, description) {
  $('confirmTitle').textContent = title;
  $('confirmText').textContent = description;
  return new Promise(resolve => {
    const dialog = $('confirm');
    dialog.returnValue = 'cancel';
    dialog.addEventListener('close', () => resolve(dialog.returnValue === 'execute'), {once:true});
    dialog.showModal();
  });
}
async function action(path, data) {
  if (busy) return;
  busy = true;
  updateButtons();
  try { await api(path, data); await refresh(); }
  catch (error) { toast(error.message); busy = false; updateButtons(); }
}
function updateButtons() {
  $('serviceGripper').disabled = busy;
  document.querySelectorAll('.execute,[data-home],[data-side]').forEach(button => button.disabled = busy);
  document.querySelectorAll('[data-gripper]').forEach(button => button.disabled = busy || !selectedSide);
}
async function loadCommands() {
  const values = {project: $('project').value.trim(), episode: $('episode').value.trim()};
  const list = await api('/api/commands', values);
  applied = values;
  commandList = list;
  $('dataPath').textContent = '录制目录  ' + [runtime.data_root, applied.project, 'collect_data'].join(runtime.separator);
  $('commands').replaceChildren();
  for (const command of list) {
    const card = document.createElement('article'); card.className = 'command';
    const title = document.createElement('h3'); title.textContent = command.title;
    const description = document.createElement('p'); description.textContent = command.description;
    const details = document.createElement('details');
    const summary = document.createElement('summary'); summary.textContent = `查看 ${runtime.shell} 指令`;
    const code = document.createElement('pre'); code.textContent = command.code;
    details.append(summary, code);
    const actions = document.createElement('div'); actions.className = 'actions';
    const copy = document.createElement('button'); copy.textContent = '复制指令';
    copy.onclick = async () => {
      try { await navigator.clipboard.writeText(command.code); toast('指令已复制'); }
      catch { toast('剪贴板不可用，请展开指令后手动复制'); }
    };
    const run = document.createElement('button'); run.className = 'execute'; run.textContent = '执行 ↗';
    run.onclick = async () => {
      if (command.confirm && !await confirmAction(command.title, command.description)) return;
      await action('/api/execute', {...applied, id: command.id});
    };
    actions.append(copy, run); card.append(title, description, details, actions); $('commands').append(card);
  }
  updateButtons();
}
async function refresh() {
  const state = await api('/api/state'); token = state.token;
  runtime = state;
  $('shellHint').textContent = `复制到 ${state.shell}，或直接执行`;
  $('terminalHint').textContent = `当前采集规格以仓库 INI 为准，执行结果请查看对应 ${state.shell} 终端。`;
  busy = state.job.busy;
  $('status').textContent = state.job.message;
  $('status').classList.toggle('error', !state.job.success);
  $('lockState').textContent = state.lock_exists ? '设备锁存在' : '设备锁空闲';
  for (const side of ['left', 'right']) {
    const endpoint = state.endpoints[side];
    $(side + 'Port').textContent = `${endpoint.port} / ID${endpoint.servo_id}`;
  }
  $('events').replaceChildren(...state.events.map(event => {
    const item = document.createElement('li'); item.textContent = `${event.time}  ${event.message}`; return item;
  }));
  $('stop').disabled = !(busy && state.job.kind === '单臂归位');
  updateButtons();
}
$('update').onclick = () => loadCommands().catch(error => toast(error.message));
document.querySelectorAll('[data-side]').forEach(button => button.onclick = () => {
  selectedSide = button.dataset.side;
  document.querySelectorAll('[data-side]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  updateButtons();
});
$('serviceGripper').onchange = () => {
  $('gripperHelp').textContent = $('serviceGripper').checked
    ? '保持新版 A 运行，先退出 B 和 ACT 执行程序，再选择实体侧开合。'
    : '先退出 A / B，再选择实体侧。';
};
document.querySelectorAll('[data-gripper]').forEach(button => button.onclick = () => action('/api/gripper', {
  side:selectedSide, action:button.dataset.gripper, via_service:$('serviceGripper').checked
}));
document.querySelectorAll('[data-home]').forEach(button => button.onclick = async () => {
  if (await confirmAction(button.textContent, '确认已退出 A / B、回程路径空闲且夹爪无持物，执行后开始慢速归位。'))
    await action('/api/home', {side:button.dataset.home});
});
$('stop').onclick = () => api('/api/stop', {}).then(() => toast('已请求停止归位')).catch(error => toast(error.message));
async function start() {
  await refresh();
  await loadCommands();
  setInterval(() => refresh().catch(() => {
    busy = true; updateButtons(); $('stop').disabled = true;
    $('status').textContent = '本地服务连接中断，请检查控制台终端。';
    $('status').classList.add('error');
  }), 1500);
}
start().catch(error => { $('status').textContent = error.message; });
