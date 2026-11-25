"""Loopback ACT console. Devices open only when the operator requests an action."""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.gripper_panel import GripperControl, load_endpoints

PYTHON = Path(sys.executable)
LOCK = Path(os.environ.get('DOBOT_DEVICE_LOCK_PATH') or ROOT.parent / '.local/runtime/device-owner')
DEFAULT_PROJECT = 'tube_insert_cup_2cam_rgb_marked_v2'
WEB = ROOT / 'tools/console_web'


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


# Copied commands and launched commands contain the same process checks.
GUARDS = r'''
function Get-ControlProcesses {
    @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and
        $_.CommandLine -match '(launch_nodes\.py|run_control\.py|5_camera_read|gripper_panel\.py)'
    })
}
function Assert-NoB {
    if (@(Get-ControlProcesses | Where-Object { $_.CommandLine -match 'run_control\.py|5_camera_read' }).Count) {
        throw '请先停止录制、关闭跟随，退出终端 B 和相机预览。'
    }
}
function Assert-NoControl {
    if (@(Get-ControlProcesses).Count) { throw '请先退出 A/B、旧控制窗口和相机预览。' }
}
'''


def commands(project=DEFAULT_PROJECT, episode=''):
    if not isinstance(project, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', project):
        raise ValueError('项目名限 1～80 位字母、数字、下划线或短横线')
    if not isinstance(episode, str) or (episode and not re.fullmatch(r'[A-Za-z0-9_.-]{1,120}', episode)):
        raise ValueError('episode 请填写目录名，限字母、数字、下划线、点和短横线')
    if episode in ('.', '..'):
        raise ValueError('episode 名称无效')
    if sys.platform.startswith('linux'):
        from tools.linux_console import commands as linux_commands
        return linux_commands(project, episode, ROOT, PYTHON, LOCK)
    data = ROOT.parent / 'datasets' / project / 'collect_data'
    base = ("$ErrorActionPreference = 'Stop'\n"
            + "$ProgressPreference = 'SilentlyContinue'\n"
            + "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n"
            + "$OutputEncoding = [Console]::OutputEncoding\n" + f'Set-Location {quote(ROOT)}\n'
            + f'$env:DOBOT_DEVICE_LOCK_PATH = {quote(LOCK)}\n'
            + "$env:DOBOT_ACT_RECORDING = '1'\n")
    py = f'& {quote(PYTHON)} -X utf8 -B '
    clear = f'''Assert-NoControl
$lockPath = {quote(LOCK)}
if (Test-Path -LiteralPath $lockPath) {{
    $record = Join-Path $lockPath 'owner.json'
    if (!(Test-Path -LiteralPath $record)) {{ throw '锁缺少 owner.json，请人工核对来源和设备停止状态。' }}
    $owner = Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
    if ($owner.host -ne [System.Net.Dns]::GetHostName() -or !$owner.pid) {{ throw '锁所有者无法确认属于本机，请人工核对。' }}
    if (Get-Process -Id ([int]$owner.pid) -ErrorAction SilentlyContinue) {{ throw '持锁进程仍在运行，请正常退出该进程。' }}
    $archiveName = (Split-Path $lockPath -Leaf) + '.closed-' + (Get-Date -Format 'yyyyMMdd-HHmmss-ffff')
    Rename-Item -LiteralPath $lockPath -NewName $archiveName -ErrorAction Stop
    Write-Host "旧锁已归档 $archiveName"
}} else {{ Write-Host '当前没有设备锁。' }}'''
    start_a = f'''Assert-NoControl
if (Test-Path -LiteralPath $env:DOBOT_DEVICE_LOCK_PATH) {{ throw '存在设备锁，请核对并归档旧锁。' }}
{py}experiments\\launch_nodes.py --hostname 127.0.0.1 --robot-port 6001'''
    start_b = f'''Assert-NoB
if (!(Test-NetConnection 127.0.0.1 -Port 6001 -InformationLevel Quiet -WarningAction SilentlyContinue)) {{ throw '请先启动终端 A 并等待 6001 服务就绪。' }}
Remove-Item Env:DOBOT_RECORD_QUEUE_SIZE -ErrorAction SilentlyContinue
Remove-Item Env:DOBOT_ALLOW_SAFE_POSE_INIT -ErrorAction SilentlyContinue
{py}experiments\\run_control.py --hostname 127.0.0.1 --robot-port 6001 --show-img --project-name {project}'''
    home = f'''Assert-NoB
if (!(Test-NetConnection 127.0.0.1 -Port 6001 -InformationLevel Quiet -WarningAction SilentlyContinue)) {{ throw '请先启动终端 A。' }}
@'
import os
os.environ['DOBOT_ALLOW_SAFE_POSE_INIT'] = '1'
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from scripts.manipulate_utils import robot_pose_init
client = ZMQClientRobot(port=6001, host='127.0.0.1')
try:
    robot_pose_init(RobotEnv(client))
    print('归位指令发送完成，请确认实臂姿态。')
finally:
    client.close()
'@ | {py}-'''
    definitions = [
        ('processes', '查看控制进程', '查看 Python 进程、PID 和完整命令行。', "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | Select-Object ProcessId,Name,CommandLine | Format-List", False),
        ('clear_lock', '归档旧锁', '先退出控制程序并确认设备停止。检查本机持锁 PID 已退出后归档，保留场景标记。', clear, True),
        ('start_a', '启动终端 A', 'ACT 反馈服务。会使能两臂并初始化夹爪，先清空夹爪活动范围。等待 6001 服务启动后再开 B。', start_a, True),
        ('start_b', '启动终端 B', 'ACT 示教采集。默认检查初始姿态，开启 top / left 两路预览。', start_b, True),
        ('home_service', '通过 A 双臂归位', '保持 A 运行，先退出 B。检查两臂回程路径后执行，夹爪保持开度。', home, True),
        ('camera', '相机预览', '先退出 B 和其他预览。当前相机配置取自 INI。Q / Esc 退出。', 'Assert-NoB\n' + py + '-m scripts.5_camera_read', False),
    ]
    if episode:
        ep = quote(data / episode)
        definitions.extend([
            ('check', '检查 episode', '生成质量报告，检查正常收尾及 ACT 数据完整性。', py + '-m dobot_control.act_dataset check ' + ep, False),
            ('preview', '生成回看视频', '生成 preview.mp4，已存在时转换器拒绝覆盖。', py + '-m dobot_control.act_dataset preview ' + ep + ' --output ' + quote(data / episode / 'preview.mp4'), False),
        ])
    definitions.append(('convert', '导出 ACT 数据集', '先退出 B。仅导出质量合格且标为 success 的记录，输出 act_v1 必须为新目录。', 'Assert-NoB\n' + py + '-m dobot_control.act_dataset convert ' + quote(data) + ' --output ' + quote(data.parent / 'act_v1') + ' --fps 15', False))
    return [dict(id=key, title=title, description=desc, confirm=confirm,
                 code=base + GUARDS + '\n' + code) for key, title, desc, code, confirm in definitions]


class Console:
    def __init__(self):
        self.control = GripperControl(load_endpoints())
        self.lock = threading.Lock()
        self.job = dict(busy=False, kind='', message='就绪', success=True)
        self.events = []

    def snapshot(self):
        with self.lock:
            return dict(job=dict(self.job), events=list(self.events), lock_exists=LOCK.exists())

    def submit(self, kind, operation, prepare_home=False):
        with self.lock:
            if self.job['busy']:
                raise ValueError('正在执行操作，请等待完成')
            if prepare_home:
                self.control.prepare_home()
            self.job = dict(busy=True, kind=kind, message='正在执行 ' + kind, success=True)
        def run():
            try:
                message = str(operation())
                success = True
            except Exception as exc:
                logging.exception('Console action %s failed', kind)
                message, success = str(exc), False
            with self.lock:
                self.job = dict(busy=False, kind=kind, message=message, success=success)
                self.events.insert(0, dict(time=time.strftime('%H:%M:%S'), message=message, success=success))
                del self.events[30:]
        threading.Thread(target=run, name='console-operation', daemon=False).start()

    def launch(self, command):
        if sys.platform.startswith('linux'):
            from tools.linux_console import launch
            return launch(command, ROOT)
        persistent = command['id'] in ('start_a', 'start_b', 'camera')
        ending = '' if persistent else '\nif ($LASTEXITCODE) { exit $LASTEXITCODE }'
        on_error = '' if persistent else '; exit 1'
        script = ('try {\n' + command['code'] + ending + '\n}'
                  + '\ncatch { [Console]::Error.WriteLine($_.Exception.Message)' + on_error + ' }')
        encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
        powershell = ['powershell.exe', '-NoLogo', '-NoProfile', '-OutputFormat', 'Text']
        if persistent:
            subprocess.Popen([*powershell, '-NoExit', '-EncodedCommand', encoded],
                             cwd=ROOT, creationflags=subprocess.CREATE_NEW_CONSOLE)
            time.sleep(3)
            return command['title'] + '的 PowerShell 已打开，请在该窗口查看执行结果并按 Ctrl+C 正常退出。'
        result = subprocess.run([*powershell, '-EncodedCommand', encoded],
                                cwd=ROOT, creationflags=subprocess.CREATE_NEW_CONSOLE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = (result.stdout + result.stderr).decode('utf-8', errors='replace').strip()
        logging.info('Command %s exit=%s output=%s', command['id'], result.returncode, output)
        if result.returncode:
            raise RuntimeError(command['title'] + '失败 ' + output[-3000:])
        return command['title'] + '执行结束 ' + output[-3000:]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logging.info(fmt, *args)

    def reply(self, status, data, content_type='application/json; charset=utf-8'):
        body = json.dumps(data, ensure_ascii=False).encode() if isinstance(data, (dict, list)) else data
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def allowed(self):
        if self.headers.get('Host') != f'127.0.0.1:{self.server.server_port}':
            self.reply(403, {'error': '仅允许本地访问'})
            return False
        return True

    def do_GET(self):
        if not self.allowed():
            return
        if self.path == '/api/state':
            self.reply(200, dict(self.server.console.snapshot(), token=self.server.token,
                                project=DEFAULT_PROJECT, root=str(ROOT), lock=str(LOCK),
                                shell='Bash' if sys.platform.startswith('linux') else 'PowerShell',
                                data_root=str(ROOT.parent / 'datasets'), separator=os.sep,
                                endpoints={key: vars(value) for key, value in self.server.console.control.endpoints.items()}))
            return
        routes = {'/': ('index.html', 'text/html; charset=utf-8'),
                  '/app.js': ('app.js', 'application/javascript; charset=utf-8'),
                  '/style.css': ('style.css', 'text/css; charset=utf-8')}
        if self.path not in routes:
            self.reply(404, {'error': '入口不存在'})
            return
        filename, mime = routes[self.path]
        self.reply(200, (WEB / filename).read_bytes(), mime)

    def do_POST(self):
        if not self.allowed():
            return
        origin = f'http://127.0.0.1:{self.server.server_port}'
        if self.headers.get('Origin') != origin or self.headers.get('X-Console-Token') != self.server.token:
            self.reply(403, {'error': '请从本地控制页面操作'})
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 8192:
                raise ValueError('请求长度无效')
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError('请求必须是对象')
            console = self.server.console
            if self.path == '/api/commands':
                self.reply(200, commands(data.get('project', DEFAULT_PROJECT), data.get('episode', '')))
                return
            if self.path == '/api/execute':
                command = next((item for item in commands(data.get('project', DEFAULT_PROJECT), data.get('episode', '')) if item['id'] == data.get('id')), None)
                if command is None:
                    raise ValueError('未知指令')
                console.submit(command['title'], lambda: console.launch(command))
            elif self.path == '/api/gripper':
                side, action = data.get('side'), data.get('action')
                if side not in ('left', 'right') or action not in ('open', 'close'):
                    raise ValueError('夹爪操作无效')
                via_service = data.get('via_service', False)
                if not isinstance(via_service, bool):
                    raise ValueError('夹爪连接模式无效')
                if via_service:
                    from tools.service_gripper import recover
                    console.submit('通过 A 恢复夹爪', lambda: f'夹爪到位，读回 {recover(side, action)}')
                else:
                    console.submit('夹爪动作', lambda: f'夹爪到位，读回 {console.control.execute(side, action)}')
            elif self.path == '/api/home':
                side = data.get('side')
                if side not in ('left', 'right'):
                    raise ValueError('侧别无效')
                def home():
                    angles = console.control.home(side)
                    return '已归位，J1～J6 ' + ', '.join(f'{x:.2f}°' for x in angles)
                console.submit('单臂归位', home, prepare_home=True)
            elif self.path == '/api/stop':
                console.control.cancel_home()
            else:
                self.reply(404, {'error': '入口不存在'})
                return
            self.reply(202, {'message': '操作已提交'})
        except (ValueError, TypeError, KeyError) as exc:
            self.reply(400, {'error': str(exc)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    log = ROOT / '.local/operator-console.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log, encoding='utf-8', level=logging.INFO)
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    except OSError as exc:
        raise SystemExit(f'网页端口不可用 {exc}。请检查已有控制台，或使用 --port 指定其他端口。')
    server.console = Console()
    server.token = secrets.token_urlsafe(32)
    url = f'http://127.0.0.1:{server.server_port}'
    print('Dobot 本地 ACT 控制台 ' + url, flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.console.control.cancel_home()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
