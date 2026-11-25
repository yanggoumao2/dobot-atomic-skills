"""Operator console regression checks. No real device connections or motion."""
import json
import os
import base64
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from tools import local_console as app


class ConsoleTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows PowerShell-specific test")
    def test_commands_use_act_mode_and_validated_project(self):
        commands = {item['id']: item for item in app.commands('test_project', 'episode_001')}
        for key in ('start_a', 'start_b'):
            self.assertIn("$env:DOBOT_ACT_RECORDING = '1'", commands[key]['code'])
        self.assertIn('--project-name test_project', commands['start_b']['code'])
        self.assertIn('Remove-Item Env:DOBOT_ALLOW_SAFE_POSE_INIT', commands['start_b']['code'])
        self.assertIn('Assert-NoB', commands['home_service']['code'])
        self.assertIn('Get-Process -Id', commands['clear_lock']['code'])
        for value in ('../other', "a';calc;#", '', 'a b'):
            with self.assertRaises(ValueError):
                app.commands(value)
        with self.assertRaises(ValueError):
            app.commands(episode='..')

    @unittest.skipUnless(os.name == "nt", "Windows PowerShell-specific test")
    def test_all_generated_powershell_parses_without_execution(self):
        with tempfile.TemporaryDirectory(prefix='dobot-console-') as directory:
            for item in app.commands(episode='episode_001'):
                Path(directory, item['id'] + '.ps1').write_text(item['code'], encoding='utf-8-sig')
            script = f"Get-ChildItem -LiteralPath {app.quote(directory)} -Filter '*.ps1' | ForEach-Object {{ $tokens=$null; $errors=$null; [void][System.Management.Automation.Language.Parser]::ParseFile($_.FullName,[ref]$tokens,[ref]$errors); if ($errors) {{ Write-Output $errors; exit 1 }} }}"
            result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', script], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(os.name == "nt", "Windows PowerShell-specific test")
    def test_archive_rejects_live_owner_and_preserves_marker(self):
        import os
        with tempfile.TemporaryDirectory(prefix='dobot-console-lock-') as directory:
            lock = Path(directory, 'device-owner')
            lock.mkdir()
            marker = Path(directory, 'device-owner.scene-invalid')
            marker.write_text('keep')
            owner = lock / 'owner.json'
            owner.write_text(json.dumps(dict(pid=os.getpid(), host=socket.gethostname())))
            with patch.object(app, 'LOCK', lock):
                command = next(item for item in app.commands() if item['id'] == 'clear_lock')['code']
            # Isolate the fixture from unrelated real control processes.
            command = command.replace(app.GUARDS, 'function Assert-NoControl {}\n')
            script = Path(directory, 'clear.ps1')
            script.write_text(command, encoding='utf-8-sig')
            encoded = base64.b64encode(command.encode('utf-16-le')).decode('ascii')
            result = subprocess.run(['powershell.exe', '-NoProfile', '-EncodedCommand', encoded], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(owner.exists())
            # Obtain a terminated PID instead of assuming an arbitrary PID is unused.
            proc = subprocess.Popen(['cmd.exe', '/c', 'exit', '0'])
            proc.wait()
            owner.write_text(json.dumps(dict(pid=proc.pid, host=socket.gethostname())))
            result = subprocess.run(['powershell.exe', '-NoProfile', '-EncodedCommand', encoded], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(lock.exists())
            self.assertEqual(len(list(Path(directory).glob('device-owner.closed-*'))), 1)
            self.assertEqual(marker.read_text(), 'keep')

    def test_busy_rejects_second_action_and_cancel_is_preserved(self):
        console = app.Console()
        started, release = threading.Event(), threading.Event()
        def operation():
            started.set()
            release.wait(2)
            return 'done'
        console.submit('test', operation, prepare_home=True)
        started.wait(2)
        try:
            with self.assertRaises(ValueError):
                console.submit('second', operation)
            console.control.cancel_home()
            self.assertTrue(console.control._cancel.is_set())
        finally:
            release.set()

    def test_http_security_and_whitelist(self):
        server = app.ThreadingHTTPServer(('127.0.0.1', 0), app.Handler)
        server.console, server.token = app.Console(), 'test-token'
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        root = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(root + '/api/state') as response:
                self.assertEqual(json.load(response)['token'], 'test-token')
            request = Request(root + '/api/execute', data=b'{"id":"arbitrary"}', headers={'Content-Type':'application/json'})
            with self.assertRaises(HTTPError) as context:
                urlopen(request)
            self.assertEqual(context.exception.code, 403)
            request.add_header('Origin', root)
            request.add_header('X-Console-Token', 'test-token')
            with self.assertRaises(HTTPError) as context:
                urlopen(request)
            self.assertEqual(context.exception.code, 400)
            with urlopen(root + '/app.js') as response:
                self.assertIn(b'navigator.clipboard.writeText', response.read())
            headers = {'Content-Type': 'application/json', 'Origin': root, 'X-Console-Token': 'test-token'}
            with patch.object(server.console, 'submit') as submit, patch.object(server.console.control, 'execute', return_value=255) as execute:
                request = Request(root + '/api/gripper', data=b'{"side":"left","action":"open"}', headers=headers)
                with urlopen(request) as response:
                    self.assertEqual(response.status, 202)
                operation = submit.call_args.args[1]
                self.assertIn('255', operation())
                execute.assert_called_once_with('left', 'open')
            with patch.object(server.console, 'submit') as submit, patch('tools.service_gripper.recover', return_value=0) as recover, patch.object(server.console.control, 'execute') as standalone:
                request = Request(root + '/api/gripper', data=b'{"side":"right","action":"close","via_service":true}', headers=headers)
                with urlopen(request) as response:
                    self.assertEqual(response.status, 202)
                self.assertIn('0', submit.call_args.args[1]())
                recover.assert_called_once_with('right', 'close')
                standalone.assert_not_called()
            with patch.object(server.console, 'submit') as submit, patch.object(server.console.control, 'home', return_value=[0]*6) as home:
                request = Request(root + '/api/home', data=b'{"side":"right"}', headers=headers)
                with urlopen(request) as response:
                    self.assertEqual(response.status, 202)
                self.assertTrue(submit.call_args.kwargs['prepare_home'])
                submit.call_args.args[1]()
                home.assert_called_once_with('right')
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == '__main__':
    unittest.main()
