"""Exercise Linux branches with temporary configuration and simulated /proc."""
import configparser
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
from dobot_control import config, linux_processes

ROOT = Path(__file__).resolve().parents[1]
class LinuxPortability(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "linux.ini"
        self.c = configparser.ConfigParser()
        self.c.read(ROOT / "config/linux.example.ini", encoding="utf-8")
        self.env = patch.dict(os.environ, {"DOBOT_CONFIG": str(self.path)})
        self.env.start(); self.addCleanup(self.env.stop)
        self.platform = patch('sys.platform', 'linux')
        self.platform.start(); self.addCleanup(self.platform.stop)
        self.save()
    def save(self):
        with self.path.open('w', encoding='utf-8') as f: self.c.write(f)
    def fill(self):
        for index, section in enumerate(('HAND_LEFT','HAND_RIGHT','GRIPPER_LEFT','GRIPPER_RIGHT')):
            self.c[section]['port'] = f'/dev/serial/by-id/test-{index}'
        for section in ('HAND_LEFT','HAND_RIGHT'): self.c[section]['absolute_mapping_verified']='1'
        self.c['CAMERA']['top']='123';self.c['CAMERA']['left']='456';self.save()
    def test_blank_config_rejected_before_server_opens_devices(self):
        from experiments import launch_nodes
        with patch.object(launch_nodes, '_launch_owned_robot_server') as start:
            with self.assertRaises(ValueError): launch_nodes.launch_robot_server(launch_nodes.Args())
            start.assert_not_called()
    def test_linux_config_and_metadata_follow_device_bindings(self):
        self.fill(); config.validate_hardware_config()
        self.assertEqual(config.hardware_metadata()['gripper_ports']['192.168.5.2'], '/dev/serial/by-id/test-2/ID21')
        self.c['GRIPPER_LEFT']['port']=self.c['GRIPPER_RIGHT']['port'];self.save()
        with self.assertRaises(ValueError): config.validate_hardware_config()
    def test_linux_console_uses_current_python_and_config(self):
        from tools.local_console import commands
        entries={e['id']:e for e in commands('linux_test', 'episode_001')}
        self.assertIn('-m experiments.run_control',entries['start_b']['code'])
        self.assertIn('DOBOT_CONFIG=',entries['start_a']['code'])
        self.assertNotIn('powershell',entries['start_a']['code'].lower())
        self.assertIn('no-b',entries['home_service']['code'])
    def test_proc_detects_module_and_script_without_matching_shell(self):
        root=Path(self.temp.name)/'proc';root.mkdir()
        for pid,cmd in [(987654,'python3\0-m\0experiments.run_control\0'),(987655,'bash\0-c\0python run_control.py\0')]:
            p=root/str(pid);p.mkdir();(p/'cmdline').write_bytes(cmd.encode())
        self.assertEqual([p['pid'] for p in linux_processes.control_processes(root)], [987654])
    def test_recovery_refuses_active_linux_teleop(self):
        from dobot_control import gripper_recovery
        with patch.object(gripper_recovery.os, 'name', 'posix'), patch.object(linux_processes,'control_processes',return_value=[dict(pid=1,argv=['python','-m','experiments.run_control'])]):
            with self.assertRaises(RuntimeError): gripper_recovery.require_teleop_stopped()
    def test_dead_local_lock_archives_and_live_lock_refuses(self):
        path=Path(self.temp.name)/'device-owner';path.mkdir()
        (path/'owner.json').write_text(json.dumps(dict(host=socket.gethostname(),pid=12345)))
        with patch.object(linux_processes,'assert_stopped'), patch.object(linux_processes.os,'kill',return_value=None):
            with self.assertRaises(RuntimeError): linux_processes.archive_lock(path)
        with patch.object(linux_processes,'assert_stopped'), patch.object(linux_processes.os,'kill',side_effect=ProcessLookupError):
            archived=linux_processes.archive_lock(path)
        self.assertTrue(Path(archived,'owner.json').exists())
