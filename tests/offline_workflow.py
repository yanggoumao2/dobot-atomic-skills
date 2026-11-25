"""No hardware: HTTP console + real loopback RPC + synthetic recording/data CLI."""
import configparser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from unittest.mock import patch
from urllib.request import Request, ProxyHandler, build_opener
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def cli(*args):
    subprocess.run([sys.executable, '-m', *map(str, args)], cwd=ROOT, check=True, timeout=60)

def run(directory):
    config = configparser.ConfigParser()
    config.read(ROOT / 'config/linux.example.ini', encoding='utf-8')
    for i, section in enumerate(('HAND_LEFT','HAND_RIGHT','GRIPPER_LEFT','GRIPPER_RIGHT')):
        config[section]['port'] = f'/dev/serial/by-id/offline-{i}'
    for section in ('HAND_LEFT','HAND_RIGHT'): config[section]['absolute_mapping_verified']='1'
    config['CAMERA']['top']='111';config['CAMERA']['left']='222'
    path = directory / 'linux.ini'
    with path.open('w',encoding='utf-8') as file: config.write(file)
    os.environ['DOBOT_CONFIG'] = str(path)
    from dobot_control.config import validate_hardware_config
    validate_hardware_config()
    from tools import local_console as app
    from tools.linux_console import commands
    from dobot_control.robots.robot_node import ZMQServerRobot, ZMQClientRobot
    import zmq
    # Local RPC passes a 14-D command to a memory-only robot and reads it back.
    class Robot:
        def __init__(self): self.q=np.zeros(14)
        def num_dofs(self): return 14
        def get_joint_state(self): return self.q.copy()
        def command_joint_state(self, joint_state, **kwargs): self.q=np.asarray(joint_state).copy()
        def get_observations(self): return dict(joint_positions=self.q.copy())
    server = ZMQServerRobot(Robot(), port=0)
    port=int(server._socket.getsockopt(zmq.LAST_ENDPOINT).decode().rsplit(':',1)[1])
    worker=threading.Thread(target=server.serve);worker.start()
    client=ZMQClientRobot(port=port)
    client._socket.setsockopt(zmq.RCVTIMEO,3000)
    client._socket.setsockopt(zmq.SNDTIMEO,3000)
    try:
        assert client.num_dofs()==14
        target=np.arange(14)/100
        client.command_joint_state(target,[1,1],[1,1])
        np.testing.assert_allclose(client.get_observations()['joint_positions'],target)
    finally:
        client.close();client._context.term();server.stop();worker.join(5)
        assert not worker.is_alive()
        server._socket.close();server._context.term()
    print('PASS: A/B loopback command and feedback',flush=True)
    # Real HTTP page, authentication, platform metadata and every command card.
    http=app.ThreadingHTTPServer(('127.0.0.1',0),app.Handler)
    http.console=app.Console();http.token='offline-token'
    thread=threading.Thread(target=http.serve_forever);thread.start()
    origin=f'http://127.0.0.1:{http.server_port}'
    opener=build_opener(ProxyHandler({}))
    try:
        with patch.object(app.sys,'platform','linux'):
            with opener.open(origin+'/api/state') as response: state=json.load(response)
            assert state['shell']=='Bash' and state['endpoints']['left']['port'].startswith('/dev/')
            request=Request(origin+'/api/commands',data=json.dumps(dict(project='offline_flow',episode='episode_001')).encode(),headers={'Origin':origin,'X-Console-Token':'offline-token','Content-Type':'application/json'})
            with opener.open(request) as response: cards=json.load(response)
        assert len(cards)==9
        for card in cards:
            assert 'powershell' not in card['code'].lower()
            if os.name=='posix': subprocess.run(['bash','-n'],input=card['code'],text=True,check=True)
        with opener.open(origin+'/app.js') as response: script=response.read().decode()
        assert 'E:\\Dobot' not in script and 'runtime.shell' in script and 'runtime.data_root' in script
        with opener.open(origin) as response: html=response.read().decode()
        assert 'id="shellHint"' in html and 'id="terminalHint"' in html
    finally:
        http.shutdown();http.server_close();thread.join()
    print('PASS: HTTP console, Linux paths, all 9 command cards and shell syntax',flush=True)
    # Actual asynchronous recorder and CLI stages, synthetic RGB/state only.
    from test_act_dataset import fixture
    from dobot_control.act_dataset import load_act_window
    raw=directory/'raw'
    for name in ('episode_001','episode_002'):
        ep=raw/name
        fixture(ep,names=('top','left'))
        cli('dobot_control.act_dataset','check',ep)
        cli('dobot_control.act_dataset','preview',ep,'--output',ep/'preview.mp4')
        cli('dobot_control.act_dataset','label',ep,'--status','success','--note','synthetic offline flow')
    dataset=directory/'act'
    cli('dobot_control.act_dataset','convert',raw,'--output',dataset,'--fps','15')
    cli('tools.prepare_training',dataset)
    images,qpos,action,mask=load_act_window(dataset/'episode_0.hdf5',0,30)
    assert images.shape==(2,480,640,3) and qpos.shape==(14,) and action.shape==(30,14)
    assert mask[-1]
    stats=json.loads((dataset/'train_normalization_stats.json').read_text())
    assert len(stats['qpos_mean'])==14
    print('PASS: recording -> quality -> preview -> label -> HDF5 -> train-only statistics -> ACT input',flush=True)

if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='dobot-flow-') as directory: run(Path(directory))
    print('COMPLETE OFFLINE WORKFLOW PASSED',flush=True)
