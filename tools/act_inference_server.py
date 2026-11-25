"""Loopback ACT inference on the laptop or GPU host, without robot interfaces."""
import argparse
import base64
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_act_host import make_policy


class Predictor:
    def __init__(self, run, checkpoint):
        self.run = run
        self.checkpoint = checkpoint
        # A local bundle carries the fixed ACT source and ResNet cache for offline startup.
        if (run / 'act').is_dir():
            sys.path.insert(0, str(run / 'act'))
            sys.path.insert(0, str(run / 'act/detr'))
            os.environ['TORCH_HOME'] = str(run / 'cache')
        settings = json.loads((run / 'config.json').read_text())
        self.image_preprocess = settings.get('image_preprocess', {})
        if settings['policy']['camera_names'] != ['top', 'left']:
            raise ValueError('Expected top,left model')
        self.stats = {k: np.asarray(v, dtype=np.float32) for k, v in
                      json.loads((run / 'normalization.json').read_text()).items()}
        self.policy = make_policy(settings['policy'], run)
        saved = torch.load(run / checkpoint, map_location='cpu', weights_only=False)
        self.policy.load_state_dict(saved['policy'])
        self.step = saved['step']
        self.policy.eval()
        self.lock = threading.Lock()
        print('ACT_INFERENCE_WARMUP: preparing CUDA kernels', flush=True)
        with torch.inference_mode():
            self.policy(torch.zeros(1,14,device='cuda'),
                        torch.zeros(1,2,3,480,640,device='cuda'))
            torch.cuda.synchronize()

    def infer(self, payload):
        qpos = np.asarray(payload['qpos'], dtype=np.float32)
        if qpos.shape != (14,) or not np.isfinite(qpos).all():
            raise ValueError('Expected finite 14-D qpos')
        if payload['camera_names'] != ['top', 'left']:
            raise ValueError('Camera order mismatch')
        images = []
        for encoded in payload['rgb_images']:
            data = base64.b64decode(encoded, validate=True)
            if len(data) != 480*640*3:
                raise ValueError('Expected 480x640 image')
            images.append(np.frombuffer(data, np.uint8).reshape(480,640,3))
        if len(images) != 2:
            raise ValueError('Two camera views required')
        mask_x = self.image_preprocess.get('top_mask_x', 640)
        if not isinstance(mask_x, int) or not 1 <= mask_x <= 640:
            raise ValueError('Invalid model top_mask_x')
        if mask_x < 640:
            images[0] = images[0].copy()
            images[0][:, mask_x:] = 0
        tensor = torch.from_numpy(np.stack(images).transpose(0,3,1,2).copy()).float()[None] / 255
        state = torch.from_numpy((qpos-self.stats['qpos_mean'])/self.stats['qpos_std'])[None]
        with self.lock, torch.inference_mode():
            action = self.policy(state.cuda(), tensor.cuda()).cpu().numpy()[0]
        action = action * self.stats['action_std'] + self.stats['action_mean']
        if action.shape != (30,14) or not np.isfinite(action).all():
            raise ValueError('Invalid model prediction')
        return {'request_id': payload['request_id'], 'action': action.tolist(), 'step': self.step}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', choices=['best.pt', 'last.pt'], default='last.pt')
    parser.add_argument('--port', type=int, default=8787)
    args = parser.parse_args()
    predictor = Predictor(args.run, args.checkpoint)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != '/health':
                return self.respond(404, {'error':'Unknown path'})
            self.respond(200, {'protocol':1, 'camera_names':['top','left'],
                              'image_encoding':'rgb8',
                              'checkpoint':predictor.checkpoint, 'step':predictor.step,
                              'run':str(predictor.run)})

        def do_POST(self):
            try:
                if self.path != '/predict' or self.headers.get('Content-Type') != 'application/json':
                    return self.respond(400, {'error':'Expected JSON /predict'})
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 8_000_000:
                    raise ValueError('Request too large or empty')
                self.connection.settimeout(3)
                payload = json.loads(self.rfile.read(length))
                self.respond(200, predictor.infer(payload))
            except Exception as error:
                self.respond(400, {'error':str(error)})

        def log_message(self, *_):
            pass

    print(f'ACT_INFERENCE_READY 127.0.0.1:{args.port} step={predictor.step}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
