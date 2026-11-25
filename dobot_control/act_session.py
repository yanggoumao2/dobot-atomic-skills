"""A-service ownership for the ACT client, without opening hardware."""
import secrets
import time

import numpy as np


class ACTSession:
    def __init__(self):
        self.token = None

    def begin(self):
        if self.token is not None:
            raise RuntimeError('ACT client already owns motion; close that client first')
        self.token = secrets.token_hex(24)
        return {'token': self.token, 'protocol': 1}

    def end(self, token):
        if not secrets.compare_digest(str(token), str(self.token)):
            raise RuntimeError('ACT session token mismatch')
        self.token = None
        return {'released': True}

    def authorize(self, method, args):
        args = dict(args)
        token = args.pop('_act_token', None)
        deadline = args.pop('_act_deadline', None)
        if self.token is None:
            if token is not None:
                raise RuntimeError('ACT session was lost; restart the trial')
            return args
        if method not in ('command_joint_state', 'set_do_status'):
            return args
        if method != 'command_joint_state' or not secrets.compare_digest(str(token), self.token):
            raise RuntimeError('Motion is owned by ACT; other control writes are blocked')
        if deadline is None or not np.isfinite(deadline) or time.perf_counter() > deadline:
            raise RuntimeError('Expired ACT command')
        joints = np.asarray(args['joint_state'], dtype=float)
        arms = np.asarray(args['arm_flags'])
        grippers = np.asarray(args['gripper_flags'])
        if joints.shape != (14,) or not np.isfinite(joints).all():
            raise ValueError('ACT requires a finite 14-D command')
        if not np.array_equal(arms, [0, 1]) or grippers.shape != (2,) or grippers[0] or grippers[1] not in (0, 1):
            raise ValueError('ACT can command only physical LEFT .2')
        return args
