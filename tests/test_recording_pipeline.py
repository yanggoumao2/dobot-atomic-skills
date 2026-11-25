import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from dobot_control.async_writer import BoundedAsyncWriter
from dobot_control.act_recording import RecordingCadence
from dobot_control.act_task import load_act_task
from dobot_control.act_dataset import inspect_episode, convert_episode, load_act_window
from tests.test_act_dataset import fixture


class RecordingPipelineTests(unittest.TestCase):
    @patch("experiments.run_control.validate_hardware_config", new=lambda: None)
    def test_act_default_queue_has_capacity_for_filesystem_stalls(self):
        from experiments import run_control as control
        from dobot_control.cameras.settings import CameraSettings
        settings = CameraSettings({'top':'123','left':'456'},15)
        camera_result = ({}, threading.Event(), threading.Event(), Mock())
        with patch.dict(os.environ, {'DOBOT_ACT_RECORDING':'1'}), patch.object(control,'mk_dir'), patch.object(control,'BoundedAsyncWriter') as writer, patch.object(control.atexit,'register'), patch.object(control,'load_camera_settings',return_value=settings), patch.object(control,'start_configured_cameras',return_value=camera_result), patch.object(control,'load_ini_data_hands',side_effect=RuntimeError('stop before hardware')):
            os.environ.pop('DOBOT_RECORD_QUEUE_SIZE',None)
            with self.assertRaisesRegex(RuntimeError,'stop before hardware'):
                control.main(control.Args())
            self.assertEqual(writer.call_args.kwargs['max_queue_size'],64)

    def test_15hz_sampling_does_not_follow_100hz_control_loop(self):
        cadence = RecordingCadence(15)
        times = []
        for tick in range(1000):
            now = tick / 100.
            if cadence.ready(now):
                times.append(now)
                cadence.advance(now)
        self.assertEqual(len(times),150)
        self.assertLessEqual(max(np.diff(times)),.071)
        cadence.reset()
        self.assertTrue(cadence.ready(20.))

    def test_delay_skips_catchup_burst(self):
        cadence = RecordingCadence(15)
        cadence.advance(0.)
        self.assertTrue(cadence.ready(1.))
        cadence.advance(1.)
        self.assertFalse(cadence.ready(1.001))

    def test_full_sample_queue_cannot_drop_or_reorder_finalization(self):
        entered, release = threading.Event(), threading.Event()
        output = []
        def write(item):
            if item == 'first':
                entered.set()
                release.wait(2)
            output.append(item)
        writer = BoundedAsyncWriter(write,1)
        try:
            self.assertTrue(writer.submit('first'))
            self.assertTrue(entered.wait(1))
            self.assertTrue(writer.submit('last'))
            self.assertFalse(writer.submit('overflow'))
            begin = time.monotonic()
            self.assertTrue(writer.submit_final('closed'))
            self.assertLess(time.monotonic()-begin,.1)
        finally:
            release.set()
            writer.close(2)
        self.assertEqual(output,['first','last','closed'])

    def test_failed_factory_returns_capacity(self):
        writer = BoundedAsyncWriter(lambda item: None,1)
        def fail():
            raise ValueError('fixture failure')
        try:
            with self.assertRaises(ValueError): writer.submit_factory(fail)
            self.assertTrue(writer.submit('good'))
        finally: writer.close()

    def test_two_camera_left_only_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep=Path(tmp)/'raw';meta=load_act_task()
            self.assertEqual(meta['required_camera_names'],['top','left'])
            fixture(ep,names=('top','left'),metadata=meta)
            import pickle
            for path in (ep/'observation').glob('*.pkl'):
                with path.open('rb') as f: row=pickle.load(f)
                row['_act_arm_flags']=np.array([0,1])
                row['_act_gripper_flags']=np.array([0,1])
                with path.open('wb') as f: pickle.dump(row,f)
            self.assertTrue(inspect_episode(ep)[0]['passed'])
            target=Path(tmp)/'act.hdf5';convert_episode(ep,target)
            images,qpos,action,mask=load_act_window(target)
            self.assertEqual(images.shape,(2,480,640,3))
            self.assertEqual(qpos.shape,(14,))
            np.testing.assert_allclose(action[0],qpos+.01,atol=1e-6)
