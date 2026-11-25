import json
import contextlib
import io
import pickle
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import h5py
import numpy as np

from dobot_control.act_recording import CameraSnapshots, GripperFeedback, enrich_observation
from dobot_control.act_dataset import inspect_episode, convert_episode, load_act_window
from dobot_control.robots.dobot import DobotRobot
from dobot_control.env import RobotEnv
from scripts.act_training_adapter import training_stats


def fixture(ep, count=24, names=("top",), metadata=None):
    from experiments.run_control import RecordingSample, write_recording_sample
    from dobot_control.async_writer import BoundedAsyncWriter
    writer = BoundedAsyncWriter(write_recording_sample, count + 2)
    for i in range(count):
        t = 10 + i / 30
        image = np.zeros((480, 640, 3), np.uint8)
        image[..., 2] = 200  # BGR red -> RGB red.
        image[0, 0, 0] = i
        jpeg = cv2.imencode('.jpg', image)[1].tobytes()
        q = np.arange(14, dtype=float) / 20 + i / 100
        a = q + .01
        row = dict(joint_positions=q, joint_velocities=np.zeros(14), _act_schema=2,
                   _act_command_time=t, _teleop_capture_monotonic_s=t, _act_state_time=np.array([t-.01, t-.01]),
                   _act_gripper_time=np.array([t-.02, t-.02]), _act_gripper_valid=np.array([True, True]),
                   _act_camera_time=np.array([t-.02]*len(names)), _act_camera_id=np.array([i//2]*len(names)),
                   _act_arm_flags=np.ones(2), _act_gripper_flags=np.ones(2), _act_send_ok=True,
                   _act_send_end=t+.01, _teleop_source_index=i)
        images = []
        for c, name in enumerate(names):
            view = np.zeros_like(image)
            view[..., (2-c) % 3] = 200
            images.append(cv2.imencode('.jpg', view)[1].tobytes())
        assert writer.submit(RecordingSample(str(ep), i, tuple(images), row, a, names, metadata))
    assert writer.submit(dict(episode_dir=str(ep), samples=count, closed=True))
    writer.close()
    meta = json.loads((ep/'episode.json').read_text())
    meta['status'] = 'success'
    (ep/'episode.json').write_text(json.dumps(meta))


class ACTDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ep = self.root/'raw'/'1'
        fixture(self.ep)

    def tearDown(self):
        self.tmp.cleanup()

    def mutate(self, fn, index=2):
        path = self.ep/'observation'/f'{index}.pkl'
        with path.open('rb') as f:
            row = pickle.load(f)
        fn(row)
        with path.open('wb') as f:
            pickle.dump(row, f)

    def test_real_writer_converter_loader_roundtrip(self):
        target = self.root/'act'/'episode_0.hdf5'
        self.assertTrue(inspect_episode(self.ep)[0]['passed'])
        convert_episode(self.ep, target)
        image, q, action, mask = load_act_window(target, 0, 30)
        self.assertEqual(image.shape, (1,480,640,3))
        self.assertGreater(image[0,10,10,0], 190)
        self.assertLess(image[0,10,10,2], 5)
        np.testing.assert_allclose(action[0], q+.01, atol=1e-6)
        self.assertTrue(mask[-1])
        with h5py.File(target) as f:
            np.testing.assert_allclose(np.diff(f['timestamp']), 1/15)
            self.assertFalse(f.attrs['compress'])
            meta=json.loads(f.attrs['metadata_json'])
            self.assertIsNone(meta['top_crop'])
            self.assertTrue(meta['top_flip'])
            self.assertEqual(meta['hardware_layout_version'], 2)
            self.assertEqual(meta['gripper_ports'], {'192.168.5.1': 'COM4/ID22',
                                                   '192.168.5.2': 'COM3/ID21'})
            self.assertIn('0=fully closed, 1=fully open',meta['gripper'])
            np.testing.assert_array_equal(f['source_index'][:3], [0,2,4])

    def test_reject_stale_feedback(self):
        self.mutate(lambda r:r.update(_act_gripper_time=np.zeros(2)))
        self.assertFalse(inspect_episode(self.ep)[0]['passed'])

    def test_reject_future_camera(self):
        self.mutate(lambda r:r.update(_act_camera_time=np.array([100.])))
        self.assertFalse(inspect_episode(self.ep)[0]['passed'])

    def test_reject_nonfinite(self):
        self.mutate(lambda r:r['joint_positions'].__setitem__(0, np.nan))
        self.assertFalse(inspect_episode(self.ep)[0]['passed'])

    def test_reject_disabled_arm(self):
        self.mutate(lambda r:r.update(_act_arm_flags=np.array([0,1])))
        self.assertFalse(inspect_episode(self.ep)[0]['passed'])

    def test_reject_failed_send(self):
        self.mutate(lambda r:r.update(_act_send_ok=False))
        self.assertFalse(inspect_episode(self.ep)[0]['passed'])

    def test_reject_unclosed_episode(self):
        (self.ep/'closed.json').unlink()
        self.assertFalse(inspect_episode(self.ep)[0]['passed'])

    def test_reject_missing_image_without_deleting_raw(self):
        (self.ep/'topImg'/'2.jpg').unlink()
        self.assertFalse(inspect_episode(self.ep)[0]['passed'])
        self.assertTrue((self.ep/'observation'/'2.pkl').exists())

    def test_failure_label_excluded(self):
        path = self.ep/'episode.json'
        meta = json.loads(path.read_text()); meta['status']='failure'
        path.write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            convert_episode(self.ep,self.root/'fail.hdf5')

    def test_destination_never_overwritten(self):
        target=self.root/'out.hdf5'
        target.write_bytes(b'keep')
        with self.assertRaises(FileExistsError): convert_episode(self.ep,target)
        self.assertEqual(target.read_bytes(),b'keep')

    def test_train_stats_exclude_other_episode(self):
        out=self.root/'act'
        convert_episode(self.ep,out/'episode_0.hdf5')
        convert_episode(self.ep,out/'episode_1.hdf5')
        with h5py.File(out/'episode_1.hdf5','a') as f: f['action'][...]=999
        stats=training_stats(out,[0])
        self.assertLess(stats['action_mean'].max(),1.)
        self.assertTrue((stats['action_std']>=.01).all())

    def test_window_invalid_start(self):
        out=self.root/'out.hdf5';convert_episode(self.ep,out)
        with self.assertRaises(ValueError):load_act_window(out,999)

    def test_cli_check_label_export_preview(self):
        from dobot_control.act_dataset import main
        fixture(self.root/'raw'/'2')
        for arguments in [
            ['check', str(self.ep)],
            ['label', str(self.ep), '--status','success','--note','functional test'],
            ['preview',str(self.ep),'--output',str(self.root/'preview.mp4')],
            ['convert',str(self.root/'raw'),'--output',str(self.root/'export')],
        ]:
            with patch('sys.argv',['act_dataset',*arguments]), contextlib.redirect_stdout(io.StringIO()):
                main()
        self.assertTrue((self.root/'preview.mp4').stat().st_size>0)
        split=json.loads((self.root/'export'/'splits.json').read_text())
        self.assertEqual(len(split['train']),1)
        self.assertEqual(len(split['validation']),1)
        self.assertTrue((self.root/'export'/'quality_report.html').exists())

    def test_cli_rejects_mixed_image_geometry(self):
        from dobot_control.act_dataset import main
        ep2=self.root/'raw'/'2'
        fixture(ep2)
        path=ep2/'episode.json'
        meta=json.loads(path.read_text())
        meta['top_crop']=[150,420,220,480]
        path.write_text(json.dumps(meta))
        with patch('sys.argv',['act_dataset','convert',str(self.root/'raw'),'--output',str(self.root/'mixed')]), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'image geometry'):main()

    def test_reject_aborted_even_if_marked_success(self):
        path=self.ep/'closed.json'
        path.write_text(json.dumps(dict(closed=True,samples=24,aborted=True)))
        self.assertFalse(inspect_episode(self.ep)[0]['passed'])

    def test_cli_rejects_mixed_gripper_bindings(self):
        from dobot_control.act_dataset import main
        ep2 = self.root/'raw'/'2'
        fixture(ep2)
        path = ep2/'episode.json'
        meta = json.loads(path.read_text())
        meta.pop('hardware_layout_version')
        meta['gripper_ports'] = {'192.168.5.1': 'COM3/ID21', '192.168.5.2': 'COM4/ID22'}
        path.write_text(json.dumps(meta))
        with patch('sys.argv', ['act_dataset', 'convert', str(self.root/'raw'),
                                '--output', str(self.root/'mixed-binding')]), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                main()

    def test_reject_source_time_jump(self):
        self.mutate(lambda r:r.update(_act_command_time=100.))
        self.assertFalse(inspect_episode(self.ep)[0]['passed'])

    def test_async_writer_failure_reaches_caller(self):
        from dobot_control.async_writer import BoundedAsyncWriter
        failed=threading.Event()
        def write(item):
            failed.set()
            raise OSError('disk full')
        writer=BoundedAsyncWriter(write,2)
        writer.submit('sample')
        self.assertTrue(failed.wait(1))
        with self.assertRaisesRegex(RuntimeError,'disk full'):
            writer.close()


class ACTTelemetryTests(unittest.TestCase):
    def test_camera_snapshot_is_immutable_pair(self):
        cam=CameraSnapshots();cam.publish(0,b'first',1.)
        old=cam.snapshot([0]);cam.publish(0,b'next',2.)
        self.assertEqual(old,((b'first',1.,1),))
        self.assertEqual(cam.snapshot([0]),((b'next',2.,2),))

    def test_enrichment_freezes_measured_values(self):
        obs={'joint_positions':np.zeros(14),'dataset_joint_positions':np.ones(14)}
        row=enrich_observation(obs,((b'jpeg',1.,2),),[1,1],[1,1],2.)
        obs['dataset_joint_positions'][:]=0
        np.testing.assert_array_equal(row['joint_positions'],np.ones(14))

    def test_feedback_cache_does_not_replace_legacy_control_state(self):
        robot=DobotRobot.__new__(DobotRobot);robot.robot_is_err=False
        robot.get_joint_state=Mock(return_value=np.ones(7))
        robot._act_feedback=Mock();robot._act_feedback.snapshot.return_value=(.2,10.,True)
        obs=RobotEnv(robot).get_obs()
        self.assertEqual(obs['joint_positions'][-1],1.)
        self.assertEqual(obs['dataset_joint_positions'][-1],.2)
        self.assertIn('_act_gripper_time',obs)

    def test_cache_samples_scale_and_stops(self):
        stop=threading.Event();grip=Mock();grip.command_lock=threading.RLock()
        grip.get_current_position.return_value=51
        cache=GripperFeedback(grip,stop)
        deadline=time.monotonic()+1
        while not cache.snapshot()[2] and time.monotonic()<deadline:time.sleep(.001)
        stop.set();cache.thread.join(1)
        value,t,valid=cache.snapshot()
        self.assertAlmostEqual(value,.2);self.assertTrue(valid);self.assertGreater(t,0)
        self.assertFalse(cache.thread.is_alive())

    def test_failed_feedback_never_becomes_command_value(self):
        stop=threading.Event();grip=Mock();grip.command_lock=threading.RLock()
        grip.get_current_position.side_effect=OSError('serial unavailable')
        cache=GripperFeedback(grip,stop)
        time.sleep(.02);stop.set();cache.thread.join(1)
        value,t,valid=cache.snapshot()
        self.assertTrue(np.isnan(value));self.assertFalse(valid);self.assertEqual(t,0)



class ThreeCameraTests(unittest.TestCase):
    def setUp(self):
        from dobot_control.act_task import load_act_task
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ep = self.root/'raw'/'1'
        self.meta = dict(load_act_task(), task_protocol_version=1,
                         required_camera_names=["top","left","right"],
                         required_arm_flags=[True,True],required_gripper_flags=[True,True])
        fixture(self.ep, names=('top','left','right'), metadata=self.meta)

    def tearDown(self):
        self.tmp.cleanup()

    def mutate(self, fn):
        for path in (self.ep/'observation').glob('*.pkl'):
            with path.open('rb') as f: row=pickle.load(f)
            fn(row)
            with path.open('wb') as f: pickle.dump(row,f)

    def test_three_view_roundtrip_and_both_arm_values(self):
        target=self.root/'episode.hdf5'
        report=convert_episode(self.ep,target)
        self.assertTrue(report['passed'])
        images,q,actions,mask=load_act_window(target)
        self.assertEqual(images.shape,(3,480,640,3))
        for i in range(3):
            self.assertGreater(images[i,100,100,i],190)
        np.testing.assert_allclose(actions[0],q+.01,atol=1e-6)
        self.assertEqual(q.shape,(14,))
        self.assertGreater(q[1],0)  # right side is recorded, not zeroed
        self.assertIn('camera_skew_p50_p95_max',report['telemetry'])

    def test_camera_faults_and_enable_contract(self):
        from dobot_control.act_recording import validate_camera_packet
        meta=dict(self.meta,camera_names=['top','left','right'])
        for frames in [((b'x',9.99,1),)*2,
                       ((b'',9.99,1),)*3,
                       ((b'x',10.01,1),)*3,
                       ((b'x',9.7,1),)*3,
                       ((b'x',9.85,1),(b'x',9.99,1),(b'x',9.99,1))]:
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                validate_camera_packet(meta,frames,10.)
        self.mutate(lambda r:r.update(_act_arm_flags=np.array([0,1])))
        self.assertFalse(inspect_episode(self.ep)[0]['passed'])

    def test_frozen_camera_rejected(self):
        self.mutate(lambda r:r.update(_act_camera_id=np.array([0,0,0])))
        self.assertFalse(inspect_episode(self.ep)[0]['passed'])

    def test_missing_image_and_aborted_rejected(self):
        (self.ep/'leftImg'/'0.jpg').unlink()
        self.assertFalse(inspect_episode(self.ep)[0]['passed'])
        closed=json.loads((self.ep/'closed.json').read_text())
        closed['aborted']=True
        (self.ep/'closed.json').write_text(json.dumps(closed))
        self.assertFalse(inspect_episode(self.ep)[0]['passed'])

    def test_row_camera_order_rejected(self):
        self.mutate(lambda r:r.update(_teleop_camera_names=np.array(['left','top','right'])))
        self.assertFalse(inspect_episode(self.ep)[0]['passed'])

    def test_three_view_preview(self):
        from dobot_control.act_dataset import main
        writer=Mock();writer.isOpened.return_value=True
        with patch('sys.argv',['act','preview',str(self.ep),'--output',str(self.root/'preview.mp4')]), patch('cv2.VideoWriter',return_value=writer) as create:
            main()
        self.assertEqual(create.call_args.args[-1],(1920,480))
        self.assertEqual(writer.write.call_args.args[0].shape,(480,1920,3))
        writer.release.assert_called_once()

    def test_mixed_task_export_rejected(self):
        from dobot_control.act_dataset import main
        other=dict(self.meta,fixture='different_fixture')
        fixture(self.root/'raw'/'2',names=('top','left','right'),metadata=other)
        with patch('sys.argv',['act','convert',str(self.root/'raw'),'--output',str(self.root/'export')]):
            with self.assertRaisesRegex(ValueError,'incompatible'):
                main()

if __name__=='__main__':unittest.main()
