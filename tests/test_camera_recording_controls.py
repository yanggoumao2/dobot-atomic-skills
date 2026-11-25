import importlib
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from dobot_control.cameras.settings import CameraSettings
from experiments import run_control as control


class CameraTests(unittest.TestCase):
    @patch("experiments.run_control.validate_hardware_config", new=lambda: None)
    def test_preview_initialization_precedes_master_construction(self):
        order = []
        settings = CameraSettings({'top':'123'},15)
        def start(settings):
            order.append('cameras')
            return {}, threading.Event(), threading.Event(), Mock()
        def hands():
            order.append('hands')
            raise RuntimeError('stop before constructing hardware')
        with patch.object(control,'mk_dir'), patch.object(control,'BoundedAsyncWriter'), patch.object(control.atexit,'register'), patch.object(control,'load_camera_settings',return_value=settings), patch.object(control,'start_configured_cameras',side_effect=start), patch.object(control,'initialize_camera_preview',side_effect=lambda indices:order.append('preview')), patch.object(control,'load_ini_data_hands',side_effect=hands), patch.dict('os.environ',{'DOBOT_ACT_RECORDING':'0'}):
            with self.assertRaisesRegex(RuntimeError,'stop before constructing hardware'):
                control.main(control.Args(show_img=True))
        self.assertEqual(order,['cameras','preview','hands'])

    def test_preview_warmup_draws_same_window_and_all_views(self):
        with patch.object(control.cv2,'namedWindow') as window, patch.object(control.cv2,'imshow') as show, patch.object(control.cv2,'waitKey') as wait:
            control.initialize_camera_preview([0,1,2])
            window.assert_called_once_with('0',cv2.WINDOW_AUTOSIZE)
            self.assertEqual(show.call_args.args[0],'0')
            self.assertEqual(show.call_args.args[1].shape,(480,1920,3))
            self.assertEqual(show.call_args.args[1].dtype,np.uint8)
            wait.assert_called_once_with(1)

    def test_rgb_only_driver_returns_no_depth_and_keeps_rotation(self):
        from dobot_control.cameras.realsense_camera import RealSenseCamera
        camera = RealSenseCamera.__new__(RealSenseCamera)
        camera._flip = True
        camera._enable_depth = False
        bgr = np.arange(4*6*3, dtype=np.uint8).reshape(4,6,3)
        frames = Mock()
        frames.get_color_frame.return_value.get_data.return_value = bgr
        camera._pipeline = Mock()
        camera._pipeline.wait_for_frames.return_value = frames
        rgb, depth = camera.read()
        self.assertIsNone(depth)
        frames.get_depth_frame.assert_not_called()
        np.testing.assert_array_equal(rgb, bgr[::-1,::-1,::-1])

    def test_depth_stream_switch_preserves_driver_default(self):
        import pyrealsense2 as rs
        from dobot_control.cameras.realsense_camera import RealSenseCamera
        for enabled in (False, True):
            with self.subTest(enabled=enabled), patch.object(rs,'pipeline'), patch.object(rs,'config') as config, patch.object(RealSenseCamera,'read'):
                camera = RealSenseCamera('123', fps=15, enable_depth=enabled)
                streams = [c.args[0] for c in config.return_value.enable_stream.call_args_list]
                self.assertIn(rs.stream.color, streams)
                self.assertEqual(rs.stream.depth in streams, enabled)
                camera.close()

    def test_rgb_only_preview_uses_configured_switch(self):
        preview = importlib.import_module('scripts.5_camera_read')
        settings = CameraSettings({'top':'123'},15,depth_enabled=False)
        camera = Mock()
        camera.read.return_value = (np.zeros((480,640,3),np.uint8),None)
        with patch.object(preview,'load_camera_settings',return_value=settings), patch.object(preview,'RealSenseCamera',return_value=camera) as create, patch('sys.argv',['camera','--frames','1','--no-display']):
            preview.main()
            create.assert_called_once_with(device_id='123',flip=True,fps=15,enable_depth=False)
            camera.close.assert_called_once()

    def test_full_frame_corners_survive_recording_encoding(self):
        rgb = np.zeros((480,640,3), np.uint8)
        colors = [(240,20,20),(20,240,20),(20,20,240),(240,240,20)]
        rgb[:64,:64],rgb[:64,-64:],rgb[-64:,:64],rgb[-64:,-64:] = colors
        stop,ready = threading.Event(),threading.Event()
        def read():
            stop.set()
            return rgb, np.zeros((480,640,1))
        control.run_thread_cam(SimpleNamespace(read=read),0,ready,threading.Event(),stop)
        image=cv2.imdecode(np.frombuffer(control.jpeg_list[0],np.uint8),cv2.IMREAD_COLOR)
        self.assertEqual(image.shape,(480,640,3))
        for (y,x),color in zip([(20,20),(20,620),(460,20),(460,620)],colors):
            np.testing.assert_allclose(image[y,x],color[::-1],atol=15)
        self.assertTrue(ready.is_set())

    def test_driver_preserves_existing_180_degree_rotation(self):
        from dobot_control.cameras.realsense_camera import RealSenseCamera
        camera=RealSenseCamera.__new__(RealSenseCamera)
        camera._flip=True
        bgr=np.arange(4*6*3,dtype=np.uint8).reshape(4,6,3)
        depth=np.arange(24,dtype=np.uint16).reshape(4,6)
        frames=Mock()
        frames.get_color_frame.return_value.get_data.return_value=bgr
        frames.get_depth_frame.return_value.get_data.return_value=depth
        camera._pipeline=Mock()
        camera._pipeline.wait_for_frames.return_value=frames
        rgb,z=camera.read()
        np.testing.assert_array_equal(rgb,bgr[::-1,::-1,::-1])
        np.testing.assert_array_equal(z[:,:,0],depth[::-1,::-1])

    def test_collector_keeps_top_flip_true(self):
        class ReadyThread:
            def __init__(self,target,args,daemon): self.ready=args[2]
            def start(self): self.ready.set()
            def is_alive(self):return True
            def join(self,timeout):pass
        settings=CameraSettings({'top':'123'},15)
        with patch.object(control,'RealSenseCamera') as camera, patch.object(control.threading,'Thread',ReadyThread):
            _,_,_,close=control.start_configured_cameras(settings)
            camera.assert_called_once_with(device_id='123',flip=True,fps=15)
            close()

    def test_preview_keeps_same_top_flip_true(self):
        preview=importlib.import_module('scripts.5_camera_read')
        camera=Mock()
        camera.read.return_value=(np.zeros((480,640,3),np.uint8),np.zeros((480,640,1),np.uint16))
        with patch.object(preview,'load_camera_settings',return_value=CameraSettings({'top':'123'},15)), patch.object(preview,'RealSenseCamera',return_value=camera) as create, patch('sys.argv',['camera','--frames','1','--no-display']):
            preview.main()
            create.assert_called_once_with(device_id='123',flip=True,fps=15)
            camera.close.assert_called_once()


class SharedRecordingButtonTests(unittest.TestCase):
    def run_keys(self,frames,servo=True):
        stop,ready=threading.Event(),threading.Event()
        state=np.zeros((2,3),int);state[:,1]=int(servo)
        agent=Mock()
        iterator=iter(enumerate(frames))
        def keys():
            index,value=next(iterator)
            if index==len(frames)-1:stop.set()
            return value.copy()
        agent.get_keys.side_effect=keys
        with patch.object(control,'what_to_do',state),patch.object(control,'runtime_log') as log,patch.object(control,'is_falling',np.array([0])),patch.object(control,'using_sensor_protection',False):
            control.button_monitor_realtime(agent,stop,ready)
            return [call.kwargs['action'] for call in log.call_args_list if call.args[0]=='BUTTON_B'],state.copy()

    @staticmethod
    def frame(*pressed):
        result=np.ones((2,3),int)
        for hand in pressed:result[hand,1]=0
        return result

    def test_right_start_left_stop_right_restart(self):
        f=self.frame
        actions,state=self.run_keys([f(),f(1),f(),f(0),f(),f(1),f()])
        self.assertEqual(actions,['record_start','record_stop','record_start'])
        self.assertEqual(state[0,2],1)

    def test_simultaneous_release_toggles_once(self):
        f=self.frame
        actions,state=self.run_keys([f(),f(0,1),f()])
        self.assertEqual(actions,['record_start'])

    def test_no_follow_ignores_recording(self):
        f=self.frame
        actions,state=self.run_keys([f(),f(1),f()],servo=False)
        self.assertEqual(actions,['record_ignored'])
        self.assertEqual(state[0,2],0)

    def test_initial_held_button_release_does_not_toggle(self):
        actions,state=self.run_keys([self.frame(1),self.frame()])
        self.assertEqual(actions,[])


if __name__=='__main__':unittest.main()
