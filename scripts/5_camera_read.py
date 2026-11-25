"""Preview configured cameras; no reset, robot connection or servo commands."""
import argparse
import time
import cv2
import numpy as np
from dobot_control.cameras.settings import CameraSettings, load_camera_settings
from dobot_control.cameras.realsense_camera import RealSenseCamera


def list_devices():
    import pyrealsense2 as rs
    devices = []
    for device in rs.context().query_devices():
        def info(key):
            return device.get_info(key) if device.supports(key) else 'unknown'
        devices.append(dict(model=info(rs.camera_info.name),
                            serial=info(rs.camera_info.serial_number),
                            usb=info(rs.camera_info.usb_type_descriptor)))
    return devices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=0,
                        help="Stop after N frame sets; 0 previews until Q/Esc/Ctrl+C")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument('--list-devices', action='store_true')
    parser.add_argument('--serial', help='Preview one USB serial without opening configured cameras')
    args = parser.parse_args()
    if args.list_devices:
        for device in list_devices():
            print(device)
        return
    if args.frames < 0 or (args.no_display and args.frames == 0):
        parser.error("Headless checks require --frames > 0")
    settings = (CameraSettings({'device': args.serial}, 15, {'device': False})
                if args.serial else load_camera_settings())
    cameras = {}
    try:
        for name, serial in settings.serials.items():
            depth_options = {} if settings.depth_enabled else {'enable_depth': False}
            cameras[name] = RealSenseCamera(device_id=serial, flip=settings.flips[name], fps=settings.fps,
                                           **depth_options)
        count, begin = 0, time.monotonic()
        while args.frames == 0 or count < args.frames:
            images = []
            for name, camera in cameras.items():
                rgb, depth = camera.read()
                depth_ok = depth is not None and depth.shape == (480, 640, 1) if settings.depth_enabled else depth is None
                if rgb.shape != (480, 640, 3) or not depth_ok:
                    raise RuntimeError(f"Unexpected frame shapes for {name}: {rgb.shape}, {getattr(depth, 'shape', None)}")
                bgr = rgb[:, :, ::-1].copy()
                cv2.putText(bgr, f'{name} {settings.serials[name]}', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,0), 1)
                images.append(bgr)
            count += 1
            if not args.no_display:
                cv2.imshow(" / ".join(cameras), np.concatenate(images, axis=1))
                if cv2.waitKey(1) & 0xff in (27, ord("q")):
                    break
        print(f"PASS: {list(cameras)}; requested={settings.fps}fps; framesets={count}; elapsed={time.monotonic()-begin:.3f}s")
    finally:
        for camera in cameras.values():
            camera.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
