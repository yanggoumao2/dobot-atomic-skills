"""Preview configured cameras; no reset, robot connection or servo commands."""
import argparse
import time
import cv2
import numpy as np
from dobot_control.cameras.settings import load_camera_settings
from dobot_control.cameras.realsense_camera import RealSenseCamera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=0,
                        help="Stop after N frame sets; 0 previews until Q/Esc/Ctrl+C")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    if args.frames < 0 or (args.no_display and args.frames == 0):
        parser.error("Headless checks require --frames > 0")
    settings = load_camera_settings()
    cameras = {}
    try:
        for name, serial in settings.serials.items():
            cameras[name] = RealSenseCamera(device_id=serial, flip=name != "left", fps=settings.fps)
        count, begin = 0, time.monotonic()
        while args.frames == 0 or count < args.frames:
            images = []
            for name, camera in cameras.items():
                rgb, depth = camera.read()
                if rgb.shape != (480, 640, 3) or depth.shape != (480, 640, 1):
                    raise RuntimeError(f"Unexpected frame shapes for {name}: {rgb.shape}, {depth.shape}")
                images.append(rgb[:, :, ::-1])
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
