import os.path, stat
import datetime
import time
import cv2
import numpy as np
import serial.tools.list_ports as serial_stl
import subprocess
from pathlib import Path
import configparser


def time_print(st):
    current_time = time.strftime("%Y-%m-%d %H-%M-%S:", time.localtime())
    print(str(current_time) +
          str(datetime.datetime.now().strftime("%f")[:-3]) +
          str(st))

def free_limit_and_set_one(file_name):
    """Optional FTDI tuning. Permissions belong to the system administrator."""
    import warnings
    path = Path(file_name)
    if not path.exists():
        return
    try:
        if path.read_text().strip() != "1":
            path.write_text("1")
    except OSError as exc:
        warnings.warn(f"FTDI latency unchanged ({path}): {exc}; configure udev permissions")


def scan_port():
    return [p.device for p in serial_stl.comports() if p.vid is not None]


# make new dir
def mk_dir(path_dir):
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir, exist_ok=True)
        return True
    else:
        return False

# log maker
def log_write(file_name, data):
    log_path = os.path.dirname(__file__)+"/logs/"
    mk_dir(log_path)
    current_time = time.strftime("%Y-%m-%d %H-%M-%S:", time.localtime())
    with open(log_path+"log.txt", 'a') as f:
        f.writelines(str(current_time)+str(datetime.datetime.now().strftime("%f")[:-3])
                     + " [" + file_name.split("/")[-1] + "] "
                     + str(data))
        f.writelines("\n")
    f.close()


def mismatch_data_write(ppp, data):
    str_path = str(ppp)+"/pose.txt"
    if not os.path.exists(str_path):
        with open(str_path, 'w') as f:
            print("ok")
    with open(str_path, 'a') as f:
        f.writelines(str(data[6])+str(data[13]))
        f.writelines("\n")
    f.close()


def time_stamp():
    s1 = time.strftime("%H %M %S", time.localtime()).split(" ")
    s2 = datetime.datetime.now().strftime("%f")[:-3]
    sec = ((int(s1[0])*60+int(s1[1]))*60+int(s1[2]))*1000+int(s2)
    return sec


def gripper_cacheData_writein(writePath, data):
    with open(writePath,  'w') as f:
        f.writelines(str(time_stamp())+" " + str(data))
        f.writelines("\n")
    f.close()


def gripper_cacheData_readPosition(writePath, data, last_read_time):
    with open(writePath,  'w') as f:
        f.writelines(str(data))
        f.writelines("\n")
    f.close()


def wait_period(delay_time, start_t) -> None:
    delta_time_ = delay_time/1000
    start, end = 0, 0  # 声明变量
    start = time.time()  # servoJ发送结束时间; 精度延时开始计时时间
    # print("sss: ", start - start_t)
    if (start - start_t) < delta_time_:
        t = (delta_time_ - (start-start_t))   # 将输入t的单位转换为秒，-3是时间补偿
        # print(t)
        while end - start < t:  # 循环至时间差值大于或等于设定值时
            end = time.time()  # 记录结束时间


def save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        print("you")
        cam_names = list(video[0].keys())
        cam_names = sorted(cam_names)
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [0, 1, 2]] # swap B and R channel
                # image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        cam_names = sorted(cam_names)
        print(cam_names)

        show_canvas = np.zeros((480, 640 * 3, 3), dtype=np.uint8)
        h, w, _ = show_canvas.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for i in range(len(video[cam_names[0]])):
            show_canvas[:, :640] = np.asarray(video[cam_names[0]][i], dtype="uint8")
            show_canvas[:, 640:640 * 2] = np.asarray(video[cam_names[1]][i], dtype="uint8")
            show_canvas[:, 640 * 2:640 * 3] = np.asarray(video[cam_names[2]][i], dtype="uint8")
            out.write(show_canvas)
        out.release()
        print(f'Saved video to: {video_path}')

if __name__ == "__main__":
    # left_usb = "/sys/bus/usb-serial/devices"
    # free_limit_and_set_one("/sys/bus/usb-serial/devices/ttyUSB2/latency_timer")
    scan_port()
