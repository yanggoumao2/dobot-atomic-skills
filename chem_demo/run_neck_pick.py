"""One-start left-arm pick of the known blue-rimmed empty bottle on the white bench."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from chem_demo.bottle_observation import read_gripper_position, read_passive_robot_state
from chem_demo.camera_drift import load_corrected_camera
from chem_demo.cameras import TOP_ROTATE_180, load_roles
from chem_demo.neck_lift import verify_neck_lift
from chem_demo.stations import photo_pose
from chem_demo.wrist_camera import LiveWristCamera
from chem_demo.wrist_runtime import ROOT
from task_skills.run_cartesian_jog import DEFAULT_PROFILE, Jog, Limits, LiveArm, _rotation_angle_deg
from vision_pick.cup_locate import DEFAULT_SESSION

DATA = ROOT / "calibration-data" / "chem-demo" / "neck-pick"
SETUP = DATA / "commissioned.json"
REFERENCE = ROOT / "calibration-data" / "chem-demo" / "bottle-observations" / "hover-20261121_170459" / "hovered"
BOTTLE = "白台面直立空玻璃试剂瓶，蓝色瓶口"
HOME = [117.31, -344.60, 415.41]
RIGHT_PARK = [-117.58, -344.79, 415.50, 180.0, 0.0, 89.97]
RIGHT_PHOTO = photo_pose("right")
ORIENTATION = [180.0, 0.0, -89.94]
HOVER_Z = 485.0


def _save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"图像写入失败 {path}")


_SPACE_SINCE = [None]
STOP_HOLD_S = 0.3


def _stop_key():
    """Space HELD for STOP_HOLD_S. GetAsyncKeyState is global: a pinyin IME uses quick space taps to
    pick characters, which stopped a live run on 2026-11-22 while the operator typed in another window.
    """
    import ctypes
    if not ctypes.windll.user32.GetAsyncKeyState(0x20) & 0x8000:
        _SPACE_SINCE[0] = None
        return False
    now = time.monotonic()
    if _SPACE_SINCE[0] is None:
        _SPACE_SINCE[0] = now
    return now - _SPACE_SINCE[0] >= STOP_HOLD_S


class NeckArm(LiveArm):
    """Jog motion plus slow, cancellable jaw adjustments for the bottle neck."""

    def set_linear_speed(self, percent):
        if type(percent) is not int or not 1 <= percent <= 10:
            raise ValueError("速度须为 1–10%")
        self.driver._configure_speed(self.rig, percent, linear=True)
        self.speed = percent

    def grip_to(self, position, speed):
        if type(position) is not int or not 0 <= position <= 255 or not 1 <= speed <= 10:
            raise ValueError("夹爪目标或速度超出范围")
        initial = self.gripper_position()
        if _stop_key():
            raise RuntimeError("操作员已停止")
        deadline = time.monotonic() + 8
        readings = []
        started = time.monotonic()
        try:
            self.driver._gripper_move(self.rig, self.profile, position, speed)
            while time.monotonic() < deadline:
                actual = self.gripper_position()
                if not 0 <= actual <= 255:
                    raise RuntimeError("夹爪开度反馈越界")
                readings.append(actual)
                if _stop_key():
                    raise RuntimeError("操作员停止了夹爪动作")
                stable = len(readings) >= 5 and max(readings[-5:]) - min(readings[-5:]) <= 1
                responded = abs(actual - position) <= 2 or abs(initial - actual) >= 3
                if responded and stable and time.monotonic() - started >= .7:
                    return {"start": initial, "target": position, "final": actual, "samples": readings[-5:]}
                time.sleep(.10)
            raise RuntimeError("夹爪未产生稳定的到位或接触反馈")
        except BaseException:
            # Stop an unverified in-flight command at its actual travel. Once a
            # normal close is confirmed, leave the full-close command active.
            try:
                actual = self.gripper_position()
            except Exception:
                actual = readings[-1] if readings else None
            if actual is None or not 0 <= actual <= 255:
                print("夹爪读数丢失，无法确认夹爪已停止。请使用现场急停。", flush=True)
            else:
                self.driver._gripper_move(self.rig, self.profile, actual, speed)
            raise


def _quiet_stop(stream):
    try:
        stream.stop()
    except RuntimeError:  # already stopped after a stall restart
        pass


class Cameras:
    """Keep both serial-bound streams open for the duration of one invocation."""

    def __init__(self, stack, directory, *, replay=None):
        self.stack = stack
        self.directory = directory
        self.replay = replay
        self.overhead = self.wrist = None
        self.top_to_reference = None
        self.roles = load_roles()
        self.left_camera = load_corrected_camera(DEFAULT_SESSION, "left")
        self.right_camera = load_corrected_camera(DEFAULT_SESSION, "right")
        self.count = 0

    def top_frame(self):
        """Top image; a stalled stream ("Frame didn't arrive") is reopened once (3 D405s share USB)."""
        try:
            return self._top_frame()
        except RuntimeError as error:
            if self.replay or "didn't arrive" not in str(error):
                raise
            print("顶部相机取帧超时，重开数据流后重试一次", flush=True)
            stream, self.overhead = self.overhead, None
            if stream is not None:
                try:
                    stream.stop()
                except RuntimeError:
                    pass
            return self._top_frame()

    def _top_frame(self):
        if self.replay:
            return self._load("top.png")
        import pyrealsense2 as rs
        if self.overhead is None:
            stream, config = rs.pipeline(), rs.config()
            config.enable_device(self.roles["top"])
            config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 15)  # shared hub: 15 fps
            active = stream.start(config)
            self.stack.callback(_quiet_stop, stream)
            self.overhead = stream
            low = active.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            # The low-resolution mode crops the sensor horizontally. Reconstruct the
            # reference pixel grid from the two factory intrinsics before applying
            # the independently calibrated overhead camera transform.
            calibrated_mode = next(profile.as_video_stream_profile()
                                   for sensor in active.get_device().query_sensors()
                                   for profile in sensor.get_stream_profiles()
                                   if profile.stream_type() == rs.stream.color and profile.format() == rs.format.bgr8
                                   and profile.as_video_stream_profile().width() == 1280
                                   and profile.as_video_stream_profile().height() == 720)
            high = calibrated_mode.get_intrinsics()
            if low.model != high.model or not np.allclose(low.coeffs, high.coeffs, atol=1e-6):
                raise ValueError("顶部相机两种分辨率的畸变模型不一致")
            sx, sy = high.fx / low.fx, high.fy / low.fy
            self.top_to_reference = np.float32([[sx, 0, high.ppx - sx * low.ppx],
                                                [0, sy, high.ppy - sy * low.ppy]])
            for _ in range(4):
                stream.wait_for_frames(5000)
        self.overhead.wait_for_frames(5000)
        image = np.asanyarray(self.overhead.wait_for_frames(5000).get_color_frame().get_data()).copy()
        image = cv2.warpAffine(image, self.top_to_reference, (1280, 720))
        return cv2.rotate(image, cv2.ROTATE_180) if TOP_ROTATE_180 else image

    def pause_streams(self):
        """Stop the top and left-wrist streams when they are no longer needed (USB bandwidth)."""
        if self.wrist is not None and hasattr(self.wrist, "pause"):
            self.wrist.pause()
        stream, self.overhead = self.overhead, None
        if stream is not None:
            _quiet_stop(stream)

    def wrist_frame(self):
        if self.replay:
            return {"color": self._load("ref.png")}
        if self.wrist is None:
            # 15 fps: the three D405s share one USB 3 hub; halving the rate halves its load.
            self.wrist = LiveWristCamera(self.roles["left_wrist"], fps=15, warmup=6)
            self.stack.callback(self.wrist.close)
        return self.wrist.grab(flush=2, require_fresh_pair=True)

    def _load(self, name):
        image = cv2.imread(str(self.replay / name))
        if image is None:
            raise ValueError(f"缺少回放图像 {self.replay / name}")
        return image

    def record(self, name, image):
        self.count += 1
        _save_image(self.directory / f"{self.count:02d}_{name}.png", image)

    def localize(self, stage, pose):
        from chem_demo.neck_vision import locate_top_rim
        t = time.perf_counter()
        image = self.top_frame()
        captured = time.perf_counter()
        measured = locate_top_rim(image, right_camera=self.right_camera, left_camera=self.left_camera)
        elapsed_ms = round((time.perf_counter() - captured) * 1000, 1)
        self.record("top", image)
        return {"ok": measured.ok, "reason": measured.reason,
                "xy": list(measured.left_base_xyz_mm[:2]) if measured.ok else None,
                "rim_px": list(measured.rim.center_px) if measured.ok else None,
                "source": "top camera", "processing_ms": elapsed_ms,
                "capture_s": round(captured - t, 3)}

    def check_wrist(self, pose, target):
        from chem_demo.neck_vision import compare_wrist_to_hover
        if abs(pose[2] - HOVER_Z) > 1 or _rotation_angle_deg(pose[3:], ORIENTATION) > 1:
            raise ValueError("腕部对准参考只适用于已核对的悬停高度和朝向")
        frame = self.wrist_frame()
        t = time.perf_counter()
        gate = compare_wrist_to_hover(frame["color"])
        elapsed_ms = round((time.perf_counter() - t) * 1000, 1)
        self.record("wrist", frame["color"])
        result = {"ok": gate.ok, "reason": gate.reason, "center_error_px": gate.center_error_px,
                  "axes_error_px": gate.axes_error_px, "jaw_errors_px": gate.jaw_errors_px,
                  "offset_px": gate.offset_px, "camera_shift_px": gate.camera_shift_px,
                  "jaws_used": gate.jaws_used, "notes": list(gate.notes), "processing_ms": elapsed_ms}
        if gate.offset_px is not None:
            result["correction_xy_mm"] = hover_correction_mm(gate.offset_px, frame.get("depth"), gate.rim)
        return result


# Left wrist camera at the hover pose (yaw -90, board model 2026-11-21): image u -> base +X, image v ->
# base -Y with the optical axis ~23 deg off vertical. A rim that appears right of / below its jaw-centred
# target sits at +X / -Y of the jaws, so the arm moves that way.
WRIST_RANGE_FALLBACK_MM = 270.0   # camera -> rim at hover (cw arithmetic; depth used when valid)
WRIST_FX_PX, WRIST_FY_PX, WRIST_V_FORESHORTEN = 435.0, 434.6, 0.917


def hover_correction_mm(offset_px, depth=None, rim=None):
    """Rim offset (px, relative to the jaws) -> left base XY move (mm) at the hover pose."""
    range_mm = WRIST_RANGE_FALLBACK_MM
    if depth is not None and rim is not None:
        cx, cy = rim.center_px
        r = max(rim.axes_px) / 2.0
        h, w = depth.shape[:2]
        ys, xs = np.mgrid[max(int(cy - r - 4), 0):min(int(cy + r + 5), h), max(int(cx - r - 4), 0):min(int(cx + r + 5), w)]
        ring = np.abs(np.hypot(xs - cx, ys - cy) - r) <= 4
        values = np.asarray(depth, float)[ys[ring], xs[ring]]
        values = values[(values > 120) & (values < 450)]
        if values.size >= 30:
            range_mm = float(np.median(values))
    du, dv = offset_px
    return [round(du * range_mm / WRIST_FX_PX, 2), round(-dv * range_mm / (WRIST_FY_PX * WRIST_V_FORESHORTEN), 2)]


def make_read_wrist(arm, sight, directory):
    """Grab a fresh left-wrist RGB-D pair with the arm pose, and save it as evidence."""
    def read_wrist():
        frame = sight.wrist_frame()
        for retry in range(3):
            skew = frame["capture_metadata"]["sdk"].get("color_depth_timestamp_delta_ms")
            depth_usable = skew is not None and skew <= 1000 / sight.wrist.fps
            if depth_usable or retry == 2:
                break
            frame = sight.wrist.grab(flush=1, require_fresh_pair=True)
        frame["pose"] = arm.read()["pose"]
        sight.record("lift", frame["color"])
        stem = f"{sight.count:02d}_lift"
        _save_image(directory / (stem + "_top.png"), sight.top_frame())
        np.save(directory / f"{stem}_depth_mm.npy", frame["depth"], allow_pickle=False)
        metadata = {"color": stem + ".png", "depth_mm": stem + "_depth_mm.npy",
                    "overhead": stem + "_top.png", "pose": frame["pose"], "camera": sight.wrist.describe(),
                    "capture": frame["capture_metadata"], "depth_usable": depth_usable}
        (directory / (stem + ".json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        frame["evidence"] = stem + ".json"
        if not depth_usable:
            frame["depth"] = np.zeros_like(frame["depth"])
        return frame
    return read_wrist


def _check_idle(profile):
    states = {}
    for name, binding in profile["arms"].items():
        state = read_passive_robot_state(binding["ip"])
        if (state["mode"] != 4 or state["enabled"] or state["running"] or
                state["tool"] != 0 or state["user"] != 0):
            raise ValueError(f"{name} 臂需停稳在下使能状态，且为 User0/Tool0")
        states[name] = state
    right = states["right"]["pose"]
    # Platform start is the photo pose; the old low park pose is still accepted.
    candidates = [RIGHT_PARK, [*RIGHT_PHOTO[0], *RIGHT_PHOTO[1]]]
    if not any(math.dist(right[:3], c[:3]) <= 4 and _rotation_angle_deg(right[3:], c[3:]) <= 2
               for c in candidates):
        raise ValueError("右臂不在拍摄位（初始位）或已验证的旧停放位")
    left = profile["arms"]["left"]
    jaw = read_gripper_position(left["gripper_port"], left["gripper_id"], left["servo_pos"])
    if not 250 <= jaw["position"] <= 255:
        raise ValueError("请先将上次的瓶子放稳，再从左爪张开的状态开始")
    return states["left"]


def _close_arm(arm, attempted):
    try:
        if attempted and not arm.session_open:
            state = read_passive_robot_state(arm.profile["arms"]["left"]["ip"])
            if state["enabled"]:
                arm.disable()
    finally:
        arm.close()
    state = read_passive_robot_state(arm.profile["arms"]["left"]["ip"])
    if attempted and (state["mode"] != 4 or state["enabled"]):
        raise RuntimeError("未确认左臂已下使能，请查看现场控制器")


def _load_profile():
    profile = json.loads(DEFAULT_PROFILE.read_text(encoding="utf-8"))
    if SETUP.exists():
        saved = json.loads(SETUP.read_text(encoding="utf-8"))
        if saved.get("profile_id") != profile["profile_id"]:
            raise ValueError("瓶颈记录与本机配置不符")
        digest = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":"),
                                           ensure_ascii=False).encode("utf-8")).hexdigest()
        teaching = saved["neck_grasp"]["commissioning"]
        if (teaching["profile_identity"]["sha256"] != digest or
                teaching["same_bottle_designation"] != BOTTLE):
            raise ValueError("瓶型或硬件配置与首次实测不符，请核对首设记录")
        profile["neck_grasp"] = saved["neck_grasp"]
    return profile


def _checked_target(report):
    if not report.get("ok"):
        raise ValueError(f"顶视定位失败：{report.get('reason')}")
    x, y = report["xy"]
    if not (220 <= x <= 305 and -425 <= y <= -350):
        raise ValueError("瓶口位于已验证白台面工作区以外")
    return x, y


def _prepare_teaching(arm, sight, profile, stop, accepted_contact_z=None):
    from chem_demo.neck_commission import commission
    pose = arm.read()["pose"]
    target = _checked_target(sight.localize("commission", pose))
    if math.dist(pose[:2], target) > 6:
        raise ValueError("首次记录请从已经对准的张爪悬停位启动")
    wrist = sight.check_wrist(pose, target)
    if wrist.get("ok") is not True:
        raise ValueError(f"首次悬停腕图未匹配：{wrist}")
    jog = Jog(arm, Limits(pose[:3], 2.0, 350, HOVER_Z + 1), record_path=sight.directory / "jog.csv",
              stop_pressed=stop)
    def ask_operator(prompt):
        status_file = sight.directory / "operator_state.json"
        def update(phase):
            temporary = status_file.with_suffix(".tmp")
            temporary.write_text(json.dumps({"phase": phase, "prompt": prompt},
                                            ensure_ascii=False), encoding="utf-8")
            temporary.replace(status_file)
        update("awaiting_operator")
        answer = input(prompt)
        update("answered")
        return answer

    return commission(arm, jog, profile=profile, profile_path=SETUP,
                      same_bottle_designation=BOTTLE, ask=ask_operator, out=print, stop_pressed=stop,
                      accepted_contact_z=accepted_contact_z, defer_save=True)


def _execute(args, directory, outcome):
    from chem_demo.neck_pick_cycle import NeckPickCycle, preflight
    profile = _load_profile()
    if args.replay:
        capture = json.loads((args.replay / "capture.json").read_text(encoding="utf-8"))
        pose = capture["after"]["pose"]
    else:
        pose = _check_idle(profile)["pose"]
    with ExitStack() as stack:
        sight = Cameras(stack, directory, replay=args.replay)
        if args.preview or args.replay:
            predicted = sight.localize("preview", pose)
            outcome["top"] = predicted
            target = _checked_target(predicted)
            if abs(pose[2] - HOVER_Z) < 1 and math.dist(pose[:2], target) <= 6:
                outcome["wrist"] = sight.check_wrist(pose, target)
                if not outcome["wrist"]["ok"]:
                    raise ValueError(f"腕图复核未通过：{outcome['wrist']}")
            print("只读结果", json.dumps(outcome, ensure_ascii=False), flush=True)
            return True

        missing_setup = "neck_grasp" not in profile
        if args.accepted_contact_z is not None and not missing_setup:
            raise ValueError("已有瓶颈首设记录，不能覆盖其实测高度")
        if missing_setup:
            print("首次运行：先在悬停位测空夹，再下降夹住瓶颈，保持闭合试提 15 mm。", flush=True)
            target = _checked_target(sight.localize("initial", pose))
            if math.dist(pose[:2], target) > 6 or abs(pose[2] - HOVER_Z) > 1:
                raise ValueError("首次设置须从已对准的高位悬停位开始")
            if not sight.check_wrist(pose, target)["ok"]:
                raise ValueError("当前腕图与已接受的悬停对准不一致")
        else:
            preflight(profile, _check_idle(profile))

        arm = NeckArm(profile, "left", 8)
        attempted = False
        try:
            current = _check_idle(profile)["pose"]
            if math.dist(current[:3], pose[:3]) > .7 or _rotation_angle_deg(current[3:], pose[3:]) > .5:
                raise RuntimeError("启动期间左臂起点发生变化")
            if _stop_key():
                raise RuntimeError("已取消")
            attempted = True
            arm.open_session(home=False)
            if missing_setup:
                profile = _prepare_teaching(arm, sight, profile, _stop_key, args.accepted_contact_z)
                (directory / "candidate.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2),
                                                          encoding="utf-8")
            read_wrist = make_read_wrist(arm, sight, directory)
            cycle = NeckPickCycle(arm, profile, localize=sight.localize, check_wrist=sight.check_wrist,
                                  read_wrist=read_wrist, verify_held=verify_neck_lift, stop_pressed=_stop_key,
                                  log_path=directory / "cycle.json", lift_mm=args.lift_mm)
            if missing_setup:
                from chem_demo.neck_commission import _atomic_save
                contact_pose = [*current[:2], profile["neck_grasp"]["tcp_z"], *current[3:]]
                result = cycle.verify_first_closed(contact_pose)["result"]
                if result.get("held") is True and result.get("success") is True:
                    profile["neck_grasp"]["commissioning"]["verified_lift_run"] = directory.name
                    _atomic_save(SETUP, profile, profile["profile_id"])
            else:
                result = cycle.run()["result"]
            outcome.update(status=result.get("status"), held=result.get("held"))
            print(f"本次抓取：{outcome['status']}；试提核验：{outcome['held']}", flush=True)
            return result.get("success") is True
        except BaseException:
            if attempted and arm.session_open:
                arm.stop()
            raise
        finally:
            _close_arm(arm, attempted)


def main(argv=None):
    parser = argparse.ArgumentParser(description="自动识别白台面上的蓝口空瓶，用左臂夹瓶颈并试提")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="在现场实际执行")
    parser.add_argument("--lift-mm", type=int, choices=(15, 30), default=15,
                        help="闭爪后的竖直抬升量（毫米），默认 15")
    parser.add_argument("--accepted-contact-z", type=float, metavar="MM",
                        help="从已复核悬停位重用操作员此前在同一瓶型、同一位置接受的 TCP 高度")
    mode.add_argument("--preview", action="store_true", help="只取相机和设备反馈，输出本次定位")
    mode.add_argument("--replay", type=Path, help="回放含 top.png、ref.png 和 capture.json 的实拍目录")
    args = parser.parse_args(argv)
    if args.accepted_contact_z is not None and (not args.live or not 350 <= args.accepted_contact_z < HOVER_Z - 15):
        parser.error("重用的实测高度须在现场首设行程内")
    directory = DATA / (datetime.now().strftime("run-%Y%m%d_%H%M%S") + "-" + uuid4().hex[:10])
    directory.mkdir(parents=True)
    started = time.perf_counter()
    result = {"mode": "live" if args.live else "preview" if args.preview else "replay"}
    print(f"左臂瓶口抓取 | {BOTTLE}\n记录：{directory}", flush=True)
    if args.live:
        print("运行时按住空格停止。使用同款空瓶，保持工作区净空，实体急停在手边。", flush=True)
    try:
        passed = _execute(args, directory, result)
    except (Exception, KeyboardInterrupt) as exc:
        result["error"] = str(exc) or type(exc).__name__
        print(f"已停止：{result['error']}", flush=True)
        passed = False
    result.update(success=passed, elapsed_s=round(time.perf_counter() - started, 3))
    (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
