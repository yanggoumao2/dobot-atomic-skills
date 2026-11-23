"""Find the reagent bottle, line up on it with the wrist camera, grasp it, lift, put it back. No teaching.

Sequence (each motion after a typed y; Enter alone aborts; space stops the arm; the arm is
disabled on every exit):
  1. coarse: top-camera frame -> chem_demo.locate (Grounding DINO + SAM 2 + upright-cylinder fit)
     -> bottle axis in this arm's base frame (about 1 cm error).
  2. hover: fingertips HOVER mm above the table over that point, tool straight down.
  3. refine (closed loop, no fixed points): wrist frame -> bottle mask -> upright-cylinder fit with
     the wrist camera pose (flange pose x run_wrist_calib mount, height re-measured on the white
     platform every frame) -> move over the new axis estimate; repeat until the correction is
     below --tol. Then the same again closer, just above the bottle.
  4. grasp: check the bottle fits between the open jaws, go straight down so the fingertips sit
     --grasp-height above the table (default 40 % of the measured bottle height), close, lift
     --lift mm and check with the wrist camera that the bottle came up with the gripper.
  5. put back where it was, open, rise, return to the start pose.

Everything is logged with the frames, masks and fits under calibration-data/chem-demo/bottle-pick.

    C:/software/anaconda3/envs/dobot-xtrainer/python.exe -X utf8 -m chem_demo.run_bottle_pick --arm left --live
    ... --sim                (synthetic camera, simulated arm and vision; no hardware, no GPU)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from chem_demo.wrist_geometry import (fit_upright_cylinder, jaw_gap, pose_matrix, transform, usable_contour)
from chem_demo.wrist_runtime import (ROOT, TABLE_Z, VisionClient, camera_serial, load_calibration,
                                     platform_height, save_frame)
from chem_demo.bottle_observation import DEFAULT_OPEN_WRIST_MODEL, run_observation
from task_skills.run_cartesian_jog import (DEFAULT_PROFILE, Jog, JogRefused, JogStopped, Limits, LiveArm, SimArm,
                                           down_tilt_deg, snap_down_rpy, start_refusal)
from vision_pick.cup_locate import Camera, frustum_silhouette

TRIALS_DIR = ROOT / "calibration-data" / "chem-demo" / "bottle-pick"
SANITY = {"radius_mm": (12.0, 60.0), "height_mm": (40.0, 260.0), "rms_px": 3.0, "usable_share": 0.35,
          "height_drift_mm": 4.0, "top_vs_wrist_mm": 40.0}
GRIPPER_OPEN = 255


class Aborted(RuntimeError):
    """The operator declined a step, or a check failed before anything moved."""


def _left_live_calibration_problem(calibration: dict[str, Any]) -> str | None:
    """Only permit left live motion with a passing fixed legacy calibration for this camera."""
    from chem_demo.run_wrist_calib import ACCEPT

    if calibration.get("arm") != "left":
        return "标定记录不是左臂"
    camera = calibration.get("camera")
    if not isinstance(camera, dict) or camera.get("source") != "realsense":
        return "标定不是来自真实 RealSense 相机"
    try:
        if camera.get("serial") != camera_serial("left_wrist"):
            return "标定的相机序列号与当前左腕相机不一致"
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        return f"无法核对左腕相机序列号：{error}"
    if not isinstance(calibration.get("checks"), dict):
        return "缺少旧腕部标定的质量检查记录"
    if calibration.get("problems") != []:
        return "标定记录未明确通过全部旧腕部检查"
    checks = calibration["checks"]
    try:
        pair_error = float(checks["pair_median_worst_mm"])
        plane_spread = float(checks["plane_height_spread_mm"])
        fewest_points = int(checks["pair_points_fewest"])
    except (KeyError, TypeError, ValueError):
        return "旧腕部标定缺少配对残差或台面高度检查"
    if not all(math.isfinite(value) for value in (pair_error, plane_spread)):
        return "旧腕部标定检查值不是有限数字"
    if (pair_error > ACCEPT["median_mm"] or plane_spread > ACCEPT["height_spread_mm"] or
            fewest_points < ACCEPT["min_points"]):
        return "旧腕部标定的配对残差、台面高度或有效点数未达通过阈值"
    jaw_gap_check = checks.get("jaw_gap")
    if not isinstance(jaw_gap_check, dict) or jaw_gap_check.get("ok") is not True:
        return "旧腕部标定缺少通过的张开夹爪检查"
    transform = calibration.get("T_flange_camera")
    intrinsics = calibration.get("K")
    if (not isinstance(transform, np.ndarray) or transform.shape != (4, 4) or
            not np.isfinite(transform).all() or not isinstance(intrinsics, np.ndarray) or
            intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()):
        return "旧腕部标定缺少有效的相机外参或内参"
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or
            not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)):
        return "旧腕部标定外参不是有效的刚体变换"
    camera_intrinsics = camera.get("intrinsics", {})
    try:
        recorded_K = np.asarray(camera_intrinsics["matrix_px"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return "旧腕部标定没有相机内参来源记录"
    if recorded_K.shape != (3, 3) or not np.allclose(recorded_K, intrinsics, rtol=1e-6, atol=1e-5):
        return "旧腕部标定的内参与其 RealSense 采集记录不一致"
    return None


class BottlePick:
    def __init__(self, arm: Any, camera: Any, vision: Any, calibration: dict, args: argparse.Namespace,
                 out_dir: Path, *, ask: Callable[[str], str], out: Callable[[str], None],
                 stop_pressed: Callable[[], bool], top_frame: Callable[[], np.ndarray] | None = None):
        self.arm, self.camera, self.vision, self.cal, self.args = arm, camera, vision, calibration, args
        self.out_dir, self.ask, self.out, self.stop_pressed = out_dir, ask, out, stop_pressed
        self.top_frame = top_frame
        self.table_z = args.table_z
        self.K = calibration["K"]
        self.gripper = calibration["gripper"]
        self.gripper_path = str(out_dir / "gripper_mask.png")
        cv2.imwrite(self.gripper_path, self.gripper * 255)
        self.height_offset: float | None = None  # camera height correction from the white platform
        self.frame_count = 0
        self.log: dict[str, Any] = {"start": datetime.now().isoformat(timespec="seconds"), "arm": args.arm,
                                    "calibration": calibration.get("path"), "table_z": self.table_z,
                                    "params": {k: getattr(args, k) for k in ("hover", "tol", "grasp_height", "lift",
                                                                                "jaw_margin", "pre_clear")},
                                    "steps": []}
        self.jog: Jog | None = None

    # ------------------------------------------------------------------ helpers
    def step(self, name: str, **fields: Any) -> None:
        self.log["steps"].append({"step": name, **fields})

    def checkpoint(self) -> None:
        if self.stop_pressed():
            self.arm.stop()
            raise JogStopped("急停：已停止，程序结束后下使能")

    def confirm(self, prompt: str) -> None:
        self.checkpoint()
        if self.args.yes:
            self.step("auto_confirm", prompt=prompt)
            return
        if self.ask(f"{prompt}（输入 y 继续，回车中止）: ").strip().lower() != "y":
            raise Aborted(prompt)

    def travel_to(self, target: Sequence[float], label: str, prompt: str, *, ask: bool = True) -> None:
        legs, finals, travel_z = self.jog._plan_travel([round(v, 3) for v in target], label, "-")
        if ask:
            self.confirm(prompt + f"（{len(legs)} 段直线，平移高度 Z={travel_z:.0f}，路线上不要有物体）")
        else:
            self.checkpoint()
        self.jog.run_route(legs, finals)
        state = self.arm.settle()
        self.step("move", label=label, target=[round(v, 2) for v in target],
                  measured=[round(v, 3) for v in state["pose"][:3]])
        self.out(f"{label}：到达 {[round(v, 1) for v in target]}")

    def straight(self, target: Sequence[float], label: str, prompt: str, *, ask: bool = True) -> None:
        """One checked straight MovL (no up-across-down route): small corrections, descents, lifts."""
        target = [round(float(v), 3) for v in target]
        finals = self.jog.check_route([target])
        if ask:
            self.confirm(prompt)
        else:
            self.checkpoint()
        self.jog.run_route([target], finals)
        state = self.arm.settle()
        self.step("move", label=label, target=target, measured=[round(v, 3) for v in state["pose"][:3]])
        self.out(f"{label}：到达 {[round(v, 1) for v in target]}")

    def vertical(self, z: float, label: str, prompt: str) -> None:
        self.straight([self.jog.xyz[0], self.jog.xyz[1], z], label, prompt)

    def tip_height(self) -> float:
        return self.jog.xyz[2] - self.table_z

    def wrist_camera_model(self, pose: Sequence[float]) -> Camera:
        T = pose_matrix(pose) @ self.cal["T_flange_camera"]
        if self.height_offset is not None:
            T = T.copy()
            T[2, 3] -= self.height_offset
        return Camera(self.K, np.zeros(5), T)

    # ------------------------------------------------------------------ vision
    def coarse(self) -> dict[str, Any]:
        args = self.args
        if args.bottle_xy:
            self.out(f"粗位置用命令行给定的 {args.bottle_xy}")
            return {"xy": list(args.bottle_xy), "height": None, "source": "command line"}
        image = self.top_frame()
        path = self.out_dir / "top.png"
        cv2.imwrite(str(path), image)
        self.out("顶部相机定位中……")
        reply = self.vision.locate_top(str(path), str(self.out_dir / "top-locate"))
        if not reply.get("ok"):
            raise Aborted(f"顶部相机定位失败：{reply.get('error')}")
        bottle = reply["vessels"].get("bottle")
        if bottle is None:
            raise Aborted(f"顶部相机没找到瓶子（{'; '.join(reply.get('notes', []))}）；定位图 {reply.get('overlay')}")
        if args.arm == "left":
            xy = bottle["left_base"]["axis_xyz_at_base"][:2]
        else:
            xy = [bottle["x"], bottle["y"]]
        self.out(f"顶部相机：瓶子在 {args.arm} 臂基座系约 ({xy[0]:.1f}, {xy[1]:.1f})，"
                 f"拟合高 {bottle['height']:.0f} mm（只作参考），定位图 {reply.get('overlay')}")
        return {"xy": [float(v) for v in xy], "height": float(bottle["height"]), "source": "top camera",
                "score": bottle.get("score"), "overlay": reply.get("overlay")}

    def look(self, label: str, guess: Sequence[float]) -> dict[str, Any]:
        """One wrist frame -> bottle axis (x, y), height and radius in base frame."""
        self.checkpoint()
        state = self.arm.settle()
        pose = [float(v) for v in state["pose"]]
        frame = self.camera.grab()
        if frame["color"].shape[:2] != self.gripper.shape:
            raise Aborted(f"相机分辨率 {frame['color'].shape[1]}x{frame['color'].shape[0]} 与标定时不同，请重新标定")
        self.frame_count += 1
        name = f"{self.frame_count:02d}_{label}"
        paths = save_frame(frame, self.out_dir / "frames", name)
        camera = self.wrist_camera_model(pose)
        guess_px = camera.project(np.array([[guess[0], guess[1], self.table_z + 0.5 * guess[2]]]))[0]
        reply = self.vision.detect_wrist(frame, paths["color"], str(self.out_dir / "frames" / f"{name}_mask.png"),
                                         near=guess_px.tolist(), exclude_path=self.gripper_path)
        if not reply.get("ok"):
            raise Aborted(f"腕部画面（{name}）里没认出瓶子：{reply.get('error')}")
        mask = (cv2.imread(reply["mask"], cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        mask[self.gripper > 0] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        # Camera height along the tool axis, from the white platform around the bottle.
        exclude = cv2.dilate(((self.gripper > 0) | (mask > 0)).astype(np.uint8), np.ones((21, 21), np.uint8))
        base_camera = Camera(self.K, np.zeros(5), pose_matrix(pose) @ self.cal["T_flange_camera"])
        plane = platform_height(base_camera.T_base_camera, self.K, frame["depth"], exclude, self.table_z,
                                window=60.0 if self.height_offset is None else 25.0 + abs(self.height_offset))
        if plane.get("ok"):
            offset = plane["z"] - self.table_z
            if self.height_offset is None:
                self.height_offset = offset
                self.out(f"台面深度：按标定算出的台面高度偏 {offset:+.1f} mm，已据此修正相机高度"
                         f"（{plane['points']} 点，倾斜 {plane['tilt_deg']:.1f}°）")
                if plane["tilt_deg"] > 3.0:
                    raise Aborted(f"看到的台面倾斜 {plane['tilt_deg']:.1f}°：标定可能不对，停止")
            elif abs(offset - self.height_offset) > SANITY["height_drift_mm"]:
                raise Aborted(f"台面高度前后不一致（{self.height_offset:+.1f} → {offset:+.1f} mm）：标定或深度有问题，停止")
        elif self.height_offset is None:
            raise Aborted(f"测不到白色台面高度：{plane.get('reason')}")
        camera = self.wrist_camera_model(pose)

        points, share = usable_contour(mask, self.gripper)
        if share < SANITY["usable_share"] or len(points) < 40:
            raise Aborted(f"瓶子轮廓大部分被夹爪或画面边缘挡住（可用 {share:.0%}）；{paths['color']}")
        start = [guess[0], guess[1], guess[2], guess[3]]
        params, rms, kept = fit_upright_cylinder(camera, points, self.table_z, start)
        x, y, h, r = (float(v) for v in params)
        result = {"frame": name, "pose": [round(v, 3) for v in pose], "xy": [round(x, 2), round(y, 2)],
                  "height": round(h, 1), "radius": round(r, 1), "rms_px": round(rms, 2), "kept": round(kept, 2),
                  "usable_share": round(share, 2), "score": reply.get("score"), "platform": plane,
                  "height_offset": None if self.height_offset is None else round(self.height_offset, 2)}
        self.draw(frame["color"], camera, mask, params, self.out_dir / "frames" / f"{name}_fit.jpg")
        self.step("look", **result)
        problems = []
        if not SANITY["radius_mm"][0] <= r <= SANITY["radius_mm"][1]:
            problems.append(f"半径 {r:.1f} mm")
        if not SANITY["height_mm"][0] <= h <= SANITY["height_mm"][1]:
            problems.append(f"高度 {h:.1f} mm")
        if rms > SANITY["rms_px"]:
            problems.append(f"轮廓残差 {rms:.2f} px")
        if problems:
            raise Aborted("腕部拟合不合理（" + "，".join(problems) + f"）；见 {name}_fit.jpg")
        return result

    def draw(self, image: np.ndarray, camera: Camera, mask: np.ndarray, params: Sequence[float], path: Path) -> None:
        view = image.copy()
        overlay = view.copy()
        overlay[mask > 0] = (0, 200, 255)
        overlay[self.gripper > 0] = (80, 80, 80)
        view = cv2.addWeighted(overlay, 0.35, view, 0.65, 0)
        x, y, h, r = params
        outline = frustum_silhouette(camera, [x, y, h, r, r], self.table_z)
        cv2.polylines(view, [np.round(outline).astype(np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
        tip = camera.project(np.array([[self.jog.xyz[0], self.jog.xyz[1], self.table_z + 0.4 * h]]))[0]
        cv2.drawMarker(view, (int(tip[0]), int(tip[1])), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.imwrite(str(path), view)

    def refine(self, tip_above_table: float, guess: list[float], label: str) -> list[float]:
        """Closed loop at one height: look, move over the fitted axis, until the correction < tol."""
        args = self.args
        z = self.table_z + tip_above_table
        self.confirm(f"开始{label}闭环对准：每次看一眼再在同一高度平移（每步 ≤ {args.max_step:.0f} mm，"
                     f"累计离粗位置 ≤ {args.max_shift:.0f} mm），中途不再逐步确认，空格随时停止")
        for iteration in range(args.max_iter):
            seen = self.look(f"{label}{iteration}", guess)
            guess = [seen["xy"][0], seen["xy"][1], seen["height"], seen["radius"]]
            error = math.dist(seen["xy"], self.jog.xyz[:2])
            self.out(f"  {label} 第 {iteration + 1} 次：瓶轴 ({seen['xy'][0]:.1f}, {seen['xy'][1]:.1f})，"
                     f"高 {seen['height']:.0f}，直径 {2 * seen['radius']:.1f}，残差 {seen['rms_px']:.2f} px；"
                     f"指尖偏离瓶轴 {error:.1f} mm")
            if math.dist(seen["xy"], self.coarse_xy) > args.max_shift:
                raise Aborted(f"腕部结果离顶部相机粗位置 {math.dist(seen['xy'], self.coarse_xy):.0f} mm "
                              f"> {args.max_shift:.0f}：可能认错了物体，停止")
            if error <= args.tol and iteration > 0:
                return guess
            target = list(seen["xy"])
            if error > args.max_step:  # move part of the way and look again
                k = args.max_step / error
                target = [self.jog.xyz[i] + k * (seen["xy"][i] - self.jog.xyz[i]) for i in range(2)]
            if error > args.tol:
                self.straight([target[0], target[1], z], f"{label}对准", "", ask=False)
        raise Aborted(f"{args.max_iter} 次后仍未对准到 {args.tol} mm 以内")

    # ------------------------------------------------------------------ sequence
    def run(self) -> str:
        args = self.args
        start = self.arm.read()
        xyz = [float(v) for v in start["pose"][:3]]
        limits = Limits(xyz, args.radius, self.table_z + args.min_tip, max(xyz[2], self.table_z + args.hover) + 60.0)
        self.jog = Jog(self.arm, limits, record_path=self.out_dir / "unused.csv", ask=self.ask, out=self.out,
                       stop_pressed=self.stop_pressed)
        if down_tilt_deg(self.jog.rpy) > 0.05:
            self.confirm(f"工具偏离竖直 {down_tilt_deg(self.jog.rpy):.2f}°，原地摆正")
            self.jog.snap_down()
        home_xyz = list(self.jog.xyz)
        self.step("ready", pose=[*home_xyz, *self.jog.rpy])

        coarse = self.coarse()
        self.log["coarse"] = coarse
        self.coarse_xy = coarse["xy"]
        if self.arm.gripper_position() < GRIPPER_OPEN - 5:
            self.confirm(f"夹爪要张到最大（当前 {self.arm.gripper_position()}），确认手里没有东西")
            self.out(f"夹爪：{self.arm.set_gripper(GRIPPER_OPEN)}")

        hover_z = self.table_z + args.hover
        self.travel_to([*self.coarse_xy, hover_z], "悬停",
                       f"移到瓶子粗位置上方（指尖离台面 {args.hover:.0f} mm）")
        guess = [*self.coarse_xy, coarse.get("height") or 120.0, 30.0]
        guess = self.refine(args.hover, guess, "高位")

        bottle_h, radius = guess[2], guess[3]
        pre_tip = bottle_h + args.pre_clear
        if pre_tip < args.hover - 5:
            self.vertical(self.table_z + pre_tip, "降到瓶口上方",
                          f"下降到瓶口上方约 {args.pre_clear:.0f} mm（指尖离台面 {pre_tip:.0f} mm）")
            guess = self.refine(pre_tip, guess, "低位")
            bottle_h, radius = guess[2], guess[3]

        # Will the bottle fit between the open jaws? Gap measured in the image at fingertip depth.
        camera = self.wrist_camera_model([*self.jog.xyz, *self.jog.rpy])
        tip_cam = transform(np.linalg.inv(camera.T_base_camera), np.array([self.jog.xyz]))[0]
        gap = jaw_gap(self.gripper, self.K, tip_cam)
        self.log["jaw_gap"] = gap
        diameter = 2 * radius
        if gap.get("ok"):
            self.out(f"瓶身直径约 {diameter:.1f} mm；夹爪张开间距估计 {gap['gap_mm']:.1f} mm")
            if diameter + 2 * args.jaw_margin > gap["gap_mm"]:
                raise Aborted(f"瓶身 {diameter:.1f} mm + 两侧余量 {2 * args.jaw_margin:.0f} mm 大于夹爪开口估计 "
                              f"{gap['gap_mm']:.1f} mm：不能从上方夹瓶身（可改夹瓶颈或侧面夹）")
        else:
            self.out(f"注意：没量出夹爪开口（{gap.get('reason')}），下降前请从侧面确认两指在瓶身外侧")

        grasp_tip = args.grasp_height if args.grasp_height else min(max(0.4 * bottle_h, 25.0), bottle_h - 20.0)
        if grasp_tip < args.min_tip:
            raise Aborted(f"抓取高度 {grasp_tip:.0f} mm 低于下限 {args.min_tip:.0f} mm")
        grasp_z = self.table_z + grasp_tip
        self.log["plan"] = {"bottle_xy": [round(v, 2) for v in guess[:2]], "bottle_height": round(bottle_h, 1),
                            "diameter": round(diameter, 1), "grasp_tip_above_table": round(grasp_tip, 1),
                            "grasp_flange_z": round(grasp_z, 2)}
        self.out(f"计划：直线下降到指尖离台面 {grasp_tip:.0f} mm（瓶高 {bottle_h:.0f}），合爪，抬起 {args.lift:.0f} mm")
        self.vertical(grasp_z, "下降到抓取高度",
                      "从侧面看：两指是否分在瓶身两侧、瓶颈从两指之间穿过？对了就下降")
        self.confirm("合爪夹瓶")
        feedback = self.arm.set_gripper(args.close_to)
        self.step("close", feedback=feedback)
        final = feedback.get("final")
        self.out(f"夹爪：{feedback.get('start')} -> {final}（目标 {args.close_to}）")
        if final is not None and final <= args.close_to + args.empty_band:
            self.out("夹爪几乎合到底：没夹到东西。张开并退回悬停高度")
            self.arm.set_gripper(GRIPPER_OPEN)
            self.vertical(hover_z, "退回", "张开后竖直退回")
            raise Aborted("夹空")

        # Reference frame with the jaws already closed, so the fingers look the same in both frames.
        before = self.camera.grab()
        self.vertical(grasp_z + args.lift, f"抬起 {args.lift:.0f} mm", "试提")
        after = self.camera.grab()
        save_frame(after, self.out_dir / "frames", "lifted")
        held = self.held_check(before, after)
        self.log["held_check"] = held
        self.out(f"腕部相机判断：{'瓶子随夹爪抬起' if held.get('held') else '瓶子没跟上来'}（{held}）")

        outcome = "s" if (self.args.yes and held.get("held")) else ("f" if self.args.yes else self.ask_outcome())
        self.confirm("放回：下降到原抓取高度")
        self.vertical(grasp_z, "放回", "放回原处")
        self.confirm("张开夹爪松开瓶子")
        self.step("open", feedback=self.arm.set_gripper(GRIPPER_OPEN))
        self.vertical(hover_z, "抬到悬停高度", "竖直抬起离开瓶子")
        self.travel_to(home_xyz, "回到起点", "回到起点")
        return "成功" if outcome == "s" else "失败"

    def held_check(self, before: dict, after: dict) -> dict[str, Any]:
        """The camera rides with the gripper: a held bottle stays put in the image after the lift,
        a bottle left on the table moves (farther away and lower in the view)."""
        reply_a = self.vision.detect_wrist(before, *self._paths("held_a", before), exclude_path=self.gripper_path)
        reply_b = self.vision.detect_wrist(after, *self._paths("held_b", after), exclude_path=self.gripper_path)
        if not (reply_a.get("ok") and reply_b.get("ok")):
            return {"held": None, "reason": "前后画面没都认出瓶子"}
        a = cv2.imread(reply_a["mask"], cv2.IMREAD_GRAYSCALE) > 0
        b = cv2.imread(reply_b["mask"], cv2.IMREAD_GRAYSCALE) > 0
        a &= self.gripper == 0
        b &= self.gripper == 0
        iou = float((a & b).sum() / max(1, (a | b).sum()))
        return {"held": iou >= self.args.held_iou, "iou": round(iou, 3)}

    def _paths(self, name: str, frame: dict) -> tuple[dict, str, str]:
        paths = save_frame(frame, self.out_dir / "frames", name)
        return frame, paths["color"], str(self.out_dir / "frames" / f"{name}_mask.png")

    def ask_outcome(self) -> str:
        outcome = ""
        while outcome not in ("s", "f"):
            outcome = self.ask("结果：s = 夹稳抬起 / f = 失败（滑落、夹歪、碰倒等）: ").strip().lower()
        note = self.ask("备注（可空）: ").strip()
        self.log["outcome"] = {"success": outcome == "s", "note": note}
        return outcome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="识别 + 腕部相机闭环对准后抓瓶（不示教），试提后放回")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--sim", action="store_true", help="合成相机 + 模拟臂 + 模拟识别，不连硬件")
    mode.add_argument("--observe-only", action="store_true",
                      help="只读取左腕相机并核对已标定姿态；不会移动机械臂")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--speed", type=int, default=10)
    parser.add_argument("--radius", type=float, default=350.0, help="以起点为圆心的水平软限位 mm")
    parser.add_argument("--table-z", type=float, help="指尖触台面时的法兰 Z（默认按臂取）")
    parser.add_argument("--calibration", type=Path, help="腕部相机标定文件（默认 wrist-calibration/<arm>_wrist.json）")
    parser.add_argument("--open-wrist-model", type=Path,
                        help="--observe-only 使用的已验证张开夹爪棋盘模型")
    parser.add_argument("--replay", type=Path,
                        help="离线重放已保存的采集目录（capture.json、ref.png 和对齐深度；不连接设备）")
    parser.add_argument("--bottle-xy", type=float, nargs=2, help="跳过顶部相机，直接给粗位置（本臂基座系 mm）")
    parser.add_argument("--hover", type=float, default=230.0, help="高位对准时指尖离台面 mm")
    parser.add_argument("--pre-clear", type=float, default=40.0, help="低位对准时指尖在瓶顶上方 mm")
    parser.add_argument("--tol", type=float, default=1.0, help="对准容差 mm")
    parser.add_argument("--max-step", type=float, default=40.0, help="单次对准最大平移 mm")
    parser.add_argument("--max-shift", type=float, default=60.0, help="腕部结果离粗位置的最大允许距离 mm")
    parser.add_argument("--max-iter", type=int, default=6)
    parser.add_argument("--grasp-height", type=float, help="抓取时指尖离台面 mm（默认瓶高的 40%%）")
    parser.add_argument("--min-tip", type=float, default=20.0, help="指尖离台面的最低高度 mm")
    parser.add_argument("--jaw-margin", type=float, default=6.0, help="瓶身每侧到手指的最小余量 mm")
    parser.add_argument("--close-to", type=int, default=0, help="合爪目标 0-255")
    parser.add_argument("--empty-band", type=int, default=8, help="合到目标附近这么多以内算夹空")
    parser.add_argument("--lift", type=float, default=20.0, help="试提高度 mm")
    parser.add_argument("--held-iou", type=float, default=0.7, help="试提前后瓶子掩码重合度下限")
    parser.add_argument("--yes", action="store_true", help=argparse.SUPPRESS)  # offline tests only
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out", type=Path, default=TRIALS_DIR)
    return parser


def main(argv: Sequence[str] | None = None, *, ask: Callable[[str], str] = input,
         out: Callable[[str], None] = print, arm: Any = None, camera: Any = None, vision: Any = None,
         stop_pressed: Callable[[], bool] | None = None, top_frame: Callable[[], np.ndarray] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.table_z is None:
        args.table_z = TABLE_Z[args.arm]
    if args.yes and args.live:
        out("--yes 只用于离线测试")
        return 2
    if not args.observe_only and (args.replay is not None or args.open_wrist_model is not None):
        out("--replay 和 --open-wrist-model 只能与 --observe-only 一起使用")
        return 2
    live_left_calibration: dict[str, Any] | None = None
    if args.live and args.arm == "left":
        try:
            live_left_calibration = load_calibration(args.arm, args.calibration)
        except (OSError, ValueError, RuntimeError) as error:
            out(f"左臂实时抓瓶未启动：左腕旧标定缺失或无效（{error}）。"
                "可先用 --observe-only --replay <采集目录> 核对新模型支持的标定姿态。")
            return 2
        problem = _left_live_calibration_problem(live_left_calibration)
        if problem:
            out(f"左臂实时抓瓶未启动：{problem}。可用 --observe-only 获取记录姿态证据；"
                "该阶段不授权任意悬停或机械臂运动。")
            return 2
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f" if args.observe_only else "%Y%m%d_%H%M%S")
    run_prefix = "observation-" if args.observe_only else ""
    trial_dir = args.out / f"{args.arm}-{run_prefix}{run_stamp}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    if stop_pressed is None:
        if sys.platform == "win32" and args.live:
            from task_skills.run_cartesian_jog import _windows_keys
            stop_pressed = _windows_keys()[1]
        else:
            stop_pressed = lambda: False  # noqa: E731
    if args.observe_only:
        if args.arm != "left":
            out("--observe-only 目前只支持 --arm left")
            return 2
        try:
            profile = None if args.replay is not None else json.loads(args.profile.read_text(encoding="utf-8"))
            _, code = run_observation(args, trial_dir, out=out, camera=camera, profile=profile)
        except (OSError, ValueError, RuntimeError) as error:
            out(f"只读观测未通过：{error}")
            code = 1
        out(f"观测记录：{trial_dir}")
        return code
    pick: BottlePick | None = None
    result, code = "未开始", 1
    try:
        calibration = live_left_calibration or load_calibration(args.arm, args.calibration)
        if arm is None:
            if args.sim:
                raise RuntimeError("--sim 由测试注入模拟臂/相机/识别（见 tests/test_bottle_pick.py）")
            out("启动识别进程（加载模型约 30 s）……")
            vision = VisionClient(out=out)
            profile = json.loads(args.profile.read_text(encoding="utf-8"))
            arm = LiveArm(profile, args.arm, args.speed, log=out)
            problem = start_refusal(arm.read().get("mode"))
            if problem:
                out(f"拒绝启动，没有发送任何指令：{problem}")
                return 5
            if ask(f"确认 {args.arm} 臂周围无人、急停在手边、另一只臂停在初始位，输入 {args.arm} 继续: ").strip() != args.arm:
                out("未确认，已退出，没有发送任何指令")
                return 1
            out(arm.open_session(home=True))
            from chem_demo.cameras import TOP_ROTATE_180, grab_realsense
            from chem_demo.wrist_camera import LiveWristCamera
            top_image = None
            if not args.bottle_xy:
                # Arm at home, out of the top view. Taken before the wrist stream starts: the top
                # camera's 1280x720 (its calibration resolution) is slow when another D405 streams.
                out("顶部相机拍照……")
                color, _, _ = grab_realsense(camera_serial("top"))
                top_image = cv2.rotate(color, cv2.ROTATE_180) if TOP_ROTATE_180 else color
            camera = LiveWristCamera(camera_serial(f"{args.arm}_wrist"))

            def top_frame() -> np.ndarray:
                return top_image
        pick = BottlePick(arm, camera, vision, calibration, args, trial_dir, ask=ask, out=out,
                          stop_pressed=stop_pressed, top_frame=top_frame)
        result = pick.run()
        code = 0
    except Aborted as reason:
        result, code = f"中止：{reason}", 1
        out(f"已中止（{reason}），机械臂停在当前位置")
    except JogRefused as refusal:
        result, code = f"拒绝：{refusal}", 4
        out(f"拒绝，未发送这一步的运动：{refusal}")
    except JogStopped as stopped:
        result, code = "空格停止", 3
        out(str(stopped))
    except RuntimeError as error:
        result, code = f"错误：{error}", 4
        out(f"运行错误，已停止：{error}")
    finally:
        for thing in (camera, vision, arm):
            if thing is not None:
                try:
                    thing.close()
                except Exception as error:  # noqa: BLE001 - keep closing the rest (the arm last)
                    out(f"关闭 {type(thing).__name__} 出错：{error}")
        if pick is not None:
            pick.log.update({"end": datetime.now().isoformat(timespec="seconds"), "result": result})
            (trial_dir / "trial.json").write_text(json.dumps(pick.log, ensure_ascii=False, indent=2, default=str),
                                                  encoding="utf-8")
    out(f"本次结果：{result}；记录在 {trial_dir}")
    return code


if __name__ == "__main__":
    sys.exit(main())
