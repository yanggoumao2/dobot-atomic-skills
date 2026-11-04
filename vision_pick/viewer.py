"""Live view for a vision pick: camera feed + fitted cup + plan + where the fingertips are now.

Runs in .venv-calibration (the only env that can read the camera) and owns the camera for
the whole trial, so it also takes the photo for locating. The pick flow drives it with one
JSON object per line on stdin and reads JSON lines from stdout:

  in : {"cmd": "locate", "table_z": .., "arm": "right", "rpy": [..]}
       {"cmd": "status", "step": "..", "tcp": [x, y, z], "plan": {"hover": [..], "grasp": [..]}}
       {"cmd": "quit"}
  out: {"type": "ready"} | {"type": "locate", "report": {...}} | {"type": "error", "error": ".."}
       {"type": "stop"}   (space or Esc pressed in the window: the flow stops the arm)

Everything drawn is also written to <out>/trial_view.mp4 for the report.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import cv2
import numpy as np

from .cup_locate import (DEFAULT_SESSION, OBJECT_MEMORY, WORKSPACE_RADIUS_MM, Camera, load_camera, locate_and_save,
                         rim_pixel, segment_saturated_cyan)

WINDOW = "vision pick"
FPS = 15.0
GREEN, CYAN, MAGENTA, RED, BLUE, WHITE = (0, 220, 0), (255, 220, 0), (255, 0, 255), (0, 0, 255), (255, 80, 0), (255, 255, 255)
YELLOW = (0, 255, 255)
LIVE_EVERY = 2          # live cup outline on every 2nd frame
DEFAULT_TABLE_Z = 240.18


def live_outline(frame: np.ndarray, camera: Camera, table_z: float) -> list[np.ndarray] | None:
    """Where the cup is in this frame (same segmenter as the locate step), or None if not visible."""
    def in_workspace(u: float, v: float) -> bool:
        point = camera.ray_to_plane((u, v), table_z + 50.0)
        return WORKSPACE_RADIUS_MM[0] <= float(np.hypot(point[0], point[1])) <= WORKSPACE_RADIUS_MM[1]

    try:
        _, cup = segment_saturated_cyan(frame, accept=in_workspace)
    except (RuntimeError, cv2.error):
        return None
    contours, _ = cv2.findContours(cup, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


TRACK_QUALITY = (0.6, 1.35)  # visible opening area / expected rim area that counts as a clean view


GRIPPER_RADIUS_MM = 65.0     # gripper + wrist, as a vertical cylinder around the fingertip axis
GRIPPER_LENGTH_MM = 320.0    # from the fingertips up past the wrist
ARM_OVERLAP_LIMIT = 0.02     # share of the cup's image area the gripper may cover before a frame is spoiled


def gripper_footprint(camera: Camera, tcp: Sequence[float]) -> np.ndarray:
    """Image polygon of the gripper and wrist above the fingertip position (convex hull of a cylinder)."""
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.column_stack([GRIPPER_RADIUS_MM * np.cos(angles), GRIPPER_RADIUS_MM * np.sin(angles)])
    points = [[tcp[0] + dx, tcp[1] + dy, tcp[2] + dz] for dz in (0.0, GRIPPER_LENGTH_MM) for dx, dy in ring]
    return cv2.convexHull(camera.project(np.array(points, float)).astype(np.float32)).reshape(-1, 2)


def cup_footprint(camera: Camera, x: float, y: float, table_z: float, height: float, top_radius: float,
                  bottom_radius: float) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    points = [[x + r * np.cos(a), y + r * np.sin(a), z] for r, z in ((top_radius, table_z + height),
                                                                      (bottom_radius, table_z)) for a in angles]
    return cv2.convexHull(camera.project(np.array(points, float)).astype(np.float32)).reshape(-1, 2)


def arm_overlap(camera: Camera, tcp: Sequence[float] | None, cup_polygon: np.ndarray, shape: tuple[int, int]) -> float:
    """Share of the cup's image area that the gripper covers (0 when the fingertip position is unknown)."""
    if tcp is None:
        return 0.0
    cup = np.zeros(shape, np.uint8)
    arm = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(cup, np.round(cup_polygon).astype(np.int32), 1)
    cv2.fillConvexPoly(arm, np.round(gripper_footprint(camera, tcp)).astype(np.int32), 1)
    area = int(cup.sum())
    return float(np.logical_and(cup, arm).sum()) / area if area else 0.0


def track_cup(frame: np.ndarray, camera: Camera, table_z: float, height: float, top_radius: float,
              bottom_radius: float | None = None, tcp: Sequence[float] | None = None) -> dict[str, Any]:
    """Fast per-frame cup position for following: rim ellipse at the learned rim height, no full fit.

    Quality = visible opening area / area of the rim circle projected at that position. A hand or
    the gripper covering the rim drops it, and the follower then holds still instead of chasing.
    arm_overlap: the gripper (from the live fingertip position) covers part of the cup in the
    image, so the position is biased even when the quality still looks fine: on 2026-11-04 such
    frames shifted the rim by 3-4 mm and made the arm cycle between approaching and following.
    """
    rim_z = table_z + height
    bottom_radius = top_radius if bottom_radius is None else bottom_radius

    def in_workspace(u: float, v: float) -> bool:
        point = camera.ray_to_plane((u, v), table_z + 50.0)
        return WORKSPACE_RADIUS_MM[0] <= float(np.hypot(point[0], point[1])) <= WORKSPACE_RADIUS_MM[1]

    try:
        opening, _ = segment_saturated_cyan(frame, accept=in_workspace)
        u, v = rim_pixel(opening)
    except (RuntimeError, cv2.error):
        return {"t": time.time(), "ok": False, "why": "not visible"}
    point = camera.ray_to_plane((u, v), rim_z)
    expected = cv2.contourArea(circle_points(camera, point[0], point[1], top_radius, rim_z).astype(np.float32))
    quality = float(opening.sum()) / max(expected, 1.0)
    overlap = arm_overlap(camera, tcp, cup_footprint(camera, point[0], point[1], table_z, height, top_radius,
                                                     bottom_radius), frame.shape[:2])
    covered = overlap > ARM_OVERLAP_LIMIT
    ok = TRACK_QUALITY[0] <= quality <= TRACK_QUALITY[1]
    why = "" if ok and not covered else ("gripper over the cup" if covered else "rim partly hidden")
    return {"t": time.time(), "ok": ok, "xy": [round(float(point[0]), 2), round(float(point[1]), 2)],
            "rim_z": round(rim_z, 2), "quality": round(quality, 3), "arm_overlap": covered,
            "overlap": round(overlap, 3), "why": why}


def put_back_error(camera: Camera, start: dict[str, Any], end: dict[str, Any],
                   reference_xy: Sequence[float] | None = None) -> dict[str, Any]:
    """Rim-centre shift between the start and end photos, both back-projected at the same rim height.

    reference_xy (follow mode): compare with where the cup was grasped instead of the start photo.
    """
    rim_z = end["rim_z"]
    a = (np.asarray(reference_xy, float) if reference_xy is not None
         else camera.ray_to_plane(start["rim_pixel"], rim_z)[:2])
    b = camera.ray_to_plane(end["rim_pixel"], rim_z)[:2]
    return {"start_xy": [round(float(v), 2) for v in a], "end_xy": [round(float(v), 2) for v in b],
            "dx": round(float(b[0] - a[0]), 2), "dy": round(float(b[1] - a[1]), 2),
            "distance_mm": round(float(np.hypot(*(b - a))), 2), "rim_z": round(float(rim_z), 2),
            "pixel_shift": round(float(np.hypot(end["rim_pixel"][0] - start["rim_pixel"][0],
                                                end["rim_pixel"][1] - start["rim_pixel"][1])), 2)}


def emit(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def circle_points(camera: Camera, x: float, y: float, radius: float, z: float, count: int = 120) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, count)
    ring = np.column_stack([x + radius * np.cos(angles), y + radius * np.sin(angles), np.full(count, z)])
    return np.round(camera.project(ring)).astype(np.int32)


def pixel(camera: Camera, point: Sequence[float]) -> tuple[int, int]:
    u, v = camera.project(np.array([point], float))[0]
    return int(round(u)), int(round(v))


def render(frame: np.ndarray, camera: Camera, state: dict[str, Any]) -> np.ndarray:
    """Pure drawing; state holds the last report, plan, tcp and step text."""
    view = frame.copy()
    if state.get("live"):
        cv2.drawContours(view, state["live"], -1, YELLOW, 1, cv2.LINE_AA)
    report = state.get("report")
    if report:
        fit = report["fit"]
        top = fit["table_z"] + fit["height"]
        cv2.polylines(view, [circle_points(camera, fit["x"], fit["y"], fit["top_radius"], top)], True, RED, 2,
                      cv2.LINE_AA)
        cv2.polylines(view, [circle_points(camera, fit["x"], fit["y"], fit["bottom_radius"], fit["table_z"])],
                      True, BLUE, 1, cv2.LINE_AA)
        cv2.line(view, pixel(camera, [fit["x"], fit["y"], fit["table_z"]]), pixel(camera, [fit["x"], fit["y"], top]),
                 GREEN, 1, cv2.LINE_AA)
        cv2.putText(view, f"cup ({fit['x']:.1f}, {fit['y']:.1f})  h {fit['height']:.1f}  rim d {2 * fit['top_radius']:.1f} mm",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
    plan = state.get("plan") or {}
    for name, colour in (("hover", CYAN), ("grasp", GREEN)):
        if plan.get(name):
            centre = pixel(camera, plan[name])
            cv2.circle(view, centre, 6, colour, 2, cv2.LINE_AA)
            cv2.putText(view, name, (centre[0] + 8, centre[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    tcp = state.get("tcp")
    if tcp:
        u, v = pixel(camera, tcp)
        cv2.drawMarker(view, (u, v), MAGENTA, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        text = f"fingertips ({tcp[0]:.1f}, {tcp[1]:.1f}, {tcp[2]:.1f})"
        target = plan.get("grasp") or plan.get("hover")
        if target:
            text += f"  to grasp: dxy {np.hypot(tcp[0] - target[0], tcp[1] - target[1]):.1f}  dz {tcp[2] - target[2]:+.1f} mm"
        cv2.putText(view, text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MAGENTA, 2, cv2.LINE_AA)
    track = state.get("track")
    if track and state.get("tracking"):
        if track.get("xy"):
            centre = pixel(camera, [*track["xy"], track["rim_z"]])
            colour = YELLOW if track["ok"] and not track.get("arm_overlap") else (0, 0, 255)
            cv2.drawMarker(view, centre, colour, cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
        clean = track.get("ok") and not track.get("arm_overlap")
        if track.get("xy"):
            label = f"tracking cup ({track['xy'][0]:.1f}, {track['xy'][1]:.1f})  quality {track['quality']:.2f}"
            if not clean:
                label += f"  - ignored: {track.get('why')}"
        else:
            label = f"tracking: {track.get('why', 'lost')}"
        cv2.putText(view, label, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW if clean else RED, 2, cv2.LINE_AA)
        if state.get("tcp") and track.get("arm_overlap"):
            cv2.polylines(view, [np.round(gripper_footprint(camera, state["tcp"])).astype(np.int32)], True, RED, 1,
                          cv2.LINE_AA)
    if state.get("put_back"):
        error = state["put_back"]
        cv2.putText(view, f"put back: moved {error['distance_mm']:.1f} mm (dx {error['dx']:+.1f}, dy {error['dy']:+.1f})",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2, cv2.LINE_AA)
    cv2.putText(view, "yellow = cup now   red = cup at the start photo", (20, view.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1, cv2.LINE_AA)
    cv2.rectangle(view, (0, 0), (view.shape[1], 40), (30, 30, 30), -1)
    cv2.putText(view, state.get("step", "waiting"), (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2, cv2.LINE_AA)
    cv2.putText(view, "SPACE/Esc = STOP", (view.shape[1] - 250, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
                cv2.LINE_AA)
    return view


def camera_frames(index: int) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开相机 {index}：请关闭占用它的程序")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    failures = 0
    try:
        while True:
            ok, raw = capture.read()
            if not ok or raw is None:
                failures += 1
                if failures > 60:
                    raise RuntimeError("连续读不到相机帧")
                time.sleep(0.03)
                continue
            failures = 0
            if raw.shape[:2] != (720, 1280):
                raise RuntimeError(f"相机分辨率 {raw.shape[1]}x{raw.shape[0]}，不是标定时的 1280x720")
            yield cv2.rotate(raw, cv2.ROTATE_180)
    finally:
        capture.release()


def image_frames(path: Path) -> Iterator[np.ndarray]:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"读不到图像 {path}")
    while True:
        time.sleep(1.0 / FPS)
        yield image.copy()


def serve(frames: Iterator[np.ndarray], commands: "queue.Queue[dict]", out_dir: Path, session: Path, arm: str, *,
          show: Callable[[np.ndarray], int], send: Callable[[dict], None] = emit, record: bool = True,
          warmup: int = 15, memory_path: Path | None = OBJECT_MEMORY) -> None:
    camera = load_camera(session, arm)
    state: dict[str, Any] = {"step": "camera warming up"}
    table_z = DEFAULT_TABLE_Z
    writer = None
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, frame in enumerate(frames):
            if index == warmup:
                send({"type": "ready"})
                state["step"] = "ready"
            while index >= warmup:
                try:
                    command = commands.get_nowait()
                except queue.Empty:
                    break
                if command.get("cmd") == "quit":
                    return
                if command.get("cmd") in ("locate", "relocate"):
                    # relocate: after the put-back, arm at home; measure where the cup ended up.
                    end = command["cmd"] == "relocate"
                    stem = "frame_end" if end else "frame"
                    cv2.imwrite(str(out_dir / f"{stem}.png"), frame)
                    try:
                        camera = load_camera(session, command.get("arm", arm))
                        table_z = float(command["table_z"])
                        report = locate_and_save(frame, out_dir / f"{stem}.png", camera, table_z,
                                                 command.get("arm", arm), command["rpy"], out_dir, stem,
                                                 memory_path=memory_path)
                        if end:
                            if not state.get("report"):
                                raise RuntimeError("没有开始时的定位结果，无法比较放回偏差")
                            state["put_back"] = put_back_error(camera, state["report"], report,
                                                               command.get("reference_xy"))
                            send({"type": "relocate", "report": report, "put_back": state["put_back"]})
                        else:
                            state["report"] = report
                            send({"type": "locate", "report": report})
                    except (RuntimeError, cv2.error) as error:
                        send({"type": "error", "error": str(error)})
                elif command.get("cmd") == "track":
                    state["tracking"] = bool(command.get("on"))
                    if state["tracking"]:
                        state["track_size"] = (float(command["table_z"]), float(command["height"]),
                                               float(command["top_radius"]),
                                               float(command.get("bottom_radius", command["top_radius"])))
                elif command.get("cmd") == "status":
                    state.update({key: command[key] for key in ("step", "tcp", "plan") if key in command})
            if state.get("tracking") and index >= warmup:
                state["track"] = track_cup(frame, camera, *state["track_size"], tcp=state.get("tcp"))
                state["live"] = None  # the tracking marker replaces the outline while following
                send({"type": "track", **state["track"]})
            elif index % LIVE_EVERY == 0:
                state["live"] = live_outline(frame, camera, table_z)
            view = render(frame, camera, state)
            if record:
                if writer is None:
                    writer = cv2.VideoWriter(str(out_dir / "trial_view.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                             (view.shape[1], view.shape[0]))
                writer.write(view)
            if show(view) in (ord(" "), 27):  # space or Esc
                send({"type": "stop"})
                state["step"] = "STOP requested"
    finally:
        if writer is not None:
            writer.release()
            cv2.imwrite(str(out_dir / "trial_view_last.png"), view)


def read_stdin(commands: "queue.Queue[dict]") -> None:
    for line in sys.stdin:
        line = line.strip()
        if line:
            try:
                commands.put(json.loads(line))
            except json.JSONDecodeError:
                pass
    commands.put({"cmd": "quit"})  # the flow went away


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="视觉抓取实时画面（由 run_cup_pick 启动）")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int)
    source.add_argument("--image", type=Path, help="用静态图代替相机（离线演练）")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--headless", action="store_true", help="不开窗口，只录像（测试用）")
    args = parser.parse_args(argv)
    commands: "queue.Queue[dict]" = queue.Queue()
    threading.Thread(target=read_stdin, args=(commands,), daemon=True).start()

    def show(view: np.ndarray) -> int:
        if args.headless:
            return -1
        cv2.imshow(WINDOW, view)
        return cv2.waitKey(1) & 0xFF

    if not args.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 720)
    try:
        frames = camera_frames(args.camera) if args.image is None else image_frames(args.image)
        # A replayed image is not a new view of the cup: it must not vote on the learned size.
        serve(frames, commands, args.out, args.session, args.arm, show=show,
              memory_path=OBJECT_MEMORY if args.image is None else None)
    except RuntimeError as error:
        emit({"type": "error", "error": str(error)})
        return 2
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
