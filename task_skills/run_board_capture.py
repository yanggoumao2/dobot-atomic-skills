"""Capture chessboard photos from the top camera for the calibration wizard.

The wizard's step 2 (intrinsics) needs 20-30 photos of a hand-held board in
varied poses, and step 3 needs one photo of the board fixed on the table. This
program shows a live preview with the detected corners, and space saves the
current frame as PNG for upload into the wizard.

The top camera is mounted upside down, so every frame is rotated by --rotate
(default 180) before preview, detection and saving. That rotation has to be the
same for every image the calibration ever sees - intrinsics, fixed board and
later runtime frames - so it is written into manifest.json next to the images.

--park first homes each named arm (the same layout home and session rules as
run_cartesian_jog: no ClearError, refuse on an alarm) and then disables it, so
the arms stay parked out of the camera's way while the board is moved around.

The two halves need different environments on this PC: the robot SDK needs
pyserial (dobot-xtrainer env), while only OpenCV 4.8.1 in .venv-calibration reads
the top camera (OpenCV 5.0 in dobot-xtrainer opens it but gets no frames). So
parking runs with --park-only in the robot env and capture runs in the wizard env.

    <dobot-xtrainer python> -X utf8 -m task_skills.run_board_capture --set intrinsics --park left,right --park-only
    <.venv-calibration python> -X utf8 -m task_skills.run_board_capture --set intrinsics
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "calibration-data" / "board-images"
DEFAULT_PROFILE = Path(__file__).resolve().parent / "chemistry" / "profile.json"
ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}
WINDOW = "board capture"
SUGGESTED = {"intrinsics": 25, "fixed": 1}
MAX_READ_FAILURES = 60

HELP = """\
按键（焦点在预览窗口上）:
  空格   保存当前画面（必须检测到完整棋盘）
  f      强制保存（未检测到棋盘也存）
  u      删除上一张
  q/Esc  退出"""


def orient(frame: np.ndarray, rotate: int) -> np.ndarray:
    code = ROTATIONS[rotate]
    return frame if code is None else cv2.rotate(frame, code)


def detect(frame: np.ndarray, pattern: tuple[int, int], *, fast: bool) -> np.ndarray | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast:
        flags |= cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    return corners if found else None


class CaptureSet:
    """One folder of saved frames plus everything needed to resume it and show the work.

    The folder holds the board_###.png files the wizard uploads and manifest.json.
    evidence/ keeps what a later report needs and the wizard must not pick up: the
    annotated preview of each save, undone frames, coverage images and the
    capture_log.jsonl event log. Resuming reloads the manifest; each saved board
    outline comes from the manifest, or is re-detected on its PNG for entries
    written before outlines were stored.
    """

    def __init__(self, folder: Path, meta: dict[str, Any], pattern: tuple[int, int]):
        self.folder = folder
        self.meta = meta
        self.pattern = pattern
        self.evidence = folder / "evidence"
        self.images: list[dict[str, Any]] = []
        self.outlines: list[np.ndarray | None] = []

    @classmethod
    def open(cls, folder: Path, meta: dict[str, Any], pattern: tuple[int, int]) -> "CaptureSet":
        capture = cls(folder, meta, pattern)
        manifest_path = folder / "manifest.json"
        if not manifest_path.exists():
            return capture
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("rotate_deg", "board_inner_corners", "set", "camera_device"):
            if manifest.get(key) != meta[key]:
                raise RuntimeError(f"{folder} 的 {key}={manifest.get(key)} 与本次 {meta[key]} 不同，"
                                   "不能接着拍；用 --new 新开一组")
        for entry in manifest.get("images", []):
            path = folder / entry["file"]
            if not path.exists():
                capture.log("missing_on_resume", file=entry["file"])
                continue
            outline = entry.get("outline")
            preview = capture.evidence / entry["file"].replace(".png", "_preview.jpg")
            if (outline is None and entry.get("board_found")) or not preview.exists():
                image = cv2.imread(str(path))
                corners = detect(image, pattern, fast=False)
                if outline is None and corners is not None:
                    outline = cv2.convexHull(corners).reshape(-1, 2).astype(int).tolist()
                    entry["outline"] = outline
                if not preview.exists():  # frames saved before previews were kept
                    if corners is not None:
                        cv2.drawChessboardCorners(image, pattern, corners, True)
                    cv2.putText(image, "preview regenerated on resume", (12, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    capture.evidence.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(preview), image)
                    capture.log("preview_regenerated", file=entry["file"])
            capture.images.append(entry)
            capture.outlines.append(None if outline is None else np.array(outline, np.int32))
        capture._write_manifest()
        return capture

    def log(self, event: str, **fields: Any) -> None:
        self.evidence.mkdir(parents=True, exist_ok=True)
        record = {"time": datetime.now().isoformat(timespec="seconds"), "event": event, **fields}
        with (self.evidence / "capture_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save(self, frame: np.ndarray, corners: np.ndarray | None, preview: np.ndarray) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        self.evidence.mkdir(parents=True, exist_ok=True)
        index = max((image["index"] for image in self.images), default=0) + 1
        path = self.folder / f"board_{index:03d}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"写入失败：{path}")
        cv2.imwrite(str(self.evidence / f"board_{index:03d}_preview.jpg"), preview)
        outline = None if corners is None else cv2.convexHull(corners).reshape(-1, 2).astype(int).tolist()
        self.images.append({"index": index, "file": path.name, "board_found": corners is not None,
                            "saved_at": datetime.now().isoformat(timespec="seconds"),
                            "outline": outline})
        self.outlines.append(None if outline is None else np.array(outline, np.int32))
        self._write_manifest()
        self.log("save", file=path.name, board_found=corners is not None, total=len(self.images))
        return path

    def undo(self) -> str | None:
        """Move the last frame out of the upload folder; it stays under evidence/undone."""
        if not self.images:
            return None
        entry = self.images.pop()
        self.outlines.pop()
        undone = self.evidence / "undone"
        undone.mkdir(parents=True, exist_ok=True)
        source = self.folder / entry["file"]
        if source.exists():
            source.replace(undone / entry["file"])
        preview = self.evidence / entry["file"].replace(".png", "_preview.jpg")
        if preview.exists():
            preview.replace(undone / preview.name)
        self._write_manifest()
        self.log("undo", file=entry["file"], total=len(self.images))
        return entry["file"]

    def write_coverage(self, frame: np.ndarray) -> Path:
        """Every saved board outline over one live frame - the coverage picture for a report."""
        view = frame.copy()
        for outline in self.outlines:
            if outline is not None:
                cv2.polylines(view, [outline], True, (255, 200, 0), 2)
        found = sum(1 for image in self.images if image["board_found"])
        cv2.putText(view, f"{self.meta['set']}: {found} boards, rotate {self.meta['rotate_deg']} deg",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        self.evidence.mkdir(parents=True, exist_ok=True)
        path = self.evidence / f"coverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(path), view)
        return path

    def _write_manifest(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        manifest = {**self.meta, "images": self.images}
        (self.folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")


def latest_set(root: Path, name: str) -> Path | None:
    """Newest <name>-<timestamp> folder that has a manifest, to resume by default."""
    folders = sorted(path for path in root.glob(f"{name}-*") if (path / "manifest.json").exists())
    return folders[-1] if folders else None


def draw(frame: np.ndarray, corners: np.ndarray | None, capture: CaptureSet,
         pattern: tuple[int, int], target: int, message: str) -> np.ndarray:
    """Preview with the live corners and the outlines of every saved board, for coverage."""
    view = frame.copy()
    for outline in capture.outlines:
        if outline is not None:
            cv2.polylines(view, [outline], True, (255, 200, 0), 1)
    if corners is not None:
        cv2.drawChessboardCorners(view, pattern, corners, True)
    status = "board OK - space to save" if corners is not None else "board NOT found"
    color = (0, 200, 0) if corners is not None else (0, 0, 255)
    cv2.putText(view, f"{status}   saved {len(capture.images)}/{target}", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if message:
        cv2.putText(view, message, (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return view


def park(arms: Sequence[str], profile: dict[str, Any], speed: int,
         ask: Callable[[str], str], out: Callable[[str], None],
         arm_factory: Callable[..., Any] | None = None) -> bool:
    """Home each arm, then disable it. False means nothing was sent."""
    from .layout import physical_nova_home_degrees
    from .run_cartesian_jog import LiveArm, start_refusal

    factory = arm_factory or LiveArm
    rigs = [(name, factory(profile, name, speed, log=out)) for name in arms]
    try:
        for name, arm in rigs:
            state = arm.read()
            problem = start_refusal(state.get("mode"))
            if problem:
                out(f"{name} 臂拒绝启动，没有发送任何指令：{problem}")
                return False
            out(f"{name} 臂：RobotMode={state.get('mode')}，将使能 -> JointMovJ 回初始位 "
                f"{list(physical_nova_home_degrees(name))} -> 下使能")
        if ask(f"确认 {'、'.join(arms)} 臂周围无人无障碍、急停在手边，输入 park 继续: ").strip() != "park":
            out("未确认，没有发送任何指令")
            return False
        for name, arm in rigs:
            out(f"{name} 臂：{arm.open_session(home=True)}")
            arm.close()  # disables, then releases the lease
            rigs = [(other, rig) for other, rig in rigs if other != name]
        return True
    finally:
        for _, arm in rigs:
            arm.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="顶部相机棋盘拍照，供标定向导上传")
    parser.add_argument("--set", choices=("intrinsics", "fixed"), required=True,
                        help="intrinsics=手持棋盘多姿态；fixed=固定在桌上的棋盘")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV 设备编号，顶部相机是 0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--rotate", type=int, choices=sorted(ROTATIONS), default=180,
                        help="每帧旋转角度，默认 180（相机倒装）")
    parser.add_argument("--cols", type=int, default=11, help="内角点列数")
    parser.add_argument("--rows", type=int, default=8, help="内角点行数")
    parser.add_argument("--out", type=Path,
                        help="保存目录；默认接着最近一组 <set>-<时间> 拍，没有就新建")
    parser.add_argument("--new", action="store_true", help="不接着上一组，新建一组")
    parser.add_argument("--park", help="拍照前先回初始位再下使能的臂，如 left,right")
    parser.add_argument("--park-only", action="store_true",
                        help="只归位并下使能，不开相机（机器人依赖在 dobot-xtrainer 环境里）")
    parser.add_argument("--speed", type=int, default=10, help="回初始位速度百分比，默认 10")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    return parser


def main(argv: Sequence[str] | None = None, *, ask: Callable[[str], str] = input,
         out: Callable[[str], None] = print, camera: Any = None,
         show: Callable[[np.ndarray], None] | None = None,
         wait_key: Callable[[], int] | None = None,
         arm_factory: Callable[..., Any] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pattern = (args.cols, args.rows)
    if args.park_only and not args.park:
        out("--park-only 需要同时给 --park left,right")
        return 2
    if args.park:
        arms = [name.strip() for name in args.park.split(",") if name.strip()]
        if not arms or any(name not in ("left", "right") for name in arms):
            out("--park 只能是 left、right 或 left,right")
            return 2
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        park_log = (args.out.parent if args.out else DEFAULT_OUT) / "park_log.txt"
        park_log.parent.mkdir(parents=True, exist_ok=True)
        plain_out = out

        def out(text: str) -> None:  # keep the parking transcript for the report
            plain_out(text)
            with park_log.open("a", encoding="utf-8") as handle:
                handle.write(f"{datetime.now().isoformat(timespec='seconds')} {text}\n")
        try:
            if not park(arms, profile, args.speed, ask, out, arm_factory):
                return 1
        except RuntimeError as error:
            out(f"归位失败，已停止（已使能的臂会被下使能）：{error}")
            return 4
        if args.park_only:
            out("已归位并下使能，相机未打开")
            return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    previous = None if args.new or args.out else latest_set(DEFAULT_OUT, args.set)
    folder = args.out or previous or DEFAULT_OUT / f"{args.set}-{stamp}"
    meta = {"camera_device": args.camera, "rotate_deg": args.rotate,
            "board_inner_corners": [args.cols, args.rows], "set": args.set,
            "note": "every frame was rotated by rotate_deg before saving; "
                    "runtime frames must be rotated the same way"}
    try:
        session = CaptureSet.open(folder, meta, pattern)
    except RuntimeError as error:
        out(str(error))
        return 2
    resumed = bool(session.images)
    out(f"{'接着上次继续' if resumed else '新建一组'}：{folder}（已有 {len(session.images)} 张）")
    capture = camera or cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        out(f"无法打开相机 {args.camera}：请关闭占用它的程序（遥操作、标定向导预览等）")
        return 3
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    show = show or (lambda image: cv2.imshow(WINDOW, image))
    wait_key = wait_key or (lambda: cv2.waitKey(1) & 0xFF)
    session.log("resume" if resumed else "start", existing=len(session.images),
                requested_wh=[args.width, args.height],
                argv=list(argv) if argv is not None else sys.argv[1:])
    target = SUGGESTED[args.set]
    message = ""
    out(HELP)
    failures = 0
    last_frame: np.ndarray | None = None
    try:
        while True:
            ok, raw = capture.read()
            if not ok or raw is None:
                failures += 1  # MSMF often drops the first frames while the stream starts
                if failures >= MAX_READ_FAILURES:
                    session.log("camera_read_failed", failures=failures)
                    out(f"连续 {failures} 次读不到相机帧，已退出（相机可能被别的程序占用）")
                    return 3
                cv2.waitKey(30) if camera is None else None
                continue
            failures = 0
            frame = orient(raw, args.rotate)
            height, width = frame.shape[:2]
            if (width, height) != (args.width, args.height):
                out(f"相机实际分辨率 {width}x{height}（旋转后）与要求的 {args.width}x{args.height} 不符，已退出")
                return 3
            last_frame = frame
            corners = detect(frame, pattern, fast=True)
            show(draw(frame, corners, session, pattern, target, message))
            key = wait_key()
            if key in (ord("q"), 27):
                break
            if key in (ord(" "), ord("f")):
                if key == ord(" "):
                    corners = detect(frame, pattern, fast=False)
                    if corners is None:
                        message = "not saved: full board not found (f = force)"
                        continue
                path = session.save(frame, corners,
                                    draw(frame, corners, session, pattern, target, "saved"))
                message = f"saved {path.name}"
                out(f"已保存 {path}" + ("" if corners is not None else "（未检测到棋盘）"))
            elif key == ord("u"):
                removed = session.undo()
                message = f"removed {removed}" if removed else "nothing to undo"
                out(f"已删除 {removed}" if removed else "没有可删除的照片")
    finally:
        if camera is None:
            capture.release()
            cv2.destroyAllWindows()
    found = sum(1 for image in session.images if image["board_found"])
    coverage = session.write_coverage(last_frame) if last_frame is not None and session.images else None
    session.log("exit", total=len(session.images), board_found=found,
                coverage=None if coverage is None else coverage.name)
    out(f"共 {len(session.images)} 张（{found} 张检测到完整棋盘）-> {folder}")
    if coverage is not None:
        out(f"覆盖图与过程记录在 {session.evidence}")
    if args.set == "intrinsics" and found < 13:
        out("提示：内参至少需要 10 张训练 + 3 张验证，建议 25 张以上")
    return 0


if __name__ == "__main__":
    sys.exit(main())
