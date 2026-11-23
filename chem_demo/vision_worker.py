"""GPU vision worker for the robot process: Grounding DINO + SAM 2 loaded once, JSON lines on stdio.

The robot code runs in the dobot-xtrainer env (serial, Dobot SDK, RealSense); the models live in
.venv-perception. The robot process starts this worker, writes frames to disk and sends requests:

    {"id": 1, "cmd": "locate", "image": "<top.png>", "out": "<dir>"}      top-camera scene (chem_demo.locate)
    {"id": 2, "cmd": "detect", "image": "<wrist.png>", "mask": "<out.png>",
     "near": [u, v] | null, "exclude": "<gripper.png>" | null}          wrist bottle mask
    {"id": 3, "cmd": "held_bottle", "image": "<top.png>", "pose": [left flange pose],
     "overlay": "<out.png>"}                                             bottle hanging from the left jaws

Each reply carries the same id. stdout is reserved for replies; anything the libraries print goes
to stderr. Read-only: never touches the arm.

    .venv-perception/Scripts/python.exe -X utf8 -m chem_demo.vision_worker
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

BOTTLE_PROMPTS = {"glass bottle": "bottle", "glass reagent bottle": "bottle", "bottle": "bottle"}


def choose(detections: list, near: list | None, exclude: np.ndarray | None, shape: tuple[int, int]):
    """Best bottle candidate: high score, near the expected pixel, not mostly the gripper."""
    h, w = shape
    best, best_cost = None, None
    for d in detections:
        if d.role != "bottle":
            continue
        mask = d.mask.astype(bool)
        area = mask.sum()
        if area < 0.002 * h * w:
            continue
        if exclude is not None and (mask & exclude).sum() > 0.5 * area:
            continue  # a finger answered to "bottle"
        cost = -d.score
        if near is not None:
            ys, xs = np.nonzero(mask)
            cost += 0.4 * np.hypot(xs.mean() - near[0], ys.mean() - near[1]) / max(h, w)
        if best_cost is None or cost < best_cost:
            best, best_cost = d, cost
    return best


def handle(request: dict, segmenter, cameras: dict) -> dict:
    cmd = request.get("cmd")
    if cmd == "detect":
        image = cv2.imread(request["image"])
        if image is None:
            return {"ok": False, "error": f"读不到图像 {request['image']}"}
        exclude = None
        if request.get("exclude"):
            raw = cv2.imread(request["exclude"], cv2.IMREAD_GRAYSCALE)
            exclude = None if raw is None else raw > 0
        t0 = time.time()
        found = segmenter.detect(image, vocabulary=BOTTLE_PROMPTS, box_threshold=request.get("threshold", 0.2),
                                 size_filter=False)
        best = choose(found, request.get("near"), exclude, image.shape[:2])
        candidates = [{"role": d.role, "score": round(d.score, 3), "box": [round(v, 1) for v in d.box]} for d in found]
        if best is None:
            return {"ok": False, "error": "腕部画面里没找到瓶子", "candidates": candidates}
        cv2.imwrite(request["mask"], best.mask.astype(np.uint8) * 255)
        return {"ok": True, "mask": request["mask"], "score": round(best.score, 3),
                "box": [round(v, 1) for v in best.box], "candidates": candidates,
                "seconds": round(time.time() - t0, 2)}
    if cmd == "locate":
        from chem_demo.locate import draw as draw_scene, locate_scene, to_left

        right, left = cameras["right"], cameras["left"]
        T_right_left = right.T_base_camera @ np.linalg.inv(left.T_base_camera)
        image = cv2.imread(request["image"])
        if image is None:
            return {"ok": False, "error": f"读不到图像 {request['image']}"}
        out = Path(request["out"])
        out.mkdir(parents=True, exist_ok=True)
        report = locate_scene(image, right, segmenter)
        cv2.imwrite(str(out / "locate.jpg"), draw_scene(image, right, report))
        vessels = {}
        for v in report["vessels"]:
            v.pop("_mask")
            v["left_base"] = {"axis_xyz_at_rim": to_left([v["x"], v["y"], v["rim_z"]], T_right_left),
                              "axis_xyz_at_base": to_left([v["x"], v["y"], v["base_z"]], T_right_left)}
            vessels[v["name"]] = v
        (out / "locate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "vessels": vessels, "notes": report["notes"], "overlay": str(out / "locate.jpg")}
    if cmd == "held_bottle":
        from chem_demo.held_bottle_top import draw as draw_held, measure_held_bottle

        image = cv2.imread(request["image"])
        if image is None:
            return {"ok": False, "error": f"读不到图像 {request['image']}"}
        t0 = time.time()
        result, mask = measure_held_bottle(image, cameras["left"], request["pose"], segmenter)
        overlay = request.get("overlay")
        if overlay:
            cv2.imwrite(overlay, draw_held(image, cameras["left"], result, mask))
        return {"ok": True, "measurement": result.to_dict(), "seconds": round(time.time() - t0, 2)}
    if cmd == "ping":
        return {"ok": True}
    return {"ok": False, "error": f"未知命令 {cmd}"}


def main() -> int:
    replies = sys.stdout
    sys.stdout = sys.stderr  # library prints must not corrupt the reply stream
    from chem_demo.camera_drift import load_corrected_camera
    from chem_demo.segment import Segmenter
    from vision_pick.cup_locate import DEFAULT_SESSION

    t0 = time.time()
    segmenter = Segmenter()
    cameras = {"right": load_corrected_camera(DEFAULT_SESSION, "right"),
               "left": load_corrected_camera(DEFAULT_SESSION, "left")}
    replies.write(json.dumps({"type": "ready", "seconds": round(time.time() - t0, 1)}) + "\n")
    replies.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        if request.get("cmd") == "quit":
            break
        try:
            reply = handle(request, segmenter, cameras)
        except Exception as error:  # noqa: BLE001 - one bad frame must not end the worker
            reply = {"ok": False, "error": f"{type(error).__name__}: {error}",
                     "trace": traceback.format_exc()[-1500:]}
        reply.update({"type": "reply", "id": request.get("id")})
        replies.write(json.dumps(reply, ensure_ascii=False) + "\n")
        replies.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
