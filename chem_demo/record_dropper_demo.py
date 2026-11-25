"""Record the dropper demo in one go: C920 recording on, then run_dropper lift (--suck), drip, put-back,
recording off, then build the side-by-side video and a 3x copy.

  python -X utf8 -m chem_demo.record_dropper_demo            # with the robot env (dobot-xtrainer)

Each step is the normal run_dropper command with its own safety checks; a failed step stops the
sequence (put-back still runs if the dropper was lifted). The recorder owns the C920, so run_dropper's
own lifted_front.png grab fails harmlessly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from chem_demo.run_dropper import DATA
from chem_demo.run_handoff import RECORDER_OPEN_DELAY_MS, _stop_recording

ROOT = Path(__file__).resolve().parents[1]


def _start_recording(out: Path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("找不到 ffmpeg，无法录像")
    start_ms = time.time() * 1000.0 + RECORDER_OPEN_DELAY_MS
    process = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "dshow", "-rtbufsize", "200M",
                                "-video_size", "1280x720", "-framerate", "30", "-i", "video=HD Pro Webcam C920",
                                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-y", str(out)],
                               stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return process, start_ms


def _step(args: list[str], tag: str) -> tuple[int, Path | None]:
    before = set(DATA.glob(f"{tag}-*"))
    print(f"\n=== run_dropper {' '.join(args)} ===", flush=True)
    code = subprocess.run([sys.executable, "-X", "utf8", "-m", "chem_demo.run_dropper", *args], cwd=str(ROOT)).returncode
    new = sorted(set(DATA.glob(f"{tag}-*")) - before)
    return code, (new[-1] if new else None)


def main() -> int:
    session = DATA / f"demo-{datetime.now():%Y%m%d_%H%M%S}"
    session.mkdir(parents=True)
    video = session / "c920.mkv"
    recorder, start_ms = _start_recording(video)
    print(f"正面相机录像中：{video}", flush=True)
    time.sleep(3.0)   # let the C920 open before the arm moves
    runs, ok = [], False
    try:
        code, d = _step(["--live", "--suck"], "live-lift")
        runs.append({"step": "lift", "code": code, "dir": str(d)})
        lifted = d is not None and "lifted_pose" in json.loads((d / "dropper_run.json").read_text(encoding="utf-8"))
        if code == 0:
            code, d = _step(["--drip"], "drip")
            runs.append({"step": "drip", "code": code, "dir": str(d)})
        if lifted:
            code, d = _step(["--put-back"], "putback")
            runs.append({"step": "put_back", "code": code, "dir": str(d)})
        ok = all(r["code"] == 0 for r in runs) and len(runs) == 3
        time.sleep(2.0)
    finally:
        _stop_recording(recorder)
        (session / "session.json").write_text(json.dumps({"start_ms": start_ms, "video": str(video), "runs": runs,
                                                          "success": ok}, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(f"\n录像结束，{'全部成功' if ok else '有步骤失败'}：{[(r['step'], r['code']) for r in runs]}", flush=True)
    from chem_demo.wrist_runtime import PERCEPTION_PYTHON
    out = session / "dropper_side_by_side.mp4"
    subprocess.run([str(PERCEPTION_PYTHON), "-X", "utf8", "-m", "chem_demo.make_dropper_video", str(session),
                    "--video", str(video), "--start-ms", str(start_ms), "--out", str(out)],
                   check=True, timeout=900, cwd=str(ROOT))
    fast = session / "dropper_side_by_side_3x.mp4"
    subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(out),
                    "-filter:v", "setpts=PTS/3", "-an", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-r", "30", str(fast)], check=True, timeout=600)
    print(f"对比视频：{out}\n3 倍速：{fast}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
