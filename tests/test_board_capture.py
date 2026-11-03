import json
import unittest
import unittest.mock
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from task_skills import run_board_capture as capmod


def board_frame(width=1280, height=720, square=40, cols=11, rows=8):
    """A flat 12x9-square board on white, with a red marker in the raw top-left corner."""
    frame = np.full((height, width, 3), 255, np.uint8)
    x0, y0 = 300, 150
    for row in range(rows + 1):
        for col in range(cols + 1):
            if (row + col) % 2 == 0:
                cv2.rectangle(frame, (x0 + col * square, y0 + row * square),
                              (x0 + (col + 1) * square - 1, y0 + (row + 1) * square - 1), (0, 0, 0), -1)
    frame[0:4, 0:4] = (0, 0, 255)
    return frame


class FakeCamera:
    def __init__(self, frames):
        self.frames = list(frames)

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def read(self):
        return (True, self.frames.pop(0).copy()) if self.frames else (False, None)


def run(frames, keys, extra=(), answers=(), arm_factory=None):
    keys = iter(keys)
    replies = iter(answers)
    lines = []
    directory = TemporaryDirectory()
    out_dir = Path(directory.name) / "set"
    code = capmod.main(["--set", "intrinsics", "--out", str(out_dir), *extra],
                       ask=lambda _: next(replies), out=lines.append, camera=FakeCamera(frames),
                       show=lambda _: None, wait_key=lambda: next(keys, ord("q")),
                       arm_factory=arm_factory)
    return code, out_dir, lines, directory


class BoardCaptureTests(unittest.TestCase):
    """Saved images are what the wizard will calibrate on, so their content is what matters."""

    def test_space_saves_only_a_detected_board_and_force_saves_anything(self):
        blank = np.full((720, 1280, 3), 255, np.uint8)
        code, folder, lines, keep = run([board_frame(), blank, blank, board_frame(), blank],
                                        [ord(" "), ord(" "), ord("f"), ord(" ")])
        with keep:
            self.assertEqual(code, 0)
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([image["board_found"] for image in manifest["images"]], [True, False, True])
            self.assertEqual(sorted(p.name for p in folder.glob("*.png")),
                             ["board_001.png", "board_002.png", "board_003.png"])
            self.assertEqual(manifest["rotate_deg"], 180)

    def test_saved_frames_are_rotated_180_degrees(self):
        code, folder, _, keep = run([board_frame()], [ord(" ")])
        with keep:
            saved = cv2.imread(str(folder / "board_001.png"))
            self.assertEqual(saved.shape[:2], (720, 1280))
            self.assertEqual(tuple(saved[-1, -1]), (0, 0, 255))  # raw top-left is now bottom-right
            self.assertEqual(tuple(saved[0, 0]), (255, 255, 255))

    def test_undo_takes_the_last_image_out_of_the_upload_folder_but_keeps_it(self):
        code, folder, _, keep = run([board_frame()] * 4, [ord(" "), ord(" "), ord("u")])
        with keep:
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([image["file"] for image in manifest["images"]], ["board_001.png"])
            self.assertFalse((folder / "board_002.png").exists())
            self.assertTrue((folder / "evidence" / "undone" / "board_002.png").exists())
            self.assertTrue((folder / "evidence" / "undone" / "board_002_preview.jpg").exists())

    def test_wrong_resolution_stops_instead_of_saving_mismatched_images(self):
        code, folder, lines, keep = run([np.zeros((480, 640, 3), np.uint8)], [ord(" ")])
        with keep:
            self.assertEqual(code, 3)
            self.assertEqual(list(folder.glob("*.png")), [])

    def test_every_save_leaves_report_evidence(self):
        code, folder, _, keep = run([board_frame()] * 3, [ord(" "), ord(" ")])
        with keep:
            evidence = folder / "evidence"
            self.assertTrue((evidence / "board_001_preview.jpg").exists())
            self.assertTrue((evidence / "board_002_preview.jpg").exists())
            self.assertEqual(len(list(evidence.glob("coverage_*.jpg"))), 1)
            events = [json.loads(line)["event"]
                      for line in (evidence / "capture_log.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events, ["start", "save", "save", "exit"])
            # The upload folder holds only what the wizard should take.
            self.assertEqual(sorted(p.name for p in folder.glob("*.png")),
                             ["board_001.png", "board_002.png"])


class ResumeTests(unittest.TestCase):
    """Stopping and restarting continues the same set, with the saved outlines back."""

    def _run_in(self, folder, frames, keys, extra=()):
        keys = iter(keys)
        lines = []
        code = capmod.main(["--set", "intrinsics", "--out", str(folder), *extra], out=lines.append,
                           camera=FakeCamera(frames), show=lambda _: None,
                           wait_key=lambda: next(keys, ord("q")))
        return code, lines

    def test_restart_continues_numbering_and_restores_outlines(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "set"
            self._run_in(folder, [board_frame()] * 3, [ord(" "), ord(" ")])
            code, lines = self._run_in(folder, [board_frame()] * 2, [ord(" ")])
            self.assertIn("接着上次继续", lines[0])
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([image["file"] for image in manifest["images"]],
                             ["board_001.png", "board_002.png", "board_003.png"])
            session = capmod.CaptureSet.open(folder, {key: manifest[key] for key in
                                                      ("camera_device", "rotate_deg",
                                                       "board_inner_corners", "set", "note")}, (11, 8))
            self.assertEqual(sum(outline is not None for outline in session.outlines), 3)
            events = [json.loads(line)["event"] for line in
                      (folder / "evidence" / "capture_log.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events.count("resume"), 1)

    def test_resume_redetects_outlines_for_images_saved_before_outlines_were_kept(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "set"
            folder.mkdir()
            cv2.imwrite(str(folder / "board_001.png"), cv2.rotate(board_frame(), cv2.ROTATE_180))
            (folder / "manifest.json").write_text(json.dumps({
                "camera_device": 0, "rotate_deg": 180, "board_inner_corners": [11, 8],
                "set": "intrinsics", "images": [{"index": 1, "file": "board_001.png",
                                                 "board_found": True}]}), encoding="utf-8")
            code, _ = self._run_in(folder, [board_frame()] * 2, [ord(" ")])
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["images"][0]["outline"]) >= 4, True)
            self.assertEqual(manifest["images"][1]["file"], "board_002.png")

    def test_a_set_saved_with_other_settings_is_not_mixed_in(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "set"
            self._run_in(folder, [board_frame()] * 2, [ord(" ")])
            code, lines = self._run_in(folder, [board_frame()], [], ["--rotate", "0"])
            self.assertEqual(code, 2)
            self.assertIn("--new", lines[-1])

    def test_default_folder_is_the_latest_set_and_new_starts_another(self):
        with TemporaryDirectory() as directory, \
                unittest.mock.patch.object(capmod, "DEFAULT_OUT", Path(directory)):
            def go(extra=()):
                keys = iter([ord(" ")])
                capmod.main(["--set", "intrinsics", *extra], out=lambda _: None,
                            camera=FakeCamera([board_frame()] * 2), show=lambda _: None,
                            wait_key=lambda: next(keys, ord("q")))
                return sorted(Path(directory).glob("intrinsics-*"))
            first = go()
            self.assertEqual(len(first), 1)
            self.assertEqual(go(), first)                 # resumed, no new folder
            with unittest.mock.patch.object(capmod, "datetime") as clock:
                clock.now.return_value.strftime.return_value = "29991231_235959"
                clock.now.return_value.isoformat.return_value = "2999-12-31T23:59:59"
                self.assertEqual(len(go(["--new"])), 2)



class ParkTests(unittest.TestCase):
    """Parking homes then disables each arm, and refuses before sending anything."""

    def _factory(self, modes, events, fail=None):
        class Arm:
            def __init__(self, profile, name, speed, log=print):
                self.name = name

            def read(self):
                return {"mode": modes[self.name]}

            def open_session(self, home=True):
                events.append(("home", self.name, home))
                if fail == self.name:
                    raise RuntimeError("arrival timeout")
                return "已回到初始位"

            def close(self):
                events.append(("close", self.name))
        return Arm

    def test_park_homes_then_disables_each_arm_in_turn(self):
        events = []
        code, _, _, keep = run([board_frame()], [], ["--park", "left,right"], ["park"],
                               self._factory({"left": 4, "right": 5}, events))
        with keep:
            self.assertEqual(code, 0)
            self.assertEqual(events, [("home", "left", True), ("close", "left"),
                                      ("home", "right", True), ("close", "right")])

    def test_park_only_never_opens_the_camera(self):
        events = []
        code, folder, lines, keep = run([], [], ["--park", "right", "--park-only"], ["park"],
                                        self._factory({"right": 4}, events))
        with keep:
            self.assertEqual(code, 0)
            self.assertEqual(events, [("home", "right", True), ("close", "right")])
            self.assertFalse(folder.exists())

    def test_alarm_or_missing_confirmation_sends_nothing(self):
        for modes, answers in (({"left": 9, "right": 5}, ()), ({"left": 5, "right": 5}, ("no",))):
            events = []
            code, _, _, keep = run([board_frame()], [], ["--park", "left,right"], answers,
                                   self._factory(modes, events))
            with keep:
                self.assertEqual(code, 1)
                self.assertNotIn("home", [event[0] for event in events])

    def test_a_failed_home_still_closes_every_arm(self):
        events = []
        code, _, lines, keep = run([board_frame()], [], ["--park", "left,right"], ["park"],
                                   self._factory({"left": 5, "right": 5}, events, fail="left"))
        with keep:
            self.assertEqual(code, 4)
            self.assertEqual(sorted(name for kind, name, *_ in events if kind == "close"),
                             ["left", "right"])


if __name__ == "__main__":
    unittest.main()
