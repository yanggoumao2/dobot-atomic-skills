"""Synthetic, hardware-free checks for the handoff wrist vision (no models, no robot)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from chem_demo.handoff_vision import (
    alignment_error, jaw_geometry, locate_held_bottle, transfer_check,
)

ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = ROOT / "calibration-data" / "chem-demo" / "bottle-observations"

# Right wrist D405 (serial 409122273461) values copied from checker_multiview_validation.json,
# right-20261121_155150-recovery-20261121_161320_332390. Validated for downward postures only.
K = np.array([[434.329345703125, 0.0, 424.657958984375],
              [0.0, 433.95782470703125, 240.8871307373047],
              [0.0, 0.0, 1.0]])
T_TOOL_CAM = np.array([
    [-0.032299064639426237, 0.8342678671954933, 0.5504124782274661, -75.68321918198086],
    [-0.9989169019354083, -0.045399315479235786, 0.010194370102291461, 10.401021637606418],
    [0.0334931851453969, -0.5494870589186789, 0.8348306287084324, -132.34360182862181],
    [0.0, 0.0, 0.0, 1.0]])
W, H = 848, 480
D_MM = 68.0


def background(seed: int = 1) -> np.ndarray:
    """Textured, mildly shaded scene so ORB has corners and edges have a flat context."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 205, np.uint8)
    for _ in range(400):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        s = int(rng.integers(6, 26))
        shade = int(rng.integers(90, 235))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + s, y + s), (shade, shade, shade), -1)
        else:
            cv2.circle(img, (x, y), s // 2, (shade, shade, int(shade * 0.9)), -1)
    return img


def smooth_background() -> np.ndarray:
    """Clean backdrop for the gradient-edge fallback (it is not meant for cluttered scenes)."""
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 1.5, (H, W, 1))
    return np.clip(205 + noise + np.zeros((H, W, 3)), 0, 255).astype(np.uint8)


def shifted(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), borderMode=cv2.BORDER_REFLECT)


def draw_bottle(img: np.ndarray, u: float, v: float, width_px: float, half_len: int = 130,
                rim_major: float | None = None) -> np.ndarray:
    """Vertical bottle: translucent body, dark glass edges, blue rim ring at the top.

    (u, v) is the body axis at the grasp row, i.e. the middle of the rim-to-bottom span.
    """
    img = img.copy()
    x0, x1 = int(round(u - width_px / 2)), int(round(u + width_px / 2))
    top, bottom = int(round(v - half_len)), int(round(v + half_len))
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, top), (x1, bottom), (215, 225, 205), -1)
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    cv2.line(img, (x0, top), (x0, bottom), (110, 120, 115), 2)
    cv2.line(img, (x1, top), (x1, bottom), (110, 120, 115), 2)
    cv2.line(img, (x0, bottom), (x1, bottom), (110, 120, 115), 2)
    major = rim_major or width_px
    cv2.ellipse(img, (int(round(u)), top), (int(major / 2), int(0.35 * major / 2)), 0, 0, 360,
                (200, 90, 20), 8)  # BGR blue
    return img


def body_mask(u: float, v: float, width_px: float, half_len: int = 130) -> np.ndarray:
    mask = np.zeros((H, W), np.uint8)
    cv2.rectangle(mask, (int(round(u - width_px / 2)), int(round(v - half_len))),
                  (int(round(u + width_px / 2)), int(round(v + half_len))), 1, -1)
    return mask


def width_for_range(z_mm: float) -> float:
    return K[0, 0] * D_MM / z_mm


def scene(error_tool: np.ndarray):
    """Render a bottle whose true axis point sits ``error_tool`` mm from the pad midpoint (tool frame)."""
    jaw = jaw_geometry(K, T_TOOL_CAM)
    assert jaw.ok
    R = T_TOOL_CAM[:3, :3]
    p_cam = np.array(jaw.pad_midpoint_cam_mm) + R.T @ np.asarray(error_tool, float)
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    w = width_for_range(p_cam[2])
    image = draw_bottle(background(), u, v, w)
    mask = body_mask(u, v, w)
    return jaw, image, [{"mask": mask, "score": 0.9}], (u, v, w, p_cam)


class LocateTests(unittest.TestCase):
    def test_range_from_width_nested_ring_and_edges(self):
        # Small bottle (~245 mm): the nested-ring detector sees the rim; no segmenter, edges only.
        w = width_for_range(245.0)
        img = draw_bottle(smooth_background(), 400.0, 240.0, w, half_len=95, rim_major=w * 0.9)
        view = locate_held_bottle(img, None, K, grasp_height_ratio=0.35)
        self.assertTrue(view.ok, view.reason)
        self.assertEqual(view.calibration_validity, "unverified_pose")
        self.assertEqual(view.range_source, "width")
        self.assertAlmostEqual(view.width_px, w, delta=4.0)
        self.assertLess(abs(view.range_mm - 245.0), 0.08 * 245.0)
        self.assertAlmostEqual(view.axis_col_px, 400.0, delta=3.0)
        json.dumps(view.to_dict())

    def test_segmenter_mask_large_bottle_blue_band_rim(self):
        _, image, masks, (u, v, w, p_cam) = scene([0.0, 0.0, 0.0])
        calls = []

        def segmenter(color, prompt):
            calls.append(prompt)
            return masks

        view = locate_held_bottle(image, None, K, segmenter=segmenter)
        self.assertTrue(view.ok, view.reason)
        self.assertEqual(calls, ["glass bottle"])
        self.assertEqual(view.silhouette_source, "segmenter_mask")
        self.assertEqual(view.rim_method, "blue_band")
        self.assertAlmostEqual(view.width_px, w, delta=2.0)
        self.assertAlmostEqual(view.grasp_point_px[0], u, delta=2.0)
        self.assertAlmostEqual(view.grasp_point_px[1], v, delta=3.0)
        self.assertLess(abs(view.range_mm - p_cam[2]), 0.08 * p_cam[2])
        self.assertGreater(view.range_sigma_mm, 0.0)

    def test_valid_depth_used_only_when_consistent(self):
        _, image, masks, (u, v, w, p_cam) = scene([0.0, 0.0, 0.0])
        good = np.zeros((H, W), np.uint16)
        good[:] = int(p_cam[2] - D_MM / 2 + 1)
        view = locate_held_bottle(image, good, K, segmenter=lambda c, p: masks)
        self.assertEqual(view.range_source, "depth")
        self.assertAlmostEqual(view.range_mm, p_cam[2] + 1.0, delta=2.0)
        far = np.full((H, W), 390, np.uint16)  # glass shows the far background
        view = locate_held_bottle(image, far, K, segmenter=lambda c, p: masks)
        self.assertEqual(view.range_source, "width")
        self.assertTrue(any("disagrees" in n for n in view.notes))
        none_valid = np.zeros((H, W), np.uint16)
        view = locate_held_bottle(image, none_valid, K, segmenter=lambda c, p: masks)
        self.assertEqual(view.range_source, "width")
        self.assertEqual(view.depth_valid_fraction, 0.0)

    def test_weak_evidence_returns_reasoned_failure(self):
        blank = background()
        view = locate_held_bottle(blank, None, K)
        self.assertFalse(view.ok)
        self.assertIn("rim", view.reason)
        _, image, masks, (u, v, w, _) = scene([0.0, 0.0, 0.0])
        tiny = [{"mask": body_mask(u, v - 130, 20, half_len=20), "score": 0.9}]  # covers the rim only
        view = locate_held_bottle(image, None, K, segmenter=lambda c, p: tiny)
        self.assertFalse(view.ok)
        self.assertIn("too small", view.reason)
        far_mask = [{"mask": body_mask(100, 400, 60, half_len=40), "score": 0.9}]
        view = locate_held_bottle(image, None, K, segmenter=lambda c, p: far_mask)
        self.assertFalse(view.ok)
        view = locate_held_bottle(image, None, K, segmenter=lambda c, p: masks, body_diameter_mm=10.0)
        self.assertFalse(view.ok)

    def test_two_bottles_are_ambiguous(self):
        img = draw_bottle(background(), 220.0, 240.0, 120.0, half_len=95, rim_major=110)
        img = draw_bottle(img, 600.0, 240.0, 120.0, half_len=95, rim_major=110)
        view = locate_held_bottle(img, None, K)
        self.assertFalse(view.ok)
        self.assertIn("ambiguous", view.reason)

    def test_range_outside_limits_rejected(self):
        # A 22 mm wide-looking bottle at this width would be ~1 m away: outside [60, 400] mm.
        img = draw_bottle(background(), 400.0, 240.0, 30.0, half_len=60, rim_major=30)
        masks = [{"mask": body_mask(400, 240, 30, half_len=60), "score": 0.9}]
        view = locate_held_bottle(img, None, K, segmenter=lambda c, p: masks)
        self.assertFalse(view.ok)

    def test_truncated_at_border(self):
        w = width_for_range(150.0)
        u = w / 2 - 6  # left edge outside the image
        img = draw_bottle(background(), u, 240.0, w)
        masks = [{"mask": body_mask(u, 240.0, w), "score": 0.9}]
        view = locate_held_bottle(img, None, K, segmenter=lambda c, p: masks)
        self.assertFalse(view.ok)
        self.assertTrue(view.truncated_by_border or "rim" in view.reason)

    @unittest.skipUnless(OBSERVATIONS.exists(), "saved bottle observations not present")
    def test_real_saved_frames_return_structured_result(self):
        frames = sorted(OBSERVATIONS.rglob("wrist_observation.png"))[:3]
        if not frames:
            self.skipTest("no wrist_observation.png frames")
        for path in frames:
            frame = cv2.imread(str(path))
            self.assertIsNotNone(frame)
            view = locate_held_bottle(frame, None, K)
            data = view.to_dict()
            json.dumps(data)
            self.assertIn(data["ok"], (True, False))
            self.assertEqual(data["calibration_validity"], "unverified_pose")
            self.assertTrue(data["ok"] or data["reason"])


class JawTests(unittest.TestCase):
    def test_projection_matches_calibration_geometry(self):
        jaw = jaw_geometry(K, T_TOOL_CAM)
        self.assertTrue(jaw.ok)
        self.assertEqual(jaw.source, "projected_from_calibration")
        self.assertEqual(jaw.calibration_validity, "unverified_pose")
        # Tool origin (fingertips) sits ~152 mm in front of the camera; pads open 91.6 mm along tool Y.
        self.assertAlmostEqual(jaw.pad_midpoint_cam_mm[2], 152.0, delta=1.0)
        gap = np.linalg.norm(np.subtract(*jaw.pad_tips_cam_mm))
        self.assertAlmostEqual(gap, 91.6, delta=0.01)
        tips = np.array(jaw.pad_tips_px)
        self.assertAlmostEqual(float(abs(tips[0, 0] - tips[1, 0])), 261.0, delta=4.0)
        np.testing.assert_allclose(jaw.pad_midpoint_tool_mm, [0, 0, 0], atol=1e-6)
        json.dumps(jaw.to_dict())

    def test_measured_tips_agree_with_projection_within_mount_error(self):
        # Pad tips measured on the right reference frame with wrist_mount's flipped detector:
        # (323.9, 203.0) and (599.0, 212.0) in the original image orientation.
        measured = jaw_geometry(K, T_TOOL_CAM, measured_tips_px=[(323.9, 203.0), (599.0, 212.0)])
        projected = jaw_geometry(K, T_TOOL_CAM)
        self.assertEqual(measured.source, "measured_tips")
        self.assertLess(np.hypot(*(np.array(measured.pad_midpoint_px) - projected.pad_midpoint_px)), 15.0)
        self.assertAlmostEqual(measured.pad_midpoint_cam_mm[2], 145.0, delta=8.0)

    def test_invalid_inputs(self):
        self.assertFalse(jaw_geometry(np.eye(2), T_TOOL_CAM).ok)
        self.assertFalse(jaw_geometry(K, np.eye(4) * 2).ok)
        self.assertFalse(jaw_geometry(K, T_TOOL_CAM, jaw_open_mm=-1.0).ok)
        self.assertFalse(jaw_geometry(K, T_TOOL_CAM, measured_tips_px=[(10, 10), (12, 10)]).ok)


class AlignmentTests(unittest.TestCase):
    def run_case(self, error_tool, **kwargs):
        jaw, image, masks, _ = scene(error_tool)
        view = locate_held_bottle(image, None, K, segmenter=lambda c, p: masks, **kwargs)
        self.assertTrue(view.ok, view.reason)
        return view, jaw, alignment_error(view, jaw, tool_from_camera=T_TOOL_CAM)

    def test_centred_bottle_has_near_zero_error_and_passes_gates(self):
        view, jaw, err = self.run_case([0.0, 0.0, 0.0])
        self.assertTrue(err.ok and err.gates_pass, err.reason)
        for value in (err.lateral_mm, err.vertical_mm, err.range_mm):
            self.assertLess(abs(value), 6.0)
        self.assertAlmostEqual(err.margin_mm, (91.6 - 68.0) / 2, places=6)
        self.assertEqual(err.calibration_validity, "unverified_pose")
        self.assertTrue(all(s > 0 for s in err.sigma_mm))
        json.dumps(err.to_dict())

    def test_shift_sign_and_magnitude_per_axis(self):
        # Components are (lateral = tool Y, vertical = tool X, range = tool Z) in mm.
        cases = {"left": ([0.0, 7.0, 0.0], "lateral_mm", 7.0), "right": ([0.0, -7.0, 0.0], "lateral_mm", -7.0),
                 "up": ([8.0, 0.0, 0.0], "vertical_mm", 8.0), "down": ([-8.0, 0.0, 0.0], "vertical_mm", -8.0),
                 "far": ([0.0, 0.0, 12.0], "range_mm", 12.0)}
        for name, (truth, attribute, expected) in cases.items():
            with self.subTest(name):
                _, _, err = self.run_case(truth)
                self.assertAlmostEqual(getattr(err, attribute), expected, delta=3.5)
                self.assertTrue(err.ok)

    def test_image_directions_follow_the_calibration(self):
        # tool +Y is image-left in this camera: moving the bottle right in the image is lateral < 0.
        jaw, _, masks, (u, v, w, _) = scene([0.0, 0.0, 0.0])
        image = draw_bottle(background(), u + 20, v, w)
        view = locate_held_bottle(image, None, K, segmenter=lambda c, p: [
            {"mask": body_mask(u + 20, v, w), "score": 0.9}])
        err = alignment_error(view, jaw, tool_from_camera=T_TOOL_CAM)
        self.assertLess(err.lateral_mm, -4.0)

    def test_gate_bottle_too_far_to_one_side(self):
        _, _, err = self.run_case([0.0, 14.0, 0.0])
        self.assertTrue(err.ok)
        self.assertFalse(err.gates["inside_pad_span"])
        self.assertFalse(err.gates_pass)
        self.assertIn("inside_pad_span", err.reason)

    def test_gate_bottle_too_wide_for_opening(self):
        jaw, image, masks, _ = scene([0.0, 0.0, 0.0])
        view = locate_held_bottle(image, None, K, segmenter=lambda c, p: masks, body_diameter_mm=95.0)
        err = alignment_error(view, jaw, tool_from_camera=T_TOOL_CAM)
        self.assertFalse(err.gates["opening_wide_enough"])
        self.assertFalse(err.gates_pass)
        narrow = jaw_geometry(K, T_TOOL_CAM, jaw_open_mm=70.0)
        view = locate_held_bottle(image, None, K, segmenter=lambda c, p: masks)
        err = alignment_error(view, narrow, tool_from_camera=T_TOOL_CAM)
        self.assertFalse(err.gates["opening_wide_enough"])
        self.assertFalse(err.gates["inside_pad_span"])

    def test_gate_bottle_near_image_border(self):
        jaw = jaw_geometry(K, T_TOOL_CAM)
        w = width_for_range(150.0)
        u = W - w / 2 - 6  # right edge 6 px from the border: measurable but inside the gate margin
        image = draw_bottle(background(), u, 240.0, w)
        masks = [{"mask": body_mask(u, 240.0, w), "score": 0.9}]
        view = locate_held_bottle(image, None, K, segmenter=lambda c, p: masks)
        self.assertTrue(view.ok, view.reason)
        err = alignment_error(view, jaw, tool_from_camera=T_TOOL_CAM)
        self.assertFalse(err.gates["inside_image"])
        self.assertFalse(err.gates_pass)

    def test_gate_bottle_touching_pad_in_image(self):
        _, _, err = self.run_case([0.0, -13.0, 0.0])
        self.assertFalse(err.gates["not_touching_pads"])

    def test_missing_bottle_or_jaw_gives_not_ok(self):
        jaw = jaw_geometry(K, T_TOOL_CAM)
        view = locate_held_bottle(background(), None, K)
        err = alignment_error(view, jaw, tool_from_camera=T_TOOL_CAM)
        self.assertFalse(err.ok)
        self.assertFalse(err.gates_pass)
        self.assertEqual(err.calibration_validity, "unverified_pose")
        _, image, masks, _ = scene([0.0, 0.0, 0.0])
        good = locate_held_bottle(image, None, K, segmenter=lambda c, p: masks)
        bad_jaw = jaw_geometry(np.eye(2), T_TOOL_CAM)
        self.assertFalse(alignment_error(good, bad_jaw, tool_from_camera=T_TOOL_CAM).ok)
        self.assertFalse(alignment_error(good, jaw, tool_from_camera=np.eye(4) * 3).ok)


class TransferTests(unittest.TestCase):
    BOTTLE = (420.0, 240.0, 170.0)

    def frame(self, bg_shift, bottle_shift):
        bg = shifted(background(7), *bg_shift)
        u, v, w = self.BOTTLE
        return draw_bottle(bg, u + bottle_shift[0], v + bottle_shift[1], w, half_len=110, rim_major=w * 0.9)

    def segmenter(self, bottle_shift):
        # Called once per frame: first the "before" frame, then the "after" frame.
        u, v, w = self.BOTTLE
        masks = [[{"mask": body_mask(u, v, w, 110), "score": 0.9}],
                 [{"mask": body_mask(u + bottle_shift[0], v + bottle_shift[1], w, 110), "score": 0.9}]]
        it = iter(masks)
        return lambda color, prompt: next(it)

    def test_bottle_static_background_shifted_is_held(self):
        before, after = self.frame((0, 0), (0, 0)), self.frame((34, 14), (0, 0))
        verdict = transfer_check(before, after, K, segmenter=self.segmenter((0, 0)))
        self.assertIs(verdict.held, True, verdict.reason)
        self.assertGreater(verdict.bg_shift_px, 30.0)
        self.assertLess(verdict.bottle_shift_px, 3.0)
        self.assertEqual(verdict.calibration_validity, "unverified_pose")
        json.dumps(verdict.to_dict())

    def test_bottle_moves_with_background_is_not_held(self):
        before, after = self.frame((0, 0), (0, 0)), self.frame((34, 14), (34, 14))
        verdict = transfer_check(before, after, K, segmenter=self.segmenter((34, 14)))
        self.assertIs(verdict.held, False, verdict.reason)
        self.assertGreater(verdict.bottle_shift_px, 30.0)

    def test_no_background_shift_is_uncertain_never_held(self):
        before, after = self.frame((0, 0), (0, 0)), self.frame((0, 0), (0, 0))
        verdict = transfer_check(before, after, K, segmenter=self.segmenter((0, 0)))
        self.assertIsNone(verdict.held)
        self.assertFalse(verdict.ok)
        self.assertIn("did not", verdict.reason)
        self.assertLess(verdict.bg_shift_px, 2.0)

    def test_small_background_shift_below_threshold_is_uncertain(self):
        before, after = self.frame((0, 0), (0, 0)), self.frame((3, 0), (0, 0))
        verdict = transfer_check(before, after, K, segmenter=self.segmenter((0, 0)))
        self.assertIsNone(verdict.held)

    def test_bottle_missing_in_one_frame_is_uncertain(self):
        before = self.frame((0, 0), (0, 0))
        verdict = transfer_check(before, shifted(background(7), 30, 0), K)
        self.assertIsNone(verdict.held)
        self.assertIn("rim", verdict.reason)

    def test_accepts_frame_dicts(self):
        before, after = self.frame((0, 0), (0, 0)), self.frame((34, 14), (0, 0))
        verdict = transfer_check({"color": before}, {"color": after}, K,
                                 segmenter=self.segmenter((0, 0)))
        self.assertIs(verdict.held, True, verdict.reason)


class MergedTailTest(unittest.TestCase):
    """2026-11-23 16:07 real right-wrist frames: in frame 6 SAM merged a white cylinder under the bottle."""

    FIX = Path(__file__).parent / "fixtures" / "handoff_20261123"
    K = np.array([[434.33, 0.0, 424.0], [0.0, 434.33, 240.0], [0.0, 0.0, 1.0]])
    DOWN = (-0.0647508933133587, 0.9979014589703344)

    def view(self, index):
        color = cv2.imread(str(self.FIX / f"right_obs_{index:03d}.png"))
        mask = cv2.imread(str(self.FIX / f"seg_{index:03d}_mask.png"), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(color)
        self.assertIsNotNone(mask)
        return locate_held_bottle(color, None, self.K, segmenter=lambda _img, _p: [{"mask": mask > 0, "score": 0.9}],
                                  down_px=self.DOWN)

    def test_merged_tail_is_trimmed_and_both_frames_agree(self):
        clean, merged = self.view(5), self.view(6)
        self.assertTrue(clean.ok, clean.reason)
        self.assertTrue(merged.ok, merged.reason)
        self.assertFalse(any("trimmed" in note for note in clean.notes))
        self.assertTrue(any("trimmed" in note for note in merged.notes))
        # One approach step (12 mm) apart: same grasp row and width within a few pixels / 10 %.
        self.assertLess(abs(merged.grasp_point_px[1] - clean.grasp_point_px[1]), 12.0)
        self.assertLess(abs(merged.width_px - clean.width_px) / clean.width_px, 0.10)

    def test_rounded_bottom_at_close_range_is_not_trimmed(self):
        # 16:48 run, ~9 cm from the bottle: 142 px wide; the rounded bottom must stay.
        color = cv2.imread(str(self.FIX / "right_obs_b010.png"))
        mask = cv2.imread(str(self.FIX / "seg_b010_mask.png"), cv2.IMREAD_GRAYSCALE)
        view = locate_held_bottle(color, None, self.K, segmenter=lambda _i, _p: [{"mask": mask > 0, "score": 0.9}],
                                  down_px=self.DOWN)
        self.assertTrue(view.ok, view.reason)
        self.assertFalse(any("trimmed" in note for note in view.notes))


if __name__ == "__main__":
    unittest.main()


class SideViewTest(unittest.TestCase):
    """Jaws horizontal: no rim visible, axis from the arm pose, grasp row measured from the bottom."""

    def frame(self, top=60, bottom=380, left=300, right=500):
        image = np.full((480, 848, 3), 200, np.uint8)       # plain grey, no blue anywhere
        mask = np.zeros((480, 848), np.uint8)
        mask[top:bottom, left:right] = 1
        return image, mask

    def test_grasp_row_is_measured_up_from_the_bottom(self):
        image, mask = self.frame()
        view = locate_held_bottle(image, None, K, segmenter=lambda c, p: [{"mask": mask, "score": 0.9}],
                                  down_px=(0.0, 1.0), body_diameter_mm=68.0, grasp_from_bottom_mm=40.0)
        self.assertTrue(view.ok, view.reason)
        self.assertEqual(view.rim_method, "side_view")
        px_per_mm = 200.0 / 68.0                               # mask width 200 px (299 .. 499)
        self.assertAlmostEqual(view.grasp_point_px[1], 379 - 40 * px_per_mm, delta=3.0)
        self.assertAlmostEqual(view.grasp_point_px[0], 399.5, delta=1.5)
        self.assertGreater(view.range_mm, 100.0)

    def test_bottom_outside_the_view_is_refused(self):
        image, mask = self.frame(top=100, bottom=480)
        view = locate_held_bottle(image, None, K, segmenter=lambda c, p: [{"mask": mask, "score": 0.9}],
                                  down_px=(0.0, 1.0))
        self.assertFalse(view.ok)
        self.assertIn("bottom", view.reason)

    def test_no_mask_says_no_bottle(self):
        image, _ = self.frame()
        view = locate_held_bottle(image, None, K, segmenter=lambda c, p: [], down_px=(0.0, 1.0))
        self.assertFalse(view.ok)
        self.assertIn("no bottle", view.reason)

    def test_nearest_of_two_masks_to_the_expected_pixel_wins(self):
        image, near = self.frame(left=500, right=640)
        _, far = self.frame(left=40, right=180)
        view = locate_held_bottle(image, None, K, down_px=(0.0, 1.0), expected_px=(570.0, 250.0),
                                  segmenter=lambda c, p: [{"mask": far, "score": 0.9}, {"mask": near, "score": 0.9}])
        self.assertTrue(view.ok, view.reason)
        self.assertGreater(view.grasp_point_px[0], 500.0)
