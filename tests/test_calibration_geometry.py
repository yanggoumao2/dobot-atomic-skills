"""Migrated original geometry/protocol tests; no camera, robot, or network."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from task_skills.calibration import calibrate as c
from task_skills.calibration import nova2_read_pose as nova
from task_skills.calibration import table_calibration as t
from task_skills.calibration.synthetic_demo import make_T, quaternion


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def random_T(self):
        return make_T(self.rng.uniform(-.6, .6, 3), self.rng.uniform(-.5, .5, 3))

    def test_pose_quaternion_and_inverse(self):
        transform = self.random_T()
        recovered = c.pose([*transform[:3, 3], *quaternion(transform[:3, :3])])
        np.testing.assert_allclose(transform, recovered, atol=1e-12)
        np.testing.assert_allclose(transform @ c.inverse(transform), np.eye(4), atol=1e-12)

    def test_both_handeye_modes_recover_ground_truth(self):
        for mode in ("eye-in-hand", "eye-to-hand"):
            expected_x, expected_y = self.random_T(), self.random_T()
            robot_poses = [self.random_T() for _ in range(16)]
            camera_poses = [
                c.inverse(expected_x) @ c.inverse(robot) @ expected_y
                if mode == "eye-in-hand" else c.inverse(expected_x) @ robot @ expected_y
                for robot in robot_poses
            ]
            for method in ("park", "tsai", "horaud"):
                recovered = c.solve_handeye(robot_poses, camera_poses, mode, method)
                np.testing.assert_allclose(recovered, expected_x, atol=1e-8)
                for robot, camera in zip(robot_poses, camera_poses):
                    np.testing.assert_allclose(
                        c.fixed_target(robot, recovered, camera, mode), expected_y, atol=1e-8)

    def test_single_axis_motion_rejected(self):
        robot_poses = [make_T([0, 0, index * .1], [index * .01, 0, 0]) for index in range(8)]
        with self.assertRaises(ValueError):
            c.excitation(robot_poses)

    def test_invalid_quaternion_and_transform_rejected(self):
        with self.assertRaises(ValueError):
            c.pose([0] * 7)
        with self.assertRaises(ValueError):
            c.pose([float("nan"), 0, 0, 0, 0, 0, 1])
        bad = np.eye(4)
        bad[0, 0] = -1
        with self.assertRaises(ValueError):
            c.validate_transform(bad)

    def test_distorted_pixel_roundtrip_in_both_modes(self):
        camera_matrix = np.array([[800., 0, 480], [0, 790., 360], [0, 0, 1]])
        distortion = np.array([.1, -.02, .001, -.001, .005])
        point = np.array([[.08, -.03, .7]])
        uv = cv2.projectPoints(point, np.zeros(3), np.zeros(3), camera_matrix, distortion)[0].reshape(2)
        expected_x, robot = self.random_T(), self.random_T()
        camera = {"image_size_wh": [960, 720]}
        for mode, key, transform in [
                ("eye-to-hand", "T_base_camera", expected_x),
                ("eye-in-hand", "T_tool_camera", robot @ expected_x)]:
            result = c.point_in_base(*uv, .7, camera, camera_matrix, distortion,
                                     {"mode": mode, key: expected_x.tolist()}, robot)
            np.testing.assert_allclose(
                result, (transform @ np.r_[point[0], 1])[:3], atol=1e-8)

    def test_missing_pose_and_bad_depth_rejected(self):
        camera = {"image_size_wh": [960, 720]}
        camera_matrix = np.array([[800., 0, 480], [0, 790., 360], [0, 0, 1]])
        hand = {"mode": "eye-in-hand", "T_tool_camera": np.eye(4).tolist()}
        for depth in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                c.point_in_base(100, 100, depth, camera, camera_matrix, np.zeros(5), hand)
        with self.assertRaises(ValueError):
            c.point_in_base(100, 100, .5, camera, camera_matrix, np.zeros(5), hand)

    def test_duplicate_images_across_splits_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.png").write_bytes(b"same-content")
            (root / "b.png").write_bytes(b"same-content")
            (root / "manifest.csv").write_text("image,split\na.png,train\nb.png,val\n")
            with self.assertRaises(ValueError):
                c.manifest(root / "manifest.csv", ["image", "split"])

    def test_nonempty_output_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "existing.txt").write_text("keep")
            with self.assertRaises(ValueError):
                c.output_dir(tmp)


class TableTests(unittest.TestCase):
    def test_board_dimensions_and_planar_rigid_fit(self):
        points = c.board_points(**t.BOARD)
        self.assertEqual(len(points), 88)
        np.testing.assert_allclose(points[-1], [.30, .21, 0], atol=1e-7)
        transform = c.transform(cv2.Rodrigues(np.array([.1, -.2, .3]))[0], [.2, -.1, .05])
        recovered = t.rigid_fit(points, t.apply(transform, points))
        np.testing.assert_allclose(recovered, transform, atol=1e-8)

    def test_collinear_and_nonfinite_rejected(self):
        for points in (np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), np.full((4, 3), np.nan)):
            with self.assertRaises(ValueError):
                t.rigid_fit(points, points)

    def test_distorted_tilted_plane_and_offset_roundtrip(self):
        camera_matrix = np.array([[800., 0, 480], [0, 810., 360], [0, 0, 1.]])
        distortion = np.array([.12, -.03, .001, -.002, .01])
        camera_to_board = c.transform(
            cv2.Rodrigues(np.array([.25, -.15, .1]))[0], [-.15, -.10, .8])
        base_to_board = c.transform(
            cv2.Rodrigues(np.array([-.1, .2, -.3]))[0], [.3, .1, .2])
        base_to_camera = base_to_board @ c.inverse(camera_to_board)
        for z in (0, .05, -.03):
            board_point = np.array([[.13, .07, z]])
            uv = cv2.projectPoints(
                board_point, cv2.Rodrigues(camera_to_board[:3, :3])[0],
                camera_to_board[:3, 3], camera_matrix, distortion)[0].reshape(2)
            predicted, recovered_board = t.pixel_to_plane(
                uv, {"image_size_wh": [960, 720]}, camera_matrix, distortion,
                base_to_camera, base_to_board, z)
            np.testing.assert_allclose(recovered_board, board_point[0], atol=1e-8)
            np.testing.assert_allclose(predicted, t.apply(base_to_board, board_point)[0], atol=1e-8)

    def test_parallel_behind_outside_rejected(self):
        camera = {"image_size_wh": [100, 100]}
        camera_matrix = np.array([[100., 0, 50], [0, 100, 50], [0, 0, 1.]])
        for base_to_board, pixel in (
                (c.transform(np.eye(3), [0, 0, -1]), [50, 50]),
                (c.transform(cv2.Rodrigues(np.array([0, np.pi / 2, 0]))[0], [1, 0, 0]), [50, 50]),
                (c.transform(np.eye(3), [0, 0, 1]), [100, 50])):
            with self.assertRaises(ValueError):
                t.pixel_to_plane(pixel, camera, camera_matrix, np.zeros(5), np.eye(4), base_to_board)

    def test_mm_conversion_and_duplicate_split_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "teach.csv"
            path.write_text("corner_id,split,x,y,z\n" + "".join(
                f"{index},{'train' if index < 6 else 'val'},100,200,300\n" for index in range(9)))
            _, _, xyz = t.taught_points(path, 88, "mm")
            np.testing.assert_allclose(xyz[0], [.1, .2, .3])
            path.write_text(path.read_text() + "0,val,100,200,300\n")
            with self.assertRaises(ValueError):
                t.taught_points(path, 88, "mm")

    def test_nova_reply_validation(self):
        self.assertEqual(
            nova.parse_pose("0,{100,-200,300,0,90,-180},GetPose();"),
            [100, -200, 300, 0, 90, -180])
        for reply in ("-1,{},GetPose();", "0,{1,2,3},GetPose();",
                      "0,{nan,2,3,4,5,6},GetPose();", "0,{1,2,3,4,5,6},MovJ();"):
            with self.assertRaises(ValueError):
                nova.parse_pose(reply)

    def test_nova_fragmented_response_only_sends_getpose(self):
        sock = MagicMock()
        sock.__enter__.return_value = sock
        sock.recv.side_effect = [b"0,{1,2,3,", b"4,5,6},Get", b"Pose();"]
        with patch("task_skills.calibration.nova2_read_pose.socket.create_connection",
                   return_value=sock):
            pose, _ = nova.read_pose("example.invalid")
        self.assertEqual(pose, [1, 2, 3, 4, 5, 6])
        sock.sendall.assert_called_once_with(b"GetPose()")


if __name__ == "__main__":
    unittest.main(verbosity=2)
