"""Offline camera-to-base calibration package for the Dobot atomic-skill project.

Importing this package never opens a camera, robot connection, serial port or
device driver. Hardware access is limited to an explicit GetPose() call through
``nova2_read_pose.read_pose`` and an explicit OpenCV camera capture started by
the local operator UI.
"""

from .integration import CalibrationTransform, load_validated_calibration

__all__ = ["CalibrationTransform", "load_validated_calibration"]
