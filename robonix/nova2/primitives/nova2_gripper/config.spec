# Runtime configuration for the native Linux Nova2 mock gripper.
#
# Supply the same package as two provider instances to model both sides:
#   provider instance: left_gripper  with side: left
#   provider instance: right_gripper with side: right
#
# Width feedback is synthetic with a fixed 0.08 m stroke. The mock has no
# object, contact, grasp-force, or physical-device model.

config:
  # string; must be left or right.
  side: left
