# Nova2 mock arm primitive

This primitive runs entirely in software. It uses Robonix's shared lifecycle
Driver and the built-in arm contracts over gRPC:

- robonix/primitive/arm/joint_command: client stream of standard
  sensor_msgs/JointState frames. Each frame must contain all six canonical
  joint names and six finite positions. Names can arrive in any order.
- robonix/primitive/arm/joint_states: server stream of standard
  sensor_msgs/JointState frames while the provider is ACTIVE.

Each process is configured with side set to left or right, which scopes the
joint names and initial home pose. Give the two deployment entries unique
names, for example nova2_arm_left and nova2_arm_right. Robonix uses each entry
name as its Atlas provider identity.

The pure model can be imported on Windows without Robonix, ROS, gRPC, or
hardware SDKs:

~~~python
from nova2_arm.model import ArmModel

arm = ArmModel("left")
frame = arm.apply_command(
    ("left_joint1", "left_joint2", "left_joint3",
     "left_joint4", "left_joint5", "left_joint6"),
    (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
)
print(frame.name, frame.position)
print(arm.feedback())
~~~

The default model pose reads radians from task_skills/layout.py directly,
without importing the hardware adapter. Initialization fails if that layout
or the Nova2 URDF is missing or malformed. The model validates each whole
target before updating any joint. Velocity and effort commands are rejected;
feedback reports positions only. Joint angles are constrained by the six
revolute joint ranges in robonix/nova2/urdf/nova2_dual.urdf.

Build natively with rbnx codegen -p followed by the package path. Start through
Robonix or run scripts/start.sh in an environment where rbnx and robonix-api
are available. The gRPC entrypoint is python -m nova2_arm.driver from this
package root.
