# Runtime configuration passed to Driver(CMD_INIT).

config:
  # string, required: left or right.
  # Determines this process's six joint names and initial home pose.
  side: left

  # number, Hz, default: 10.0, range: 1.0 to 50.0.
  # Maximum rate for the active joint_states server stream.
  feedback_rate_hz: 10.0
