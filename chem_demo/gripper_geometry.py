"""Gripper and platform geometry shared by the handoff planner and the right-wrist observer.

Two coordinate conventions exist in this repo and must not be mixed:

* Controller poses (Tool0 = the real flange, each arm's own base frame).
* "Fingertip convention" (cw): the top-camera calibration (11-04) and both wrist hand-eyes put the
  Tool0 origin at the closed fingertips. A point given in cw coordinates sits FINGERTIP_FROM_FLANGE_MM
  higher in number than the same physical point in controller coordinates:
      real = cw - FINGERTIP_FROM_FLANGE_MM * z_hat
  TABLE_Z (240.18 right / 243.2 left) is a cw number: the flange Z at which the closed tips touch
  the white table.

FINGERTIP_FROM_FLANGE_MM is an ESTIMATE (2026-11-22, two independent image checks):
  1. 05_lift_top.png of run-20261122_213554: the left flange pose projected with the top camera
     lands on the fingertips/neck; points 80-130 mm above it land on the gripper adapter and the
     end of the J6 housing, i.e. flange -> tips ~ 110-130 mm.
  2. Left wrist depth at the neck grasp (04_lift): table ~255 mm from the camera while the tips are
     115.7 mm above the table, so the camera (clamped on the J6 housing) sits ~140 mm above the tips.
The earlier planner value (240 mm, "table at the base plane") put the right pads ~120 mm below
the held bottle at 21:35 on 2026-11-22. Measure flange face -> closed tips with a ruler and
replace the estimate; every derived number below follows from it.

2026-11-23: Kunyu asked to try 180 mm. Re-projecting the flange ring (phi 63) on 05_lift_top.png and
hang_top.png of run-20261122_230032 (calibration-data/chem-demo/handoff/fingertip-check-20261123)
puts L=180 at the end of the J6 housing, L=120 on the silver gripper adapter below it. Still not a
ruler measurement.
"""

from __future__ import annotations

FINGERTIP_FROM_FLANGE_MM = 180.0   # Kunyu 2026-11-23: try 180; top-camera overlay agrees better than 120 (see below)
PAD_CENTRE_ABOVE_TIPS_MM = 16.0     # right-wrist hand-eye: pad midpoint at Tool0(tip) [+3, 0, -16]
PAD_CENTRE_TOOL_X_MM = 3.0

# cw table heights (fingertip touch), each arm's base frame
TABLE_Z_CW = {"right": 240.18, "left": 243.2}

# Commissioned neck grasp and bottle (cw, left base): flange 358.86 with the bottle on the table,
# neck-ring plane 377.49 (chem_demo.neck_vision.RING_PLANE_Z_MM).
NECK_GRASP_FLANGE_Z_CW = 358.86
RING_PLANE_Z_CW = 377.49
GRASP_ABOVE_BOTTOM_MM = 40.0        # right pads close on the body this far above the bottle bottom


def table_z_real(arm: str, fingertip_mm: float = FINGERTIP_FROM_FLANGE_MM) -> float:
    """Table height in controller coordinates of ``arm`` (base frame, mm)."""
    return TABLE_Z_CW[arm] - fingertip_mm


def pad_offset_flange(fingertip_mm: float = FINGERTIP_FROM_FLANGE_MM) -> tuple[float, float, float]:
    """Right pad midpoint in the real flange frame (Tool0), for the planner."""
    return (PAD_CENTRE_TOOL_X_MM, 0.0, fingertip_mm - PAD_CENTRE_ABOVE_TIPS_MM)


def pad_offset_tip_frame() -> tuple[float, float, float]:
    """Right pad midpoint in the hand-eye (fingertip-origin) frame, for the wrist observer."""
    return (PAD_CENTRE_TOOL_X_MM, 0.0, -PAD_CENTRE_ABOVE_TIPS_MM)


def bottle_hang(fingertip_mm: float = FINGERTIP_FROM_FLANGE_MM) -> dict[str, float]:
    """Held bottle below the left flange (real mm), from the commissioned neck grasp.

    Tips grasp the neck NECK_GRASP_FLANGE_Z_CW - TABLE_Z_CW["left"] = 115.7 mm above the bottle bottom,
    so the bottom hangs that plus the flange->tips length below the flange. Only the flange->tips
    length is an estimate; the rest is measured.
    """
    tips_above_bottom = NECK_GRASP_FLANGE_Z_CW - TABLE_Z_CW["left"]
    bottom_below = tips_above_bottom + fingertip_mm
    height = RING_PLANE_Z_CW - TABLE_Z_CW["left"]
    return {"bottom_below_flange_mm": bottom_below,
            "total_height_mm": height,
            "rim_below_flange_mm": bottom_below - height,
            "grasp_drop_mm": bottom_below - GRASP_ABOVE_BOTTOM_MM,
            "tips_above_bottom_mm": tips_above_bottom}
