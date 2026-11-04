"""Corner targets and step-7 acceptance rows from a fitted calibration-wizard session.

Read-only over the wizard's files: session.json names each arm's current fit, the fit's
extrinsics.json holds T_base_board and the ids it was fitted on, and the board observation
with the same sha256 holds each corner's detected pixel. Nothing here talks to a robot.

    python -X utf8 -m task_skills.calibration_targets --session DIR --arm left \
        --touches calibration-data/jog-points/left_check.csv --out left_verify.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

VERIFY_FIELDS = ["u_px", "v_px", "board_z_m", "reference_x_m", "reference_y_m", "reference_z_m"]


class TargetError(RuntimeError):
    """The session cannot provide a trustworthy target."""


@dataclass(frozen=True)
class ArmFit:
    arm: str
    robot_id: str
    tcp_name: str
    cols: int
    rows: int
    square_m: float
    T_base_board: tuple[tuple[float, ...], ...]
    corners_uv: tuple[tuple[float, float], ...]
    fit_ids: frozenset[int]
    job_id: str

    def check_id(self, corner_id: int) -> None:
        if not 0 <= corner_id < self.cols * self.rows:
            raise TargetError(f"角点编号必须在 0-{self.cols * self.rows - 1} 之间")

    def corner_base_mm(self, corner_id: int) -> list[float]:
        """Board corner (col, row) * square on the board plane, in this arm's base, mm."""
        self.check_id(corner_id)
        point = ((corner_id % self.cols) * self.square_m, (corner_id // self.cols) * self.square_m, 0.0)
        T = self.T_base_board
        return [round(1000.0 * (sum(T[r][c] * point[c] for c in range(3)) + T[r][3]), 4)
                for r in range(3)]


def load_arm_fit(session_dir: Path, arm: str) -> ArmFit:
    session_dir = Path(session_dir)
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    fit = session["arms"][arm]["fit"]
    if not fit or not fit.get("usable") or fit.get("stale") or not fit.get("validation_passed"):
        raise TargetError(f"{arm} 臂没有当前可用且通过的拟合（先完成向导第 5 步）")
    extrinsics_path = session_dir / fit["artifact_path"]
    raw = extrinsics_path.read_bytes()
    if fit.get("artifact_sha256") and hashlib.sha256(raw).hexdigest() != fit["artifact_sha256"]:
        raise TargetError(f"{extrinsics_path} 与 session.json 记录的哈希不一致")
    extrinsics = json.loads(raw.decode("utf-8"))
    observation = None
    for path in sorted((session_dir / "board").glob("*/observation_*.json")):
        if hashlib.sha256(path.read_bytes()).hexdigest() == extrinsics["observation_sha256"]:
            observation = json.loads(path.read_text(encoding="utf-8"))
            break
    if observation is None:
        raise TargetError("找不到拟合所用的固定棋盘观测（observation sha256 不匹配）")
    board = extrinsics["board"]
    return ArmFit(
        arm=arm, robot_id=extrinsics["robot_id"], tcp_name=extrinsics["tcp_name"],
        cols=int(board["cols"]), rows=int(board["rows"]), square_m=float(board["square_m"]),
        T_base_board=tuple(tuple(float(v) for v in row) for row in extrinsics["T_base_board"]),
        corners_uv=tuple((float(u), float(v)) for u, v in observation["corners_uv"]),
        fit_ids=frozenset(int(point["corner_id"]) for point in extrinsics["points"]),
        job_id=fit["job_id"],
    )


def verify_rows(fit: ArmFit, touches: Sequence[dict[str, str]]) -> tuple[list[dict[str, float]], list[str]]:
    """Touched corners -> wizard step-7 rows; corners used by the fit are left out."""
    rows, notes = [], []
    for touch in touches:
        corner = int(touch["corner_id"])
        fit.check_id(corner)
        measured = [float(touch[axis]) for axis in "xyz"]
        if corner in fit.fit_ids:
            notes.append(f"角点 {corner}：参与了 {fit.arm} 臂拟合，不是独立点，已跳过")
            continue
        predicted = fit.corner_base_mm(corner)
        error = math.dist(predicted, measured)
        notes.append(f"角点 {corner}：实测 {[round(v, 2) for v in measured]} 预测 "
                     f"{[round(v, 2) for v in predicted]} 相差 {error:.2f} mm（按棋盘几何，向导会用像素重算）")
        u, v = fit.corners_uv[corner]
        rows.append({"u_px": u, "v_px": v, "board_z_m": 0.0, "reference_x_m": measured[0] / 1000.0,
                     "reference_y_m": measured[1] / 1000.0, "reference_z_m": measured[2] / 1000.0})
    return rows, notes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 jog 触碰的角点转成向导第 7 步的独立验收 CSV")
    parser.add_argument("--session", type=Path, required=True, help="向导会话目录 calibration-data/cal-...")
    parser.add_argument("--arm", choices=("left", "right"), required=True)
    parser.add_argument("--touches", type=Path, required=True, help="jog 的 p 键记录的 CSV")
    parser.add_argument("--out", type=Path, required=True, help="输出的第 7 步导入 CSV")
    args = parser.parse_args(argv)
    try:
        fit = load_arm_fit(args.session, args.arm)
        with args.touches.open(newline="", encoding="utf-8") as handle:
            rows, notes = verify_rows(fit, list(csv.DictReader(handle)))
    except TargetError as error:
        print(f"错误：{error}")
        return 2
    for note in notes:
        print(note)
    if len(rows) < 3:
        print(f"只有 {len(rows)} 个独立点，第 7 步至少要 3 个；未写文件")
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=VERIFY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"已写 {len(rows)} 个独立验收点 -> {args.out}（拟合 {fit.job_id}，TCP {fit.tcp_name}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
