"""Persistent calibration state, hashing and path-boundary helpers."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 2
SESSION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
ARMS = ("left", "right")
SPLITS = ("train", "val")
UNIT_CHOICES = ("mm", "m")
CORNER_ORDERS = ("normal", "reverse")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

DEFAULT_BOARD = {"cols": 11, "rows": 8, "square_m": 0.03}
DEFAULT_THRESHOLDS = {
    "intrinsics_max_rms_px": 1.0,
    "fit_max_pnp_rms_px": 1.0,
    "fit_max_error_mm": 2.0,
    "verify_max_error_mm": 2.0,
}


class CalibrationError(Exception):
    """Stable, actionable error returned by the service and HTTP layer."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError("INVALID_NUMBER", f"{label} 必须是有限数字")
    number = float(value)
    if not math.isfinite(number):
        raise CalibrationError("NONFINITE_NUMBER", f"{label} 不允许 NaN/Inf")
    if positive and number <= 0:
        raise CalibrationError("INVALID_NUMBER", f"{label} 必须大于 0")
    return number


def finite_int(value: Any, label: str, *, minimum: int | None = None,
               maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationError("INVALID_INTEGER", f"{label} 必须是整数")
    if minimum is not None and value < minimum:
        raise CalibrationError("INVALID_INTEGER", f"{label} 不能小于 {minimum}")
    if maximum is not None and value > maximum:
        raise CalibrationError("INVALID_INTEGER", f"{label} 不能大于 {maximum}")
    return value


def clean_text(value: Any, label: str, *, maximum: int = 128,
               allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise CalibrationError("INVALID_TEXT", f"{label} 必须是文本")
    text = value.strip()
    if not text and not allow_empty:
        raise CalibrationError("INVALID_TEXT", f"{label} 不能为空")
    if len(text) > maximum:
        raise CalibrationError("INVALID_TEXT", f"{label} 最多 {maximum} 个字符")
    if any(ord(ch) < 32 or ch == "\x7f" for ch in text):
        raise CalibrationError("INVALID_TEXT", f"{label} 不能包含控制字符")
    return text


def validate_board(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CalibrationError("INVALID_BOARD", "棋盘参数必须是对象")
    cols = finite_int(value.get("cols"), "棋盘列内角点", minimum=3, maximum=30)
    rows = finite_int(value.get("rows"), "棋盘行内角点", minimum=3, maximum=30)
    square_m = finite_number(value.get("square_m"), "格长（米）", positive=True)
    if not 0.001 <= square_m <= 0.5:
        raise CalibrationError("INVALID_BOARD", "格长必须在 0.001–0.5 m 之间")
    return {"cols": cols, "rows": rows, "square_m": square_m}


def validate_thresholds(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise CalibrationError("INVALID_THRESHOLDS", "阈值必须是对象")
    out: dict[str, float] = {}
    for key, label in (
        ("intrinsics_max_rms_px", "内参单图 RMS 上限"),
        ("fit_max_pnp_rms_px", "PnP RMS 上限"),
        ("fit_max_error_mm", "拟合点误差上限"),
        ("verify_max_error_mm", "独立验收误差上限"),
    ):
        out[key] = finite_number(value.get(key), label, positive=True)
        if out[key] > 100.0:
            raise CalibrationError("INVALID_THRESHOLDS", f"{label} 不应超过 100")
    return out


def validate_settings(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CalibrationError("INVALID_SETTINGS", "任务设置必须是对象")
    camera_id = clean_text(value.get("camera_id"), "相机 ID", maximum=128)
    width = finite_int(value.get("width"), "图像宽度", minimum=64, maximum=16384)
    height = finite_int(value.get("height"), "图像高度", minimum=64, maximum=16384)
    return {
        "camera_id": camera_id,
        "image_size_wh": [width, height],
        "board": validate_board(value.get("board")),
        "thresholds": validate_thresholds(value.get("thresholds")),
    }


def validate_arm_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CalibrationError("INVALID_ARM_METADATA", "机械臂元数据必须是对象")
    unit = value.get("unit", "mm")
    if unit not in UNIT_CHOICES:
        raise CalibrationError("INVALID_UNIT", "单位只能是 mm 或 m")
    host = value.get("controller_host", "")
    host = clean_text(host, "控制器地址", maximum=253, allow_empty=True)
    if host and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}", host):
        raise CalibrationError("INVALID_CONTROLLER_HOST", "控制器地址只能是 IP 或主机名")
    port = value.get("controller_port", 29999)
    port = finite_int(port, "控制器端口", minimum=1, maximum=65535)
    return {
        "robot_id": clean_text(value.get("robot_id"), "robot_id", maximum=128),
        "base_frame": clean_text(value.get("base_frame"), "base_frame", maximum=128),
        "tcp_name": clean_text(value.get("tcp_name"), "TCP 名称", maximum=128),
        "unit": unit,
        "controller_host": host,
        "controller_port": port,
    }


def validate_teach_point(value: Any, corner_count: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CalibrationError("INVALID_TEACH_POINT", "示教点必须是对象")
    corner_id = finite_int(value.get("corner_id"), "角点 ID", minimum=0,
                           maximum=corner_count - 1)
    split = value.get("split")
    if split not in SPLITS:
        raise CalibrationError("INVALID_SPLIT", "数据划分只能是 train 或 val")
    return {
        "corner_id": corner_id,
        "split": split,
        "x": finite_number(value.get("x"), "示教 X"),
        "y": finite_number(value.get("y"), "示教 Y"),
        "z": finite_number(value.get("z"), "示教 Z"),
    }


def validate_teach_points(values: Any, corner_count: int) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise CalibrationError("INVALID_TEACH_POINTS", "示教点必须是数组")
    if len(values) > corner_count:
        raise CalibrationError("INVALID_TEACH_POINTS", "示教点数量超过角点总数")
    points = [validate_teach_point(value, corner_count) for value in values]
    ids = [point["corner_id"] for point in points]
    if len(ids) != len(set(ids)):
        raise CalibrationError("DUPLICATE_CORNER_ID", "同一只手臂不能重复记录角点 ID")
    return sorted(points, key=lambda point: point["corner_id"])


def validate_verify_row(value: Any, image_size: tuple[int, int]) -> dict[str, float]:
    if not isinstance(value, dict):
        raise CalibrationError("INVALID_VERIFY_ROW", "验收点必须是对象")
    width, height = image_size
    u_px = finite_number(value.get("u_px"), "像素 u")
    v_px = finite_number(value.get("v_px"), "像素 v")
    if not 0 <= u_px < width or not 0 <= v_px < height:
        raise CalibrationError("PIXEL_OUT_OF_BOUNDS", "验收像素超出原始图像")
    return {
        "u_px": u_px,
        "v_px": v_px,
        "board_z_m": finite_number(value.get("board_z_m"), "板坐标 z（米）"),
        "reference_x_m": finite_number(value.get("reference_x_m"), "参考 X（米）"),
        "reference_y_m": finite_number(value.get("reference_y_m"), "参考 Y（米）"),
        "reference_z_m": finite_number(value.get("reference_z_m"), "参考 Z（米）"),
    }


def validate_verify_rows(values: Any, image_size: tuple[int, int]) -> list[dict[str, float]]:
    if not isinstance(values, list):
        raise CalibrationError("INVALID_VERIFY_ROWS", "验收点必须是数组")
    if len(values) > 500:
        raise CalibrationError("INVALID_VERIFY_ROWS", "验收点最多 500 个")
    rows = [validate_verify_row(value, image_size) for value in values]
    pixels = [(row["u_px"], row["v_px"]) for row in rows]
    if len(pixels) != len(set(pixels)):
        raise CalibrationError("DUPLICATE_VERIFY_PIXEL", "独立验收点不能重复使用同一像素")
    return rows


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                # Windows can briefly deny replacement when another process has
                # the destination open without delete sharing (e.g. antivirus).
                if attempt == 5:
                    raise
                time.sleep(0.02 * (attempt + 1))
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    atomic_write_bytes(path, data)


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_session_id(value: str) -> str:
    if not isinstance(value, str) or not SESSION_ID_RE.fullmatch(value):
        raise CalibrationError("INVALID_SESSION_ID", "会话 ID 格式不合法", status=400)
    return value


def safe_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CalibrationError("INVALID_PATH", "产物路径不合法")
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise CalibrationError("UNSAFE_PATH", "产物路径越界")
    if path.suffix.lower() not in IMAGE_SUFFIXES | {".json", ".csv", ".txt", ".zip"}:
        raise CalibrationError("UNSAFE_PATH", "不支持的产物文件类型")
    return normalized


def resolve_session_path(session_dir: Path, relative: str) -> Path:
    relative = safe_relative_path(relative)
    base = session_dir.resolve()
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise CalibrationError("UNSAFE_PATH", "产物路径越界") from error
    return candidate


def _walk_paths(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, str) and (key.endswith("_path") or key == "source_image"):
                yield child
            else:
                yield from _walk_paths(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_paths(child)


def known_asset_paths(state: dict[str, Any]) -> set[str]:
    paths = {path.replace("\\", "/") for path in _walk_paths(state)}
    board = state.get("board") or {}
    for key in ("previews", "observations"):
        values = board.get(key) or {}
        if isinstance(values, dict):
            paths.update(value.replace("\\", "/") for value in values.values()
                         if isinstance(value, str) and value)
    return paths


def intrinsics_input_hash(settings: dict[str, Any], images: list[dict[str, Any]]) -> str:
    payload = {
        "camera_id": settings["camera_id"],
        "image_size_wh": settings["image_size_wh"],
        "board": settings["board"],
        "max_rms_px": settings["thresholds"]["intrinsics_max_rms_px"],
        "images": [
            {"id": image["id"], "sha256": image["sha256"], "split": image["split"]}
            for image in images
        ],
    }
    return hash_json(payload)


def board_input_hash(settings: dict[str, Any], source_sha256: str,
                     corner_order: str | None = None) -> str:
    payload = {
        "camera_id": settings["camera_id"],
        "image_size_wh": settings["image_size_wh"],
        "board": settings["board"],
        "source_sha256": source_sha256,
    }
    if corner_order is not None:
        payload["corner_order"] = corner_order
    return hash_json(payload)


def teach_input_hash(metadata: dict[str, Any], points: list[dict[str, Any]]) -> str:
    payload = {
        "metadata": metadata,
        "points": [
            {key: point[key] for key in ("corner_id", "split", "x", "y", "z")}
            for point in sorted(points, key=lambda item: item["corner_id"])
        ],
    }
    return hash_json(payload)


def arm_content_hashes(arm_state: dict[str, Any]) -> dict[str, str]:
    """Recompute saved arm provenance from the actual persisted payloads."""
    metadata = arm_state.get("metadata")
    points = arm_state.get("points")
    if not isinstance(metadata, dict) or not isinstance(points, list):
        raise ValueError("机械臂元数据或示教点内容格式不正确")
    return {"metadata_hash": hash_json(metadata), "points_hash": hash_json(points)}


def arm_content_hash_status(arm_state: dict[str, Any]) -> tuple[bool, str]:
    try:
        actual = arm_content_hashes(arm_state)
    except (TypeError, ValueError) as error:
        return False, f"机械臂输入内容无法计算哈希：{error}"
    for key, label in (("metadata_hash", "机械臂元数据"),
                       ("points_hash", "示教点")):
        if arm_state.get(key) != actual[key]:
            return False, f"保存的{label}哈希与实际内容不一致，拟合输入已更改；请重新保存并重新拟合"
    return True, ""


def fit_input_hash(state: dict[str, Any], arm: str) -> str:
    settings = state["settings"]
    intrinsics = state["intrinsics"].get("result") or {}
    board = state["board"].get("confirmation") or {}
    arm_state = state["arms"][arm]
    content_hashes = arm_content_hashes(arm_state)
    payload = {
        "intrinsics_input_hash": intrinsics_input_hash(settings, state["intrinsics"]["images"]),
        "intrinsics_artifact_sha256": intrinsics.get("artifact_sha256"),
        "intrinsics_validation_passed": intrinsics.get("validation_passed"),
        "board_input_hash": board.get("input_hash"),
        "board_observation_sha256": board.get("observation_sha256"),
        "board_corner_order": board.get("corner_order"),
        "camera_id": settings["camera_id"],
        "image_size_wh": settings["image_size_wh"],
        "board": settings["board"],
        "arm_metadata": arm_state["metadata"],
        "arm_metadata_hash": content_hashes["metadata_hash"],
        "teach_points_hash": content_hashes["points_hash"],
        "thresholds": {
            "fit_max_pnp_rms_px": settings["thresholds"]["fit_max_pnp_rms_px"],
            "fit_max_error_mm": settings["thresholds"]["fit_max_error_mm"],
        },
        "synthetic_only": bool(state.get("synthetic_only")),
    }
    return hash_json(payload)


def verification_input_hash(state: dict[str, Any], arm: str) -> str:
    arm_state = state["arms"][arm]
    draft = arm_state["verification"]
    fit = arm_state.get("fit") or {}
    payload = {
        "fit_input_hash": fit_input_hash(state, arm),
        "fit_artifact_sha256": fit.get("artifact_sha256"),
        "rows": draft.get("rows", []),
        "method": draft.get("method", ""),
        "independence_confirmed": bool(draft.get("independence_confirmed")),
        "threshold_mm": state["settings"]["thresholds"]["verify_max_error_mm"],
        "robot_id": arm_state["metadata"]["robot_id"],
        "base_frame": arm_state["metadata"]["base_frame"],
        "tcp_name": arm_state["metadata"]["tcp_name"],
        "unit": arm_state["metadata"]["unit"],
        "synthetic_only": bool(state.get("synthetic_only")),
    }
    return hash_json(payload)


def add_event(state: dict[str, Any], level: str, message: str,
              detail: dict[str, Any] | None = None) -> None:
    event = {"time": utc_now(), "level": level, "message": message}
    if detail:
        event["detail"] = detail
    state.setdefault("events", []).append(event)
    state["events"] = state["events"][-300:]


def add_history(container: dict[str, Any], action: str,
                detail: dict[str, Any] | None = None) -> None:
    container.setdefault("revision", 0)
    container["revision"] += 1
    entry = {"revision": container["revision"], "time": utc_now(), "action": action}
    if detail:
        entry["detail"] = detail
    container.setdefault("history", []).append(entry)
    container["history"] = container["history"][-80:]


def mark_stale(record: dict[str, Any] | None, reason: str) -> None:
    if not isinstance(record, dict):
        return
    if record.get("status") or record.get("artifact_path"):
        record["stale"] = True
        record["stale_reason"] = reason
        record["validation_passed"] = False
        record["usable"] = False


def invalidate_arm(state: dict[str, Any], arm: str, reason: str) -> None:
    arm_state = state["arms"][arm]
    mark_stale(arm_state.get("fit"), reason)
    mark_stale(arm_state.get("verification", {}).get("result"), reason)
    if arm_state.get("fit"):
        arm_state["fit"]["usable"] = False
    add_history(arm_state, "invalidate", {"reason": reason})


def invalidate_all_arms(state: dict[str, Any], reason: str) -> None:
    for arm in ARMS:
        invalidate_arm(state, arm, reason)


def _file_digest(path: Path) -> str | None:
    try:
        return sha256_file(path)
    except OSError:
        return None


def _artifact_json(session_dir: Path, record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(record, dict) or not record.get("artifact_path"):
        return None
    try:
        path = resolve_session_path(session_dir, record["artifact_path"])
        if not path.is_file() or sha256_file(path) != record.get("artifact_sha256"):
            return None
        data = load_json(path)
        return data if isinstance(data, dict) else None
    except (CalibrationError, OSError, ValueError, json.JSONDecodeError):
        return None


def intrinsics_sources_status(state: dict[str, Any], session_dir: Path) -> tuple[bool, str]:
    """Check that every recorded intrinsic image still has its promised bytes."""
    images = state.get("intrinsics", {}).get("images", [])
    if not isinstance(images, list):
        return False, "内参图像清单格式不正确"
    seen_paths: set[str] = set()
    for index, image in enumerate(images, start=1):
        if not isinstance(image, dict):
            return False, f"内参图像第 {index} 条记录格式不正确"
        relative = image.get("original_path")
        expected = image.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
            return False, f"内参图像第 {index} 条记录缺少原始文件哈希"
        if relative in seen_paths:
            return False, f"内参图像第 {index} 条记录路径重复"
        seen_paths.add(relative)
        try:
            path = resolve_session_path(session_dir, relative)
            if not path.is_file():
                return False, f"内参原图缺失：{image.get('display_name') or relative}"
            actual = sha256_file(path)
        except (CalibrationError, OSError):
            return False, f"内参原图不可读取：{image.get('display_name') or relative}"
        if actual != expected:
            return False, f"内参原图哈希不匹配：{image.get('display_name') or relative}"
    return True, ""


def intrinsics_status(state: dict[str, Any], session_dir: Path) -> tuple[bool, str]:
    settings = state["settings"]
    images = state["intrinsics"]["images"]
    result = state["intrinsics"].get("result")
    if not result:
        return False, "尚未运行内参标定"
    if result.get("stale"):
        return False, result.get("stale_reason") or "内参结果已失效"
    if result.get("status") != "passed" or result.get("exit_code") != 0:
        return False, "内参计算未通过"
    if result.get("input_hash") != intrinsics_input_hash(settings, images):
        return False, "内参输入已更改"
    sources_ok, sources_reason = intrinsics_sources_status(state, session_dir)
    if not sources_ok:
        return False, sources_reason
    artifact = _artifact_json(session_dir, result)
    if not artifact:
        return False, "内参产物缺失或被修改"
    if artifact.get("kind") != "intrinsics":
        return False, "内参产物类型不正确"
    if artifact.get("camera_id") != settings["camera_id"]:
        return False, "内参相机 ID 与当前任务不一致"
    if artifact.get("image_size_wh") != settings["image_size_wh"]:
        return False, "内参图像尺寸与当前任务不一致"
    if artifact.get("board") != settings["board"]:
        return False, "内参棋盘参数与当前任务不一致"
    if not artifact.get("validation_passed") or not result.get("validation_passed"):
        return False, "内参未达到阈值"
    if bool(artifact.get("synthetic_only")) != bool(state.get("synthetic_only")):
        return False, "内参模拟标记不一致"
    if result.get("artifact_sha256") != _file_digest(
            resolve_session_path(session_dir, result["artifact_path"])):
        return False, "内参产物哈希不匹配"
    return True, ""


def board_status(state: dict[str, Any], session_dir: Path) -> tuple[bool, str]:
    settings = state["settings"]
    board = state["board"]
    confirmation = board.get("confirmation")
    if not confirmation:
        return False, "尚未确认固定棋盘编号"
    if confirmation.get("stale"):
        return False, confirmation.get("stale_reason") or "固定棋盘已失效"
    if board.get("state") == "stale":
        return False, "固定棋盘版本已失效"
    if confirmation.get("input_hash") != board_input_hash(
            settings, board.get("source_sha256", ""), confirmation.get("corner_order")):
        return False, "固定棋盘输入已更改"
    source_path = board.get("source_path")
    if not source_path:
        return False, "固定棋盘源图缺失"
    try:
        source_path = resolve_session_path(session_dir, source_path)
        if not source_path.is_file() or sha256_file(source_path) != board.get("source_sha256"):
            return False, "固定棋盘源图缺失或被修改"
    except (CalibrationError, OSError):
        return False, "固定棋盘源图不可读取"
    observation_path = confirmation.get("observation_path")
    if not observation_path:
        return False, "固定棋盘观测文件缺失"
    try:
        observation_path = resolve_session_path(session_dir, observation_path)
        if sha256_file(observation_path) != confirmation.get("observation_sha256"):
            return False, "固定棋盘观测哈希不匹配"
        observation = load_json(observation_path)
    except (CalibrationError, OSError, ValueError, json.JSONDecodeError):
        return False, "固定棋盘观测不可读取"
    if observation.get("kind") != "board_observation":
        return False, "固定棋盘观测类型不正确"
    if observation.get("board") != settings["board"]:
        return False, "固定棋盘参数与当前任务不一致"
    if observation.get("image_size_wh") != settings["image_size_wh"]:
        return False, "固定棋盘图像尺寸与当前任务不一致"
    if observation.get("corner_order") != confirmation.get("corner_order"):
        return False, "固定棋盘编号方向与确认值不一致"
    if bool(observation.get("synthetic_only")) != bool(state.get("synthetic_only")):
        return False, "固定棋盘模拟标记不一致"
    return True, ""


def fit_status(state: dict[str, Any], session_dir: Path, arm: str) -> tuple[bool, str]:
    intr_ok, intr_reason = intrinsics_status(state, session_dir)
    if not intr_ok:
        return False, f"内参不可用：{intr_reason}"
    board_ok, board_reason = board_status(state, session_dir)
    if not board_ok:
        return False, f"固定棋盘不可用：{board_reason}"
    arm_state = state["arms"][arm]
    content_ok, content_reason = arm_content_hash_status(arm_state)
    if not content_ok:
        return False, content_reason
    fit = arm_state.get("fit")
    if not fit:
        return False, "尚未运行拟合"
    if fit.get("stale"):
        return False, fit.get("stale_reason") or "拟合结果已失效"
    if fit.get("status") != "passed" or fit.get("exit_code") != 0:
        return False, "拟合未通过"
    if not fit.get("validation_passed") or not fit.get("usable"):
        return False, "拟合结果不可用于下游"
    if fit.get("input_hash") != fit_input_hash(state, arm):
        return False, "拟合输入已更改"
    artifact = _artifact_json(session_dir, fit)
    if not artifact:
        return False, "拟合产物缺失或被修改"
    if artifact.get("kind") != "fixed_board_extrinsics":
        return False, "拟合产物类型不正确"
    if artifact.get("fit_input_hash") != fit.get("input_hash"):
        return False, "拟合产物输入哈希不一致"
    if artifact.get("robot_id") != arm_state["metadata"]["robot_id"]:
        return False, "拟合 robot_id 不一致"
    if artifact.get("base_frame") != arm_state["metadata"]["base_frame"]:
        return False, "拟合 base_frame 不一致"
    if artifact.get("tcp_name") != arm_state["metadata"]["tcp_name"]:
        return False, "拟合 TCP 不一致"
    if artifact.get("input_unit") != arm_state["metadata"]["unit"]:
        return False, "拟合单位不一致"
    if bool(artifact.get("synthetic_only")) != bool(state.get("synthetic_only")):
        return False, "拟合模拟标记不一致"
    if not artifact.get("validation_passed"):
        return False, "拟合阈值未通过"
    return True, ""


def verification_status(state: dict[str, Any], session_dir: Path,
                        arm: str) -> tuple[bool, str]:
    ok, reason = fit_status(state, session_dir, arm)
    if not ok:
        return False, f"拟合不可用：{reason}"
    verification = state["arms"][arm]["verification"]
    result = verification.get("result")
    if not result:
        return False, "尚未运行独立验收"
    if result.get("stale"):
        return False, result.get("stale_reason") or "独立验收已失效"
    if result.get("status") != "passed" or result.get("exit_code") != 0:
        return False, "独立验收未通过"
    if result.get("input_hash") != verification_input_hash(state, arm):
        return False, "独立验收输入已更改"
    if not verification.get("independence_confirmed") or not verification.get("method"):
        return False, "独立性确认或测量方法缺失"
    artifact = _artifact_json(session_dir, result)
    if not artifact:
        return False, "独立验收产物缺失或被修改"
    if artifact.get("kind") != "plane_point_validation":
        return False, "独立验收产物类型不正确"
    if artifact.get("verification_input_hash") != result.get("input_hash"):
        return False, "独立验收产物输入哈希不一致"
    if artifact.get("robot_id") != state["arms"][arm]["metadata"]["robot_id"]:
        return False, "独立验收 robot_id 不一致"
    if artifact.get("base_frame") != state["arms"][arm]["metadata"]["base_frame"]:
        return False, "独立验收 base_frame 不一致"
    if bool(artifact.get("synthetic_only")) != bool(state.get("synthetic_only")):
        return False, "独立验收模拟标记不一致"
    if not artifact.get("validation_passed") or not result.get("validation_passed"):
        return False, "独立验收阈值未通过"
    return True, ""


def depth_status(state: dict[str, Any]) -> tuple[bool, str]:
    registration = state.get("depth_registration") or {}
    if not registration.get("confirmed"):
        return False, "深度模式未启用：尚未确认光学-Z 注册"
    if registration.get("intrinsics_input_hash") != intrinsics_input_hash(
            state["settings"], state["intrinsics"]["images"]):
        return False, "深度注册对应的内参已改变"
    return True, ""


def create_initial_state(session_id: str, display_name: str, mode: str,
                         settings: dict[str, Any]) -> dict[str, Any]:
    now = utc_now()
    synthetic_only = mode == "synthetic"
    arms = {}
    for arm in ARMS:
        metadata = {
            "robot_id": f"NOVA2_{arm.upper()}",
            "base_frame": "base",
            "tcp_name": "tip",
            "unit": "mm",
            "controller_host": "",
            "controller_port": 29999,
        }
        points: list[dict[str, Any]] = []
        arms[arm] = {
            "metadata": metadata,
            "metadata_hash": hash_json(metadata),
            "points": points,
            "points_hash": hash_json(points),
            "revision": 0,
            "history": [],
            "fit": None,
            "verification": {
                "rows": [],
                "method": "",
                "independence_confirmed": False,
                "confirmed_utc": None,
                "input_hash": None,
                "result": None,
                "revision": 0,
                "history": [],
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "display_name": display_name,
        "mode": "synthetic" if synthetic_only else "real",
        "synthetic_only": synthetic_only,
        "created_utc": now,
        "updated_utc": now,
        "revision": 0,
        "settings": settings,
        "intrinsics": {
            "images": [],
            "result": None,
            "revision": 0,
            "history": [],
        },
        "board": {
            "revision": 0,
            "state": "empty",
            "source_path": None,
            "source_sha256": None,
            "image_size_wh": None,
            "raw_corners_uv": [],
            "detected_utc": None,
            "previews": {},
            "confirmation": None,
            "history": [],
        },
        "arms": arms,
        "depth_registration": {
            "confirmed": False,
            "registration_note": "",
            "intrinsics_input_hash": None,
            "confirmed_utc": None,
        },
        "jobs": [],
        "exports": [],
        "events": [],
    }


class SessionStore:
    """One-process, atomic JSON store with per-session re-entrant locks."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.RLock] = {}
        self._locks_lock = threading.Lock()

    def _lock(self, session_id: str) -> threading.RLock:
        with self._locks_lock:
            return self._locks.setdefault(session_id, threading.RLock())

    def session_dir(self, session_id: str) -> Path:
        session_id = safe_session_id(session_id)
        path = (self.root / session_id).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise CalibrationError("UNSAFE_PATH", "会话路径越界", status=400) from error
        return path

    def state_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "session.json"

    def create(self, display_name: str, mode: str, settings: dict[str, Any]) -> dict[str, Any]:
        if mode not in ("real", "synthetic"):
            raise CalibrationError("INVALID_MODE", "任务模式只能是 real 或 synthetic")
        display_name = clean_text(display_name, "任务名称", maximum=80)
        for _ in range(12):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            suffix = os.urandom(3).hex()
            session_id = f"cal-{stamp}-{suffix}"
            directory = self.session_dir(session_id)
            try:
                directory.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                continue
        else:
            raise CalibrationError("SESSION_CREATE_FAILED", "无法分配会话目录", status=500)
        state = create_initial_state(session_id, display_name, mode, settings)
        add_event(state, "info", "任务已创建", {"mode": mode, "synthetic_only": state["synthetic_only"]})
        atomic_write_json(directory / "session.json", state)
        return state

    def load(self, session_id: str) -> dict[str, Any]:
        session_id = safe_session_id(session_id)
        path = self.state_path(session_id)
        if not path.is_file():
            raise CalibrationError("SESSION_NOT_FOUND", "标定任务不存在", status=404)
        try:
            state = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise CalibrationError("SESSION_CORRUPT", "会话文件损坏", status=500) from error
        if not isinstance(state, dict) or state.get("session_id") != session_id:
            raise CalibrationError("SESSION_CORRUPT", "会话文件身份不一致", status=500)
        return state

    def save(self, state: dict[str, Any]) -> None:
        state["updated_utc"] = utc_now()
        atomic_write_json(self.state_path(state["session_id"]), state)

    @contextlib.contextmanager
    def edit(self, session_id: str) -> Iterator[dict[str, Any]]:
        with self._lock(session_id):
            state = self.load(session_id)
            before = state.get("revision", 0)
            yield state
            state["revision"] = before + 1
            state["updated_utc"] = utc_now()
            atomic_write_json(self.state_path(session_id), state)

    def read(self, session_id: str) -> dict[str, Any]:
        with self._lock(session_id):
            return self.load(session_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions = []
        for directory in sorted(self.root.iterdir(), key=lambda path: path.name):
            if not directory.is_dir() or not (directory / "session.json").is_file():
                continue
            try:
                state = self.load(directory.name)
            except CalibrationError:
                continue
            sessions.append({
                "session_id": state["session_id"],
                "display_name": state.get("display_name", state["session_id"]),
                "mode": state.get("mode", "real"),
                "synthetic_only": bool(state.get("synthetic_only")),
                "created_utc": state.get("created_utc"),
                "updated_utc": state.get("updated_utc"),
                "revision": state.get("revision", 0),
            })
        return sorted(sessions, key=lambda item: item.get("updated_utc") or "", reverse=True)

    def delete_session(self, session_id: str) -> None:
        """Reserved for tests and explicit maintenance; not exposed over HTTP."""
        with self._lock(session_id):
            directory = self.session_dir(session_id)
            if directory.exists():
                shutil.rmtree(directory)


def artifact_paths(state: dict[str, Any]) -> set[str]:
    return known_asset_paths(state)
