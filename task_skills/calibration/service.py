"""Calibration session service, background jobs and report/export logic."""

from __future__ import annotations

import csv
import io
import json
import math
import platform
import threading
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from . import calibrate as c
from . import nova2_read_pose as nova
from . import table_calibration as t
from .synthetic_demo import make_T, render
from .state import (
    ARMS,
    SPLITS,
    CalibrationError,
    SessionStore,
    add_event,
    add_history,
    arm_content_hashes,
    artifact_paths,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    board_input_hash,
    board_status,
    canonical_json,
    clean_text,
    create_initial_state,
    depth_status,
    fit_input_hash,
    fit_status,
    hash_json,
    intrinsics_input_hash,
    intrinsics_sources_status,
    intrinsics_status,
    known_asset_paths,
    load_json,
    mark_stale,
    resolve_session_path,
    sha256_bytes,
    sha256_file,
    teach_input_hash,
    utc_now,
    validate_arm_metadata,
    validate_board,
    validate_settings,
    validate_teach_points,
    validate_thresholds,
    validate_verify_rows,
    verification_input_hash,
    verification_status,
    invalidate_all_arms,
    invalidate_arm,
)

IMAGE_CONTENT_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
}
MAX_TOTAL_IMAGE_BYTES = 600 * 1024 * 1024


def _dependency_versions() -> dict[str, Any]:
    from importlib import metadata

    def distribution(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    numpy_dist = distribution("numpy")
    opencv_dist = distribution("opencv-python")
    return {
        "python": platform.python_version(),
        "python_supported": platform.python_version_tuple()[:2] in {("3", "11"), ("3", "12")},
        "numpy": numpy_dist,
        "numpy_runtime": np.__version__,
        "opencv_python": opencv_dist,
        "opencv_runtime": cv2.__version__,
        "pinned_numpy": "1.26.4",
        "pinned_opencv_python": "4.8.1.78",
        "pinned_match": numpy_dist == "1.26.4" and opencv_dist == "4.8.1.78",
        "videoio_available": hasattr(cv2, "VideoCapture"),
        "device_opened_at_startup": False,
    }


def _sanitize_filename(value: str | None, fallback: str = "image.png") -> str:
    if not value:
        return fallback
    name = Path(str(value).replace("\\", "/")).name
    name = "".join(ch for ch in name if ord(ch) >= 32 and ch not in "\x7f")
    return name[:120] or fallback


def _encode_png(image: np.ndarray) -> bytes:
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise CalibrationError("IMAGE_ENCODE_FAILED", "无法编码图像")
    return data.tobytes()


def _decode_upload(data: bytes, content_type: str) -> np.ndarray:
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise CalibrationError("EMPTY_UPLOAD", "上传内容为空")
    if content_type not in IMAGE_CONTENT_TYPES:
        raise CalibrationError("UNSUPPORTED_IMAGE_TYPE", "只接受 PNG 或 JPEG 图像", status=415)
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise CalibrationError("INVALID_IMAGE", "图像无法解码")
    return image


def _coverage(corners: np.ndarray | None, width: int, height: int) -> dict[str, Any]:
    if corners is None:
        return {"corner_count": 0, "bbox_area_fraction": 0.0, "grid_bins_fraction": 0.0,
                "u_range_px": None, "v_range_px": None}
    uv = np.asarray(corners, dtype=float).reshape(-1, 2)
    if not len(uv):
        return {"corner_count": 0, "bbox_area_fraction": 0.0, "grid_bins_fraction": 0.0,
                "u_range_px": None, "v_range_px": None}
    u0, v0 = uv.min(axis=0)
    u1, v1 = uv.max(axis=0)
    bins = np.zeros((3, 4), dtype=bool)
    cols = np.clip((uv[:, 0] / max(width, 1) * 4).astype(int), 0, 3)
    rows = np.clip((uv[:, 1] / max(height, 1) * 3).astype(int), 0, 2)
    bins[rows, cols] = True
    return {
        "corner_count": int(len(uv)),
        "bbox_area_fraction": float(max(0.0, u1 - u0) * max(0.0, v1 - v0) / (width * height)),
        "grid_bins_fraction": float(bins.mean()),
        "u_range_px": [float(u0), float(u1)],
        "v_range_px": [float(v0), float(v1)],
    }


def _thumbnail(image: np.ndarray, maximum=(480, 360)) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(maximum[0] / width, maximum[1] / height, 1.0)
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                      interpolation=cv2.INTER_AREA)


def _write_png(path: Path, image: np.ndarray) -> None:
    atomic_write_bytes(path, _encode_png(image))


def _image_record(session_dir: Path, image: np.ndarray, settings: dict[str, Any],
                  split: str, display_name: str, image_id: str,
                  synthetic_only: bool) -> dict[str, Any]:
    """Normalize, detect and persist one intrinsic image, returning session metadata."""
    if split not in SPLITS:
        raise CalibrationError("INVALID_SPLIT", "数据划分只能是 train 或 val")
    height, width = image.shape[:2]
    expected = tuple(settings["image_size_wh"])
    if (width, height) != expected:
        raise CalibrationError(
            "IMAGE_SIZE_MISMATCH",
            f"图像尺寸 {width}×{height} 与任务设置 {expected[0]}×{expected[1]} 不一致；内参阶段不能改变分辨率",
        )
    encoded = _encode_png(image)
    digest = sha256_bytes(encoded)
    folder = session_dir / "intrinsics" / "images"
    folder.mkdir(parents=True, exist_ok=True)
    original_path = f"intrinsics/images/{image_id}.png"
    overlay_path = f"intrinsics/images/{image_id}_overlay.png"
    thumbnail_path = f"intrinsics/images/{image_id}_thumb.png"
    _write_png(session_dir / original_path, image)
    _write_png(session_dir / thumbnail_path, _thumbnail(image))
    coverage = _coverage(None, width, height)
    detected = False
    detection_error = None
    try:
        _, size, corners = c.detect(str(session_dir / original_path),
                                    settings["board"]["cols"], settings["board"]["rows"],
                                    expected)
        if tuple(size) != expected:
            raise ValueError("检测图像尺寸不一致")
        overlay = image.copy()
        cv2.drawChessboardCorners(overlay, (settings["board"]["cols"], settings["board"]["rows"]),
                                  corners, True)
        _write_png(session_dir / overlay_path, overlay)
        coverage = _coverage(corners, width, height)
        detected = True
    except (ValueError, OSError, cv2.error) as error:
        _write_png(session_dir / overlay_path, image)
        detection_error = str(error)
    return {
        "id": image_id,
        "split": split,
        "display_name": _sanitize_filename(display_name),
        "sha256": digest,
        "width": width,
        "height": height,
        "original_path": original_path,
        "overlay_path": overlay_path,
        "thumbnail_path": thumbnail_path,
        "detected": detected,
        "detection_error": detection_error,
        "coverage": coverage,
        "reprojection_rms_px": None,
        "residual_overlay_path": None,
        "synthetic_only": bool(synthetic_only),
        "created_utc": utc_now(),
    }


class JobContext:
    def __init__(self, service: "CalibrationService", session_id: str, job_id: str):
        self.service = service
        self.store = service.store
        self.session_id = session_id
        self.job_id = job_id
        self.finished = False

    def _job(self, state: dict[str, Any]) -> dict[str, Any] | None:
        return next((job for job in state.get("jobs", []) if job["id"] == self.job_id), None)

    def log(self, message: str) -> None:
        with self.store.edit(self.session_id) as state:
            job = self._job(state)
            if job is None:
                return
            path = resolve_session_path(self.store.session_dir(self.session_id), job["log_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{utc_now()}] {message}\n")

    def phase(self, phase: str, progress: float | None = None, message: str | None = None) -> None:
        with self.store.edit(self.session_id) as state:
            job = self._job(state)
            if job is None or job.get("status") in ("passed", "failed_threshold", "error", "invalidated"):
                return
            job["phase"] = phase
            if progress is not None:
                job["progress"] = max(0.0, min(1.0, float(progress)))
            job["heartbeat_utc"] = utc_now()
            path = resolve_session_path(self.store.session_dir(self.session_id), job["log_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                suffix = f" — {message}" if message else ""
                handle.write(f"[{utc_now()}] PHASE {phase}{suffix}\n")

    def set_input_hash(self, value: str) -> None:
        with self.store.edit(self.session_id) as state:
            job = self._job(state)
            if job is not None:
                job["input_hash"] = value

    def finish(self, exit_code: int, summary: dict[str, Any] | None,
               commit: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None) -> None:
        if self.finished:
            raise RuntimeError("Job already finished")
        # The terminal state is the source of truth. Only mark the context as
        # finished after the atomic store commit succeeds; otherwise the caller
        # can still persist an error and the job never becomes an orphaned
        # running task.
        with self.store.edit(self.session_id) as state:
            job = self._job(state)
            if job is None:
                raise RuntimeError("Job disappeared before terminal state could be written")
            override = commit(state) if commit else None
            override = override or {}
            effective_exit = int(override.get("exit_code", exit_code))
            status = {0: "passed", 2: "failed_threshold"}.get(effective_exit, "error")
            if override.get("status"):
                status = override["status"]
            job.update({
                "status": status,
                "exit_code": effective_exit,
                "phase": override.get("phase", "完成" if status == "passed" else "结束"),
                "progress": 1.0 if status != "error" else job.get("progress", 0.0),
                "error": override.get("error"),
                "result_summary": summary or {},
                "finished_utc": utc_now(),
                "heartbeat_utc": utc_now(),
            })
            if override.get("summary_patch"):
                job["result_summary"] = {**(summary or {}), **override["summary_patch"]}
            add_event(state, "success" if effective_exit == 0 else "error",
                      f"后台任务结束：{job.get('kind')} {job.get('arm') or ''}".strip(),
                      {"job_id": self.job_id, "exit_code": effective_exit, "status": status})
        try:
            self._append_log(
                f"FINISH exit={effective_exit} status={status}"
                + (f" error={job['error']}" if job.get("error") else "")
            )
        except Exception as error:
            message = f"任务终态日志写入失败：{type(error).__name__}: {error}"
            try:
                self._persist_error_fallback(message)
            except Exception:
                traceback.print_exc()
                raise
        else:
            self.finished = True

    def _append_log(self, message: str) -> None:
        state = self.store.read(self.session_id)
        job = self._job(state)
        if job is None:
            return
        path = resolve_session_path(self.store.session_dir(self.session_id), job["log_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{utc_now()}] {message}\n")

    def _persist_error_fallback(self, message: str) -> None:
        """Persist a terminal error without a commit callback or log device."""
        with self.store.edit(self.session_id) as state:
            job = self._job(state)
            if job is None:
                raise RuntimeError("Job disappeared before fallback error could be written")
            job.update({
                "status": "error",
                "exit_code": 1,
                "phase": "运行错误",
                "error": message,
                "result_summary": {"error": message},
                "finished_utc": utc_now(),
                "heartbeat_utc": utc_now(),
            })
            add_event(state, "error", "后台任务提交失败并已落入错误终态",
                      {"job_id": self.job_id, "error": message})
        self.finished = True

    def fail(self, error: Exception) -> None:
        if self.finished:
            return
        message = (f"[{error.code}] {error.message}" if isinstance(error, CalibrationError)
                   else f"任务运行错误：{type(error).__name__}: {error}")
        try:
            self._append_log("ERROR " + message
                             + ("" if isinstance(error, CalibrationError)
                                else "\n" + traceback.format_exc()))
        except Exception:
            traceback.print_exc()
        # Publish the terminal status and its actionable error atomically. If
        # the normal completion path failed, use a commit-free fallback so the
        # job cannot remain running merely because its commit callback failed.
        try:
            self.finish(1, {"error": message},
                        commit=lambda _state: {"error": message, "phase": "运行错误"})
        except Exception:
            traceback.print_exc()
            self._persist_error_fallback(message)


class CalibrationService:
    def __init__(self, store: SessionStore, *, fit_hook: Callable[[], None] | None = None,
                 max_workers: int = 2):
        self.store = store
        self._executor = ThreadPoolExecutor(max_workers=max_workers,
                                            thread_name_prefix="calibration-job")
        self._job_lock = threading.Lock()
        self._pose_lock = threading.Lock()
        self._fit_hook = fit_hook
        self._closed = False
        self._recover_interrupted_jobs()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=False, cancel_futures=False)

    def _recover_interrupted_jobs(self) -> None:
        for summary in self.store.list_sessions():
            try:
                with self.store.edit(summary["session_id"]) as state:
                    changed = False
                    for job in state.get("jobs", []):
                        if job.get("status") in ("queued", "running"):
                            job.update({
                                "status": "error",
                                "exit_code": 1,
                                "phase": "服务重启",
                                "error": "服务重启时任务未完成；本次结果不可用",
                                "finished_utc": utc_now(),
                            })
                            changed = True
                    if changed:
                        for arm in ARMS:
                            arm_state = state["arms"][arm]
                            if arm_state.get("fit_job_id"):
                                mark_stale(arm_state.get("fit"), "服务重启导致拟合任务中断")
                                arm_state["fit_job_id"] = None
                            if arm_state.get("verification_job_id"):
                                mark_stale(arm_state.get("verification", {}).get("result"),
                                           "服务重启导致验收任务中断")
                                arm_state["verification_job_id"] = None
                        add_event(state, "warning", "已恢复会话并作废中断的后台任务")
            except CalibrationError:
                continue

    def dependencies(self) -> dict[str, Any]:
        report = _dependency_versions()
        report["ok"] = bool(report["numpy"] and report["opencv_python"])
        report["notes"] = [
            "导入与启动不会打开相机、机械臂或串口。",
            "相机预览与 GetPose 仅由操作者显式触发。",
            "真实设备兼容性仍需现场验证。",
        ]
        return report

    def list_sessions(self) -> list[dict[str, Any]]:
        return self.store.list_sessions()

    def create_session(self, body: dict[str, Any]) -> dict[str, Any]:
        name = clean_text(body.get("display_name") or body.get("name"), "任务名称", maximum=80)
        mode = body.get("mode", "real")
        if mode not in ("real", "synthetic"):
            raise CalibrationError("INVALID_MODE", "任务模式只能是 real 或 synthetic")
        settings = validate_settings({
            "camera_id": body.get("camera_id", "overhead_camera"),
            "width": body.get("width", 1280),
            "height": body.get("height", 960),
            "board": body.get("board", {"cols": 11, "rows": 8, "square_m": 0.03}),
            "thresholds": body.get("thresholds", {
                "intrinsics_max_rms_px": 1.0,
                "fit_max_pnp_rms_px": 1.0,
                "fit_max_error_mm": 2.0,
                "verify_max_error_mm": 2.0,
            }),
        })
        if mode == "synthetic" and (
                settings["board"] != {"cols": 11, "rows": 8, "square_m": 0.03}
                or settings["image_size_wh"] != [1280, 960]):
            raise CalibrationError(
                "DEMO_SETTINGS_FIXED",
                "模拟演示固定使用 1280×960、11×8 内角点、30 mm 格长；真实任务可自定义",
            )
        state = self.store.create(name, mode, settings)
        if mode == "synthetic":
            self._submit_job(state["session_id"], "demo", None, self._run_demo)
        return self.store.read(state["session_id"])

    def delete_session(self, session_id: str) -> None:
        self.store.delete_session(session_id)

    def session_view(self, session_id: str) -> dict[str, Any]:
        state = self.store.read(session_id)
        session_dir = self.store.session_dir(session_id)
        intr_ok, intr_reason = intrinsics_status(state, session_dir)
        board_ok, board_reason = board_status(state, session_dir)
        arm_view = {}
        for arm in ARMS:
            fit_ok, fit_reason = fit_status(state, session_dir, arm)
            verify_ok, verify_reason = verification_status(state, session_dir, arm)
            arm_view[arm] = {
                "fit_ok": fit_ok,
                "fit_reason": fit_reason,
                "verification_ok": verify_ok,
                "verification_reason": verify_reason,
                "train_count": sum(p["split"] == "train" for p in state["arms"][arm]["points"]),
                "val_count": sum(p["split"] == "val" for p in state["arms"][arm]["points"]),
            }
        depth_ok, depth_reason = depth_status(state)
        counts = {
            "intrinsics_train": sum(i["split"] == "train" for i in state["intrinsics"]["images"]),
            "intrinsics_val": sum(i["split"] == "val" for i in state["intrinsics"]["images"]),
        }
        return {
            **state,
            "ui": {
                "dependencies": self.dependencies(),
                "gates": {
                    "intrinsics": {"ok": intr_ok, "reason": intr_reason},
                    "board": {"ok": board_ok, "reason": board_reason},
                    "depth": {"ok": depth_ok, "reason": depth_reason},
                    "arms": arm_view,
                },
                "counts": counts,
                "requirements": {
                    "intrinsics_min_train": 10,
                    "intrinsics_min_val": 3,
                    "intrinsics_recommended_train": [18, 24],
                    "intrinsics_recommended_val": [5, 8],
                    "teach_min_train": 6,
                    "teach_min_val": 3,
                    "teach_recommended_train": [8, 12],
                    "teach_recommended_val": [3, 5],
                    "verification_min_points": 3,
                },
                "asset_base": f"/api/sessions/{session_id}/assets/",
            },
        }

    def asset(self, session_id: str, relative: str) -> Path:
        state = self.store.read(session_id)
        normalized = relative.replace("\\", "/")
        if normalized not in known_asset_paths(state):
            raise CalibrationError("ASSET_NOT_FOUND", "该文件不在当前会话产物清单中", status=404)
        path = resolve_session_path(self.store.session_dir(session_id), normalized)
        if not path.is_file():
            raise CalibrationError("ASSET_NOT_FOUND", "产物文件不存在", status=404)
        return path

    def job_log(self, session_id: str, job_id: str, offset: int = 0) -> tuple[str, int]:
        state = self.store.read(session_id)
        job = next((item for item in state.get("jobs", []) if item["id"] == job_id), None)
        if not job:
            raise CalibrationError("JOB_NOT_FOUND", "后台任务不存在", status=404)
        path = resolve_session_path(self.store.session_dir(session_id), job["log_path"])
        if not path.is_file():
            return "", 0
        data = path.read_text(encoding="utf-8", errors="replace")
        offset = max(0, min(int(offset), len(data)))
        return data[offset:], len(data)

    def update_settings(self, session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        settings = validate_settings(body)
        with self.store.edit(session_id) as state:
            old = state["settings"]
            state["settings"] = settings
            geometry_changed = (
                old["camera_id"] != settings["camera_id"]
                or old["image_size_wh"] != settings["image_size_wh"]
                or old["board"] != settings["board"]
            )
            intrinsic_threshold_changed = (
                old["thresholds"]["intrinsics_max_rms_px"]
                != settings["thresholds"]["intrinsics_max_rms_px"]
            )
            if geometry_changed or intrinsic_threshold_changed:
                mark_stale(state["intrinsics"].get("result"), "相机/棋盘几何或内参阈值已更改")
            if geometry_changed and state["board"].get("confirmation"):
                state["board"]["confirmation"]["stale"] = True
                state["board"]["confirmation"]["stale_reason"] = "相机或棋盘几何设置已更改"
                state["board"]["state"] = "stale"
            invalidate_all_arms(state, "任务设置已更改")
            add_history(state, "settings", {"geometry_changed": geometry_changed})
            add_event(state, "warning", "任务设置已更新，相关下游结果已失效",
                      {"geometry_changed": geometry_changed})
        return self.session_view(session_id)

    def add_intrinsics_image(self, session_id: str, data: bytes, content_type: str,
                             split: str, filename: str | None) -> dict[str, Any]:
        image = _decode_upload(data, content_type)
        with self.store.edit(session_id) as state:
            if len(state["intrinsics"]["images"]) >= 400:
                raise CalibrationError("TOO_MANY_IMAGES", "内参图像最多 400 张")
            image_id = f"img-{uuid.uuid4().hex[:16]}"
            record = _image_record(self.store.session_dir(session_id), image, state["settings"],
                                   split, filename or "image.png", image_id,
                                   bool(state.get("synthetic_only")))
            if any(item["sha256"] == record["sha256"] for item in state["intrinsics"]["images"]):
                raise CalibrationError("DUPLICATE_IMAGE", "图库中存在内容完全相同的图像")
            state["intrinsics"]["images"].append(record)
            mark_stale(state["intrinsics"].get("result"), "内参图像或划分已更改")
            invalidate_all_arms(state, "内参图像已更改")
            add_history(state["intrinsics"], "add_image", {"image_id": image_id, "split": split})
            add_event(state, "info", "已添加内参图像", {"image_id": image_id, "split": split,
                                                       "detected": record["detected"]})
            return record

    def update_intrinsics_split(self, session_id: str, image_id: str,
                                split: str) -> dict[str, Any]:
        if split not in SPLITS:
            raise CalibrationError("INVALID_SPLIT", "数据划分只能是 train 或 val")
        with self.store.edit(session_id) as state:
            record = next((item for item in state["intrinsics"]["images"]
                           if item["id"] == image_id), None)
            if not record:
                raise CalibrationError("IMAGE_NOT_FOUND", "图像不存在", status=404)
            if record["split"] != split:
                record["split"] = split
                mark_stale(state["intrinsics"].get("result"), "内参图像或划分已更改")
                invalidate_all_arms(state, "内参数据划分已更改")
                add_history(state["intrinsics"], "split", {"image_id": image_id, "split": split})
            return record

    def delete_intrinsics_image(self, session_id: str, image_id: str) -> None:
        with self.store.edit(session_id) as state:
            images = state["intrinsics"]["images"]
            record = next((item for item in images if item["id"] == image_id), None)
            if not record:
                raise CalibrationError("IMAGE_NOT_FOUND", "图像不存在", status=404)
            images.remove(record)
            mark_stale(state["intrinsics"].get("result"), "内参图像或划分已更改")
            invalidate_all_arms(state, "内参图像已删除")
            add_history(state["intrinsics"], "delete_image", {"image_id": image_id})
            add_event(state, "info", "已删除内参图像", {"image_id": image_id})

    def batch_intrinsics_images(self, session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Delete or re-split many images in one edit, so downstream is invalidated once.

        ``all: true`` targets the whole gallery; otherwise ``image_ids`` must name
        existing images. Any unknown id rejects the whole request unchanged.
        """
        action = body.get("action")
        if action not in ("delete", "split"):
            raise CalibrationError("INVALID_BATCH_ACTION", "批量操作只能是 delete 或 split")
        split = body.get("split")
        if action == "split" and split not in SPLITS:
            raise CalibrationError("INVALID_SPLIT", "数据划分只能是 train 或 val")
        select_all = body.get("all") is True
        ids = body.get("image_ids")
        if not select_all:
            if (not isinstance(ids, list) or not ids or len(ids) > 400
                    or not all(isinstance(item, str) for item in ids)):
                raise CalibrationError("INVALID_IMAGE_IDS", "image_ids 必须是 1–400 个图像 ID")
        with self.store.edit(session_id) as state:
            images = state["intrinsics"]["images"]
            if select_all:
                targets = list(images)
            else:
                by_id = {item["id"]: item for item in images}
                missing = [item for item in dict.fromkeys(ids) if item not in by_id]
                if missing:
                    raise CalibrationError("IMAGE_NOT_FOUND", f"{len(missing)} 张图像不存在，未做任何修改",
                                           status=404)
                targets = [by_id[item] for item in dict.fromkeys(ids)]
            if action == "delete":
                target_ids = {item["id"] for item in targets}
                images[:] = [item for item in images if item["id"] not in target_ids]
                changed = len(targets)
            else:
                changed = 0
                for record in targets:
                    if record["split"] != split:
                        record["split"] = split
                        changed += 1
            if changed:
                mark_stale(state["intrinsics"].get("result"), "内参图像或划分已更改")
                invalidate_all_arms(state, "内参图像已批量删除" if action == "delete" else "内参数据划分已更改")
                detail = {"action": action, "count": changed, "all": select_all}
                if action == "split":
                    detail["split"] = split
                add_history(state["intrinsics"], f"batch_{action}", detail)
                add_event(state, "info", "已批量删除内参图像" if action == "delete"
                          else f"已批量设为 {split}", detail)
            return {"action": action, "count": changed, "remaining": len(images)}

    def upload_board_image(self, session_id: str, data: bytes, content_type: str,
                           filename: str | None) -> dict[str, Any]:
        image = _decode_upload(data, content_type)
        with self.store.edit(session_id) as state:
            height, width = image.shape[:2]
            expected = tuple(state["settings"]["image_size_wh"])
            if (width, height) != expected:
                raise CalibrationError(
                    "IMAGE_SIZE_MISMATCH",
                    f"固定板图像尺寸 {width}×{height} 与任务设置 {expected[0]}×{expected[1]} 不一致",
                )
            image_id = f"board-{uuid.uuid4().hex[:16]}"
            relative = f"board/uploads/{image_id}.png"
            _write_png(self.store.session_dir(session_id) / relative, image)
            state["board"].update({
                "stem_path": relative,
                "stem_sha256": sha256_bytes(_encode_png(image)),
                "stem_display_name": _sanitize_filename(filename or "fixed-board.png"),
                "source_path": relative,
                "source_sha256": sha256_bytes(_encode_png(image)),
                "image_size_wh": [width, height],
                "raw_corners_uv": [],
                "previews": {},
                "confirmation": None,
                "state": "uploaded",
                "detected_utc": None,
            })
            invalidate_all_arms(state, "固定棋盘图像已更改")
            add_history(state["board"], "upload", {"path": relative})
            add_event(state, "info", "已上传固定棋盘图像；请重新检测并确认编号")
            return self._board_view(state)

    def _board_view(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "board": state["board"],
            "gate": {"ok": board_status(state, self.store.session_dir(state["session_id"]))[0],
                     "reason": board_status(state, self.store.session_dir(state["session_id"]))[1]},
        }

    def detect_board(self, session_id: str) -> dict[str, Any]:
        with self.store.edit(session_id) as state:
            board = state["board"]
            if not board.get("source_path"):
                raise CalibrationError("BOARD_IMAGE_REQUIRED", "请先上传固定棋盘图像")
            session_dir = self.store.session_dir(session_id)
            source = resolve_session_path(session_dir, board["source_path"])
            if not source.is_file():
                raise CalibrationError("BOARD_IMAGE_MISSING", "固定棋盘图像不存在")
            if sha256_file(source) != board.get("source_sha256"):
                raise CalibrationError("BOARD_IMAGE_CHANGED", "固定棋盘图像被修改，请重新上传")
            image = c.read_image(source)
            settings = state["settings"]
            try:
                _, size, corners = c.detect(str(source), settings["board"]["cols"],
                                            settings["board"]["rows"],
                                            tuple(settings["image_size_wh"]))
            except (ValueError, OSError, cv2.error) as error:
                raise CalibrationError("BOARD_NOT_DETECTED", f"未检测到完整棋盘：{error}") from error
            revision = int(board.get("revision", 0)) + 1
            version_dir = session_dir / "board" / f"v{revision:03d}"
            version_dir.mkdir(parents=True, exist_ok=True)
            source_rel = f"board/v{revision:03d}/source.png"
            _write_png(session_dir / source_rel, image)
            source_sha = sha256_file(session_dir / source_rel)
            previews: dict[str, str] = {}
            observations: dict[str, tuple[str, str]] = {}
            raw = corners.reshape(-1, 2).astype(float).tolist()
            for order in ("normal", "reverse"):
                ordered = c.ordered(corners, order)
                annotated = image.copy()
                for index, (u, v) in enumerate(ordered.reshape(-1, 2)):
                    point = (int(round(float(u))), int(round(float(v))))
                    cv2.circle(annotated, point, 3, (0, 0, 255), -1)
                    label = (point[0] + 3, point[1] - 4)
                    cv2.putText(annotated, str(index), label, cv2.FONT_HERSHEY_SIMPLEX,
                                .40, (255, 255, 255), 3, cv2.LINE_AA)
                    cv2.putText(annotated, str(index), label, cv2.FONT_HERSHEY_SIMPLEX,
                                .40, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(annotated, "id=0 origin  x:0->1  y:0->11", (14, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 90, 255), 2, cv2.LINE_AA)
                preview_rel = f"board/v{revision:03d}/{order}_numbered.png"
                _write_png(session_dir / preview_rel, annotated)
                previews[order] = preview_rel
                observation = {
                    **c.base_result("board_observation"),
                    "synthetic_only": bool(state.get("synthetic_only")),
                    "board": settings["board"],
                    "image_size_wh": list(size),
                    "source_image": "source.png",
                    "source_sha256": source_sha,
                    "corner_order": order,
                    "raw_corners_uv": raw,
                    "corners_uv": ordered.reshape(-1, 2).astype(float).tolist(),
                    "input_hash": board_input_hash(settings, source_sha, order),
                }
                obs_rel = f"board/v{revision:03d}/observation_{order}.json"
                atomic_write_json(session_dir / obs_rel, observation)
                observations[order] = (obs_rel, sha256_file(session_dir / obs_rel))
            board.update({
                "revision": revision,
                "state": "detected",
                "source_path": source_rel,
                "source_sha256": source_sha,
                "image_size_wh": list(size),
                "raw_corners_uv": raw,
                "previews": previews,
                "observations": {key: value[0] for key, value in observations.items()},
                "detected_utc": utc_now(),
                "confirmation": None,
            })
            invalidate_all_arms(state, "固定棋盘重新检测")
            add_history(board, "detect", {"revision": revision, "corners": len(raw)})
            add_event(state, "info", "固定棋盘检测完成，等待 normal/reverse 确认",
                      {"revision": revision})
            return self._board_view(state)

    def confirm_board(self, session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        order = body.get("corner_order")
        if order not in ("normal", "reverse"):
            raise CalibrationError("INVALID_CORNER_ORDER", "请选择 normal 或 reverse 编号")
        if body.get("fixed_confirmed") is not True:
            raise CalibrationError("BOARD_CONFIRMATION_REQUIRED", "必须确认棋盘已固定且 88 个内角点完整可见")
        with self.store.edit(session_id) as state:
            board = state["board"]
            if board.get("state") != "detected" or not board.get("observations"):
                raise CalibrationError("BOARD_NOT_DETECTED", "请先检测固定棋盘")
            existing = board.get("confirmation")
            if existing and not existing.get("stale"):
                if existing.get("corner_order") == order:
                    return self._board_view(state)
                raise CalibrationError("BOARD_IMMUTABLE", "当前编号版本已确认；如需改变方向，请重新检测生成新版本", status=409)
            observation_rel = board["observations"][order]
            observation_path = resolve_session_path(self.store.session_dir(session_id), observation_rel)
            observation_sha = sha256_file(observation_path)
            settings = state["settings"]
            confirmation = {
                "corner_order": order,
                "confirmed_utc": utc_now(),
                "immutable_revision": board["revision"],
                "observation_path": observation_rel,
                "observation_sha256": observation_sha,
                "input_hash": board_input_hash(settings, board["source_sha256"], order),
                "fixed_confirmed": True,
            }
            board["confirmation"] = confirmation
            board["state"] = "confirmed"
            invalidate_all_arms(state, "固定棋盘编号已确认/变更")
            add_history(board, "confirm", {"order": order, "revision": board["revision"]})
            add_event(state, "success", "固定棋盘编号已冻结", {"order": order,
                                                               "revision": board["revision"]})
            return self._board_view(state)

    def update_arm_metadata(self, session_id: str, arm: str,
                            body: dict[str, Any]) -> dict[str, Any]:
        arm = self._arm(arm)
        metadata = validate_arm_metadata(body)
        with self.store.edit(session_id) as state:
            arm_state = state["arms"][arm]
            changed = arm_state["metadata"] != metadata
            if changed:
                arm_state["metadata"] = metadata
                arm_state["metadata_hash"] = hash_json(metadata)
                invalidate_arm(state, arm, "机械臂元数据已更改")
                add_history(arm_state, "metadata", {"metadata": metadata})
                add_event(state, "warning", f"{self._arm_cn(arm)}元数据已更新，拟合与验收失效")
            return arm_state

    def update_teach_points(self, session_id: str, arm: str,
                            body: dict[str, Any]) -> dict[str, Any]:
        arm = self._arm(arm)
        with self.store.edit(session_id) as state:
            count = state["settings"]["board"]["cols"] * state["settings"]["board"]["rows"]
            points = validate_teach_points(body.get("points"), count)
            self._teach_coverage_for_arm(state, arm, points)
            arm_state = state["arms"][arm]
            if points != arm_state["points"]:
                arm_state["points"] = points
                arm_state["points_hash"] = hash_json(points)
                invalidate_arm(state, arm, "示教点已更改")
                add_history(arm_state, "points", {"count": len(points)})
                add_event(state, "warning", f"{self._arm_cn(arm)}示教点已更新，拟合与验收失效")
            return arm_state

    def _teach_coverage_for_arm(self, state: dict[str, Any], arm: str,
                                points: list[dict[str, Any]]) -> None:
        train = [point for point in points if point["split"] == "train"]
        if len(train) < 3:
            return
        board = state["settings"]["board"]
        obj = c.board_points(board["cols"], board["rows"], board["square_m"])
        unit = state["arms"][arm]["metadata"]["unit"]
        scale = 0.001 if unit == "mm" else 1.0
        ids = np.array([point["corner_id"] for point in train], dtype=int)
        measured = np.array([[point["x"], point["y"], point["z"]] for point in train], dtype=float) * scale
        try:
            t.rigid_fit(obj[ids], measured)
        except ValueError as error:
            raise CalibrationError("TEACH_COVERAGE_INVALID", f"示教点覆盖不足：{error}") from error

    def import_teach_csv(self, session_id: str, arm: str, body: dict[str, Any]) -> dict[str, Any]:
        arm = self._arm(arm)
        text = body.get("csv")
        if not isinstance(text, str) or not text.strip():
            raise CalibrationError("INVALID_CSV", "CSV 内容为空")
        if len(text.encode("utf-8")) > 1024 * 1024:
            raise CalibrationError("CSV_TOO_LARGE", "CSV 最大 1 MiB", status=413)
        with self.store.edit(session_id) as state:
            metadata = state["arms"][arm]["metadata"]
            reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
            required = {"corner_id", "split", "x", "y", "z"}
            if not required.issubset(reader.fieldnames or []):
                raise CalibrationError("INVALID_CSV", "CSV 必须包含 corner_id,split,x,y,z")
            points = []
            for index, row in enumerate(reader, start=2):
                for key in ("unit", "robot_id", "base_frame", "tcp_name"):
                    value = (row.get(key) or "").strip()
                    if value and value != metadata[key]:
                        raise CalibrationError("CSV_METADATA_MISMATCH",
                                               f"CSV 第 {index} 行 {key} 与当前机械臂元数据不一致")
                try:
                    corner_id = int(row["corner_id"])
                    x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
                except (TypeError, ValueError) as error:
                    raise CalibrationError("INVALID_CSV", f"CSV 第 {index} 行数值不合法") from error
                points.append({"corner_id": corner_id, "split": row["split"],
                               "x": x, "y": y, "z": z})
            points = validate_teach_points(points, state["settings"]["board"]["cols"]
                                           * state["settings"]["board"]["rows"])
            self._teach_coverage_for_arm(state, arm, points)
            arm_state = state["arms"][arm]
            arm_state["points"] = points
            arm_state["points_hash"] = hash_json(points)
            invalidate_arm(state, arm, "CSV 导入示教点")
            add_history(arm_state, "import_csv", {"count": len(points)})
            add_event(state, "info", f"{self._arm_cn(arm)}已导入示教 CSV", {"count": len(points)})
            return arm_state

    def export_teach_csv(self, session_id: str, arm: str) -> str:
        arm = self._arm(arm)
        state = self.store.read(session_id)
        metadata = state["arms"][arm]["metadata"]
        output = io.StringIO()
        fields = ["corner_id", "split", "x", "y", "z", "unit", "robot_id", "base_frame", "tcp_name"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for point in state["arms"][arm]["points"]:
            writer.writerow({**point, **{key: metadata[key] for key in fields[5:]}})
        return output.getvalue()

    def import_verify_csv(self, session_id: str, arm: str, body: dict[str, Any]) -> dict[str, Any]:
        arm = self._arm(arm)
        text = body.get("csv")
        if not isinstance(text, str) or not text.strip():
            raise CalibrationError("INVALID_CSV", "CSV 内容为空")
        if len(text.encode("utf-8")) > 1024 * 1024:
            raise CalibrationError("CSV_TOO_LARGE", "CSV 最大 1 MiB", status=413)
        with self.store.edit(session_id) as state:
            reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
            required = {"u_px", "v_px", "board_z_m", "reference_x_m", "reference_y_m", "reference_z_m"}
            if not required.issubset(reader.fieldnames or []):
                raise CalibrationError("INVALID_CSV", "CSV 缺少独立验收必需列")
            rows = []
            for index, row in enumerate(reader, start=2):
                try:
                    rows.append({key: float(row[key]) for key in required})
                except (TypeError, ValueError) as error:
                    raise CalibrationError("INVALID_CSV", f"CSV 第 {index} 行数值不合法") from error
            rows = validate_verify_rows(rows, tuple(state["settings"]["image_size_wh"]))
            verification = state["arms"][arm]["verification"]
            verification["rows"] = rows
            verification["result"] = None
            add_history(verification, "import_csv", {"count": len(rows)})
            add_event(state, "info", f"{self._arm_cn(arm)}已导入独立验收 CSV", {"count": len(rows)})
            return verification

    def export_verify_csv(self, session_id: str, arm: str) -> str:
        arm = self._arm(arm)
        state = self.store.read(session_id)
        output = io.StringIO()
        fields = ["u_px", "v_px", "board_z_m", "reference_x_m", "reference_y_m", "reference_z_m"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(state["arms"][arm]["verification"]["rows"])
        return output.getvalue()

    def update_verification_draft(self, session_id: str, arm: str,
                                  body: dict[str, Any]) -> dict[str, Any]:
        arm = self._arm(arm)
        with self.store.edit(session_id) as state:
            rows = validate_verify_rows(body.get("rows", []), tuple(state["settings"]["image_size_wh"]))
            method = clean_text(body.get("method", ""), "测量方法", maximum=500, allow_empty=True)
            confirmed = body.get("independence_confirmed") is True
            verification = state["arms"][arm]["verification"]
            verification.update({
                "rows": rows,
                "method": method,
                "independence_confirmed": confirmed,
                "confirmed_utc": utc_now() if confirmed else None,
                "input_hash": None,
                "result": None,
            })
            mark_stale(verification.get("result"), "独立验收输入已更改")
            add_history(verification, "draft", {"rows": len(rows), "confirmed": confirmed})
            add_event(state, "warning", f"{self._arm_cn(arm)}独立验收草稿已更新，旧验收失效")
            return verification

    def _arm(self, arm: str) -> str:
        if arm not in ARMS:
            raise CalibrationError("INVALID_ARM", "只支持 left 或 right 臂", status=404)
        return arm

    @staticmethod
    def _arm_cn(arm: str) -> str:
        return "左臂" if arm == "left" else "右臂"

    def get_pose(self, session_id: str, arm: str, body: dict[str, Any]) -> dict[str, Any]:
        arm = self._arm(arm)
        if not all(body.get(key) is True for key in ("confirm_base", "confirm_tcp", "confirm_unit")):
            raise CalibrationError("POSE_CONFIRMATION_REQUIRED",
                                   "必须先确认示教器基座、尖端 TCP 与 GetPose 单位")
        state = self.store.read(session_id)
        metadata = state["arms"][arm]["metadata"]
        if not metadata.get("controller_host"):
            raise CalibrationError("CONTROLLER_HOST_REQUIRED", "请先填写控制器 IP/主机名")
        if not self._pose_lock.acquire(blocking=False):
            raise CalibrationError("POSE_BUSY", "已有 GetPose 请求正在执行", status=409)
        try:
            pose, raw = nova.read_pose(metadata["controller_host"], metadata["controller_port"], timeout=3.0)
        finally:
            self._pose_lock.release()
        with self.store.edit(session_id) as current:
            add_event(current, "info", f"{self._arm_cn(arm)}执行显式只读 GetPose()",
                      {"host": metadata["controller_host"], "port": metadata["controller_port"],
                       "xyz": pose[:3], "unit": metadata["unit"]})
        return {
            "robot_id": metadata["robot_id"],
            "base_frame": metadata["base_frame"],
            "tcp_name": metadata["tcp_name"],
            "unit": metadata["unit"],
            "xyz": pose[:3],
            "rpy_raw": pose[3:],
            "raw_reply": raw,
            "auto_saved": False,
            "network_command": "GetPose()",
        }

    # ------------------------------------------------------------------
    # Background job plumbing
    # ------------------------------------------------------------------

    def _active_jobs(self, state: dict[str, Any], kind: str | None = None,
                     arm: str | None = None) -> list[dict[str, Any]]:
        active = []
        for job in state.get("jobs", []):
            if job.get("status") not in ("queued", "running"):
                continue
            if kind is not None and job.get("kind") != kind:
                continue
            if arm is not None and job.get("arm") != arm:
                continue
            active.append(job)
        return active

    def _submit_job(self, session_id: str, kind: str, arm: str | None,
                    target: Callable[[JobContext], None]) -> str:
        job_id = f"job-{uuid.uuid4().hex[:16]}"
        with self.store.edit(session_id) as state:
            if kind == "demo" and not state.get("synthetic_only"):
                raise CalibrationError("DEMO_REQUIRES_SYNTHETIC", "模拟演示只能用于 synthetic 任务")
            if self._active_jobs(state, kind=kind, arm=arm):
                raise CalibrationError("JOB_ALREADY_RUNNING", "同类任务已在运行", status=409)
            if len(self._active_jobs(state)) >= 8:
                raise CalibrationError("JOB_LIMIT", "后台任务过多，请等待现有任务结束", status=429)
            if kind == "demo" and self._active_jobs(state):
                raise CalibrationError("DEMO_BUSY", "模拟演示需要独占任务队列", status=409)
            log_path = f"jobs/{job_id}/log.txt"
            log_file = self.store.session_dir(session_id) / log_path
            log_file.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(log_file, f"[{utc_now()}] created kind={kind} arm={arm or '-'}\n")
            job = {
                "id": job_id,
                "kind": kind,
                "arm": arm,
                "status": "queued",
                "phase": "排队",
                "progress": 0.0,
                "created_utc": utc_now(),
                "started_utc": None,
                "finished_utc": None,
                "heartbeat_utc": utc_now(),
                "input_hash": None,
                "exit_code": None,
                "error": None,
                "result_summary": {},
                "log_path": log_path,
            }
            state.setdefault("jobs", []).append(job)
            state["jobs"] = state["jobs"][-120:]
            if kind == "intrinsics":
                mark_stale(state["intrinsics"].get("result"), "新的内参任务已启动")
                invalidate_all_arms(state, "新的内参任务已启动")
            elif kind == "fit" and arm in ARMS:
                arm_state = state["arms"][arm]
                mark_stale(arm_state.get("fit"), "新的拟合任务已启动")
                mark_stale(arm_state.get("verification", {}).get("result"), "拟合重跑")
                arm_state["fit_job_id"] = job_id
            elif kind == "verify" and arm in ARMS:
                arm_state = state["arms"][arm]
                mark_stale(arm_state.get("verification", {}).get("result"), "新的独立验收任务已启动")
                arm_state["verification_job_id"] = job_id
            add_event(state, "info", "后台任务已排队", {"job_id": job_id, "kind": kind,
                                                       "arm": arm})
        try:
            self._executor.submit(self._execute_job, session_id, job_id, target)
        except RuntimeError as error:
            with self.store.edit(session_id) as state:
                job = next((item for item in state["jobs"] if item["id"] == job_id), None)
                if job:
                    job.update({"status": "error", "exit_code": 1, "phase": "调度失败",
                                "error": "服务正在关闭，任务未能启动", "finished_utc": utc_now()})
            raise CalibrationError("JOB_SUBMIT_FAILED", "无法启动后台任务", status=503) from error
        return job_id

    def _execute_job(self, session_id: str, job_id: str,
                     target: Callable[[JobContext], None]) -> None:
        ctx = JobContext(self, session_id, job_id)
        try:
            with self.store.edit(session_id) as state:
                job = next(item for item in state["jobs"] if item["id"] == job_id)
                job.update({"status": "running", "started_utc": utc_now(),
                            "phase": "运行中", "progress": 0.01, "heartbeat_utc": utc_now()})
            target(ctx)
            if not ctx.finished:
                ctx.fail(RuntimeError("任务函数结束但未提交结果"))
        except Exception as error:  # job boundary: persist, never kill the server
            try:
                ctx.fail(error)
            except Exception:
                traceback.print_exc()
        finally:
            try:
                with self.store.edit(session_id) as state:
                    job = next((item for item in state["jobs"] if item["id"] == job_id), None)
                    if job and job.get("status") in ("queued", "running"):
                        # Last-resort terminal write for transient failures in
                        # commit or the normal error path. This keeps the job
                        # retryable instead of leaving it running forever.
                        message = "后台任务未完成且未能提交正常结果；本次结果不可用"
                        job.update({
                            "status": "error",
                            "exit_code": 1,
                            "phase": "任务终止",
                            "error": job.get("error") or message,
                            "result_summary": {"error": job.get("error") or message},
                            "finished_utc": utc_now(),
                            "heartbeat_utc": utc_now(),
                        })
                        add_event(state, "error", "后台任务落入兜底错误终态",
                                  {"job_id": job_id, "error": job["error"]})
                        ctx.finished = True
                    for arm in ARMS:
                        arm_state = state["arms"][arm]
                        if arm_state.get("fit_job_id") == job_id:
                            arm_state["fit_job_id"] = None
                            if job and job.get("status") not in ("passed", "failed_threshold"):
                                mark_stale(arm_state.get("fit"), "拟合任务失败或中断")
                        if arm_state.get("verification_job_id") == job_id:
                            arm_state["verification_job_id"] = None
                            if job and job.get("status") not in ("passed", "failed_threshold"):
                                mark_stale(arm_state.get("verification", {}).get("result"),
                                           "独立验收任务失败或中断")
            except Exception:
                # Storage is unavailable; surface the failure rather than
                # swallowing it. A later service restart recovery will convert
                # any still-running job to an error state.
                traceback.print_exc()

    def run_intrinsics(self, session_id: str) -> str:
        return self._submit_job(session_id, "intrinsics", None, self._run_intrinsics)

    def run_fit(self, session_id: str, arm: str) -> str:
        arm = self._arm(arm)
        state = self.store.read(session_id)
        intr_ok, intr_reason = intrinsics_status(state, self.store.session_dir(session_id))
        board_ok, board_reason = board_status(state, self.store.session_dir(session_id))
        if not intr_ok:
            raise CalibrationError("INTRINSICS_NOT_READY", f"内参不可用：{intr_reason}", status=409)
        if not board_ok:
            raise CalibrationError("BOARD_NOT_READY", f"固定棋盘不可用：{board_reason}", status=409)
        metadata = state["arms"][arm]["metadata"]
        if not metadata["robot_id"] or not metadata["base_frame"] or not metadata["tcp_name"]:
            raise CalibrationError("ARM_METADATA_REQUIRED", "请填写 robot_id、base_frame 与 TCP 名称")
        return self._submit_job(session_id, "fit", arm, self._run_fit)

    def run_verification(self, session_id: str, arm: str) -> str:
        arm = self._arm(arm)
        state = self.store.read(session_id)
        ok, reason = fit_status(state, self.store.session_dir(session_id), arm)
        if not ok:
            raise CalibrationError("FIT_NOT_READY", f"拟合结果不可用：{reason}", status=409)
        draft = state["arms"][arm]["verification"]
        if len(draft.get("rows", [])) < 3:
            raise CalibrationError("VERIFY_POINTS_REQUIRED", "独立验收至少需要 3 个测量点")
        if not draft.get("method"):
            raise CalibrationError("VERIFY_METHOD_REQUIRED", "请填写独立测量方法")
        if not draft.get("independence_confirmed"):
            raise CalibrationError("VERIFY_INDEPENDENCE_REQUIRED", "必须确认参考点独立于拟合数据")
        return self._submit_job(session_id, "verify", arm, self._run_verification)

    # ------------------------------------------------------------------
    # Intrinsics job
    # ------------------------------------------------------------------

    def _compute_intrinsics_artifact(self, state: dict[str, Any], job_id: str,
                                     synthetic_only: bool,
                                     log: Callable[[str], None]) -> dict[str, Any]:
        session_dir = self.store.session_dir(state["session_id"])
        settings = state["settings"]
        images = state["intrinsics"]["images"]
        train_count = sum(item["split"] == "train" for item in images)
        val_count = sum(item["split"] == "val" for item in images)
        if train_count < 10 or val_count < 3:
            raise CalibrationError(
                "INTRINSICS_COUNTS",
                f"内参至少需要 10 train / 3 val，当前 {train_count} / {val_count}",
            )
        out_dir = session_dir / "intrinsics" / "runs" / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        obj = c.board_points(settings["board"]["cols"], settings["board"]["rows"],
                             settings["board"]["square_m"])
        corners_by_image: list[np.ndarray] = []
        manifest_rows = []
        for index, image_record in enumerate(images, start=1):
            path = resolve_session_path(session_dir, image_record["original_path"])
            if not path.is_file():
                raise CalibrationError(
                    "INTRINSICS_SOURCE_MISSING",
                    f"内参原图缺失：{image_record.get('display_name') or path.name}",
                )
            actual_sha256 = sha256_file(path)
            if actual_sha256 != image_record.get("sha256"):
                raise CalibrationError(
                    "INTRINSICS_SOURCE_CHANGED",
                    f"内参原图哈希不匹配：{image_record.get('display_name') or path.name}；"
                    "请重新上传该图或删除后重采，不能沿用旧输入哈希",
                )
            try:
                _, size, corners = c.detect(str(path), settings["board"]["cols"],
                                            settings["board"]["rows"],
                                            tuple(settings["image_size_wh"]))
            except (ValueError, OSError, cv2.error) as error:
                raise CalibrationError("INTRINSICS_DETECTION_FAILED",
                                       f"图像 {image_record['display_name']} 角点检测失败：{error}") from error
            if tuple(size) != tuple(settings["image_size_wh"]):
                raise CalibrationError("INTRINSICS_SIZE_MISMATCH", "图像尺寸与任务设置不一致")
            corners_by_image.append(corners)
            manifest_rows.append({"image": image_record["original_path"],
                                  "split": image_record["split"],
                                  "sha256": actual_sha256})
            log(f"检测角点 {index}/{len(images)}：{image_record['display_name']}")
        training = [corners for record, corners in zip(images, corners_by_image)
                    if record["split"] == "train"]
        size = tuple(settings["image_size_wh"])
        try:
            rms, K, D, _, _ = cv2.calibrateCamera(
                [obj] * len(training), training, size, None, None,
            )
        except cv2.error as error:
            raise CalibrationError("INTRINSICS_CALIBRATION_FAILED", f"OpenCV 内参拟合失败：{error}") from error
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(D)) or not math.isfinite(float(rms)):
            raise CalibrationError("NONFINITE_INTRINSICS", "内参拟合产生 NaN/Inf")
        reports = []
        overlay_map = {}
        val_errors = []
        for image_record, corners in zip(images, corners_by_image):
            try:
                _, view_rms = c.pose_from_board(obj, corners, K, D)
            except ValueError as error:
                raise CalibrationError("INTRINSICS_VIEW_FAILED",
                                       f"图像 {image_record['display_name']} 重投影失败：{error}") from error
            image = c.read_image(resolve_session_path(session_dir, image_record["original_path"]))
            annotated = image.copy()
            cv2.drawChessboardCorners(annotated, (settings["board"]["cols"], settings["board"]["rows"]),
                                      corners, True)
            cv2.putText(annotated, f"RMS {view_rms:.3f} px", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)
            overlay_rel = f"intrinsics/runs/{job_id}/overlays/{image_record['id']}.png"
            _write_png(session_dir / overlay_rel, annotated)
            overlay_map[image_record["id"]] = overlay_rel
            reports.append({
                "image_id": image_record["id"],
                "display_name": image_record["display_name"],
                "split": image_record["split"],
                "sha256": image_record["sha256"],
                "reprojection_rms_px": view_rms,
                "coverage": image_record["coverage"],
                "residual_overlay_path": overlay_rel,
            })
            if image_record["split"] == "val":
                val_errors.append(view_rms)
        threshold = settings["thresholds"]["intrinsics_max_rms_px"]
        max_view = max(row["reprojection_rms_px"] for row in reports)
        passed = bool(max_view <= threshold)
        input_hash = intrinsics_input_hash(settings, images)
        artifact = {
            **c.base_result("intrinsics"),
            "synthetic_only": bool(synthetic_only),
            "input_hash": input_hash,
            "camera_id": settings["camera_id"],
            "image_size_wh": list(size),
            "board": settings["board"],
            "model": "opencv_pinhole_5coeff",
            "K": np.asarray(K).tolist(),
            "distortion": np.asarray(D).reshape(-1).tolist(),
            "train_rms_px": float(rms),
            "max_view_rms_px": float(max_view),
            "val_max_view_rms_px": float(max(val_errors)) if val_errors else None,
            "train_count": train_count,
            "val_count": val_count,
            "views": reports,
            "max_allowed_view_rms_px": threshold,
            "validation_passed": passed,
            "validation_scope": "held-out image reprojection with per-image PnP; not metric robot accuracy",
        }
        artifact_rel = f"intrinsics/runs/{job_id}/intrinsics.json"
        artifact_path = session_dir / artifact_rel
        atomic_write_json(artifact_path, artifact)
        manifest_text = "image,split,sha256\n" + "".join(
            f"{row['image']},{row['split']},{row['sha256']}\n" for row in manifest_rows
        )
        atomic_write_text(session_dir / f"intrinsics/runs/{job_id}/manifest.csv", manifest_text)
        return {
            "artifact": artifact,
            "artifact_path": artifact_rel,
            "artifact_sha256": sha256_file(artifact_path),
            "overlay_map": overlay_map,
            "exit_code": 0 if passed else 2,
            "summary": {
                "train_rms_px": float(rms),
                "max_view_rms_px": float(max_view),
                "val_max_view_rms_px": artifact["val_max_view_rms_px"],
                "train_count": train_count,
                "val_count": val_count,
                "threshold_px": threshold,
                "validation_passed": passed,
            },
        }

    def _store_intrinsics_result(self, state: dict[str, Any], job_id: str,
                                 computed: dict[str, Any], captured_hash: str,
                                 *, stale: bool = False) -> dict[str, Any] | None:
        artifact = computed["artifact"]
        exit_code = computed["exit_code"]
        if stale:
            exit_code = 1
        status = "passed" if exit_code == 0 else ("failed_threshold" if exit_code == 2 else "invalidated")
        record = {
            "job_id": job_id,
            "status": status,
            "exit_code": exit_code,
            "validation_passed": bool(exit_code == 0 and artifact.get("validation_passed")),
            "usable": bool(exit_code == 0 and not stale),
            "stale": bool(stale),
            "stale_reason": "运行期间内参输入已更改，结果仅保留为证据" if stale else None,
            "input_hash": captured_hash,
            "artifact_path": computed["artifact_path"],
            "artifact_sha256": computed["artifact_sha256"],
            "synthetic_only": bool(artifact.get("synthetic_only")),
            "summary": computed["summary"],
            "views": artifact["views"],
            "computed_utc": utc_now(),
        }
        state["intrinsics"]["result"] = record
        view_map = {row["image_id"]: row for row in artifact["views"]}
        for image_record in state["intrinsics"]["images"]:
            row = view_map.get(image_record["id"])
            if row:
                image_record["reprojection_rms_px"] = row["reprojection_rms_px"]
                image_record["residual_overlay_path"] = row["residual_overlay_path"]
        invalidate_all_arms(state, "内参结果已更新")
        add_history(state["intrinsics"], "result", {"job_id": job_id, "exit_code": exit_code,
                                                    "stale": stale})
        add_event(state, "success" if exit_code == 0 else "error", "内参任务结束",
                  {"job_id": job_id, "exit_code": exit_code, "stale": stale})
        if stale:
            return {"exit_code": 1, "status": "invalidated",
                    "error": "运行期间内参输入已更改，结果已作废"}
        return None

    def _run_intrinsics(self, ctx: JobContext) -> None:
        state = self.store.read(ctx.session_id)
        captured = intrinsics_input_hash(state["settings"], state["intrinsics"]["images"])
        ctx.set_input_hash(captured)
        ctx.phase("检测棋盘并拟合内参", 0.08)
        ctx.log("开始内参计算；训练图用于拟合，val 图仅用于检查")
        computed = self._compute_intrinsics_artifact(state, ctx.job_id,
                                                     bool(state.get("synthetic_only")), ctx.log)
        ctx.phase("整理重投影误差与覆盖", 0.9)

        def commit(current: dict[str, Any]) -> dict[str, Any] | None:
            now_hash = intrinsics_input_hash(current["settings"], current["intrinsics"]["images"])
            stale = now_hash != captured
            return self._store_intrinsics_result(current, ctx.job_id, computed,
                                                 captured, stale=stale)

        ctx.finish(computed["exit_code"], computed["summary"], commit)

    # ------------------------------------------------------------------
    # Fit job
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_residual_chart(reports: list[dict[str, Any]], threshold_mm: float,
                             path: Path) -> None:
        width, height = 1000, 430
        canvas = np.full((height, width, 3), 248, np.uint8)
        left, right, top, bottom = 70, width - 30, 50, height - 80
        values = [float(row["max_error_mm"]) for row in reports]
        y_max = max([threshold_mm * 1.35, max(values, default=0.0) * 1.2, 0.1])
        cv2.rectangle(canvas, (left, top), (right, bottom), (225, 228, 232), 1)
        threshold_y = int(bottom - threshold_mm / y_max * (bottom - top))
        for x in range(left, right + 1, 120):
            cv2.line(canvas, (x, top), (x, bottom), (238, 240, 242), 1)
        cv2.line(canvas, (left, threshold_y), (right, threshold_y), (40, 80, 220), 2)
        cv2.putText(canvas, f"limit {threshold_mm:.2f} mm", (right - 190, max(18, threshold_y - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (40, 80, 220), 1, cv2.LINE_AA)
        count = max(len(reports), 1)
        slot = (right - left) / count
        for index, row in enumerate(reports):
            x = int(left + (index + 0.5) * slot)
            bar_w = max(3, int(slot * 0.58))
            value = float(row["max_error_mm"])
            y = int(bottom - value / y_max * (bottom - top))
            color = ((190, 115, 20) if row.get("split") in ("val", "validation")
                     else (30, 150, 180))
            cv2.rectangle(canvas, (x - bar_w // 2, y), (x + bar_w // 2, bottom), color, -1)
            if index % max(1, count // 16) == 0:
                cv2.putText(canvas, str(row["corner_id"]), (x - 8, bottom + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, .38, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Per-point residual (mm)", (left, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    .75, (35, 35, 35), 2, cv2.LINE_AA)
        cv2.putText(canvas, "train", (left + 10, top + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (30, 150, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, "held-out val", (left + 90, top + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (190, 115, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, "corner_id", (right - 80, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    .42, (60, 60, 60), 1, cv2.LINE_AA)
        _write_png(path, canvas)

    def _compute_fit_artifact(self, state: dict[str, Any], arm: str, job_id: str,
                              log: Callable[[str], None]) -> dict[str, Any]:
        session_dir = self.store.session_dir(state["session_id"])
        intr_ok, intr_reason = intrinsics_status(state, session_dir)
        board_ok, board_reason = board_status(state, session_dir)
        if not intr_ok:
            raise CalibrationError("INTRINSICS_NOT_READY", f"内参不可用：{intr_reason}")
        if not board_ok:
            raise CalibrationError("BOARD_NOT_READY", f"固定棋盘不可用：{board_reason}")
        arm_state = state["arms"][arm]
        content_hashes = arm_content_hashes(arm_state)
        if (arm_state.get("metadata_hash") != content_hashes["metadata_hash"]
                or arm_state.get("points_hash") != content_hashes["points_hash"]):
            raise CalibrationError(
                "ARM_INPUT_HASH_MISMATCH",
                "保存的机械臂元数据或示教点哈希与实际内容不一致；请重新保存并重新拟合",
                status=409,
            )
        metadata = arm_state["metadata"]
        points = arm_state["points"]
        train_count = sum(point["split"] == "train" for point in points)
        val_count = sum(point["split"] == "val" for point in points)
        if train_count < 6 or val_count < 3:
            raise CalibrationError("TEACH_COUNTS", f"示教点至少需要 6 train / 3 val，当前 {train_count} / {val_count}")
        unit = metadata["unit"]
        scale = 0.001 if unit == "mm" else 1.0
        intr_record = state["intrinsics"]["result"]
        intrinsics = load_json(resolve_session_path(session_dir, intr_record["artifact_path"]))
        K = np.asarray(intrinsics["K"], dtype=float)
        D = np.asarray(intrinsics["distortion"], dtype=float).reshape(-1)
        cam = intrinsics
        board_confirmation = state["board"]["confirmation"]
        obs_rel = board_confirmation["observation_path"]
        obs_path = resolve_session_path(session_dir, obs_rel)
        obs, obj, corners, source = t.load_observation(obs_path, cam)
        ids = np.array([point["corner_id"] for point in points], dtype=int)
        measured = np.array([[point["x"], point["y"], point["z"]] for point in points],
                            dtype=float) * scale
        train_mask = np.array([point["split"] == "train" for point in points])
        log("使用 train 点进行平面刚体拟合；val 点只参与误差检查")
        try:
            B = t.rigid_fit(obj[ids[train_mask]], measured[train_mask])
        except ValueError as error:
            raise CalibrationError("TEACH_COVERAGE_INVALID", f"示教点覆盖或几何不合法：{error}") from error
        C, pnp_rms = c.pose_from_board(obj, corners, K, D)
        X = c.validate_transform(B @ c.inverse(C))
        reports = []
        for point, corner_id, reference in zip(points, ids, measured):
            predicted, board_coord = t.pixel_to_plane(corners[corner_id, 0], cam, K, D, X, B)
            rigid_error = float(np.linalg.norm(t.apply(B, obj[corner_id]) - reference) * 1000)
            pixel_error = float(np.linalg.norm(predicted - reference) * 1000)
            reports.append({
                **point,
                "reference_base_m": reference.tolist(),
                "predicted_base_m": predicted.tolist(),
                "predicted_base_mm": (predicted * 1000).tolist(),
                "board_point_m": board_coord.tolist(),
                "rigid_error_mm": rigid_error,
                "pixel_plane_error_mm": pixel_error,
                "max_error_mm": max(rigid_error, pixel_error),
            })
            log(f"角点 {corner_id} ({point['split']}): max {reports[-1]['max_error_mm']:.3f} mm")
        threshold_pnp = state["settings"]["thresholds"]["fit_max_pnp_rms_px"]
        threshold_error = state["settings"]["thresholds"]["fit_max_error_mm"]
        max_error = max(row["max_error_mm"] for row in reports)
        passed = bool(pnp_rms <= threshold_pnp and max_error <= threshold_error)
        val_reports = [row for row in reports if row["split"] == "val"]
        train_reports = [row for row in reports if row["split"] == "train"]
        input_hash = fit_input_hash(state, arm)
        out_dir = session_dir / "jobs" / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        axes_rel = f"jobs/{job_id}/board_axes.png"
        image = c.read_image(source)
        cv2.drawFrameAxes(image, K, D, cv2.Rodrigues(C[:3, :3])[0], C[:3, 3],
                          obs["board"]["square_m"] * 2)
        _write_png(session_dir / axes_rel, image)
        chart_rel = f"jobs/{job_id}/residuals.png"
        self._draw_residual_chart(reports, threshold_error, session_dir / chart_rel)
        artifact = {
            **c.base_result("fixed_board_extrinsics"),
            "synthetic_only": bool(state.get("synthetic_only")),
            "fit_input_hash": input_hash,
            "robot_id": metadata["robot_id"],
            "robot_model": "DOBOT Nova2",
            "base_frame": metadata["base_frame"],
            "tcp_name": metadata["tcp_name"],
            "input_unit": unit,
            "camera_id": state["settings"]["camera_id"],
            "image_size_wh": state["settings"]["image_size_wh"],
            "board": state["settings"]["board"],
            "intrinsics_sha256": intr_record["artifact_sha256"],
            "intrinsics_input_hash": intr_record["input_hash"],
            "observation_sha256": board_confirmation["observation_sha256"],
            "board_input_hash": board_confirmation["input_hash"],
            "arm_metadata_hash": content_hashes["metadata_hash"],
            "teach_points_hash": content_hashes["points_hash"],
            "T_base_board": B.tolist(),
            "T_camera_board": C.tolist(),
            "T_base_camera": X.tolist(),
            "board_normal_base": B[:3, 2].tolist(),
            "pnp_rms_px": float(pnp_rms),
            "max_error_mm": float(max_error),
            "train_max_error_mm": float(max(row["max_error_mm"] for row in train_reports)),
            "held_out_max_error_mm": float(max(row["max_error_mm"] for row in val_reports)),
            "thresholds": {"max_pnp_rms_px": threshold_pnp, "max_error_mm": threshold_error},
            "train_count": train_count,
            "val_count": val_count,
            "points": reports,
            "board_axes_path": axes_rel,
            "residual_chart_path": chart_rel,
            "validation_passed": passed,
            "validation_scope": "held-out taught points on this board plane; not independent robot metrology or off-plane accuracy",
        }
        artifact_rel = f"jobs/{job_id}/extrinsics.json"
        artifact_path = session_dir / artifact_rel
        atomic_write_json(artifact_path, artifact)
        return {
            "artifact": artifact,
            "artifact_path": artifact_rel,
            "artifact_sha256": sha256_file(artifact_path),
            "axes_path": axes_rel,
            "chart_path": chart_rel,
            "exit_code": 0 if passed else 2,
            "summary": {
                "pnp_rms_px": float(pnp_rms),
                "max_error_mm": float(max_error),
                "train_max_error_mm": artifact["train_max_error_mm"],
                "held_out_max_error_mm": artifact["held_out_max_error_mm"],
                "thresholds": artifact["thresholds"],
                "train_count": train_count,
                "val_count": val_count,
                "validation_passed": passed,
                "T_base_camera": X.tolist(),
                "T_base_board": B.tolist(),
                "T_camera_board": C.tolist(),
                "board_normal_base": B[:3, 2].tolist(),
            },
        }

    def _store_fit_result(self, state: dict[str, Any], arm: str, job_id: str,
                          computed: dict[str, Any], captured_hash: str,
                          *, stale: bool = False) -> dict[str, Any] | None:
        artifact = computed["artifact"]
        exit_code = 1 if stale else computed["exit_code"]
        status = "passed" if exit_code == 0 else ("failed_threshold" if exit_code == 2 else "invalidated")
        record = {
            "job_id": job_id,
            "status": status,
            "exit_code": exit_code,
            "validation_passed": bool(exit_code == 0 and artifact.get("validation_passed")),
            "usable": bool(exit_code == 0 and not stale),
            "stale": bool(stale),
            "stale_reason": "运行期间拟合输入已更改，结果仅保留为证据" if stale else None,
            "input_hash": captured_hash,
            "artifact_path": computed["artifact_path"],
            "artifact_sha256": computed["artifact_sha256"],
            "axes_path": computed["axes_path"],
            "chart_path": computed["chart_path"],
            "synthetic_only": bool(artifact.get("synthetic_only")),
            "summary": computed["summary"],
            "points": artifact["points"],
            "computed_utc": utc_now(),
        }
        arm_state = state["arms"][arm]
        arm_state["fit"] = record
        arm_state["fit_job_id"] = None
        mark_stale(arm_state["verification"].get("result"), "拟合结果已更新")
        add_history(arm_state, "fit", {"job_id": job_id, "exit_code": exit_code, "stale": stale})
        add_event(state, "success" if exit_code == 0 else "error", f"{self._arm_cn(arm)}拟合任务结束",
                  {"job_id": job_id, "exit_code": exit_code, "stale": stale})
        if stale:
            return {"exit_code": 1, "status": "invalidated",
                    "error": "运行期间示教输入、内参或棋盘已更改，拟合结果已作废"}
        return None

    def _run_fit(self, ctx: JobContext) -> None:
        state = self.store.read(ctx.session_id)
        arm = next(job["arm"] for job in state["jobs"] if job["id"] == ctx.job_id)
        captured = fit_input_hash(state, arm)
        ctx.set_input_hash(captured)
        ctx.phase("校验内参与固定棋盘", 0.05)
        if self._fit_hook:
            self._fit_hook()
        computed = self._compute_fit_artifact(state, arm, ctx.job_id, ctx.log)
        ctx.phase("生成坐标轴与逐点残差", 0.86)

        def commit(current: dict[str, Any]) -> dict[str, Any] | None:
            stale = fit_input_hash(current, arm) != captured
            return self._store_fit_result(current, arm, ctx.job_id, computed,
                                          captured, stale=stale)

        ctx.finish(computed["exit_code"], computed["summary"], commit)

    # ------------------------------------------------------------------
    # Independent verification job
    # ------------------------------------------------------------------

    def _compute_verification_artifact(self, state: dict[str, Any], arm: str,
                                       job_id: str) -> dict[str, Any]:
        session_dir = self.store.session_dir(state["session_id"])
        fit_ok, fit_reason = fit_status(state, session_dir, arm)
        if not fit_ok:
            raise CalibrationError("FIT_NOT_READY", f"拟合结果不可用：{fit_reason}")
        draft = state["arms"][arm]["verification"]
        rows = draft.get("rows", [])
        if len(rows) < 3:
            raise CalibrationError("VERIFY_POINTS_REQUIRED", "独立验收至少需要 3 个测量点")
        if not draft.get("method") or not draft.get("independence_confirmed"):
            raise CalibrationError("VERIFY_CONFIRMATION_REQUIRED", "测量方法与独立性确认不能为空")
        fit_record = state["arms"][arm]["fit"]
        fit_artifact = load_json(resolve_session_path(session_dir, fit_record["artifact_path"]))
        intr_artifact = load_json(resolve_session_path(
            session_dir, state["intrinsics"]["result"]["artifact_path"]))
        K = np.asarray(intr_artifact["K"], dtype=float)
        D = np.asarray(intr_artifact["distortion"], dtype=float).reshape(-1)
        X = c.validate_transform(fit_artifact["T_base_camera"])
        B = c.validate_transform(fit_artifact["T_base_board"])
        board = fit_artifact["board"]
        reports = []
        for index, row in enumerate(rows, start=1):
            predicted, board_point = t.pixel_to_plane(
                (row["u_px"], row["v_px"]), intr_artifact, K, D, X, B, row["board_z_m"],
            )
            reference = np.array([row["reference_x_m"], row["reference_y_m"],
                                  row["reference_z_m"]], dtype=float)
            error_mm = float(np.linalg.norm(predicted - reference) * 1000)
            inside = bool(0 <= board_point[0] <= (board["cols"] - 1) * board["square_m"]
                          and 0 <= board_point[1] <= (board["rows"] - 1) * board["square_m"])
            reports.append({
                **row,
                "predicted_base_m": predicted.tolist(),
                "predicted_base_mm": (predicted * 1000).tolist(),
                "board_point_m": board_point.tolist(),
                "error_mm": error_mm,
                "inside_board_inner_rectangle": inside,
                "source": "operator_provided_independent_reference",
            })
        errors = np.array([row["error_mm"] for row in reports], dtype=float)
        threshold = state["settings"]["thresholds"]["verify_max_error_mm"]
        rms = float(np.sqrt(np.mean(errors ** 2)))
        maximum = float(np.max(errors))
        passed = bool(maximum <= threshold)
        input_hash = verification_input_hash(state, arm)
        out_dir = session_dir / "jobs" / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        chart_rel = f"jobs/{job_id}/validation_residuals.png"
        self._draw_residual_chart([{**row, "corner_id": index, "max_error_mm": row["error_mm"]}
                                   for index, row in enumerate(reports, start=1)],
                                  threshold, session_dir / chart_rel)
        csv_rel = f"jobs/{job_id}/verification_points.csv"
        output = io.StringIO()
        fields = ["u_px", "v_px", "board_z_m", "reference_x_m", "reference_y_m", "reference_z_m"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        atomic_write_text(session_dir / csv_rel, output.getvalue())
        artifact = {
            **c.base_result("plane_point_validation"),
            "synthetic_only": bool(state.get("synthetic_only")),
            "verification_input_hash": input_hash,
            "fit_input_hash": fit_record["input_hash"],
            "fit_artifact_sha256": fit_record["artifact_sha256"],
            "robot_id": state["arms"][arm]["metadata"]["robot_id"],
            "base_frame": state["arms"][arm]["metadata"]["base_frame"],
            "tcp_name": state["arms"][arm]["metadata"]["tcp_name"],
            "unit": state["arms"][arm]["metadata"]["unit"],
            "camera_id": state["settings"]["camera_id"],
            "method": draft["method"],
            "independence_confirmed": True,
            "points": reports,
            "rms_mm": rms,
            "max_mm": maximum,
            "threshold_mm": threshold,
            "validation_passed": passed,
            "residual_chart_path": chart_rel,
            "verification_points_csv": csv_rel,
            "validation_scope": "operator-provided independent reference points in metres; distinct from fit held-out points; not a hardware or off-plane guarantee",
        }
        artifact_rel = f"jobs/{job_id}/validation.json"
        atomic_write_json(session_dir / artifact_rel, artifact)
        return {
            "artifact": artifact,
            "artifact_path": artifact_rel,
            "artifact_sha256": sha256_file(session_dir / artifact_rel),
            "chart_path": chart_rel,
            "csv_path": csv_rel,
            "exit_code": 0 if passed else 2,
            "summary": {
                "rms_mm": rms,
                "max_mm": maximum,
                "threshold_mm": threshold,
                "validation_passed": passed,
                "point_count": len(rows),
            },
        }

    def _store_verification_result(self, state: dict[str, Any], arm: str, job_id: str,
                                   computed: dict[str, Any], captured_hash: str,
                                   *, stale: bool = False) -> dict[str, Any] | None:
        artifact = computed["artifact"]
        exit_code = 1 if stale else computed["exit_code"]
        status = "passed" if exit_code == 0 else ("failed_threshold" if exit_code == 2 else "invalidated")
        verification = state["arms"][arm]["verification"]
        verification["result"] = {
            "job_id": job_id,
            "status": status,
            "exit_code": exit_code,
            "validation_passed": bool(exit_code == 0 and artifact.get("validation_passed")),
            "usable": bool(exit_code == 0 and not stale),
            "stale": bool(stale),
            "stale_reason": "运行期间验收输入或拟合结果已更改，结果仅保留为证据" if stale else None,
            "input_hash": captured_hash,
            "artifact_path": computed["artifact_path"],
            "artifact_sha256": computed["artifact_sha256"],
            "chart_path": computed["chart_path"],
            "csv_path": computed["csv_path"],
            "synthetic_only": bool(artifact.get("synthetic_only")),
            "summary": computed["summary"],
            "points": artifact["points"],
            "computed_utc": utc_now(),
        }
        verification["input_hash"] = captured_hash
        state["arms"][arm]["verification_job_id"] = None
        add_history(verification, "result", {"job_id": job_id, "exit_code": exit_code,
                                             "stale": stale})
        add_event(state, "success" if exit_code == 0 else "error",
                  f"{self._arm_cn(arm)}独立验收结束",
                  {"job_id": job_id, "exit_code": exit_code, "stale": stale})
        if stale:
            return {"exit_code": 1, "status": "invalidated",
                    "error": "运行期间拟合或验收输入已更改，独立验收已作废"}
        return None

    def _run_verification(self, ctx: JobContext) -> None:
        state = self.store.read(ctx.session_id)
        arm = next(job["arm"] for job in state["jobs"] if job["id"] == ctx.job_id)
        captured = verification_input_hash(state, arm)
        ctx.set_input_hash(captured)
        ctx.phase("读取独立参考点", 0.08)
        ctx.log("独立验收只使用操作者提供的参考 XYZ；不复用拟合残差标签")
        computed = self._compute_verification_artifact(state, arm, ctx.job_id)
        ctx.phase("汇总 RMS、最大值与逐点误差", 0.9)

        def commit(current: dict[str, Any]) -> dict[str, Any] | None:
            stale = verification_input_hash(current, arm) != captured
            return self._store_verification_result(current, arm, ctx.job_id, computed,
                                                   captured, stale=stale)

        ctx.finish(computed["exit_code"], computed["summary"], commit)

    # ------------------------------------------------------------------
    # Pixel query, depth registration, export
    # ------------------------------------------------------------------

    def confirm_depth(self, session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        confirmed = body.get("confirmed") is True
        note = clean_text(body.get("registration_note", ""), "光学-Z 注册说明",
                          maximum=500, allow_empty=not confirmed)
        with self.store.edit(session_id) as state:
            if confirmed:
                ok, reason = intrinsics_status(state, self.store.session_dir(session_id))
                if not ok:
                    raise CalibrationError("INTRINSICS_NOT_READY", f"内参不可用：{reason}", status=409)
                if len(note) < 10:
                    raise CalibrationError("DEPTH_NOTE_REQUIRED",
                                           "请说明 RGB 与深度已在同一光学坐标系完成注册")
                state["depth_registration"] = {
                    "confirmed": True,
                    "registration_note": note,
                    "intrinsics_input_hash": intrinsics_input_hash(
                        state["settings"], state["intrinsics"]["images"]),
                    "confirmed_utc": utc_now(),
                }
            else:
                state["depth_registration"] = {
                    "confirmed": False,
                    "registration_note": note,
                    "intrinsics_input_hash": None,
                    "confirmed_utc": None,
                }
            add_event(state, "warning" if not confirmed else "info",
                      "深度模式已" + ("启用" if confirmed else "禁用"))
            return state["depth_registration"]

    def query(self, session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        state = self.store.read(session_id)
        width, height = state["settings"]["image_size_wh"]
        u = float(_finite_query(body.get("u_px"), "像素 u"))
        v = float(_finite_query(body.get("v_px"), "像素 v"))
        if not 0 <= u < width or not 0 <= v < height:
            raise CalibrationError("PIXEL_OUT_OF_BOUNDS", "像素超出原始图像")
        has_board_z = "board_z_m" in body
        has_depth = "depth_z_m" in body
        if has_board_z == has_depth:
            raise CalibrationError("QUERY_MODE_REQUIRED", "必须且只能选择板坐标 z 或已注册深度")
        allow_extrapolation = body.get("allow_extrapolation", False)
        if not isinstance(allow_extrapolation, bool):
            raise CalibrationError("INVALID_EXTRAPOLATION_FLAG", "外推选项必须是布尔值")
        if has_depth:
            depth_ok, depth_reason = depth_status(state)
            if not depth_ok:
                raise CalibrationError("DEPTH_DISABLED", depth_reason, status=409)
            depth_z = float(_finite_query(body.get("depth_z_m"), "光学-Z 深度", positive=True))
            board_z = None
        else:
            depth_z = None
            board_z = float(_finite_query(body.get("board_z_m"), "板坐标 z（米）"))
        session_dir = self.store.session_dir(session_id)
        arms_result = {}
        for arm in ARMS:
            ok, reason = fit_status(state, session_dir, arm)
            if not ok:
                arms_result[arm] = {"ok": False, "code": "FIT_NOT_READY", "error": reason}
                continue
            fit_record = state["arms"][arm]["fit"]
            fit_artifact = load_json(resolve_session_path(session_dir, fit_record["artifact_path"]))
            intr_artifact = load_json(resolve_session_path(
                session_dir, state["intrinsics"]["result"]["artifact_path"]))
            K = np.asarray(intr_artifact["K"], dtype=float)
            D = np.asarray(intr_artifact["distortion"], dtype=float).reshape(-1)
            X = c.validate_transform(fit_artifact["T_base_camera"])
            B = c.validate_transform(fit_artifact["T_base_board"])
            try:
                if has_depth:
                    predicted = c.point_in_base(u, v, depth_z, intr_artifact, K, D,
                                                {"mode": "eye-to-hand", "T_base_camera": X})
                    board_point = t.apply(c.inverse(B), predicted)
                    mode = "registered_optical_Z_depth"
                else:
                    predicted, board_point = t.pixel_to_plane(
                        (u, v), intr_artifact, K, D, X, B, board_z,
                    )
                    mode = "intersection_with_board_coordinate_plane"
            except ValueError as error:
                arms_result[arm] = {"ok": False, "code": "QUERY_GEOMETRY_FAILED",
                                    "error": str(error)}
                continue
            board = fit_artifact["board"]
            inside = bool(0 <= board_point[0] <= (board["cols"] - 1) * board["square_m"]
                          and 0 <= board_point[1] <= (board["rows"] - 1) * board["square_m"])
            if not inside and not allow_extrapolation:
                arms_result[arm] = {
                    "ok": False,
                    "code": "OUTSIDE_BOARD",
                    "error": "像素落在棋盘内角点矩形外；如确需外推，请显式勾选并另行验收",
                }
                continue
            arms_result[arm] = {
                "ok": True,
                "robot_id": state["arms"][arm]["metadata"]["robot_id"],
                "base_frame": state["arms"][arm]["metadata"]["base_frame"],
                "tcp_name": state["arms"][arm]["metadata"]["tcp_name"],
                "unit": state["arms"][arm]["metadata"]["unit"],
                "mode": mode,
                "point_base_m": predicted.tolist(),
                "point_base_mm": (predicted * 1000).tolist(),
                "point_board_m": board_point.tolist(),
                "inside_board_inner_rectangle": inside,
                "extrapolated": bool(not inside and allow_extrapolation),
            }
        return {
            "u_px": u,
            "v_px": v,
            "image_size_wh": [width, height],
            "board_z_m": board_z,
            "depth_z_m": depth_z,
            "allow_extrapolation": allow_extrapolation,
            "synthetic_only": bool(state.get("synthetic_only")),
            "arms": arms_result,
        }

    @staticmethod
    def _export_dependency_snapshot(state: dict[str, Any], arms: list[str],
                                    session_dir: Path) -> dict[str, Any]:
        def pick(record: dict[str, Any] | None, fields: tuple[str, ...]) -> dict[str, Any] | None:
            if not isinstance(record, dict):
                return None
            return {field: record.get(field) for field in fields}

        intrinsics = state.get("intrinsics", {})
        intr_record = intrinsics.get("result")
        intr_ok, intr_reason = intrinsics_status(state, session_dir)
        board = state.get("board", {})
        confirmation = board.get("confirmation")
        board_ok, board_reason = board_status(state, session_dir)
        depth_ok, depth_reason = depth_status(state)
        arm_snapshot: dict[str, Any] = {}
        for arm in arms:
            arm_state = state["arms"][arm]
            fit_ok, fit_reason = fit_status(state, session_dir, arm)
            verify_ok, verify_reason = verification_status(state, session_dir, arm)
            verification = arm_state["verification"]
            arm_snapshot[arm] = {
                "metadata": arm_state.get("metadata"),
                "metadata_hash": arm_state.get("metadata_hash"),
                "points": arm_state.get("points"),
                "points_hash": arm_state.get("points_hash"),
                "fit": pick(arm_state.get("fit"), (
                    "input_hash", "artifact_sha256", "status", "exit_code",
                    "validation_passed", "usable", "stale", "stale_reason",
                )),
                "fit_gate": {"ok": fit_ok, "reason": fit_reason},
                "verification_draft": {
                    "rows": verification.get("rows", []),
                    "method": verification.get("method", ""),
                    "independence_confirmed": bool(verification.get("independence_confirmed")),
                },
                "verification": pick(verification.get("result"), (
                    "input_hash", "artifact_sha256", "status", "exit_code",
                    "validation_passed", "stale", "stale_reason",
                )),
                "verification_gate": {"ok": verify_ok, "reason": verify_reason},
            }
        return {
            "synthetic_only": bool(state.get("synthetic_only")),
            "settings": {
                "camera_id": state["settings"].get("camera_id"),
                "image_size_wh": state["settings"].get("image_size_wh"),
                "board": state["settings"].get("board"),
                "thresholds": state["settings"].get("thresholds"),
            },
            "intrinsics": {
                "images": [{key: image.get(key) for key in (
                    "id", "split", "original_path", "sha256",
                )} for image in intrinsics.get("images", [])],
                "result": pick(intr_record, (
                    "input_hash", "artifact_sha256", "status", "exit_code",
                    "validation_passed", "stale", "stale_reason",
                )),
                "gate": {"ok": intr_ok, "reason": intr_reason},
            },
            "board": {
                "revision": board.get("revision"),
                "state": board.get("state"),
                "source_path": board.get("source_path"),
                "source_sha256": board.get("source_sha256"),
                "confirmation": pick(confirmation, (
                    "input_hash", "observation_sha256", "corner_order",
                    "immutable_revision", "fixed_confirmed", "stale", "stale_reason",
                )),
                "gate": {"ok": board_ok, "reason": board_reason},
            },
            "depth_registration": pick(state.get("depth_registration"), (
                "confirmed", "registration_note", "intrinsics_input_hash",
            )),
            "depth_gate": {"ok": depth_ok, "reason": depth_reason},
            "arms": arm_snapshot,
        }

    def create_export(self, session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        requested = body.get("arms", list(ARMS))
        if not isinstance(requested, list) or not requested:
            raise CalibrationError("EXPORT_ARMS_REQUIRED", "请选择至少一只手臂")
        if any(arm not in ARMS for arm in requested):
            raise CalibrationError("INVALID_ARM", "导出只支持 left/right")
        requested = [arm for arm in ARMS if arm in requested]
        state = self.store.read(session_id)
        session_dir = self.store.session_dir(session_id)
        dependency_snapshot = self._export_dependency_snapshot(state, requested, session_dir)
        for arm in requested:
            ok, reason = verification_status(state, session_dir, arm)
            if not ok:
                raise CalibrationError("EXPORT_NOT_READY",
                                       f"{self._arm_cn(arm)}不可导出：{reason}", status=409)
        payloads: dict[str, bytes] = {}
        arms_report: dict[str, Any] = {}
        intrinsics_record = state["intrinsics"]["result"]
        intrinsics_path = resolve_session_path(session_dir, intrinsics_record["artifact_path"])
        intrinsics_bytes = intrinsics_path.read_bytes()
        if sha256_bytes(intrinsics_bytes) != intrinsics_record.get("artifact_sha256"):
            raise CalibrationError("EXPORT_INTRINSICS_CHANGED", "导出期间内参产物哈希发生变化", status=409)
        intrinsics = load_json(intrinsics_path)
        payloads["camera/intrinsics.json"] = intrinsics_bytes
        board_confirmation = state["board"]["confirmation"]
        observation_path = resolve_session_path(session_dir, board_confirmation["observation_path"])
        observation_bytes = observation_path.read_bytes()
        if sha256_bytes(observation_bytes) != board_confirmation.get("observation_sha256"):
            raise CalibrationError("EXPORT_BOARD_CHANGED", "导出期间固定棋盘观测哈希发生变化", status=409)
        observation = load_json(observation_path)
        payloads["board/observation.json"] = observation_bytes
        board_provenance = {
            "schema_version": 1,
            "kind": "calibration_board_confirmation",
            "session_id": session_id,
            "synthetic_only": bool(state.get("synthetic_only")),
            "board_revision": state["board"].get("revision"),
            "immutable_revision": board_confirmation.get("immutable_revision"),
            "corner_order": board_confirmation.get("corner_order"),
            "fixed_confirmed": board_confirmation.get("fixed_confirmed"),
            "confirmed_utc": board_confirmation.get("confirmed_utc"),
            "source_file": "board/source.png",
            "source_sha256": state["board"].get("source_sha256"),
            "observation_file": "board/observation.json",
            "observation_sha256": board_confirmation.get("observation_sha256"),
            "board_input_hash": board_confirmation.get("input_hash"),
            "corner_count": len(observation.get("corners_uv", [])),
            "transform_convention": "T_A_B maps column coordinates from frame B to frame A; all translations in metres",
        }
        payloads["board/confirmation.json"] = (
            json.dumps(board_provenance, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8")
        for arm in requested:
            arm_state = state["arms"][arm]
            content_hashes = arm_content_hashes(arm_state)
            fit = arm_state["fit"]
            verification = arm_state["verification"]["result"]
            fit_artifact = load_json(resolve_session_path(session_dir, fit["artifact_path"]))
            validation = load_json(resolve_session_path(session_dir, verification["artifact_path"]))
            prefix = f"arms/{arm}"
            payloads[f"{prefix}/extrinsics.json"] = resolve_session_path(
                session_dir, fit["artifact_path"]).read_bytes()
            payloads[f"{prefix}/verification.json"] = resolve_session_path(
                session_dir, verification["artifact_path"]).read_bytes()
            payloads[f"{prefix}/teach_points.csv"] = self.export_teach_csv(session_id, arm).encode("utf-8-sig")
            payloads[f"{prefix}/verification_points.csv"] = self.export_verify_csv(
                session_id, arm).encode("utf-8-sig")
            for source_key, target in (("axes_path", "board_axes.png"),
                                       ("chart_path", "residuals.png")):
                if fit.get(source_key):
                    payloads[f"{prefix}/{target}"] = resolve_session_path(
                        session_dir, fit[source_key]).read_bytes()
            if verification.get("chart_path"):
                payloads[f"{prefix}/verification_residuals.png"] = resolve_session_path(
                    session_dir, verification["chart_path"]).read_bytes()
            arms_report[arm] = {
                "metadata": arm_state["metadata"],
                "metadata_hash": content_hashes["metadata_hash"],
                "teach_points_hash": content_hashes["points_hash"],
                "fit_input_hash": fit["input_hash"],
                "fit_artifact_sha256": fit["artifact_sha256"],
                "verification_input_hash": verification["input_hash"],
                "verification_artifact_sha256": verification["artifact_sha256"],
                "intrinsics_input_hash": intrinsics_record["input_hash"],
                "intrinsics_artifact_sha256": intrinsics_record["artifact_sha256"],
                "board_input_hash": board_confirmation["input_hash"],
                "board_observation_sha256": board_confirmation["observation_sha256"],
                "board_corner_order": board_confirmation["corner_order"],
                "fit_metrics": fit["summary"],
                "verification_metrics": verification["summary"],
                "T_base_camera": fit_artifact["T_base_camera"],
                "T_base_board": fit_artifact["T_base_board"],
                "T_camera_board": fit_artifact["T_camera_board"],
                "board_normal_base": fit_artifact["board_normal_base"],
                "fit_scope": fit_artifact["validation_scope"],
                "verification_scope": validation["validation_scope"],
                "verification_method": validation["method"],
                "files": sorted(path for path in payloads if path.startswith(prefix + "/")),
            }
        board_order = state["board"]["confirmation"]["corner_order"]
        for source_key, target in (("source_path", "board/source.png"),
                                   (None, "board/numbered_corners.png")):
            relative = state["board"]["previews"][board_order] if source_key is None else state["board"][source_key]
            if relative:
                payloads[target] = resolve_session_path(session_dir, relative).read_bytes()
        payloads["session_settings.json"] = (json.dumps({
            "camera_id": state["settings"]["camera_id"],
            "image_size_wh": state["settings"]["image_size_wh"],
            "board": state["settings"]["board"],
            "thresholds": state["settings"]["thresholds"],
            "mode": state["mode"],
            "synthetic_only": state["synthetic_only"],
        }, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        warnings = [
            "离线导出成功不等于真实相机内参、TCP 或真机精度验收。",
            "拟合 held-out 点与独立验收点是不同证据层；不要混用标签。",
            "不得自动写入现有示教位姿、scene_config 或 chemistry profile。",
        ]
        if state.get("synthetic_only"):
            warnings.insert(0, "本包来自 synthetic_only 演示，严禁用于真机控制或现场验收。")
        report = {
            "schema_version": 1,
            "kind": "calibration_bundle_report",
            "created_utc": utc_now(),
            "session_id": session_id,
            "display_name": state["display_name"],
            "mode": state["mode"],
            "synthetic_only": bool(state.get("synthetic_only")),
            "usable_for_real_control": bool(not state.get("synthetic_only")),
            "hardware_validation_claimed": False,
            "camera_id": state["settings"]["camera_id"],
            "image_size_wh": state["settings"]["image_size_wh"],
            "board": state["settings"]["board"],
            "board_corner_order": board_order,
            "board_provenance": board_provenance,
            "thresholds": state["settings"]["thresholds"],
            "intrinsics": {
                "file": "camera/intrinsics.json",
                "artifact_sha256": intrinsics_record["artifact_sha256"],
                "input_hash": intrinsics_record["input_hash"],
                "camera_id": intrinsics.get("camera_id"),
                "image_size_wh": intrinsics.get("image_size_wh"),
                "board": intrinsics.get("board"),
                "model": intrinsics.get("model"),
                "K": intrinsics.get("K"),
                "distortion": intrinsics.get("distortion"),
                "validation_scope": intrinsics.get("validation_scope"),
            },
            "transform_convention": "T_A_B maps column coordinates from frame B to frame A; all translations in metres",
            "arms": arms_report,
            "warnings": warnings,
        }
        payloads["report.json"] = (json.dumps(report, indent=2, ensure_ascii=False,
                                              allow_nan=False) + "\n").encode("utf-8")
        manifest = {
            "schema_version": 1,
            "kind": "calibration_export_manifest",
            "session_id": session_id,
            "synthetic_only": bool(state.get("synthetic_only")),
            "files": {name: sha256_bytes(data) for name, data in sorted(payloads.items())},
        }
        payloads["manifest.json"] = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(payloads.items()):
                archive.writestr(name, data)
        data = buffer.getvalue()
        digest = sha256_bytes(data)
        filename = f"calibration-{state['mode']}-{uuid.uuid4().hex[:10]}.zip"
        relative = f"exports/{filename}"
        with self.store.edit(session_id) as current:
            current_dependencies = self._export_dependency_snapshot(
                current, requested, self.store.session_dir(session_id))
            if canonical_json(current_dependencies) != canonical_json(dependency_snapshot):
                raise CalibrationError("EXPORT_RACE", "导出期间标定依赖已更改，请重新生成报告包", status=409)
            for arm in requested:
                ok, reason = verification_status(current, self.store.session_dir(session_id), arm)
                if not ok:
                    raise CalibrationError("EXPORT_RACE",
                                           f"导出期间输入改变：{self._arm_cn(arm)} {reason}", status=409)
            destination = resolve_session_path(session_dir, relative)
            atomic_write_bytes(destination, data)
            record = {
                "filename": filename,
                "bundle_path": relative,
                "sha256": digest,
                "size_bytes": len(data),
                "arms": requested,
                "synthetic_only": bool(current.get("synthetic_only")),
                "dependencies": dependency_snapshot,
                "created_utc": utc_now(),
            }
            current.setdefault("exports", []).append(record)
            current["exports"] = current["exports"][-30:]
            add_history(current, "export", {"filename": filename, "sha256": digest})
            add_event(current, "success", "标定报告包已生成", {"filename": filename,
                                                                "sha256": digest})
            return record

    # ------------------------------------------------------------------
    # Synthetic-only end-to-end demo
    # ------------------------------------------------------------------

    def _run_demo(self, ctx: JobContext) -> None:
        state = self.store.read(ctx.session_id)
        if not state.get("synthetic_only"):
            raise CalibrationError("DEMO_REQUIRES_SYNTHETIC", "演示任务必须标记为 synthetic_only")
        settings = state["settings"]
        width, height = settings["image_size_wh"]
        board = settings["board"]
        cols, rows, square_m = board["cols"], board["rows"], board["square_m"]
        rng = np.random.default_rng(20261020)
        K = np.array([[1000., 0, width / 2], [0, 990., height / 2], [0, 0, 1.]])
        ctx.phase("生成合成内参图像", 0.04, "合成图像，不连接相机")
        for index in range(24):
            for _attempt in range(80):
                rotation = cv2.Rodrigues(rng.uniform([-.6, -.6, -.4], [.6, .6, .4]))[0]
                centre = rng.uniform([-.12, -.08, .65], [.12, .08, .95])
                C = c.transform(rotation, centre - rotation @ np.array(
                    [(cols - 1) * square_m / 2, (rows - 1) * square_m / 2, 0]))
                try:
                    image = render(C, K, cols=cols, rows=rows, s=square_m, size=(width, height))
                    break
                except ValueError:
                    continue
            else:
                raise CalibrationError("DEMO_RENDER_FAILED", "无法生成完整的合成棋盘图像")
            split = "train" if index < 18 else "val"
            self.add_intrinsics_image(ctx.session_id, _encode_png(image), "image/png",
                                      split, f"synthetic_view_{index:02d}.png")
            if index % 4 == 3:
                ctx.phase("生成合成内参图像", 0.04 + 0.20 * (index + 1) / 24)
        state = self.store.read(ctx.session_id)
        captured = intrinsics_input_hash(state["settings"], state["intrinsics"]["images"])
        ctx.phase("拟合并检查合成内参", 0.28)
        intrinsics = self._compute_intrinsics_artifact(state, ctx.job_id, True, ctx.log)
        if intrinsics["exit_code"] != 0:
            raise CalibrationError("DEMO_INTRINSICS_FAILED", "合成内参未达到演示阈值")
        with self.store.edit(ctx.session_id) as current:
            self._store_intrinsics_result(current, ctx.job_id, intrinsics, captured, stale=False)
        ctx.phase("生成并冻结合成固定棋盘", 0.40)
        Cfix = make_T([.16, -.12, .07], [-.14, -.10, .78])
        fixed_image = render(Cfix, K, cols=cols, rows=rows, s=square_m, size=(width, height))
        self.upload_board_image(ctx.session_id, _encode_png(fixed_image), "image/png",
                                "synthetic_fixed_board.png")
        self.detect_board(ctx.session_id)
        board_state = self.store.read(ctx.session_id)["board"]
        origin = cv2.projectPoints(np.zeros((1, 3)), cv2.Rodrigues(Cfix[:3, :3])[0],
                                   Cfix[:3, 3], K, np.zeros(5))[0].reshape(2)
        raw = np.asarray(board_state["raw_corners_uv"], dtype=float)
        order = "normal" if np.linalg.norm(raw[0] - origin) < np.linalg.norm(raw[-1] - origin) else "reverse"
        self.confirm_board(ctx.session_id, {"corner_order": order, "fixed_confirmed": True})
        rotation_left = cv2.Rodrigues(np.array([0., 0., .3]))[0] @ np.diag([1., -1., -1.])
        rotation_right = cv2.Rodrigues(np.array([0., 0., -.22]))[0] @ np.diag([1., -1., -1.])
        truth = {
            "left": c.transform(rotation_left, [.20, .15, .03]),
            "right": c.transform(rotation_right, [-.35, .25, .05]),
        }
        obj = c.board_points(cols, rows, square_m)
        teach_ids = {
            "train": [0, 5, 10, 14, 33, 43, 49, 69, 77, 82, 87],
            "val": [21, 28, 61, 73],
        }
        for arm in ARMS:
            meta = {
                "robot_id": f"SIM_NOVA2_{arm.upper()}",
                "base_frame": "SIM_BASE_" + arm.upper(),
                "tcp_name": "SIM_TIP",
                "unit": "mm",
                "controller_host": "",
                "controller_port": 29999,
            }
            self.update_arm_metadata(ctx.session_id, arm, meta)
            points = []
            for split in ("train", "val"):
                for corner_id in teach_ids[split]:
                    xyz = t.apply(truth[arm], obj[corner_id]) + rng.normal(0, .00015, 3)
                    points.append({"corner_id": corner_id, "split": split,
                                   "x": xyz[0] * 1000, "y": xyz[1] * 1000, "z": xyz[2] * 1000})
            self.update_teach_points(ctx.session_id, arm, {"points": points})
        for arm_index, arm in enumerate(ARMS):
            ctx.phase(f"拟合合成{self._arm_cn(arm)}", 0.52 + arm_index * 0.08)
            state = self.store.read(ctx.session_id)
            captured_fit = fit_input_hash(state, arm)
            fit = self._compute_fit_artifact(state, arm, ctx.job_id + f"-{arm}", ctx.log)
            with self.store.edit(ctx.session_id) as current:
                override = self._store_fit_result(current, arm, ctx.job_id + f"-{arm}",
                                                  fit, captured_fit, stale=False)
            if fit["exit_code"] != 0 or override:
                raise CalibrationError("DEMO_FIT_FAILED", f"合成{self._arm_cn(arm)}拟合未通过")
        for arm_index, arm in enumerate(ARMS):
            ctx.phase(f"生成合成独立验收：{self._arm_cn(arm)}", 0.72 + arm_index * 0.06)
            measured_points = np.array([[.04 + .05 * (arm_index + 1), .05, 0.0],
                                        [.24, .06 + .02 * arm_index, 0.0],
                                        [.22, .18, 0.0],
                                        [.10, .14, 0.0]])
            uv = cv2.projectPoints(measured_points, cv2.Rodrigues(Cfix[:3, :3])[0],
                                   Cfix[:3, 3], K, np.zeros(5))[0].reshape(-1, 2)
            rows = []
            for pixel, board_point in zip(uv, measured_points):
                reference = t.apply(truth[arm], board_point) + rng.normal(0, .00015, 3)
                rows.append({
                    "u_px": float(pixel[0]),
                    "v_px": float(pixel[1]),
                    "board_z_m": 0.0,
                    "reference_x_m": float(reference[0]),
                    "reference_y_m": float(reference[1]),
                    "reference_z_m": float(reference[2]),
                })
            self.update_verification_draft(ctx.session_id, arm, {
                "rows": rows,
                "method": "独立合成真值点（不是拟合残差回填）",
                "independence_confirmed": True,
            })
            state = self.store.read(ctx.session_id)
            captured_verify = verification_input_hash(state, arm)
            validation = self._compute_verification_artifact(state, arm, ctx.job_id + f"-{arm}")
            with self.store.edit(ctx.session_id) as current:
                override = self._store_verification_result(current, arm, ctx.job_id + f"-{arm}",
                                                           validation, captured_verify, stale=False)
            if validation["exit_code"] != 0 or override:
                raise CalibrationError("DEMO_VERIFY_FAILED", f"合成{self._arm_cn(arm)}独立验收未通过")
        ctx.phase("整理演示结果", 0.95)
        final = self.store.read(ctx.session_id)
        summary = {
            "synthetic_only": True,
            "intrinsics_rms_px": final["intrinsics"]["result"]["summary"]["train_rms_px"],
            "arms": {
                arm: {
                    "fit_passed": final["arms"][arm]["fit"]["validation_passed"],
                    "verification_passed": final["arms"][arm]["verification"]["result"]["validation_passed"],
                }
                for arm in ARMS
            },
        }
        ctx.finish(0, summary)


def _finite_query(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError("INVALID_NUMBER", f"{label} 必须是有限数字")
    number = float(value)
    if not math.isfinite(number):
        raise CalibrationError("NONFINITE_NUMBER", f"{label} 不允许 NaN/Inf")
    if positive and number <= 0:
        raise CalibrationError("INVALID_NUMBER", f"{label} 必须大于 0")
    return number
