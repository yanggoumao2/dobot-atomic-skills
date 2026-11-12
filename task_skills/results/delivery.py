"""Build and deliver durable provisional experiment result bundles.

The 3.3 endpoint below is a local mock contract for integration work, not an
official receiver API. Callers should keep the captured queue directory with
their execution data so pending deliveries survive process restarts.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import mimetypes
import os
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import (
    CATEGORY_NAMES,
    ENDPOINT_PATH,
    MAX_ATTACHMENT_BYTES,
    MAX_REQUEST_BYTES,
    SCHEMA_VERSION,
    STATUS_ALIASES,
    canonical_json,
    content_digest,
    normalize_status,
    validate_bundle,
)

REQUEST_TIMEOUT_SECONDS = 5.0
_IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
_AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
_LOG_SUFFIXES = {".log", ".txt"}


class DeliveryError(RuntimeError):
    """A request failed or the receiver returned an invalid acknowledgement."""


def _json_copy(value: Any) -> Any:
    """Return a detached JSON value, rejecting non-JSON and non-finite values."""
    return json.loads(canonical_json(value).decode("utf-8"))


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _present(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _pick(source: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: _json_copy(source[key]) for key in keys if key in source}


def _nested_value(source: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in source and _present(source[key]):
            return source[key]
    return None


def _identifier(source: dict[str, Any], nested_key: str, *keys: str) -> str:
    value = _nested_value(source, *keys)
    if value is None:
        nested = source.get(nested_key)
        if isinstance(nested, dict):
            value = _nested_value(nested, *keys)
    if value is None:
        return "unknown"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))[:256]
    return str(value)[:256] or "unknown"


def _all_category_overrides(*sources: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in sources:
        for key in ("data_categories", "category_statuses"):
            value = source.get(key)
            if isinstance(value, dict):
                merged.update(value)
    return merged


def _category_record(
    name: str,
    data: Any,
    overrides: dict[str, Any],
    derived_status: str | None = None,
) -> dict[str, Any]:
    explicit = overrides.get(name)
    if isinstance(explicit, dict):
        raw_status = explicit.get("status")
        status = normalize_status(raw_status)
        if "data" in explicit:
            category_data = _json_copy(explicit["data"])
        else:
            category_data = {
                key: _json_copy(value)
                for key, value in explicit.items()
                if key not in {"status", "missing_reason", "reason", "source_status"}
            }
        reason = explicit.get("missing_reason", explicit.get("reason"))
        if reason is None and status not in {"available", "partial"}:
            reason = f"Source declared this category {raw_status or status}."
        result = {
            "status": status,
            "data": category_data,
            "missing_reason": _json_copy(reason),
            "status_source": "provided",
        }
        if raw_status is not None and normalize_status(raw_status) == "unknown" and raw_status != "unknown":
            result["source_status"] = str(raw_status)
        return result

    if isinstance(explicit, str):
        status = normalize_status(explicit)
        recognized = explicit.strip() in STATUS_ALIASES or explicit.strip().lower() in STATUS_ALIASES
        reason = None if status in {"available", "partial"} else f"Source declared this category {explicit}."
        result = {
            "status": status,
            "data": None,
            "missing_reason": reason if recognized else f"Unrecognized supplied category status: {explicit}.",
            "status_source": "provided_status",
        }
        if not recognized:
            result["source_status"] = explicit
        return result

    if explicit is not None:
        return {
            "status": "available" if _present(explicit) else "unknown",
            "data": _json_copy(explicit),
            "missing_reason": None if _present(explicit) else "No source data or collection status was provided.",
            "status_source": "provided_payload",
        }
    if data is not None:
        status = derived_status or ("available" if _present(data) else "unknown")
        return {
            "status": status,
            "data": _json_copy(data),
            "missing_reason": None if status in {"available", "partial"} else "Source data was omitted or could not be exported.",
            "status_source": "source_payload",
        }
    return {
        "status": "unknown",
        "data": None,
        "missing_reason": "No source data or collection status was provided.",
        "status_source": "not_supplied",
    }


def _source_kind(mode: Any, package: dict[str, Any], result: dict[str, Any], override: str | None) -> str:
    """Choose a conservative provenance label; simulation mode cannot claim real."""
    candidates: list[Any] = [override]
    for source in (result, package):
        provenance = source.get("provenance")
        if isinstance(provenance, dict):
            candidates.append(provenance.get("kind"))
            if provenance.get("synthetic") is True:
                candidates.append("synthetic")
        else:
            candidates.append(provenance)
        for key in ("source_kind", "data_source", "source", "mode"):
            candidates.append(source.get(key))
        if source.get("synthetic") is True or source.get("is_synthetic") is True:
            candidates.append("synthetic")

    normalized = [str(item).strip().lower() for item in candidates if item is not None]
    if any(item in {"synthetic", "fixture", "generated", "synthetic_fixture"} for item in normalized):
        return "synthetic"
    normalized_mode = str(mode).strip().lower() if mode is not None else ""
    # Runtime mode takes precedence over source text that may describe another
    # layer (for example a result payload with source="real" during simulation).
    if normalized_mode in {"synthetic", "fixture", "generated", "synthetic_fixture"}:
        return "synthetic"
    if normalized_mode in {"mock", "sim", "simulation", "simulated"}:
        return "mock"
    if normalized_mode in {"real", "hardware", "robot", "physical"}:
        return "real"
    if any(item in {"mock", "sim", "simulation", "simulated"} for item in normalized):
        return "mock"
    if any(item in {"real", "hardware", "robot", "physical"} for item in normalized):
        return "real"
    return "unknown"


def _add_attachment(
    attachment_list: list[dict[str, Any]],
    source_documents: dict[str, Any],
    missing: list[dict[str, Any]],
    *,
    attachment_id: str,
    name: str,
    content_type: str,
    raw: bytes,
    source: str,
    total_so_far: int,
) -> int:
    digest = hashlib.sha256(raw).hexdigest()
    if total_so_far + len(raw) > MAX_ATTACHMENT_BYTES:
        missing.append(
            {
                "name": name,
                "source": source,
                "size_bytes": len(raw),
                "sha256": digest,
                "status": "export_restricted",
                "reason": "The 16 MiB aggregate raw-attachment limit was reached.",
            }
        )
        return total_so_far
    attachment_list.append(
        {
            "attachment_id": attachment_id,
            "name": name,
            "content_type": content_type,
            "size_bytes": len(raw),
            "sha256": digest,
            "source": source,
            "data_base64": base64.b64encode(raw).decode("ascii"),
        }
    )
    if source in {"package", "result", "events"}:
        source_documents[source] = {"attachment_id": attachment_id, "sha256": digest, "size_bytes": len(raw)}
    return total_so_far + len(raw)


def _execution_files(execution_dir: str | os.PathLike[str] | None) -> list[tuple[str, Path]]:
    if execution_dir is None:
        return []
    root = Path(execution_dir).absolute()
    if not root.exists() or not root.is_dir():
        return []
    try:
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError):
        return []
    files: list[tuple[str, Path]] = []
    stack = [root]
    # 0x400 is FILE_ATTRIBUTE_REPARSE_POINT on Windows; use the value directly
    # as a fallback for Python builds that do not export the stat constant.
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400 if os.name == "nt" else 0)
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as scan:
                entries = sorted(scan, key=lambda item: item.name, reverse=True)
        except OSError:
            continue
        for entry in entries:
            path = Path(entry.path)
            try:
                relative = path.relative_to(root)
                if any(part.startswith(".") for part in relative.parts):
                    continue
                entry_stat = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or (reparse_flag and getattr(entry_stat, "st_file_attributes", 0) & reparse_flag):
                    continue
                resolved = path.resolve(strict=True)
                try:
                    resolved.relative_to(resolved_root)
                except ValueError:
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(path)
                elif entry.is_file(follow_symlinks=False):
                    files.append((relative.as_posix(), path))
            except (OSError, RuntimeError, ValueError):
                continue
    files.sort(key=lambda item: item[0])
    return files


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media_kind(name: str) -> str | None:
    suffix = Path(name).suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    if suffix in _VIDEO_SUFFIXES:
        return "video"
    if suffix in _AUDIO_SUFFIXES:
        return "audio"
    return None


def make_bundle(
    execution_id: str,
    package: Any,
    result: Any,
    events: Any,
    execution_dir: str | os.PathLike[str] | None,
    mode: str,
    *,
    bundle_id: str | None = None,
    source_kind: str | None = None,
) -> dict[str, Any]:
    """Build a JSON bundle while keeping source data and uncertainty explicit.

    ``package``, ``result`` and ``events`` must be JSON-serializable. Files in
    ``execution_dir`` are copied as content-addressed inline attachments; dot
    paths and symbolic links are skipped. Oversize attachments remain visible
    in ``media.missing`` with their size and SHA-256 digest.
    """
    if not isinstance(execution_id, (str, int)) or not str(execution_id):
        raise ValueError("execution_id must be a non-empty string or integer")
    execution_id = str(execution_id)[:256]
    package_copy = _json_copy(package)
    result_copy = _json_copy(result)
    events_copy = _json_copy(events)
    package_map = _as_mapping(package_copy)
    result_map = _as_mapping(result_copy)

    task_id = _identifier(package_map, "task", "task_id", "task_name", "task")
    package_id = _identifier(package_map, "package", "package_id", "id", "name", "package_name")
    identifiers = {
        "task_id": task_id,
        "package_id": package_id,
        "execution_id": execution_id,
    }
    if bundle_id is None:
        bundle_id = uuid.uuid4().hex
    # validate with the receiver's same safe-ID rule before constructing paths.
    from .common import safe_bundle_id

    safe_bundle_id(bundle_id)
    created_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    target_constraints = {
        "target": _pick(package_map, "target", "target_result", "expected_result", "task_instruction", "instruction"),
        "constraints": _pick(package_map, "constraints", "task_constraints", "completion_conditions", "timeout", "tolerance"),
    }
    equipment_config = _pick(
        package_map,
        "equipment",
        "equipment_id",
        "robot",
        "robots",
        "arm",
        "gripper",
        "equipment_config",
        "runtime_config",
        "config",
        "skill_id",
        "skill_version",
        "strategy",
        "mode",
    )

    planned_actions = _pick(package_map, "actions", "planned_actions", "action_trace")
    actual_commands = _pick(result_map, "actual_commands", "control_commands", "equipment_setpoints", "commands_sent")
    actual_state = _pick(result_map, "actual_state_feedback", "actual_state", "feedback", "equipment_actual_values", "observation")
    outcome_fields = _pick(result_map, "execution_status", "status", "task_outcome", "task_outcome_source", "measurement", "judgment", "evidence")
    logs = _pick(result_map, "logs", "execution_log", "errors", "exceptions", "error_code")
    quality_data = _pick(package_map, "data_quality", "quality", "sampling", "collection", "clock_sync")
    quality_data.update(_pick(result_map, "data_quality", "quality", "sampling", "collection", "clock_sync"))
    category_payloads: dict[str, Any] = {
        "identifiers": identifiers,
        "task_targets_constraints": target_constraints if any(_present(x) for x in target_constraints.values()) else None,
        "equipment_runtime_config": equipment_config or None,
        "action_trace": planned_actions or None,
        "control_commands_setpoints": actual_commands or None,
        "actual_state_feedback": actual_state or None,
        "logs_events_exceptions": {
            "events": events_copy,
            "result_logs": logs or None,
        } if events is not None or logs else None,
        "experiment_result": outcome_fields or None,
        "video_images": None,
        "data_quality_collection": quality_data or None,
    }

    attachments: list[dict[str, Any]] = []
    attachments_missing: list[dict[str, Any]] = []
    source_documents: dict[str, Any] = {}
    total_bytes = 0
    for name, original in (("package", package_copy), ("result", result_copy), ("events", events_copy)):
        raw = canonical_json(original)
        total_bytes = _add_attachment(
            attachments,
            source_documents,
            attachments_missing,
            attachment_id=f"source-{name}",
            name=f"{name}.json",
            content_type="application/json",
            raw=raw,
            source=name,
            total_so_far=total_bytes,
        )
        if name not in source_documents:
            source_documents[name] = {
                "status": "export_restricted",
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "reason": "The 16 MiB aggregate raw-attachment limit was reached.",
            }

    media_items: list[dict[str, Any]] = []
    log_files: list[dict[str, Any]] = []
    for relative_name, path in _execution_files(execution_dir):
        suffix = Path(relative_name).suffix.lower()
        kind = _media_kind(relative_name)
        source = "media" if kind else ("log" if suffix in _LOG_SUFFIXES else "execution_file")
        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            attachments_missing.append(
                {
                    "name": relative_name,
                    "source": source,
                    "status": "export_restricted",
                    "reason": f"File could not be read: {type(exc).__name__}.",
                }
            )
            continue
        if total_bytes + size_bytes > MAX_ATTACHMENT_BYTES:
            try:
                digest = _sha256_file(path)
            except OSError as exc:
                digest = None
                reason = f"File could not be read: {type(exc).__name__}."
            else:
                reason = "The 16 MiB aggregate raw-attachment limit was reached."
            attachments_missing.append(
                {
                    "name": relative_name,
                    "source": source,
                    "size_bytes": size_bytes,
                    "sha256": digest,
                    "status": "export_restricted",
                    "reason": reason,
                }
            )
            continue
        remaining_bytes = MAX_ATTACHMENT_BYTES - total_bytes
        try:
            with path.open("rb") as stream:
                raw = stream.read(remaining_bytes + 1)
        except OSError as exc:
            attachments_missing.append(
                {
                    "name": relative_name,
                    "source": source,
                    "size_bytes": size_bytes,
                    "status": "export_restricted",
                    "reason": f"File could not be read: {type(exc).__name__}.",
                }
            )
            continue
        if len(raw) > remaining_bytes:
            try:
                actual_size = path.stat().st_size
                digest = _sha256_file(path)
            except OSError as exc:
                actual_size = size_bytes
                digest = None
                reason = f"File could not be read: {type(exc).__name__}."
            else:
                reason = "The 16 MiB aggregate raw-attachment limit was reached."
            attachments_missing.append(
                {
                    "name": relative_name,
                    "source": source,
                    "size_bytes": actual_size,
                    "sha256": digest,
                    "status": "export_restricted",
                    "reason": reason,
                }
            )
            continue
        before = len(attachments)
        total_bytes = _add_attachment(
            attachments,
            source_documents,
            attachments_missing,
            attachment_id=f"file-{len(attachments) + 1:04d}",
            name=relative_name,
            content_type=mimetypes.guess_type(relative_name)[0] or "application/octet-stream",
            raw=raw,
            source=source,
            total_so_far=total_bytes,
        )
        if len(attachments) == before:
            continue
        record = {
            "attachment_id": attachments[-1]["attachment_id"],
            "name": relative_name,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        if kind:
            media_items.append({**record, "kind": kind})
        if source == "log":
            log_files.append(record)

    media_missing = [item for item in attachments_missing if item.get("source") == "media"]
    media_data = {
        "items": media_items,
        "missing": media_missing,
    }
    category_payloads["video_images"] = media_data if media_items or media_missing else None
    overrides = _all_category_overrides(package_map, result_map)
    video_status = None
    if "video_images" not in overrides:
        if media_items and media_missing:
            video_status = "partial"
        elif media_missing:
            video_status = "export_restricted"
        elif media_items:
            video_status = "available"
    data_categories = {
        name: _category_record(
            name,
            category_payloads[name],
            overrides,
            derived_status=video_status if name == "video_images" else None,
        )
        for name in CATEGORY_NAMES
    }

    # Make the execution status and task outcome separate even when one is absent.
    execution_record = {
        "planned_actions": planned_actions.get("actions", planned_actions) if planned_actions else None,
        "actual_commands": actual_commands or None,
        "actual_state_feedback": actual_state or None,
        "events": events_copy,
        "execution_status": result_map.get("execution_status", result_map.get("status", "unknown")),
        "task_outcome": result_map.get("task_outcome", "unknown"),
        "task_outcome_source": result_map.get("task_outcome_source", "unknown"),
        "result_fields": outcome_fields,
    }

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "created_at": created_at,
        "mode": str(mode) if mode is not None else "unknown",
        "provenance": {
            "kind": _source_kind(mode, package_map, result_map, source_kind),
            "mode": str(mode) if mode is not None else "unknown",
            "note": "Provenance is explicit source metadata or conservative mode classification; unknown remains unknown.",
        },
        "identifiers": identifiers,
        "task_targets_constraints": target_constraints,
        "equipment_runtime_config": equipment_config if equipment_config else {"status": "unknown", "data": None},
        "execution": execution_record,
        "logs": {
            "source_fields": logs,
            "attachments": log_files,
        },
        "result": {
            "execution_status": execution_record["execution_status"],
            "task_outcome": execution_record["task_outcome"],
            "task_outcome_source": execution_record["task_outcome_source"],
            "measurement": result_map.get("measurement"),
            "judgment": result_map.get("judgment"),
            "evidence": result_map.get("evidence"),
        },
        "media": media_data,
        "attachments_missing": attachments_missing,
        "data_categories": data_categories,
        "source_documents": source_documents,
        "attachments": attachments,
    }
    validate_bundle(bundle)
    return bundle


def _experiments_url(receiver_url: str) -> str:
    parsed = urllib.parse.urlsplit(receiver_url)
    if parsed.path.rstrip("/").endswith(ENDPOINT_PATH):
        path = parsed.path.rstrip("/")
    else:
        path = parsed.path.rstrip("/") + ENDPOINT_PATH
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))


def _validate_receiver_url(url: str, allow_http: bool) -> str:
    if not isinstance(url, str):
        raise ValueError("receiver_url must be a URL string")
    normalized = _experiments_url(url)
    parsed = urllib.parse.urlsplit(normalized)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise ValueError("receiver_url must use http or https and include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("receiver_url must not include embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("receiver_url must not include a query or fragment")
    if parsed.scheme == "http":
        host = parsed.hostname.lower()
        loopback = host == "localhost"
        try:
            loopback = loopback or ipaddress.ip_address(host).is_loopback
        except ValueError:
            pass
        if not loopback and not allow_http:
            raise ValueError("remote HTTP requires allow_http=True")
    return normalized


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def deliver_bundle(
    receiver_url: str,
    bundle: dict[str, Any],
    token: str | None = None,
    allow_http: bool = False,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """POST one bundle and validate the provisional receiver's ACK.

    The receiver URL may be its base URL or the full mock endpoint. Plain HTTP
    is accepted by default only for loopback hosts; remote HTTP needs explicit
    ``allow_http=True``. Redirects are rejected.
    """
    validate_bundle(bundle)
    url = _validate_receiver_url(receiver_url, allow_http)
    if token is not None and (not isinstance(token, str) or "\r" in token or "\n" in token):
        raise ValueError("token must be a single-line string")
    envelope = {"bundle": bundle, "content_digest": content_digest(bundle)}
    payload = canonical_json(envelope)
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("encoded bundle exceeds the 24 MiB request limit")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise DeliveryError(f"receiver returned HTTP {response.status}")
            response_body = response.read(64 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        raise DeliveryError(f"receiver returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DeliveryError(f"receiver request failed: {type(exc).__name__}: {exc}") from exc
    if len(response_body) > 64 * 1024:
        raise DeliveryError("receiver acknowledgement exceeds the 64 KiB limit")
    try:
        ack = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeliveryError("receiver returned a non-JSON acknowledgement") from exc
    expected_digest = envelope["content_digest"]
    if not isinstance(ack, dict) or ack.get("accepted") is not True:
        raise DeliveryError("receiver acknowledgement did not confirm accepted=true")
    if ack.get("bundle_id") != bundle["bundle_id"]:
        raise DeliveryError("receiver acknowledgement bundle_id did not match the submitted bundle")
    if ack.get("content_digest") != expected_digest:
        raise DeliveryError("receiver acknowledgement content_digest did not match the submitted bundle")
    return ack


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    raw = canonical_json(value)
    with temporary.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _stable_bundle_id(execution_id: str, package: Any) -> str:
    package_map = _as_mapping(package) if isinstance(package, dict) else {}
    task_id = _identifier(package_map, "task", "task_id", "task_name", "task")
    package_id = _identifier(package_map, "package", "package_id", "id", "name", "package_name")
    idempotency_key = canonical_json(
        {"task_id": task_id, "package_id": package_id, "execution_id": str(execution_id)}
    )
    return f"exec-{hashlib.sha256(idempotency_key).hexdigest()[:32]}"


class ResultsDelivery:
    """Durable, retrying sender for the provisional result receiver.

    Captured records are saved in ``data_dir/outbox`` before any network call.
    Re-capturing the same task/package/execution tuple reuses the existing
    bundle, including its original timestamp and contents.
    """

    def __init__(
        self,
        data_dir: str | os.PathLike[str],
        receiver_url: str,
        token: str | None = None,
        allow_http: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.outbox_dir = self.data_dir / "outbox"
        self.receiver_url = _validate_receiver_url(receiver_url, allow_http)
        self.token = token
        self.allow_http = allow_http
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_worker_error: str | None = None

    def _record_path(self, bundle_id: str) -> Path:
        filename = hashlib.sha256(bundle_id.encode("utf-8")).hexdigest() + ".json"
        return self.outbox_dir / filename

    def _records(self) -> list[tuple[Path, dict[str, Any]]]:
        if not self.outbox_dir.exists():
            return []
        records: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted(self.outbox_dir.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and isinstance(value.get("bundle"), dict):
                records.append((path, value))
        return records

    def capture(
        self,
        execution_id: str,
        package: Any,
        result: Any,
        events: Any,
        execution_dir: str | os.PathLike[str] | None,
        mode: str,
    ) -> str:
        """Persist a bundle and return its stable ID before delivery begins."""
        bundle_id = _stable_bundle_id(str(execution_id), package)
        path = self._record_path(bundle_id)
        with self._lock:
            if path.exists():
                # A retry after restart must not create a second bundle or let
                # transient marker files alter the already-captured payload.
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                    if existing.get("bundle", {}).get("bundle_id") == bundle_id:
                        return bundle_id
                except (OSError, json.JSONDecodeError, AttributeError):
                    pass
                raise DeliveryError(f"existing outbox record for {bundle_id} is unreadable")

            bundle = make_bundle(
                execution_id,
                package,
                result,
                events,
                execution_dir,
                mode,
                bundle_id=bundle_id,
            )
            record = {
                "record_version": 1,
                "bundle": bundle,
                "content_digest": content_digest(bundle),
                "state": "pending",
                "attempts": 0,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                "last_attempt_at": None,
                "ack": None,
                "last_error": None,
                "next_attempt_at": 0.0,
            }
            _atomic_write_json(path, record)
        self._wake.set()
        return bundle_id

    def start(self) -> None:
        """Start one daemon worker that retries every pending durable record."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._worker, name="results-delivery", daemon=True)
            self._thread.start()
        self._wake.set()

    def close(self) -> None:
        """Stop the background worker after its current request finishes."""
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=REQUEST_TIMEOUT_SECONDS + 1.0)
        self._thread = None

    def status(self) -> dict[str, Any]:
        """Return a snapshot of queued, delivered, and failed-to-read records."""
        items: list[dict[str, Any]] = []
        for path in sorted(self.outbox_dir.glob("*.json")) if self.outbox_dir.exists() else []:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                bundle = record.get("bundle", {})
                items.append(
                    {
                        "bundle_id": bundle.get("bundle_id", "unknown"),
                        "execution_id": bundle.get("identifiers", {}).get("execution_id", "unknown"),
                        "state": record.get("state", "unknown"),
                        "attempts": record.get("attempts", 0),
                        "last_error": record.get("last_error"),
                        "ack": record.get("ack"),
                    }
                )
            except (OSError, json.JSONDecodeError, AttributeError):
                items.append({"bundle_id": "unknown", "state": "corrupt", "record": path.name})
        return {
            "pending": sum(item.get("state") == "pending" for item in items),
            "delivered": sum(item.get("state") == "delivered" for item in items),
            "corrupt": sum(item.get("state") == "corrupt" for item in items),
            "last_worker_error": self._last_worker_error,
            "items": items,
        }

    def _worker(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            next_due: float | None = None
            try:
                records = self._records()
            except Exception as exc:
                self._last_worker_error = f"outbox scan failed: {type(exc).__name__}: {exc}"
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            for path, record in records:
                if self._stop.is_set():
                    break
                try:
                    if record.get("state") != "pending":
                        continue
                    due = float(record.get("next_attempt_at", 0.0) or 0.0)
                    if due > now:
                        next_due = due if next_due is None else min(next_due, due)
                        continue
                    self._send_record(path, record)
                except Exception as exc:
                    # A malformed retry time or disk error must not kill the
                    # daemon and strand every other pending bundle.
                    self._last_worker_error = f"{path.name}: {type(exc).__name__}: {exc}"
            wait_seconds = 1.0 if next_due is None else min(1.0, max(0.05, next_due - time.time()))
            self._wake.wait(wait_seconds)
            self._wake.clear()

    def _send_record(self, path: Path, record: dict[str, Any]) -> None:
        with self._lock:
            # Re-read under lock so capture/worker updates cannot overwrite one another.
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
            if record.get("state") != "pending":
                return
            record["attempts"] = int(record.get("attempts", 0)) + 1
            attempt = record["attempts"]
            record["last_attempt_at"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            _atomic_write_json(path, record)
        try:
            ack = deliver_bundle(
                self.receiver_url,
                record["bundle"],
                token=self.token,
                allow_http=self.allow_http,
            )
        except Exception as exc:  # Persist failures so the queue remains inspectable/retryable.
            delay = min(30.0, float(2 ** min(attempt - 1, 5)))
            with self._lock:
                try:
                    current = json.loads(path.read_text(encoding="utf-8"))
                    current["last_error"] = f"{type(exc).__name__}: {exc}"
                    current["next_attempt_at"] = time.time() + delay
                    _atomic_write_json(path, current)
                except Exception as persist_exc:
                    self._last_worker_error = (
                        f"{path.name}: could not persist delivery error: "
                        f"{type(persist_exc).__name__}: {persist_exc}"
                    )
            return
        with self._lock:
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
                current["state"] = "delivered"
                current["ack"] = ack
                current["last_error"] = None
                current["next_attempt_at"] = None
                _atomic_write_json(path, current)
            except Exception as exc:
                self._last_worker_error = (
                    f"{path.name}: could not persist delivery ACK: {type(exc).__name__}: {exc}"
                )


__all__ = ["DeliveryError", "ResultsDelivery", "deliver_bundle", "make_bundle"]
