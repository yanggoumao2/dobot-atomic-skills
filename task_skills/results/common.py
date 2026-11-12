"""Shared serialization and validation for the provisional mock contract."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from typing import Any

SCHEMA_VERSION = "provisional-3.3-mock/v1"
ENDPOINT_PATH = "/mock-3.3/v1/experiments"
MAX_ATTACHMENT_BYTES = 16 * 1024 * 1024
MAX_REQUEST_BYTES = 24 * 1024 * 1024
MAX_IDENTIFIER_LENGTH = 256

CATEGORY_NAMES = (
    "identifiers",
    "task_targets_constraints",
    "equipment_runtime_config",
    "action_trace",
    "control_commands_setpoints",
    "actual_state_feedback",
    "logs_events_exceptions",
    "experiment_result",
    "video_images",
    "data_quality_collection",
)

CATEGORY_STATUSES = frozenset(
    {
        "available",
        "partial",
        "not_recorded",
        "export_restricted",
        "unsupported",
        "not_applicable",
        "unknown",
    }
)

STATUS_ALIASES = {
    "可提供": "available",
    "available": "available",
    "partial": "partial",
    "部分可提供": "partial",
    "本次未记录": "not_recorded",
    "not_recorded": "not_recorded",
    "导出受限": "export_restricted",
    "export_restricted": "export_restricted",
    "不支持": "unsupported",
    "unsupported": "unsupported",
    "不适用": "not_applicable",
    "not_applicable": "not_applicable",
    "未知": "unknown",
    "unknown": "unknown",
}

_SAFE_BUNDLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: Any) -> bytes:
    """Serialize JSON deterministically and reject NaN/Infinity."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def normalize_status(value: Any) -> str:
    if isinstance(value, str):
        return STATUS_ALIASES.get(value.strip().lower(), STATUS_ALIASES.get(value.strip(), "unknown"))
    return "unknown"


def validate_bundle(bundle: Any) -> None:
    """Validate required contract fields and every inline attachment digest."""
    if not isinstance(bundle, dict):
        raise ValueError("bundle must be a JSON object")
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported provisional bundle schema")
    bundle_id = bundle.get("bundle_id")
    if not isinstance(bundle_id, str) or not _SAFE_BUNDLE_ID.fullmatch(bundle_id):
        raise ValueError("bundle_id must be a safe identifier")

    identifiers = bundle.get("identifiers")
    if not isinstance(identifiers, dict):
        raise ValueError("bundle.identifiers must be an object")
    for field in ("task_id", "package_id", "execution_id"):
        value = identifiers.get(field)
        if not isinstance(value, str) or not value or len(value) > MAX_IDENTIFIER_LENGTH:
            raise ValueError(f"bundle.identifiers.{field} must be a non-empty string")

    provenance = bundle.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("kind") not in {
        "real",
        "mock",
        "synthetic",
        "unknown",
    }:
        raise ValueError("bundle.provenance.kind must identify real, mock, synthetic, or unknown data")

    categories = bundle.get("data_categories")
    if not isinstance(categories, dict):
        raise ValueError("bundle.data_categories must be an object")
    for category in CATEGORY_NAMES:
        item = categories.get(category)
        if not isinstance(item, dict):
            raise ValueError(f"missing data category: {category}")
        if item.get("status") not in CATEGORY_STATUSES:
            raise ValueError(f"invalid status for data category: {category}")
        if "data" not in item:
            raise ValueError(f"data category lacks data field: {category}")
        if not isinstance(item.get("missing_reason"), (str, type(None))):
            raise ValueError(f"invalid missing_reason for data category: {category}")

    attachments = bundle.get("attachments")
    if not isinstance(attachments, list):
        raise ValueError("bundle.attachments must be an array")
    attachment_ids: set[str] = set()
    total_attachment_bytes = 0
    for attachment in attachments:
        if not isinstance(attachment, dict):
            raise ValueError("attachment must be an object")
        attachment_id = attachment.get("attachment_id")
        if not isinstance(attachment_id, str) or not attachment_id or attachment_id in attachment_ids:
            raise ValueError("attachment IDs must be unique non-empty strings")
        attachment_ids.add(attachment_id)
        encoded = attachment.get("data_base64")
        if not isinstance(encoded, str):
            raise ValueError(f"attachment {attachment_id} has no base64 payload")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError(f"attachment {attachment_id} contains invalid base64") from exc
        size = attachment.get("size_bytes")
        digest = attachment.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size != len(raw):
            raise ValueError(f"attachment {attachment_id} size does not match its payload")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"attachment {attachment_id} has an invalid sha256")
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"attachment {attachment_id} sha256 does not match its payload")
        total_attachment_bytes += len(raw)
        if total_attachment_bytes > MAX_ATTACHMENT_BYTES:
            raise ValueError("bundle attachments exceed the 16 MiB limit")

    source_documents = bundle.get("source_documents")
    if not isinstance(source_documents, dict):
        raise ValueError("bundle.source_documents must be an object")
    for document_name in ("package", "result", "events"):
        ref = source_documents.get(document_name)
        if not isinstance(ref, dict):
            raise ValueError(f"bundle.source_documents.{document_name} must reference an attachment")
        if ref.get("attachment_id") not in attachment_ids:
            if ref.get("status") != "export_restricted" or not isinstance(ref.get("reason"), str):
                raise ValueError(f"bundle.source_documents.{document_name} must reference an attachment or explain its omission")

    # Validate bundle JSON itself before accepting it into the mock receiver.
    canonical_json(bundle)


def safe_bundle_id(value: Any) -> str:
    if not isinstance(value, str) or not _SAFE_BUNDLE_ID.fullmatch(value):
        raise ValueError("bundle_id must be a safe identifier")
    return value
