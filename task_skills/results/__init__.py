"""Provisional local result-bundle delivery for future 3.3 integration.

This package defines a mock contract only. It is not an official 3.3 API.
Attachments are limited to 16 MiB total before base64 encoding; omitted files
are recorded under ``media.missing`` with an explicit reason.
"""

from .delivery import (
    DeliveryError,
    ResultsDelivery,
    deliver_bundle,
    make_bundle,
)

__all__ = ["DeliveryError", "ResultsDelivery", "deliver_bundle", "make_bundle"]
