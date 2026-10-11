"""Stable SEAM identifiers (16 lowercase hex chars)."""

import re
import uuid

_ID_RE = re.compile(r"^[0-9a-f]{16}$")


def new_id() -> str:
    """Return a new random 16-hex-char identifier."""
    return uuid.uuid4().hex[:16]


def is_valid_id(value) -> bool:
    """True if ``value`` is a well-formed SEAM id."""
    return isinstance(value, str) and bool(_ID_RE.match(value))
