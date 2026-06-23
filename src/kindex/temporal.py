"""Canonical timestamp parsing and validation for temporal boundaries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dateutil.parser import isoparse


def parse_iso_datetime(value: str, field: str = "timestamp") -> datetime:
    """Parse an ISO date or timestamp and normalize it to aware UTC."""
    try:
        parsed = isoparse(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be an ISO date or timestamp") from None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_datetime(value: datetime) -> datetime:
    """Normalize an instant to aware UTC at millisecond resolution."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    milliseconds = (value.microsecond + 500) // 1_000
    if milliseconds == 1_000:
        return value.replace(microsecond=0) + timedelta(seconds=1)
    return value.replace(microsecond=milliseconds * 1_000)


def normalize_iso_datetime(value: str, field: str = "timestamp") -> str:
    """Return a canonical UTC timestamp rounded half-up to milliseconds."""
    parsed = normalize_datetime(parse_iso_datetime(value, field))
    return parsed.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def normalize_time_range(since: str = "", until: str = "") -> tuple[str, str]:
    """Normalize optional boundaries and reject an inverted time range."""
    normalized_since = normalize_iso_datetime(since, "since") if since else ""
    normalized_until = normalize_iso_datetime(until, "until") if until else ""
    if normalized_since and normalized_until and normalized_since > normalized_until:
        raise ValueError("since must be earlier than or equal to until")
    return normalized_since, normalized_until
