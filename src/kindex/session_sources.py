"""Discover local agent sessions and resolve their source event times."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from .temporal import normalize_datetime, parse_iso_datetime

SessionFile = tuple[Path, datetime]
EventTimeResolver = Callable[[Path], datetime]


def file_event_time(path: Path) -> datetime:
    """Return a file's modification time as an aware UTC datetime."""
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def codex_event_time(path: Path) -> datetime:
    """Return Codex session metadata time, falling back to file modification time."""
    timestamp = _codex_metadata_timestamp(path)
    if timestamp:
        try:
            return parse_iso_datetime(timestamp, field="timestamp")
        except ValueError:
            pass
    return file_event_time(path)


def recent_session_files(
    root: Path,
    *,
    event_time: EventTimeResolver = file_event_time,
    since: str | None = None,
    limit: int = 10,
) -> list[SessionFile]:
    """Return newest session files, filtering by source time before limiting."""
    since_time = (
        normalize_datetime(parse_iso_datetime(since, field="since"))
        if since
        else None
    )
    candidates = (
        (path, normalize_datetime(event_time(path)))
        for path in root.rglob("*.jsonl")
    )
    eligible = (
        candidate
        for candidate in candidates
        if since_time is None or candidate[1] >= since_time
    )
    return sorted(eligible, key=lambda candidate: candidate[1], reverse=True)[:limit]


def _codex_metadata_timestamp(path: Path) -> str | None:
    """Read the timestamp from the first Codex session metadata event."""
    try:
        with path.open("r", errors="replace") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if entry.get("type") != "session_meta":
                    continue
                payload = entry.get("payload") or {}
                if isinstance(payload, dict) and payload.get("timestamp"):
                    return payload["timestamp"]
                return entry.get("timestamp")
    except OSError:
        return None
    return None
