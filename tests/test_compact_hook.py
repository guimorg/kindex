"""Tests for compact-hook envelope handling (issue #14).

Claude Code's PreCompact hook pipes a JSON envelope ({session_id,
transcript_path, cwd, hook_event_name, ...}) on stdin. transcript_path
is a file path, not conversation text — extracting from the envelope
itself minted one blank node per JSON field.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from kindex.config import Config
from kindex.store import Store

_SESSION_ID = "0f9b3a52-1111-2222-3333-444455556666"


def _env(tmp_path):
    env = dict(os.environ)
    env.pop("KIN_PROFILE", None)
    env.pop("ANTHROPIC_API_KEY", None)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env["HOME"] = str(home)
    return env


def _run_hook(tmp_path, input_text):
    cmd = [sys.executable, "-m", "kindex.cli", "compact-hook",
           "--data-dir", str(tmp_path / "data")]
    return subprocess.run(cmd, input=input_text, capture_output=True,
                          text=True, timeout=60, env=_env(tmp_path),
                          cwd=str(tmp_path))


def _write_transcript(tmp_path, texts):
    transcript = tmp_path / "transcript.jsonl"
    lines = [
        json.dumps({
            "type": "assistant",
            "message": {"role": "assistant",
                        "content": [{"type": "text", "text": text}]},
        })
        for text in texts
    ]
    transcript.write_text("\n".join(lines) + "\n")
    return transcript


def _envelope(tmp_path, transcript=None):
    payload = {
        "session_id": _SESSION_ID,
        "cwd": str(tmp_path),
        "hook_event_name": "PreCompact",
        "trigger": "auto",
        "custom_instructions": "",
    }
    if transcript is not None:
        payload["transcript_path"] = str(transcript)
    return json.dumps(payload)


def _nodes(tmp_path):
    cfg = Config(data_dir=str(tmp_path / "data"))
    store = Store(cfg)
    try:
        return {n["title"]: n for n in store.all_nodes(limit=500)}
    finally:
        store.close()


def _reinforce_queue(tmp_path):
    cfg = Config(data_dir=str(tmp_path / "data"))
    store = Store(cfg)
    try:
        raw = store.get_meta("reinforce.queue")
        return json.loads(raw) if raw else []
    finally:
        store.close()


def test_envelope_fields_do_not_become_nodes(tmp_path):
    transcript = _write_transcript(tmp_path, [
        "We learned that the cache invalidation bug was caused by a stale "
        "TTL configuration in the deploy pipeline.",
    ])
    r = _run_hook(tmp_path, _envelope(tmp_path, transcript))
    assert r.returncode == 0, r.stderr

    nodes = _nodes(tmp_path)
    for junk in ("session_id", "transcript_path", "cwd", "hook_event_name",
                 "PreCompact", "auto", _SESSION_ID, str(transcript)):
        assert junk not in nodes
    # Whatever was minted came from the transcript and carries content
    for node in nodes.values():
        assert node.get("content", "").strip()


def test_envelope_extracts_from_transcript_text(tmp_path):
    transcript = _write_transcript(tmp_path, [
        "We learned that the cache invalidation bug was caused by a stale "
        "TTL configuration in the deploy pipeline.",
    ])
    r = _run_hook(tmp_path, _envelope(tmp_path, transcript))
    assert r.returncode == 0, r.stderr

    assert any("cache invalidation" in title for title in _nodes(tmp_path))


def test_envelope_without_transcript_creates_nothing(tmp_path):
    r = _run_hook(tmp_path, _envelope(tmp_path))
    assert r.returncode == 0, r.stderr
    assert _nodes(tmp_path) == {}


def test_envelope_still_enqueues_reinforce(tmp_path):
    transcript = _write_transcript(tmp_path, ["Short."])
    r = _run_hook(tmp_path, _envelope(tmp_path, transcript))
    assert r.returncode == 0, r.stderr

    queue = _reinforce_queue(tmp_path)
    assert any(job.get("conversation_id") == _SESSION_ID for job in queue)
    assert any(job.get("transcript_path") == str(transcript) for job in queue)


def test_plain_text_still_extracts(tmp_path):
    text = ("During this refactor we learned that the retry logic never "
            "honored the backoff ceiling configured for the API client.")
    r = _run_hook(tmp_path, text)
    assert r.returncode == 0, r.stderr

    nodes = _nodes(tmp_path)
    assert any("retry logic" in title for title in nodes)
    for node in nodes.values():
        assert node.get("content", "").strip()


def test_title_only_keyword_concepts_are_not_minted(tmp_path):
    # Quoted terms produce title-only concepts in the keyword fallback;
    # those are linking hints, not knowledge worth minting as blank nodes.
    text = ('The "flux capacitor" and the "warp drive" subsystems were '
            "mentioned repeatedly during the discussion today.")
    r = _run_hook(tmp_path, text)
    assert r.returncode == 0, r.stderr
    assert _nodes(tmp_path) == {}


def test_extract_session_text_handles_nested_message_format(tmp_path):
    from kindex.ingest import _extract_session_text

    transcript = _write_transcript(tmp_path, ["First insight.", "Second insight."])
    text = _extract_session_text(transcript)
    assert "First insight." in text
    assert "Second insight." in text


def test_extract_session_text_handles_legacy_format(tmp_path):
    from kindex.ingest import _extract_session_text

    transcript = tmp_path / "legacy.jsonl"
    transcript.write_text(
        json.dumps({"role": "assistant", "content": "Legacy text."}) + "\n"
    )
    assert "Legacy text." in _extract_session_text(transcript)


def test_capture_session_end_skips_title_only_concepts(tmp_path):
    from kindex.budget import BudgetLedger
    from kindex.hooks import capture_session_end

    cfg = Config(data_dir=str(tmp_path / "data"))
    store = Store(cfg)
    try:
        ledger = BudgetLedger(cfg.ledger_path, cfg.budget)
        text = ('The "flux capacitor" and the "warp drive" subsystems were '
                "mentioned repeatedly during the discussion today.")
        capture_session_end(store, cfg, ledger, session_text=text)
        titles = {n["title"] for n in store.all_nodes(limit=100)}
        assert "flux capacitor" not in titles
        assert "warp drive" not in titles
    finally:
        store.close()
