"""The Claude plan's limits (the session window and the week, how full and when each resets), as Claude
Code's own `/usage` reports them: `claude -p "/usage"` runs locally, costs no tokens, and with a
claude.ai sign-in carries a structured ``usage_report`` (documented, marked experimental). Read on the
home Mac with its `claude` sign-in, cached for a few minutes because Anthropic rate-limits the lookup.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from .work import claude_bin

FRESH = 5 * 60         # how long a reading holds
RETRY = 15 * 60        # after one that failed: the lookup may be rate-limited for a while
TIMEOUT = 30

_cache: dict[str, tuple[float, Optional[dict]]] = {}
_lock = threading.Lock()


def _label(row: dict) -> str:
    kind, group = str(row.get("kind") or ""), str(row.get("group") or "")
    scope = row.get("scope") or {}
    named = ((scope.get("model") or {}).get("display_name") or (scope.get("surface") or {}).get("display_name") or "")
    base = "Session" if group == "session" or kind == "session" else "Week" if group == "weekly" or kind.startswith("weekly") else kind.replace("_", " ").capitalize()
    return f"{base} · {named}" if named else base


def parse(stdout: str) -> Optional[dict]:
    """The limits from `/usage`'s stream-json, in the order Claude Code sends them; None without a report."""
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        report = event.get("usage_report") if isinstance(event, dict) and event.get("type") == "assistant" else None
        rows = ((report or {}).get("rate_limits") or {}).get("limits")
        if rows:
            return {"limits": [{"kind": r.get("kind"), "label": _label(r), "percent": round(float(r.get("percent") or 0)),
                                "resets_at": r.get("resets_at"), "severity": r.get("severity") or "normal"}
                               for r in rows if isinstance(r, dict) and r.get("is_active", True)]}
    return None


def usage(home: Path, *, run: Callable = subprocess.run, now: Optional[float] = None) -> Optional[dict]:
    now = now if now is not None else time.time()
    key = str(home)
    with _lock:
        hit = _cache.get(key)
    if hit and now < hit[0]:
        return hit[1]
    found = None
    binary = claude_bin()
    if binary:
        quiet = home / "plugin-data" / "domo" / ".domo-quiet"   # Link skips this folder: the lookup never shows as a session
        quiet.mkdir(parents=True, exist_ok=True)
        try:
            out = run([binary, "-p", "/usage", "--output-format", "stream-json", "--verbose", "--no-session-persistence"],
                      stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=TIMEOUT, cwd=quiet)
            found = parse(out.stdout) if out.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            found = None
    if found:
        found["at"] = now
    with _lock:
        _cache[key] = (now + (FRESH if found else RETRY), found)
    return found
