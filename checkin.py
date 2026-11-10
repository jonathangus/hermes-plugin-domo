"""Spaces that keep moving (docs/PRODUCT.md "Anticipate, then ask"; after Nebula's autonomous channels).

A space with ``keep_moving`` on gets a check-in every couple of hours from the "Domo check-in" routine.
This module is the routine's gate: its script (scripts/domo-checkin.py) prints ``brief()``, and when
nothing needs a look it prints ``{"wakeAgent": false}``, so Hermes never calls the model and an idle
check-in costs nothing. The agent then follows the domo-check-in skill: prepare the next step and
leave it in Next up, or ask what the space is waiting on. It never posts a status update.

It stays quiet in a space while:
- a question Domo asked there is still open (a decision);
- a run there is working;
- someone was chatting there in the last 45 minutes;
- its last two check-ins left something and nobody has been back since.

Promises with a due time that someone else made (who) or Domo owns come up here once, when due, so
the agent can find out whether they happened and prepare a nudge.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import search, suggest

QUIET_AFTER_CHAT = 45 * 60
STRIKES = 2          # check-ins that left something with nobody back since, before it waits
STATE_KEY = "checkin"
ASLEEP = json.dumps({"wakeAgent": False})


def _state(store) -> dict:
    row = store._db.execute("SELECT value FROM meta WHERE key = ?", (STATE_KEY,)).fetchone()
    try:
        return json.loads(row[0]) if row else {}
    except ValueError:
        return {}


def _save(store, state: dict) -> None:
    with store._tx() as db:
        db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (STATE_KEY, json.dumps(state)))


def _chats(home: Path, ids: list[str]) -> list[dict]:
    """The space's chats, newest first, with the last thing said: the person's and the agent's."""
    db = search._hermes(home) if ids else None
    if db is None:
        return []
    try:
        marks = ",".join("?" * len(ids))
        rows = db.execute(f"SELECT id, title, source, COALESCE(last_activity_at, ended_at, started_at) AS last_active FROM sessions"
                          f" WHERE id IN ({marks}) AND source != 'cron' ORDER BY last_active DESC LIMIT 6", ids).fetchall()
        out = []
        for r in rows:
            last = db.execute("SELECT role, content FROM messages WHERE session_id = ? AND role IN ('user','assistant') AND active = 1"
                              " AND COALESCE(content, '') != '' ORDER BY id DESC LIMIT 1", (r["id"],)).fetchone()
            out.append({**dict(r), "last": search.clean(last["content"])[:280] if last else "", "last_role": last["role"] if last else ""})
        return out
    finally:
        db.close()


def _of(items: list[dict], space: dict, sessions: set[str]) -> list[dict]:
    names = {space["name"].lower()}
    return [i for i in items if (i["space"] or "").lower() in names or (i["session_id"] and i["session_id"] in sessions)]


def _section(space: dict, chats: list[dict], todos: list[dict], left: list[dict], files: list[dict]) -> str:
    lines = [f"## {space['name']} (space {space['id']})"]
    if space.get("description"):
        lines.append(f"What it's for: {space['description']}")
    if space.get("prompt"):
        lines.append(f"What Domo should know: {space['prompt']}")
    if chats:
        lines.append("Recent chats:")
        for c in chats:
            who = "they" if c["last_role"] == "user" else "Domo"
            lines.append(f"- “{c['title'] or 'Untitled'}” ({suggest._ago(c['last_active'])}), {who} last said: {c['last'] or '…'}")
    if todos:
        lines.append("Open to-dos: " + "; ".join(f"{t['title']}" + (f" (due {t['data']['due']})" if t["data"].get("due") else "") for t in todos[:8]))
    if files:
        lines.append("Files: " + ", ".join(f["name"] for f in files[:6]))
    if left:
        lines.append("Already left for them, don't repeat: " + "; ".join(s["title"] for s in left[:6]))
    return "\n".join(lines)


def brief(home: Path, store, now: Optional[float] = None) -> Optional[str]:
    """What this check-in should look at, or None when nothing needs a look (the agent isn't woken)."""
    now = now if now is not None else time.time()
    state = _state(store)
    spaces_state, raised = state.get("spaces", {}), set(state.get("promises", []))
    decisions = store.list("decision", status="open", limit=200)
    runs = store.list("run", status="open", limit=200)
    todos = store.list("todo", status="open", limit=500)
    ideas = [s for s in store.list("suggestion", status="open", limit=200) if float(s["data"].get("expires_at") or now + 1) > now]
    sections = []
    for space in store.spaces():
        if not space.get("keep_moving"):
            continue
        sessions = store.tagged(space["id"], "session")
        chats = _chats(home, sessions)
        stamp = max([c["last_active"] or 0 for c in chats] + [0])
        mine = spaces_state.get(space["id"], {})
        if _of(decisions, space, set(sessions)) or _of(runs, space, set(sessions)) or now - stamp < QUIET_AFTER_CHAT:
            continue
        left = _of(ideas, space, set(sessions))
        strikes = mine.get("strikes", 0)
        if mine:
            acted = any(i["created_at"] > mine["at"] for i in _of(ideas + decisions, space, set(sessions)))
            strikes = 0 if stamp > mine.get("stamp", 0) else strikes + (1 if acted else 0)
        if strikes >= STRIKES:
            spaces_state[space["id"]] = {**mine, "strikes": strikes}
            continue
        spaces_state[space["id"]] = {"at": now, "stamp": stamp, "strikes": strikes}
        sections.append(_section(space, chats, _of(todos, space, set(sessions)), left, store.files(space["id"], limit=6)))
    promises = []
    for t in todos:
        due = suggest.due_at(t["data"].get("due"))
        who = str(t["data"].get("who") or "").strip()
        if due is None or now < due or t["id"] in raised or not (who or t["data"].get("owner") == "domo"):
            continue
        raised.add(t["id"])
        whose = f"{who} said they'd" if who else "You (Domo) said you'd"
        promises.append(f"- {whose} {suggest._verb(t['title'])} by {t['data']['due']} (to-do {t['id']}{', space ' + t['space'] if t['space'] else ''})")
    _save(store, {"spaces": spaces_state, "promises": sorted(raised)[-200:]})
    if not sections and not promises:
        return None
    head = f"# Domo check-in · {datetime.fromtimestamp(now):%a %d %b %H:%M}"
    parts = [head, *sections]
    if promises:
        parts.append("## Promises now due\n" + "\n".join(promises))
    return "\n\n".join(parts)


def gate(home: Path, store) -> str:
    """The routine script's whole output."""
    return brief(home, store) or ASLEEP
