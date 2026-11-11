"""Whose a chat is: the person's own, or someone else's on a Hermes they share with others (the work computer's).

Home shows what came out of the person's own chats; a space stitches in everything, whoever's chat it was.
Hermes names who started a gateway chat (Slack, WhatsApp…) by the sender's id there, so the person's own
ids on that Hermes (`slack:U0123`) say which of those are theirs. A chat started through Domo's API server
(Hermes names its sender "api"), the terminal or an editor is theirs, as is one with no sender: they're the
one reaching that Hermes through Domo. A routine belongs to whoever made it (its origin), and its runs and what they leave with it; what a
subagent or a branch of a chat does belongs to the chat it came from. Something without a chat is theirs.

Only the work computer's agents are filtered, and only because the home Mac's relay names the person
(`me=`, possibly empty): the home Mac's own profiles are each one person's already.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from . import search

CRON = "cron"
OWN_SOURCES = ("api_server", "cli", "local", "tui", "acp")   # reached only by whoever runs Domo or sits at that machine


def parse(me: Optional[str]) -> Optional[set[str]]:
    """`slack:U1,whatsapp:4670…` → the set; None (no filter) when the request named nobody."""
    if me is None:
        return None
    return {x.strip().lower() for x in me.split(",") if x.strip()}


def identity(platform: object, user_id: object) -> str:
    return f"{platform}:{user_id}".lower()


def _jobs(home: Path) -> dict[str, dict]:
    path = home / "cron" / "jobs.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    jobs = data.get("jobs", []) if isinstance(data, dict) else data
    if isinstance(jobs, dict):
        jobs = [{"id": k, **v} for k, v in jobs.items() if isinstance(v, dict)]
    return {str(j.get("id")): j for j in jobs if isinstance(j, dict) and j.get("id")}


def _sessions(home: Path) -> dict[str, tuple[str, str, str]]:
    """session id → (source, sender id, parent id), from Hermes's own database. Empty on a Hermes that
    doesn't record senders: there, every chat counts as the person's."""
    db = search._hermes(home)
    if db is None:
        return {}
    try:
        cols = {r[1] for r in db.execute("PRAGMA table_info(sessions)")}
        if "user_id" not in cols:
            return {}
        parent = "parent_session_id" if "parent_session_id" in cols else "''"
        return {r[0]: (r[1] or "", r[2] or "", r[3] or "")
                for r in db.execute(f"SELECT id, source, user_id, {parent} FROM sessions")}
    finally:
        db.close()


def _cron_job(session_id: str) -> str:
    """`cron_<job id>_<YYYYmmdd>_<HHMMSS>` → the job's id."""
    parts = session_id.split("_")
    return "_".join(parts[1:-2]) if len(parts) >= 4 and parts[0] == CRON else ""


class Mine:
    def __init__(self, home: Path, me: Iterable[str]):
        self.me = {x.lower() for x in me}
        self._sessions = _sessions(home)
        self._jobs = _jobs(home)

    def job(self, job: dict) -> bool:
        """A routine made in someone else's chat is theirs; one made here, in Domo or in the terminal, the person's."""
        origin = job.get("origin") or (self._jobs.get(str(job.get("id"))) or {}).get("origin") or {}
        if not isinstance(origin, dict) or origin.get("platform") in OWN_SOURCES or not origin.get("user_id"):
            return True
        return identity(origin.get("platform"), origin["user_id"]) in self.me

    def session(self, session_id: str, _depth: int = 0) -> bool:
        if not session_id or _depth > 8:
            return True
        if session_id.startswith(CRON + "_"):
            job = self._jobs.get(_cron_job(session_id))
            return self.job(job) if job else True
        row = self._sessions.get(session_id)
        if row is None:
            return True   # gone, or older than the Hermes that records senders
        source, user, parent = row
        if parent:
            return self.session(parent, _depth + 1)
        if source == CRON or source in OWN_SOURCES:
            return True
        return not user or identity(source, user) in self.me

    def item(self, item: dict) -> bool:
        return self.session(item.get("session_id") or "")


def people(home: Path, limit: int = 30) -> list[dict]:
    """Who has started chats on this Hermes, most chats first, to pick which of them is you."""
    db = search._hermes(home)
    if db is None:
        return []
    try:
        cols = {r[1] for r in db.execute("PRAGMA table_info(sessions)")}
        if "user_id" not in cols:
            return []
        origin = "origin_json" if "origin_json" in cols else "NULL"
        rows = db.execute(
            f"SELECT source, user_id, {origin} AS origin, COUNT(*) AS chats, MAX(COALESCE(last_activity_at, started_at)) AS last"
            f" FROM sessions WHERE COALESCE(user_id, '') != '' AND source NOT IN ({','.join('?' * (len(OWN_SOURCES) + 1))})"
            " GROUP BY source, user_id ORDER BY chats DESC, last DESC LIMIT ?", (*OWN_SOURCES, CRON, limit)).fetchall()
    finally:
        db.close()
    out = []
    for r in rows:
        try:
            name = (json.loads(r["origin"] or "{}") or {}).get("user_name") or ""
        except ValueError:
            name = ""
        out.append({"id": identity(r["source"], r["user_id"]), "platform": r["source"], "user_id": r["user_id"],
                    "name": name, "chats": r["chats"], "last": r["last"]})
    return out
