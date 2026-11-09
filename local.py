"""Sessions from the person's other machines: Claude Code and Cursor, as Domo Link sends them.

Link (apps/link) runs on the MacBook, tails every transcript and pushes the new lines here with the hook
events and a snapshot of what's running. This module keeps, per profile, in ``plugin-data/domo/local.db``:

- an archive: every transcript, compressed, as written (Claude Code deletes its own after 30 days);
- an index of what was typed, said and run (not tool output), for search;
- the machines, what each has running now, and the hook events, as the trace of how the person works.

Every request is idempotent: Link sends a byte range per file, and the answer says how far this side holds
each file, so a dropped answer or a wiped store just makes Link send again from there.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import zlib
from pathlib import Path
from typing import Optional

from . import transcripts

PROTOCOL = 1
SOURCES = ("claude-code", "cursor")
ONLINE_FOR = 60          # a machine that synced this recently is there (Link syncs at least every 20 s)
HOOK_STATUS_FOR = 15 * 60
CURSOR_WORKING_FOR = 120
_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,100}$")
_STATUS = {"busy": "working", "working": "working", "waiting": "waiting", "idle": "idle"}
_HOOK_STATUS = {"UserPromptSubmit": "working", "PostToolUse": "working", "PostToolUseFailure": "working",
                "PreCompact": "working", "SubagentStop": "working", "Notification": "waiting", "Stop": "idle",
                "SessionStart": "idle", "SessionEnd": "ended"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS machines (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL DEFAULT '',
    hostname      TEXT NOT NULL DEFAULT '',
    platform      TEXT NOT NULL DEFAULT '',
    link_version  TEXT NOT NULL DEFAULT '',
    first_seen    REAL NOT NULL,
    last_seen     REAL NOT NULL,
    live          TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS files (
    source      TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    machine_id  TEXT NOT NULL,
    path        TEXT NOT NULL DEFAULT '',
    cwd         TEXT NOT NULL DEFAULT '',
    repo        TEXT NOT NULL DEFAULT 'null',
    meta        TEXT NOT NULL DEFAULT '{}',
    size        INTEGER NOT NULL DEFAULT 0,
    first_seen  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    info        TEXT NOT NULL DEFAULT '{}',
    info_size   INTEGER NOT NULL DEFAULT -1,
    thread_id   TEXT NOT NULL DEFAULT '',
    stitched    INTEGER NOT NULL DEFAULT -1,
    PRIMARY KEY (source, session_id)
);
CREATE INDEX IF NOT EXISTS files_updated ON files(updated_at);
CREATE TABLE IF NOT EXISTS chunks (
    source      TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    start       INTEGER NOT NULL,
    stop        INTEGER NOT NULL,
    data        BLOB NOT NULL,
    PRIMARY KEY (source, session_id, start)
);
CREATE TABLE IF NOT EXISTS events (
    machine_id  TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    at          REAL NOT NULL,
    source      TEXT NOT NULL DEFAULT '',
    event       TEXT NOT NULL DEFAULT '',
    session_id  TEXT NOT NULL DEFAULT '',
    cwd         TEXT NOT NULL DEFAULT '',
    idle        REAL,
    payload     TEXT NOT NULL DEFAULT '{}',
    -- with the time, so a Link whose outbox was reset (seq from 1 again) is new data, and a resend is not
    PRIMARY KEY (machine_id, seq, at)
);
CREATE INDEX IF NOT EXISTS events_session ON events(session_id, at);
CREATE VIRTUAL TABLE IF NOT EXISTS words USING fts5(text, role UNINDEXED, source UNINDEXED, session_id UNINDEXED);
CREATE TABLE IF NOT EXISTS threads (
    id           TEXT PRIMARY KEY,
    machine_id   TEXT NOT NULL,
    place        TEXT NOT NULL,
    branch       TEXT NOT NULL DEFAULT '',
    repo         TEXT NOT NULL DEFAULT 'null',
    started_at   REAL NOT NULL,
    last_active  REAL NOT NULL,
    sessions     TEXT NOT NULL DEFAULT '[]',
    size         INTEGER NOT NULL DEFAULT 0,
    digest       TEXT NOT NULL DEFAULT '{}',
    digest_size  INTEGER NOT NULL DEFAULT -1,
    digested_at  REAL
);
CREATE INDEX IF NOT EXISTS threads_active ON threads(last_active);
"""


class LocalError(ValueError):
    pass


def db_path(profile_home: Path) -> Path:
    return Path(profile_home) / "plugin-data" / "domo" / "local.db"


def _j(text: str, default):
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return default
    return value if value is not None else default


class LocalStore:
    def __init__(self, path: Path, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA busy_timeout=5000")
            self._db.executescript(_SCHEMA)
        self._parsed: dict[tuple[str, str], tuple[int, dict]] = {}

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _q(self, sql: str, params=()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    # ---- what Link sends ------------------------------------------------------------------------

    def ingest(self, body: dict) -> dict:
        """One sync from Link: the machine and what it runs, new transcript bytes, hook events."""
        if not isinstance(body, dict):
            raise LocalError("expected a JSON object")
        machine = body.get("machine") or {}
        machine_id = str(machine.get("id") or "")
        if not _ID.fullmatch(machine_id):
            raise LocalError("machine.id is missing or malformed")
        now = self._clock()
        live = [l for l in body.get("live") or [] if isinstance(l, dict)]
        with self._lock:
            self._db.execute(
                "INSERT INTO machines (id, name, hostname, platform, link_version, first_seen, last_seen, live)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,"
                " hostname=excluded.hostname, platform=excluded.platform, link_version=excluded.link_version,"
                " last_seen=excluded.last_seen, live=excluded.live",
                (machine_id, str(machine.get("name") or "")[:80], str(machine.get("hostname") or "")[:80],
                 str(machine.get("platform") or "")[:20], str(machine.get("link_version") or "")[:20], now, now,
                 json.dumps(live[:200])))
        files = [self._take_file(machine_id, f, now) for f in body.get("files") or [] if isinstance(f, dict)]
        acked = self._take_events(machine_id, [e for e in body.get("events") or [] if isinstance(e, dict)])
        return {"ok": True, "protocol": PROTOCOL, "files": [f for f in files if f], "events_acked": acked, "config": {}}

    def _take_file(self, machine_id: str, f: dict, now: float) -> Optional[dict]:
        source, session_id = str(f.get("source") or ""), str(f.get("session_id") or "")
        if source not in SOURCES or not _ID.fullmatch(session_id):
            return None
        try:
            start, stop = int(f.get("offset", 0)), int(f.get("end", 0))
        except (TypeError, ValueError):
            return None
        data = f.get("data") if isinstance(f.get("data"), str) else ""
        repo = f.get("repo") if isinstance(f.get("repo"), dict) else None
        meta = f.get("meta") if isinstance(f.get("meta"), dict) else {}
        cwd = str(f.get("cwd") or meta.get("cwd") or "")
        with self._lock:
            row = self._db.execute("SELECT size FROM files WHERE source=? AND session_id=?", (source, session_id)).fetchone()
            held = row["size"] if row else 0
            if f.get("reset") and start == 0:
                self._db.execute("DELETE FROM chunks WHERE source=? AND session_id=?", (source, session_id))
                self._db.execute("DELETE FROM words WHERE source=? AND session_id=?", (source, session_id))
                held = 0
            if row is None:
                self._db.execute(
                    "INSERT INTO files (source, session_id, machine_id, path, cwd, repo, meta, size, first_seen, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
                    (source, session_id, machine_id, str(f.get("path") or ""), cwd, json.dumps(repo), json.dumps(meta), now, now))
            if start != held or stop < start:
                return {"source": source, "session_id": session_id, "offset": held}
            self._db.execute("BEGIN IMMEDIATE")
            try:
                if stop > start:
                    self._db.execute("INSERT OR REPLACE INTO chunks (source, session_id, start, stop, data) VALUES (?, ?, ?, ?, ?)",
                                     (source, session_id, start, stop, zlib.compress(data.encode(), 6)))
                    self._db.executemany("INSERT INTO words (text, role, source, session_id) VALUES (?, ?, ?, ?)",
                                         [(text, role, source, session_id) for role, text in transcripts.searchable(data)])
                self._db.execute(
                    "UPDATE files SET machine_id=?, path=?, cwd=CASE WHEN ?='' THEN cwd ELSE ? END,"
                    " repo=CASE WHEN ?='null' THEN repo ELSE ? END, meta=CASE WHEN ?='{}' THEN meta ELSE ? END,"
                    " size=?, updated_at=CASE WHEN ?>? THEN ? ELSE updated_at END WHERE source=? AND session_id=?",
                    (machine_id, str(f.get("path") or ""), cwd, cwd, json.dumps(repo), json.dumps(repo),
                     json.dumps(meta), json.dumps(meta), stop, stop, start, now, source, session_id))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return {"source": source, "session_id": session_id, "offset": stop}

    def _take_events(self, machine_id: str, events: list[dict]) -> int:
        top = 0
        rows = []
        for e in events:
            try:
                seq = int(e.get("seq"))
            except (TypeError, ValueError):
                continue
            top = max(top, seq)
            payload = e.get("payload") if isinstance(e.get("payload"), dict) else {}
            idle = e.get("idle_seconds")
            rows.append((machine_id, seq, float(e.get("at") or self._clock()), str(e.get("source") or "claude-code")[:20],
                         str(e.get("event") or payload.get("hook_event_name") or "")[:40],
                         str(e.get("session_id") or payload.get("session_id") or "")[:100],
                         str(e.get("cwd") or payload.get("cwd") or "")[:500],
                         float(idle) if isinstance(idle, (int, float)) else None, json.dumps(payload)[:20000]))
        if rows:
            with self._lock:
                self._db.executemany("INSERT OR IGNORE INTO events (machine_id, seq, at, source, event, session_id, cwd, idle, payload)"
                                     " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        return top

    # ---- reading --------------------------------------------------------------------------------

    def machines(self) -> list[dict]:
        now = self._clock()
        return [{"id": r["id"], "name": r["name"] or r["hostname"] or "Mac", "hostname": r["hostname"],
                 "platform": r["platform"], "link_version": r["link_version"], "last_seen": r["last_seen"],
                 "online": now - r["last_seen"] < ONLINE_FOR, "running": len(_j(r["live"], []))}
                for r in self._q("SELECT * FROM machines ORDER BY last_seen DESC")]

    def text(self, source: str, session_id: str) -> str:
        rows = self._q("SELECT data FROM chunks WHERE source=? AND session_id=? ORDER BY start", (source, session_id))
        return "".join(zlib.decompress(r["data"]).decode(errors="replace") for r in rows)

    def _file(self, session_id: str) -> Optional[sqlite3.Row]:
        rows = self._q("SELECT * FROM files WHERE session_id=? ORDER BY updated_at DESC LIMIT 1", (session_id,))
        return rows[0] if rows else None

    def parsed(self, row: sqlite3.Row) -> dict:
        """The session read whole, cached by its size, so a live session is re-read only when it grew."""
        key = (row["source"], row["session_id"])
        hit = self._parsed.get(key)
        if hit and hit[0] == row["size"]:
            return hit[1]
        result = transcripts.parse(self.text(*key))
        if len(self._parsed) > 64:
            self._parsed.pop(next(iter(self._parsed)))
        self._parsed[key] = (row["size"], result)
        return result

    def info(self, row: sqlite3.Row) -> dict:
        """What a session says about itself (title, prompts, commits…), kept with the row once read."""
        if row["info_size"] == row["size"]:
            return _j(row["info"], {})
        info = self.parsed(row)["info"]
        info["message_count"] = len(self.parsed(row)["messages"])
        with self._lock:
            self._db.execute("UPDATE files SET info=?, info_size=? WHERE source=? AND session_id=?",
                             (json.dumps(info), row["size"], row["source"], row["session_id"]))
        return info

    def stale(self, limit: int = 50) -> list[sqlite3.Row]:
        return self._q("SELECT * FROM files WHERE info_size != size ORDER BY updated_at DESC LIMIT ?", (limit,))

    def _status(self, row: sqlite3.Row, machine: Optional[sqlite3.Row], now: float) -> tuple[str, str]:
        online = bool(machine) and now - machine["last_seen"] < ONLINE_FOR
        if not online:
            return "ended", ""
        for entry in _j(machine["live"], []):
            if entry.get("session_id") == row["session_id"]:
                return _STATUS.get(str(entry.get("status") or ""), "working"), str(entry.get("waiting_for") or "")
        if row["source"] == "cursor":
            return ("working" if now - row["updated_at"] < CURSOR_WORKING_FOR else "ended"), ""
        last = self._q("SELECT event, payload, at FROM events WHERE session_id=? ORDER BY at DESC LIMIT 1", (row["session_id"],))
        if last and now - last[0]["at"] < HOOK_STATUS_FOR and _HOOK_STATUS.get(last[0]["event"]) in ("working", "waiting"):
            status = _HOOK_STATUS[last[0]["event"]]
            return status, (_j(last[0]["payload"], {}).get("message") or "") if status == "waiting" else ""
        return "ended", ""

    def describe(self, row: sqlite3.Row, machines: Optional[dict] = None, now: Optional[float] = None) -> dict:
        now = now if now is not None else self._clock()
        if machines is None:
            machines = {m["id"]: m for m in self._q("SELECT * FROM machines")}
        machine = machines.get(row["machine_id"])
        info = self.info(row)
        meta = _j(row["meta"], {})
        repo = _j(row["repo"], None)
        status, waiting_for = self._status(row, machine, now)
        title = (meta.get("title") if row["source"] == "cursor" else "") or info.get("title") or "Untitled session"
        return {
            "id": row["session_id"], "source": row["source"], "machine": row["machine_id"],
            "machine_name": (machine["name"] or machine["hostname"]) if machine else "",
            "machine_online": bool(machine) and now - machine["last_seen"] < ONLINE_FOR,
            "title": title[:120], "cwd": row["cwd"] or info.get("cwd") or "",
            "repo": repo, "branch": (repo or {}).get("branch") or info.get("branch") or "",
            "started_at": info.get("started_at") or row["first_seen"], "last_active": info.get("last_at") or row["updated_at"],
            "message_count": info.get("message_count", 0), "prompts": info.get("prompts", 0),
            "preview": " ".join((info.get("last_text") or info.get("last_prompt") or "").split())[:200],
            "status": status, "waiting_for": waiting_for[:200], "model": info.get("model") or "",
            "cost_usd": info.get("cost_usd"), "commits": len(info.get("commits") or []), "prs": info.get("prs") or [],
            "thread_id": row["thread_id"], "size": row["size"],
        }

    def sessions(self, *, limit: int = 50, machine: Optional[str] = None, since: float = 0) -> list[dict]:
        sql, params = "SELECT * FROM files WHERE updated_at >= ?", [since]
        if machine:
            sql += " AND machine_id = ?"
            params.append(machine)
        rows = self._q(sql + " ORDER BY updated_at DESC LIMIT ?", (*params, limit))
        machines = {m["id"]: m for m in self._q("SELECT * FROM machines")}
        now = self._clock()
        out = [self.describe(r, machines, now) for r in rows]
        # what needs the person first, then what's running, then the rest newest first (as listed)
        rank = {"waiting": 0, "working": 1, "idle": 2}
        return sorted(out, key=lambda s: (rank.get(s["status"], 3), -(s["last_active"] or 0)))

    def session(self, session_id: str) -> Optional[dict]:
        row = self._file(session_id)
        return self.describe(row) if row else None

    def messages(self, session_id: str) -> Optional[dict]:
        row = self._file(session_id)
        if row is None:
            return None
        return {"session": self.describe(row), "messages": self.parsed(row)["messages"], "revision": row["size"]}

    def revision(self, session_id: str) -> Optional[int]:
        row = self._file(session_id)
        return row["size"] if row else None

    def events(self, *, since: float = 0, session_id: str = "", limit: int = 500) -> list[dict]:
        sql, params = "SELECT * FROM events WHERE at >= ?", [since]
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        rows = self._q(sql + " ORDER BY at DESC LIMIT ?", (*params, limit))
        return [{"machine": r["machine_id"], "seq": r["seq"], "at": r["at"], "source": r["source"], "event": r["event"],
                 "session_id": r["session_id"], "cwd": r["cwd"], "idle_seconds": r["idle"], "payload": _j(r["payload"], {})}
                for r in rows]

    def search(self, q: str, limit: int = 20) -> list[dict]:
        """Sessions by their words: one row per session, its best line."""
        words = [w for w in re.findall(r"\w+", q, flags=re.U) if len(w) > 1]
        if not words:
            return []
        query = " ".join(f'"{w}"' for w in words)
        try:
            rows = self._q("SELECT session_id, source, role, text, bm25(words) AS rank FROM words WHERE words MATCH ?"
                           " ORDER BY rank LIMIT 400", (query,))
        except sqlite3.OperationalError:
            return []
        best: dict[str, sqlite3.Row] = {}
        for r in rows:
            best.setdefault(r["session_id"], r)
        out = []
        for session_id, r in list(best.items())[:limit]:
            s = self.session(session_id)
            if s:
                out.append({**s, "snippet": _snippet(r["text"], words), "role": r["role"]})
        return out


def _snippet(text: str, words: list[str], width: int = 200) -> str:
    flat = " ".join(text.split())
    low = flat.lower()
    at = min((low.find(w.lower()) for w in words if low.find(w.lower()) >= 0), default=0)
    start = max(0, at - width // 3)
    piece = flat[start:start + width]
    return ("…" if start else "") + piece + ("…" if start + width < len(flat) else "")


_stores: dict[str, LocalStore] = {}
_stores_lock = threading.Lock()


def store_for(profile_home: Path) -> LocalStore:
    path = str(db_path(profile_home))
    with _stores_lock:
        if path not in _stores:
            _stores[path] = LocalStore(Path(path))
        return _stores[path]
