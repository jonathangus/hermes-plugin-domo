"""Domo's store: decisions, to-dos, receipts, runs, briefs, facts and spaces for one Hermes profile.

One SQLite file per profile (``<profile home>/plugin-data/domo/domo.db``). Items share one table
with a JSON ``data`` column, so adding a field never needs a migration. Every write appends to
``changes`` so clients can poll ``since=<cursor>`` instead of re-reading everything. A space groups
chats, routines (Hermes cron jobs), files and items; items still name theirs in ``space``.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import unicodedata
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

KINDS = ("decision", "todo", "receipt", "run", "brief", "fact", "suggestion")

# Statuses a client may filter on; "open" items are the ones still asking for attention.
OPEN_STATUSES = {
    "decision": ("open",),
    "todo": ("open",),
    "run": ("running", "stalled"),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    space       TEXT NOT NULL DEFAULT '',
    title       TEXT NOT NULL,
    body        TEXT NOT NULL DEFAULT '',
    data        TEXT NOT NULL DEFAULT '{}',
    session_id  TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS items_kind_status ON items(kind, status);
CREATE TABLE IF NOT EXISTS changes (
    seq      INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id  TEXT NOT NULL,
    at       REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT '',
    space       TEXT NOT NULL DEFAULT '',
    asked       INTEGER NOT NULL DEFAULT 0,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS spaces (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    icon         TEXT NOT NULL DEFAULT '',
    color        TEXT NOT NULL DEFAULT '',
    description  TEXT NOT NULL DEFAULT '',
    prompt       TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    position     INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS session_spaces (
    session_id  TEXT PRIMARY KEY,
    space_id    TEXT NOT NULL,
    updated_at  REAL NOT NULL
);
-- Memory, in three layers (after DeepTutor): what happened, what's known, and what it adds up to.
CREATE TABLE IF NOT EXISTS events (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    at     REAL NOT NULL,
    kind   TEXT NOT NULL,
    text   TEXT NOT NULL,
    ref    TEXT NOT NULL DEFAULT '',
    space  TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS instincts (
    id          TEXT PRIMARY KEY,
    pattern     TEXT NOT NULL,
    norm        TEXT NOT NULL UNIQUE,
    kind        TEXT NOT NULL DEFAULT 'habit',
    confidence  REAL NOT NULL,
    evidence    INTEGER NOT NULL DEFAULT 1,
    origin      TEXT NOT NULL DEFAULT 'inferred',
    space       TEXT NOT NULL DEFAULT '',
    prompt      TEXT NOT NULL DEFAULT '',
    hours       TEXT NOT NULL DEFAULT '',
    days        TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS synthesis (
    section     TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS suggestion_votes (
    key      TEXT NOT NULL,
    kind     TEXT NOT NULL,
    verdict  TEXT NOT NULL,
    at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS suggestion_votes_key ON suggestion_votes(key);
CREATE TABLE IF NOT EXISTS suggestion_snoozes (
    key    TEXT PRIMARY KEY,
    until  REAL NOT NULL
);
-- Which chats the person has read, on any device: in their own profile's store, for every agent they see
-- (`ours`, `jonathan@work` too). A chat is unread once it moves past what's here, or past `since` (the first
-- look) when it was never opened.
CREATE TABLE IF NOT EXISTS seen (
    agent       TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    at          REAL NOT NULL,
    PRIMARY KEY (agent, session_id)
);
CREATE TABLE IF NOT EXISTS seen_since (
    agent  TEXT PRIMARY KEY,
    since  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS archived_sessions (
    session_id   TEXT PRIMARY KEY,
    archived_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS job_spaces (
    job_id    TEXT PRIMARY KEY,
    space_id  TEXT NOT NULL
);
-- Spaces as tags: a chat (or a file, routine, item) can be in an area and in any topics, each tag saying
-- who put it there and how sure Domo was. state 'off' is a tag the person took away: Domo leaves it off.
CREATE TABLE IF NOT EXISTS space_links (
    space_id     TEXT NOT NULL,
    target_type  TEXT NOT NULL,
    target_id    TEXT NOT NULL,
    by           TEXT NOT NULL DEFAULT 'person',
    confidence   REAL,
    reason       TEXT NOT NULL DEFAULT '',
    state        TEXT NOT NULL DEFAULT 'on',
    at           REAL NOT NULL,
    PRIMARY KEY (space_id, target_type, target_id)
);
CREATE INDEX IF NOT EXISTS space_links_target ON space_links(target_type, target_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
    path        TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    mime        TEXT NOT NULL DEFAULT '',
    size        INTEGER NOT NULL DEFAULT 0,
    space_id    TEXT,
    session_id  TEXT,
    created_at  REAL NOT NULL
);
"""

SPACE_FIELDS = ("id", "name", "icon", "color", "description", "prompt", "created_at", "updated_at", "parent_id", "origin", "kind", "quiet", "keep_moving")
KINDS_OF_SPACE = ("area", "topic")
TARGETS = ("session", "file", "job", "item")
PALETTE = ("#4CC9F0", "#FF6B2C", "#3DD68C", "#FF2D8A", "#FFB020", "#2F5BFF")   # the brand spectrum (docs/BRAND.md §3.2)

# Seeded spaces: icon, description, and the prompt the agent gets when a chat starts in the space.
SPACE_PRESETS = {
    "work": ("briefcase", "Your job: projects, colleagues, meetings and deadlines.",
             "You're helping with work. Keep answers crisp and decision-ready, and track commitments as to-dos. "
             "Ask before sending anything to colleagues or clients."),
    "coding": ("code", "Code, repos and long coding runs on this Mac.",
               "You're pairing on code. Work in the repo the person names, on a branch; track long jobs with domo_run; "
               "ask before pushing or merging.\n"
               "- Think first: say what you assume and what you'll change before you change it; ask when it's ambiguous.\n"
               "- Keep it simple: the smallest change that does the job, in the code's own style; no extras nobody asked for.\n"
               "- Touch only what the task needs; leave unrelated code alone, and say what you noticed instead.\n"
               "- Done means proven: run the tests or the program and show the result before you say it works.\n"
               "Method, when it fits: brainstorming then writing-plans for anything new, systematic-debugging for bugs, "
               "test-driven-development for logic, verification-before-completion before you report, "
               "to-questionnaire when they must choose, handoff when the chat gets long, archify to explain a system, "
               "impeccable for interface work."),
    "personal": ("leaf", "Health, errands, money and everything that's just yours.",
                 "You're helping with personal life. Be warm and brief, keep private things private, "
                 "and turn loose ends into to-dos."),
    "apartment hunt": ("house", "Listings, viewings, bids and the paperwork of finding a new home.",
                       "You're helping the two of them find a new apartment. Compare listings with what you know "
                       "they want and put viewings and bids up as decisions. Never contact agents or bid without asking."),
    "daughter": ("child", "School, health, activities and everything about our daughter.",
                 "You're helping with their daughter's life: school, health and activities. Keep both parents in "
                 "the loop and note dates as to-dos. Treat her details as private."),
    "trips": ("plane", "Travel plans, bookings and itineraries.",
              "You're planning trips. Gather options with prices and times and propose one plan. "
              "Ask before booking or paying for anything."),
    "home": ("house", "The household: bills, repairs, chores and shopping.",
             "You're running the household. Keep track of bills, repairs and chores, and split the work fairly. "
             "Ask before paying or ordering anything."),
    "family": ("people", "The extended family: birthdays, visits and plans together.",
               "You're helping the extended family coordinate. Keep replies short and friendly and sum plans up "
               "clearly for everyone. Ask before committing anyone to anything."),
}


SYNTHESIS_SECTIONS = ("who", "now", "preferences")


class _Rows(list):
    """A statement's rows, already fetched: safe to read after the lock is released."""

    def __init__(self, rows, lastrowid, rowcount):
        super().__init__(rows)
        self.lastrowid, self.rowcount = lastrowid, rowcount

    def fetchone(self):
        return self[0] if self else None

    def fetchall(self):
        return list(self)


class _Locked:
    """sqlite3.Connection, one statement at a time across threads."""

    def __init__(self, conn: sqlite3.Connection, lock):
        self._conn, self._lock = conn, lock

    def execute(self, sql: str, params=()) -> _Rows:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return _Rows(cur.fetchall(), cur.lastrowid, cur.rowcount)

    def executemany(self, sql: str, seq) -> _Rows:
        with self._lock:
            cur = self._conn.executemany(sql, seq)
            return _Rows([], cur.lastrowid, cur.rowcount)

    def executescript(self, script: str):
        with self._lock:
            return self._conn.executescript(script)

    def close(self):
        self._conn.close()


class StoreError(ValueError):
    """A request the store refuses (unknown item, wrong kind, bad status)."""


def store_path(profile_home: Path) -> Path:
    return Path(profile_home) / "plugin-data" / "domo" / "domo.db"


class Store:
    def __init__(self, path: Path, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        # One connection, used from request threads and background ones (the filer, the indexer): every
        # statement holds the lock, and its rows are read before it lets go.
        self._lock = threading.RLock()
        conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        self._db = _Locked(conn, self._lock)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.executescript(_SCHEMA)
        for sql in (
            # stores from before facts knew whether they were said or guessed
            "ALTER TABLE facts ADD COLUMN origin TEXT NOT NULL DEFAULT 'stated'",
            # sub-spaces (Work › Baltazar), and whether the person or Domo made a space
            "ALTER TABLE spaces ADD COLUMN parent_id TEXT",
            "ALTER TABLE spaces ADD COLUMN origin TEXT NOT NULL DEFAULT 'person'",
            # who filed a chat, how sure, and why; the person's filing is never moved by Domo
            "ALTER TABLE session_spaces ADD COLUMN filed_by TEXT NOT NULL DEFAULT 'person'",
            "ALTER TABLE session_spaces ADD COLUMN confidence REAL",
            "ALTER TABLE session_spaces ADD COLUMN reason TEXT NOT NULL DEFAULT ''",
            # areas (a few, the person's) and topics (many, mostly Domo's); quiet ones fade from view
            "ALTER TABLE spaces ADD COLUMN kind TEXT NOT NULL DEFAULT 'area'",
            "ALTER TABLE spaces ADD COLUMN quiet INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE spaces ADD COLUMN aliases TEXT NOT NULL DEFAULT '[]'",
            # a space Domo checks in on by itself, to keep its work moving (checkin.py)
            "ALTER TABLE spaces ADD COLUMN keep_moving INTEGER NOT NULL DEFAULT 0",
        ):
            try:
                self._db.execute(sql)
            except sqlite3.OperationalError:
                pass   # already there
        self._db.execute("CREATE TABLE IF NOT EXISTS filing_attempts (session_id TEXT PRIMARY KEY, at REAL NOT NULL, revision INTEGER NOT NULL DEFAULT 0)")
        self._migrate_to_tags()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    # ---- items -------------------------------------------------------------------------------

    def create(self, kind: str, title: str, *, body: str = "", space: str = "", status: str = "",
               data: Optional[dict] = None, session_id: str = "") -> dict:
        if kind not in KINDS or kind == "fact":
            raise StoreError(f"unknown item kind {kind!r}")
        title = (title or "").strip()
        if not title:
            raise StoreError("title is required")
        now = self._clock()
        item_id = f"{kind[:3]}_{uuid.uuid4().hex[:12]}"
        status = status or _default_status(kind)
        with self._tx() as db:
            db.execute(
                "INSERT INTO items (id, kind, status, space, title, body, data, session_id, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (item_id, kind, status, space or "", title, body or "", json.dumps(data or {}),
                 session_id or "", now, now),
            )
            db.execute("INSERT INTO changes (item_id, at) VALUES (?, ?)", (item_id, now))
        return self.get(item_id)

    def get(self, item_id: str) -> dict:
        row = self._db.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            raise StoreError(f"no item {item_id!r}")
        return _row(row)

    def update(self, item_id: str, *, status: Optional[str] = None, title: Optional[str] = None,
               body: Optional[str] = None, data: Optional[dict] = None) -> dict:
        """Patch an item. ``data`` is merged into the existing data, not replaced."""
        current = self.get(item_id)
        merged = {**current["data"], **(data or {})}
        now = self._clock()
        with self._tx() as db:
            db.execute(
                "UPDATE items SET status = ?, title = ?, body = ?, data = ?, updated_at = ? WHERE id = ?",
                (status or current["status"], title if title is not None else current["title"],
                 body if body is not None else current["body"], json.dumps(merged), now, item_id),
            )
            db.execute("INSERT INTO changes (item_id, at) VALUES (?, ?)", (item_id, now))
        return self.get(item_id)

    def list(self, kind: Optional[str] = None, *, status: Optional[str] = None, space: Optional[str] = None,
             since: float = 0.0, limit: int = 100) -> list[dict]:
        sql, args = "SELECT * FROM items WHERE updated_at >= ?", [since]
        if kind:
            sql += " AND kind = ?"; args.append(kind)
        if status == "open" and kind in OPEN_STATUSES:
            marks = ",".join("?" for _ in OPEN_STATUSES[kind])
            sql += f" AND status IN ({marks})"; args.extend(OPEN_STATUSES[kind])
        elif status:
            sql += " AND status = ?"; args.append(status)
        if space:
            sql += " AND space = ?"; args.append(space)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"; args.append(max(1, min(limit, 500)))
        return [_row(r) for r in self._db.execute(sql, args).fetchall()]

    def cursor(self) -> int:
        row = self._db.execute("SELECT MAX(seq) FROM changes").fetchone()
        return int(row[0] or 0)

    def changed_since(self, seq: int) -> tuple[list[dict], int]:
        """Items changed after change ``seq``, newest state only, and the new cursor."""
        rows = self._db.execute(
            "SELECT DISTINCT item_id FROM changes WHERE seq > ? ORDER BY seq", (seq,)).fetchall()
        items = []
        for r in rows:
            try:
                items.append(self.get(r["item_id"]))
            except StoreError:
                continue
        return items, self.cursor()

    # ---- decisions ---------------------------------------------------------------------------

    def answer(self, decision_id: str, choice: str, *, note: str = "", by: str = "") -> dict:
        item = self.get(decision_id)
        if item["kind"] != "decision":
            raise StoreError(f"{decision_id} is a {item['kind']}, not a decision")
        if item["status"] != "open":
            raise StoreError(f"decision {decision_id} is already {item['status']}")
        options = item["data"].get("options") or []
        if options and choice not in options:
            raise StoreError(f"choice {choice!r} is not one of {options}")
        return self.update(decision_id, status="answered",
                           data={"choice": choice, "note": note, "answered_by": by,
                                 "answered_at": self._clock()})

    # ---- runs --------------------------------------------------------------------------------

    def run_beat(self, run_id: str, *, progress: Optional[float] = None, note: str = "",
                 cost_usd: Optional[float] = None) -> dict:
        item = self._require(run_id, "run")
        data: dict[str, Any] = {"last_beat_at": self._clock()}
        if progress is not None:
            data["progress"] = max(0.0, min(1.0, float(progress)))
        if note:
            data["note"] = note
        if cost_usd is not None:
            data["cost_usd"] = float(cost_usd)
        status = "running" if item["status"] == "stalled" else None
        return self.update(run_id, status=status, data=data)

    def run_finish(self, run_id: str, *, ok: bool, note: str = "", proof: str = "",
                   cost_usd: Optional[float] = None) -> dict:
        self._require(run_id, "run")
        data: dict[str, Any] = {"finished_at": self._clock(), "note": note, "proof": proof}
        if cost_usd is not None:
            data["cost_usd"] = float(cost_usd)
        if ok:
            data["progress"] = 1.0
        return self.update(run_id, status="done" if ok else "failed", data=data)

    def mark_stalled(self, now: Optional[float] = None) -> list[dict]:
        """Running runs whose heartbeat is older than their allowed silence become 'stalled'."""
        now = now if now is not None else self._clock()
        stalled = []
        for run in self.list("run", status="running", limit=500):
            limit_s = float(run["data"].get("stall_after_minutes") or 30) * 60
            last = float(run["data"].get("last_beat_at") or run["created_at"])
            if now - last > limit_s:
                stalled.append(self.update(run["id"], status="stalled"))
        return stalled

    # ---- briefs ------------------------------------------------------------------------------

    def latest_brief(self) -> Optional[dict]:
        briefs = self.list("brief", limit=1)
        return briefs[0] if briefs else None

    # ---- facts -------------------------------------------------------------------------------

    def set_fact(self, key: str, value: str, *, source: str = "", space: str = "", origin: str = "stated") -> dict:
        """``origin``: 'stated' when the person said it, 'inferred' when Domo worked it out (after Graphify)."""
        key = _fact_key(key)
        if not value:
            raise StoreError("value is required")
        if origin not in ("stated", "inferred"):
            raise StoreError("origin is stated or inferred")
        with self._tx() as db:
            db.execute(
                "INSERT INTO facts (key, value, source, space, origin, updated_at) VALUES (?,?,?,?,?,?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value, source=excluded.source,"
                " space=excluded.space, origin=excluded.origin, updated_at=excluded.updated_at",
                (key, value, source, space, origin, self._clock()))
        return self.get_fact(key, count=False)

    def forget_fact(self, key: str) -> None:
        with self._tx() as db:
            db.execute("DELETE FROM facts WHERE key = ?", (_fact_key(key),))

    # ---- memory: events, instincts, synthesis -----------------------------------------------

    def note(self, kind: str, text: str, *, ref: str = "", space: str = "") -> dict:
        """The trace: append-only, so a later consolidation can always go back to what happened."""
        text = " ".join((text or "").split())[:400]
        if not text:
            raise StoreError("text is required")
        with self._tx() as db:
            cur = db.execute("INSERT INTO events (at, kind, text, ref, space) VALUES (?,?,?,?,?)", (self._clock(), kind, text, ref, space))
        return dict(self._db.execute("SELECT * FROM events WHERE id = ?", (cur.lastrowid,)).fetchone())

    def events(self, *, since: float = 0, limit: int = 100) -> list[dict]:
        rows = self._db.execute("SELECT * FROM events WHERE at > ? ORDER BY id DESC LIMIT ?", (since, limit)).fetchall()
        return [dict(r) for r in rows]

    def learn(self, pattern: str, *, kind: str = "habit", origin: str = "inferred", space: str = "", prompt: str = "",
              hours: str = "", days: str = "") -> dict:
        """An instinct (after ECC): a pattern with a confidence that evidence raises and feedback lowers.
        Seen again, it gets stronger; told by the person, it starts strong."""
        pattern = " ".join((pattern or "").split())[:240]
        if not pattern:
            raise StoreError("pattern is required")
        if kind not in ("habit", "preference", "avoid"):
            raise StoreError("kind is habit, preference or avoid")
        norm = _fact_key(pattern)
        now = self._clock()
        row = self._db.execute("SELECT * FROM instincts WHERE norm = ?", (norm,)).fetchone()
        with self._tx() as db:
            if row:
                origin = "stated" if "stated" in (origin, row["origin"]) else "inferred"
                db.execute("UPDATE instincts SET confidence = MIN(0.95, confidence + 0.1), evidence = evidence + 1, origin = ?,"
                           " prompt = COALESCE(NULLIF(?, ''), prompt), hours = COALESCE(NULLIF(?, ''), hours),"
                           " days = COALESCE(NULLIF(?, ''), days), space = COALESCE(NULLIF(?, ''), space), updated_at = ? WHERE id = ?",
                           (origin, prompt, hours, days, space, now, row["id"]))
                iid = row["id"]
            else:
                iid = f"in_{uuid.uuid4().hex[:12]}"
                db.execute("INSERT INTO instincts (id, pattern, norm, kind, confidence, origin, space, prompt, hours, days, created_at, updated_at)"
                           " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                           (iid, pattern, norm, kind, 0.8 if origin == "stated" else 0.35, origin, space, prompt, hours, days, now, now))
        return self.instinct(iid)

    def instinct(self, instinct_id: str) -> dict:
        row = self._db.execute("SELECT * FROM instincts WHERE id = ?", (instinct_id,)).fetchone()
        if row is None:
            raise StoreError(f"no instinct {instinct_id!r}")
        return dict(row)

    def weaken(self, instinct_id: str, by: float = 0.2) -> dict:
        """Counter-evidence or a 'less like this'; below 0.1 it's gone."""
        current = self.instinct(instinct_id)
        with self._tx() as db:
            if current["confidence"] - by < 0.1:
                db.execute("DELETE FROM instincts WHERE id = ?", (instinct_id,))
                return {**current, "confidence": 0.0, "forgotten": True}
            db.execute("UPDATE instincts SET confidence = confidence - ?, updated_at = ? WHERE id = ?", (by, self._clock(), instinct_id))
        return self.instinct(instinct_id)

    def forget_instinct(self, instinct_id: str) -> None:
        with self._tx() as db:
            db.execute("DELETE FROM instincts WHERE id = ?", (instinct_id,))

    def instincts(self, *, at_least: float = 0.0) -> list[dict]:
        rows = self._db.execute("SELECT * FROM instincts WHERE confidence >= ? ORDER BY confidence DESC, updated_at DESC", (at_least,))
        return [dict(r) for r in rows]

    def set_synthesis(self, section: str, text: str) -> None:
        """The top layer: a short, rewritten picture of the person (who, now, preferences)."""
        if section not in SYNTHESIS_SECTIONS:
            raise StoreError(f"section is one of {', '.join(SYNTHESIS_SECTIONS)}")
        with self._tx() as db:
            if text.strip():
                db.execute("INSERT INTO synthesis (section, text, updated_at) VALUES (?,?,?) ON CONFLICT(section)"
                           " DO UPDATE SET text=excluded.text, updated_at=excluded.updated_at", (section, text.strip()[:1200], self._clock()))
            else:
                db.execute("DELETE FROM synthesis WHERE section = ?", (section,))

    def synthesis(self) -> dict[str, dict]:
        return {r["section"]: dict(r) for r in self._db.execute("SELECT * FROM synthesis")}

    def get_fact(self, key: str, *, count: bool = True) -> dict:
        """Look a fact up. ``count`` records that someone had to ask again (a memory miss)."""
        key = _fact_key(key)
        if count:
            with self._tx() as db:
                db.execute("UPDATE facts SET asked = asked + 1 WHERE key = ?", (key,))
        row = self._db.execute("SELECT * FROM facts WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise StoreError(f"no fact {key!r}")
        return dict(row)

    def facts(self) -> list[dict]:
        return [dict(r) for r in self._db.execute("SELECT * FROM facts ORDER BY key").fetchall()]

    # ---- spaces ------------------------------------------------------------------------------
    # A space's changes are logged as ``space:<id>`` so /changes pollers notice them too.

    def seed_spaces(self, names: list[str]) -> list[dict]:
        """Defaults for a new profile; never again once spaces have existed (deleting all stays deleted)."""
        if self._db.execute("SELECT 1 FROM spaces UNION ALL SELECT 1 FROM changes WHERE item_id LIKE 'space:%'"
                            " LIMIT 1").fetchone():
            return []
        seeded = []
        for i, name in enumerate(n.strip() for n in names if n.strip()):
            icon, description, prompt = SPACE_PRESETS.get(name.lower(), ("", "", ""))
            seeded.append(self.create_space(name, icon=icon, color=PALETTE[i % len(PALETTE)],
                                            description=description, prompt=prompt))
        return seeded

    def spaces(self, *, include_quiet: bool = True) -> list[dict]:
        """Areas first (in their order), then topics, most recently active first; each with its counts.
        Adopts space names items use that have no record yet."""
        known = {k for r in self._db.execute("SELECT id, name FROM spaces") for k in (r["id"], r["name"].lower())}
        for (name,) in self._db.execute("SELECT DISTINCT space FROM items WHERE space != ''").fetchall():
            if name.lower() not in known:
                known.update((self.create_space(name)["id"], name.lower()))
        rows = self._db.execute("SELECT * FROM spaces ORDER BY position, name").fetchall()
        last = {r[0]: r[1] for r in self._db.execute("SELECT space_id, MAX(at) FROM space_links WHERE state = 'on' GROUP BY space_id")}
        out = [{**_space(r), "counts": self._space_counts(r), "last_active": last.get(r["id"])} for r in rows if include_quiet or not r["quiet"]]
        areas = [x for x in out if x["kind"] == "area"]
        topics = sorted((x for x in out if x["kind"] == "topic"), key=lambda x: -(x["last_active"] or x["updated_at"] or 0))
        return areas + topics

    def children(self, space_id: str) -> list[dict]:
        """An area's topics: the ones whose chats are mostly in it."""
        rows = self._db.execute("SELECT * FROM spaces WHERE parent_id = ? AND kind = 'topic' ORDER BY name", (space_id,)).fetchall()
        return [{**_space(r), "counts": self._space_counts(r)} for r in rows]

    def lineage(self, space_id: str) -> list[dict]:
        """The space and its home area, outermost first: [Work, Baltazar]."""
        s = self.space(space_id)
        if not s:
            return []
        parent = self.space(s["parent_id"]) if s.get("parent_id") else None
        return [parent, s] if parent else [s]

    def space(self, space_id: str) -> Optional[dict]:
        row = self._db.execute("SELECT * FROM spaces WHERE id = ?", (space_id or "",)).fetchone()
        return _space(row) if row else None

    def _space_counts(self, row: sqlite3.Row) -> dict:
        n = lambda t: self._db.execute("SELECT COUNT(*) FROM space_links WHERE space_id = ? AND target_type = ? AND state = 'on'", (row["id"], t)).fetchone()[0]  # noqa: E731
        return {"chats": n("session"), "files": len(self.files(row["id"])), "routines": n("job"), "open": len(self.space_items(row["id"]))}

    def space_items(self, space_id: str) -> list[dict]:
        """Open to-dos and decisions in the space: tagged, filed under its name, or asked in one of its chats."""
        space = self.space(space_id)
        if not space:
            return []
        rows = self._db.execute(
            "SELECT * FROM items WHERE kind IN ('todo', 'decision') AND status = 'open' AND (space IN (?, ?)"
            " OR id IN (SELECT target_id FROM space_links WHERE space_id = ? AND target_type = 'item' AND state = 'on')"
            " OR (session_id != '' AND session_id IN (SELECT target_id FROM space_links WHERE space_id = ? AND target_type = 'session' AND state = 'on')))"
            " ORDER BY created_at DESC, rowid DESC", (space["name"], space["id"], space_id, space_id)).fetchall()
        return [_row(r) for r in rows]

    def create_space(self, name: str, *, icon: str = "", color: str = "", description: str = "",
                     prompt: str = "", parent_id: Optional[str] = None, origin: str = "person", kind: Optional[str] = None) -> dict:
        """An area (the person's few big parts of life) or a topic (a client, a trip, a repo); a topic's
        parent is its home area, a hint for grouping, not a container."""
        name = " ".join((name or "").split())
        if not name:
            raise StoreError("name is required")
        kind = kind or ("topic" if parent_id else "area")
        if kind not in KINDS_OF_SPACE:
            raise StoreError("kind is area or topic")
        parent = None
        if parent_id:
            parent = self.space(parent_id)
            if not parent:
                raise StoreError(f"no space {parent_id!r}")
            if parent["kind"] != "area":
                raise StoreError("a topic's home is an area")
        if self._space_named(name, kind):
            raise StoreError(f"space {name!r} already exists")
        if parent:   # a topic wears its area's look unless it's given one
            color = color or parent["color"]
            icon = icon or parent["icon"]
        now = self._clock()
        with self._tx() as db:
            base = _slug(name) if kind == "area" or not db.execute("SELECT 1 FROM spaces WHERE id = ?", (_slug(name),)).fetchone() else f"{parent_id or 'topic'}-{_slug(name)}"
            space_id, n = base, 1
            while db.execute("SELECT 1 FROM spaces WHERE id = ?", (space_id,)).fetchone():
                n += 1
                space_id = f"{base}-{n}"
            position = db.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM spaces").fetchone()[0]
            db.execute("INSERT INTO spaces (id, name, icon, color, description, prompt, created_at, updated_at, position, parent_id, origin, kind)"
                       " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (space_id, name, icon or "", color or "", description or "", prompt or "", now, now, position,
                        parent_id or None, origin if origin in ("person", "domo") else "person", kind))
            _touch(db, space_id, now)
        return self.space(space_id)

    def update_space(self, space_id: str, **fields: Optional[str]) -> dict:
        """Patch a space. A rename carries the items and facts filed under the old name along."""
        current = self.space(space_id)
        if not current:
            raise StoreError(f"no space {space_id!r}")
        patch = {k: v for k, v in fields.items() if v is not None and k in SPACE_FIELDS[1:6]}
        if fields.get("keep_moving") is not None:
            patch["keep_moving"] = int(bool(fields["keep_moving"]))
        if "name" in patch:
            patch["name"] = " ".join(patch["name"].split())
            if not patch["name"]:
                raise StoreError("name is required")
            other = self._space_named(patch["name"], current["kind"])
            if other and other["id"] != space_id:
                raise StoreError(f"space {patch['name']!r} already exists")
        now = self._clock()
        if patch and current.get("origin") == "domo":
            patch["origin"] = "person"   # touched by the person: theirs now
        with self._tx() as db:
            if patch:
                sets = ", ".join(f"{k} = ?" for k in patch)
                db.execute(f"UPDATE spaces SET {sets}, updated_at = ? WHERE id = ?", (*patch.values(), now, space_id))
            if patch.get("name", current["name"]) != current["name"]:
                _refile(db, current["name"], patch["name"], now)
            _touch(db, space_id, now)
        return self.space(space_id)

    def delete_space(self, space_id: str) -> None:
        """Remove a space and its tags. Chats, files and items stay, without it; an area's topics stay too."""
        current = self.space(space_id)
        if not current:
            raise StoreError(f"no space {space_id!r}")
        now = self._clock()
        with self._tx() as db:
            db.execute("UPDATE spaces SET parent_id = NULL WHERE parent_id = ?", (space_id,))
            db.execute("DELETE FROM spaces WHERE id = ?", (space_id,))
            db.execute("DELETE FROM space_links WHERE space_id = ?", (space_id,))
            db.execute("UPDATE files SET space_id = NULL WHERE space_id = ?", (space_id,))
            _refile(db, current["name"], "", now)
            _refile(db, space_id, "", now)
            _touch(db, space_id, now)

    def merge_spaces(self, source: str, into: str) -> dict:
        """Two names for one thing (Lissabon, Lisbon): everything tagged with one moves to the other, and
        the old name is kept as an alias so it's recognised next time."""
        a, b = self.space(source), self.space(into)
        if not a or not b or source == into:
            raise StoreError("merge two different spaces")
        now = self._clock()
        aliases = sorted({*self.aliases(into), a["name"], *self.aliases(source)} - {b["name"]})
        with self._tx() as db:
            for r in db.execute("SELECT * FROM space_links WHERE space_id = ?", (source,)).fetchall():
                db.execute("INSERT INTO space_links (space_id, target_type, target_id, by, confidence, reason, state, at) VALUES (?,?,?,?,?,?,?,?)"
                           " ON CONFLICT(space_id, target_type, target_id) DO NOTHING",
                           (into, r["target_type"], r["target_id"], r["by"], r["confidence"], r["reason"], r["state"], r["at"]))
            db.execute("UPDATE spaces SET aliases = ?, updated_at = ?, quiet = 0 WHERE id = ?", (json.dumps(aliases), now, into))
            db.execute("UPDATE files SET space_id = ? WHERE space_id = ?", (into, source))
            db.execute("DELETE FROM space_links WHERE space_id = ?", (source,))
            db.execute("DELETE FROM spaces WHERE id = ?", (source,))
            _refile(db, a["name"], b["name"], now)
            _touch(db, into, now)
            _touch(db, source, now)
        return self.space(into)

    def set_space_meta(self, space_id: str, *, parent_id: Optional[str] = ..., quiet: Optional[bool] = None) -> None:
        """What the stitcher keeps up to date: a topic's home area, and whether it has gone quiet."""
        sets, args = [], []
        if parent_id is not ...:
            sets.append("parent_id = ?"); args.append(parent_id)
        if quiet is not None:
            sets.append("quiet = ?"); args.append(1 if quiet else 0)
        if sets:
            with self._tx() as db:
                db.execute(f"UPDATE spaces SET {', '.join(sets)} WHERE id = ?", (*args, space_id))

    def aliases(self, space_id: str) -> list[str]:
        row = self._db.execute("SELECT aliases FROM spaces WHERE id = ?", (space_id,)).fetchone()
        return json.loads(row[0] or "[]") if row else []

    def _space_named(self, name: str, kind: Optional[str] = None) -> Optional[dict]:
        row = self._db.execute("SELECT * FROM spaces WHERE lower(name) = lower(?)" + (" AND kind = ?" if kind else ""),
                               (name, kind) if kind else (name,)).fetchone()
        return _space(row) if row else None

    def spaces_changed_since(self, seq: int) -> list[str]:
        rows = self._db.execute("SELECT DISTINCT item_id FROM changes WHERE seq > ? AND item_id LIKE 'space:%'"
                                " ORDER BY item_id", (seq,)).fetchall()
        return [r[0].removeprefix("space:") for r in rows]

    # ---- tags: what belongs to which spaces ------------------------------------------------

    def tag(self, target: str, target_id: str, space_id: str, *, by: str = "person", confidence: Optional[float] = None, reason: str = "") -> bool:
        """Put a chat (file, routine, item) in a space. Domo never puts back a tag the person took off,
        and never overwrites one the person put on."""
        if target not in TARGETS:
            raise StoreError(f"target is one of {', '.join(TARGETS)}")
        if not self.space(space_id):
            raise StoreError(f"no space {space_id!r}")
        row = self._db.execute("SELECT * FROM space_links WHERE space_id = ? AND target_type = ? AND target_id = ?", (space_id, target, target_id)).fetchone()
        if by == "domo" and row and (row["state"] == "off" or row["by"] == "person"):
            return row["state"] == "on"
        now = self._clock()
        with self._tx() as db:
            db.execute("INSERT INTO space_links (space_id, target_type, target_id, by, confidence, reason, state, at) VALUES (?,?,?,?,?,?, 'on', ?)"
                       " ON CONFLICT(space_id, target_type, target_id) DO UPDATE SET by=excluded.by, confidence=excluded.confidence,"
                       " reason=excluded.reason, state='on', at=excluded.at", (space_id, target, target_id, by, confidence, reason[:200], now))
            db.execute("UPDATE spaces SET quiet = 0 WHERE id = ?", (space_id,))
            _touch(db, space_id, now)
        return True

    def untag(self, target: str, target_id: str, space_id: str, *, by: str = "person") -> None:
        """Take a tag off. The person's removal is remembered, so Domo doesn't put it back."""
        now = self._clock()
        with self._tx() as db:
            if by == "person":
                db.execute("INSERT INTO space_links (space_id, target_type, target_id, by, state, at) VALUES (?,?,?, 'person', 'off', ?)"
                           " ON CONFLICT(space_id, target_type, target_id) DO UPDATE SET by='person', state='off', confidence=NULL, at=excluded.at",
                           (space_id, target, target_id, now))
            else:
                db.execute("DELETE FROM space_links WHERE space_id = ? AND target_type = ? AND target_id = ? AND by = 'domo'", (space_id, target, target_id))
            _touch(db, space_id, now)

    def tags(self, target: str, target_id: str, *, include_off: bool = False) -> list[dict]:
        """A thing's spaces, its area first, then topics, surest first."""
        rows = self._db.execute(
            "SELECT l.*, s.kind, s.name FROM space_links l JOIN spaces s ON s.id = l.space_id WHERE l.target_type = ? AND l.target_id = ?"
            + ("" if include_off else " AND l.state = 'on'") + " ORDER BY s.kind = 'topic', COALESCE(l.confidence, 1) DESC", (target, target_id)).fetchall()
        return [dict(r) for r in rows]

    def tag_map(self, target: str) -> dict[str, list[dict]]:
        """Every tagged thing of one kind: id → its tags (area first)."""
        out: dict[str, list[dict]] = {}
        for r in self._db.execute(
                "SELECT l.space_id, l.target_id, l.by, l.confidence, l.reason, s.kind FROM space_links l JOIN spaces s ON s.id = l.space_id"
                " WHERE l.target_type = ? AND l.state = 'on' ORDER BY s.kind = 'topic', COALESCE(l.confidence, 1) DESC", (target,)):
            out.setdefault(r["target_id"], []).append({"space_id": r["space_id"], "by": r["by"], "confidence": r["confidence"], "reason": r["reason"], "kind": r["kind"]})
        return out

    def tagged(self, space_id: str, target: str) -> list[str]:
        return [r[0] for r in self._db.execute("SELECT target_id FROM space_links WHERE space_id = ? AND target_type = ? AND state = 'on' ORDER BY at DESC", (space_id, target))]

    def related(self, space_id: str, limit: int = 6) -> list[dict]:
        """The spaces this one's chats are also in, most shared first."""
        rows = self._db.execute(
            "SELECT b.space_id, COUNT(*) AS n FROM space_links a JOIN space_links b ON a.target_type = b.target_type AND a.target_id = b.target_id"
            " WHERE a.space_id = ? AND b.space_id != ? AND a.state = 'on' AND b.state = 'on' GROUP BY b.space_id ORDER BY n DESC LIMIT ?",
            (space_id, space_id, limit)).fetchall()
        return [{**self.space(r[0]), "shared": r[1]} for r in rows if self.space(r[0])]

    # the single-space view the rest of the app grew up with: a chat's area (or its first tag)

    def tag_session(self, session_id: str, space_id: Optional[str], *, by: str = "person",
                    confidence: Optional[float] = None, reason: str = "") -> Optional[str]:
        """Move a chat to a space: an area replaces its area, a topic is added (with its home area when
        the chat has none). None, by the person, takes every tag off and keeps them off."""
        current = self.tags("session", session_id)
        if not space_id:
            for t in current:
                self.untag("session", session_id, t["space_id"], by=by)
            return None
        target = self.space(space_id)
        if not target:
            raise StoreError(f"no space {space_id!r}")
        if target["kind"] == "area":
            for t in current:
                if t["kind"] == "area" and t["space_id"] != space_id:
                    self.untag("session", session_id, t["space_id"], by=by)
            self.tag("session", session_id, space_id, by=by, confidence=confidence, reason=reason)
        else:
            self.tag("session", session_id, space_id, by=by, confidence=confidence, reason=reason)
            if target.get("parent_id") and not any(t["kind"] == "area" for t in current):
                self.tag("session", session_id, target["parent_id"], by=by, confidence=confidence, reason=reason)
        return (self.session_space(session_id) or {}).get("space_id")

    def filing_attempted(self, session_id: str, revision: int) -> None:
        with self._tx() as db:
            db.execute("INSERT OR REPLACE INTO filing_attempts (session_id, at, revision) VALUES (?, ?, ?)", (session_id, self._clock(), revision))

    def filing_attempts(self) -> dict[str, int]:
        return {r[0]: r[1] for r in self._db.execute("SELECT session_id, revision FROM filing_attempts")}

    def filings(self) -> dict[str, dict]:
        """Every chat with a tag on or off: its main space, who put it there, how sure, why."""
        out = {}
        for sid in {r[0] for r in self._db.execute("SELECT DISTINCT target_id FROM space_links WHERE target_type = 'session'")}:
            out[sid] = self.session_space(sid) or {"session_id": sid, "space_id": "", "filed_by": "person", "confidence": None, "reason": ""}
        return out

    def session_space(self, session_id: str) -> Optional[dict]:
        t = self.tags("session", session_id)
        if not t:
            return None
        main = t[0]
        return {"session_id": session_id, "space_id": main["space_id"], "filed_by": main["by"], "confidence": main["confidence"], "reason": main["reason"]}

    # ---- feedback on suggestions -----------------------------------------------------------

    def vote(self, key: str, kind: str, verdict: str) -> None:
        """done, dismiss, more or less: hides a suggestion, or tunes how much its kind counts."""
        if verdict not in ("done", "dismiss", "more", "less"):
            raise StoreError("verdict is done, dismiss, more or less")
        with self._tx() as db:
            db.execute("INSERT INTO suggestion_votes (key, kind, verdict, at) VALUES (?, ?, ?, ?)", (key, kind, verdict, self._clock()))

    def hidden_suggestions(self, *, dismissed_for: float = 7 * 86400) -> set[str]:
        """Done is done; a dismissal lasts a week, then the thing may come back if it's still true."""
        now = self._clock()
        rows = self._db.execute("SELECT key, verdict, MAX(at) FROM suggestion_votes WHERE verdict IN ('done', 'dismiss') GROUP BY key, verdict")
        snoozed = self._db.execute("SELECT key FROM suggestion_snoozes WHERE until > ?", (now,))
        return {k for k, v, at in rows if v == "done" or now - at < dismissed_for} | {r[0] for r in snoozed}

    def snooze(self, key: str, until: float) -> None:
        """Put a suggestion away until a moment; it comes back then if it's still true. It doesn't tune its kind."""
        if until <= self._clock():
            raise StoreError("snooze until a moment still to come")
        with self._tx() as db:
            db.execute("INSERT OR REPLACE INTO suggestion_snoozes (key, until) VALUES (?, ?)", (key, until))

    def kind_weights(self) -> dict[str, float]:
        """More like this lifts a kind, less like this lowers it; dismissals nudge it down."""
        w: dict[str, float] = {}
        for kind, verdict in self._db.execute("SELECT kind, verdict FROM suggestion_votes WHERE at > ?", (self._clock() - 60 * 86400,)):
            w[kind] = w.get(kind, 1.0) * {"more": 1.15, "less": 0.8, "dismiss": 0.97, "done": 1.03}[verdict]
        return {k: max(0.3, min(1.6, v)) for k, v in w.items()}

    # ---- what the person has read ----------------------------------------------------------

    def seen_state(self, agent: Optional[str] = None) -> dict[str, dict]:
        """agent → {since, seen: {chat: its last_active when opened}}, for agents looked at before."""
        where, args = (" WHERE agent = ?", (agent,)) if agent else ("", ())
        out = {a: {"since": since, "seen": {}} for a, since in self._db.execute(f"SELECT agent, since FROM seen_since{where}", args)}
        for a, sid, at in self._db.execute(f"SELECT agent, session_id, at FROM seen{where}", args):
            if a in out:
                out[a]["seen"][sid] = at
        return out

    def mark_seen(self, agent: str, *, since: Optional[float] = None, seen: Optional[dict[str, float]] = None) -> dict:
        """Only ever forward: a device that's behind can't make a read chat unread again. ``since`` sets the
        first look once; the first device to look wins."""
        with self._tx() as db:
            if since is not None:
                db.execute("INSERT OR IGNORE INTO seen_since (agent, since) VALUES (?, ?)", (agent, since))
            elif seen:   # opened before any first look: what's older than it counts as read
                db.execute("INSERT OR IGNORE INTO seen_since (agent, since) VALUES (?, ?)", (agent, max(map(float, seen.values()))))
            db.executemany("INSERT INTO seen (agent, session_id, at) VALUES (?, ?, ?)"
                           " ON CONFLICT (agent, session_id) DO UPDATE SET at = MAX(at, excluded.at)",
                           [(agent, sid, float(at)) for sid, at in (seen or {}).items()])
        return self.seen_state(agent).get(agent) or {"since": since, "seen": {}}

    def set_archived(self, session_id: str, archived: bool) -> None:
        """Remember what was archived here: Hermes's own listing leaves archived chats out entirely."""
        with self._tx() as db:
            if archived:
                db.execute("INSERT OR REPLACE INTO archived_sessions (session_id, archived_at) VALUES (?, ?)",
                           (session_id, self._clock()))
            else:
                db.execute("DELETE FROM archived_sessions WHERE session_id = ?", (session_id,))

    def archived(self) -> list[str]:
        return [r[0] for r in self._db.execute("SELECT session_id FROM archived_sessions ORDER BY archived_at DESC")]

    def session_tags(self) -> dict[str, str]:
        """chat → its main space (area first): the single-space view."""
        return {sid: t[0]["space_id"] for sid, t in self.tag_map("session").items()}

    def tag_job(self, job_id: str, space_id: Optional[str]) -> Optional[str]:
        for t in self.tags("job", job_id):
            if space_id is None or (self.space(space_id) or {}).get("kind") == t["kind"]:
                self.untag("job", job_id, t["space_id"])
        if space_id:
            self.tag("job", job_id, space_id)
        return space_id or None

    def job_tags(self) -> dict[str, str]:
        return {jid: t[0]["space_id"] for jid, t in self.tag_map("job").items()}

    def add_file(self, path: str, name: str, mime: str, size: int, *, space_id: Optional[str] = None,
                 session_id: Optional[str] = None) -> dict:
        now = self._clock()
        with self._tx() as db:
            db.execute("INSERT OR REPLACE INTO files (path, name, mime, size, space_id, session_id, created_at)"
                       " VALUES (?,?,?,?,?,?,?)", (path, name, mime, size, space_id or None, session_id or None, now))
            if space_id:
                _touch(db, space_id, now)
        return self.file(path)

    def file(self, path: str) -> Optional[dict]:
        row = self._db.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
        return dict(row) if row else None

    def link_file(self, path: str, *, session_id: str, space_id: Optional[str]) -> None:
        """An attachment sent in a chat belongs to that chat (and its space) unless it was filed already."""
        with self._tx() as db:
            db.execute("UPDATE files SET session_id = COALESCE(session_id, ?), space_id = COALESCE(space_id, ?)"
                       " WHERE path = ?", (session_id or None, space_id or None, path))

    def files(self, space_id: Optional[str] = None, limit: int = 200) -> list[dict]:
        """Uploads, newest first; for a space, those filed in it, tagged with it, or sent in its chats."""
        sql, args = "SELECT * FROM files", []
        if space_id:
            sql += (" WHERE space_id = ? OR path IN (SELECT target_id FROM space_links WHERE space_id = ? AND target_type = 'file' AND state = 'on')"
                    " OR session_id IN (SELECT target_id FROM space_links WHERE space_id = ? AND target_type = 'session' AND state = 'on')")
            args += [space_id, space_id, space_id]
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"; args.append(max(1, min(limit, 500)))
        return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def _migrate_to_tags(self) -> None:
        """Once: spaces with a parent become topics, and every filing becomes a tag (a topic's chats in
        its area too). The old tables stay, unread, as they were."""
        if self._db.execute("SELECT 1 FROM space_links LIMIT 1").fetchone() or self._db.execute("SELECT value FROM meta WHERE key = 'tags_migrated'").fetchone():
            return
        now = self._clock()
        with self._tx() as db:
            db.execute("UPDATE spaces SET kind = 'topic' WHERE parent_id IS NOT NULL")
            parents = {r[0]: r[1] for r in db.execute("SELECT id, parent_id FROM spaces")}
            for r in db.execute("SELECT * FROM session_spaces WHERE space_id != ''").fetchall():
                for sid in (r["space_id"], parents.get(r["space_id"])):
                    if sid and sid in parents:
                        db.execute("INSERT OR IGNORE INTO space_links (space_id, target_type, target_id, by, confidence, reason, state, at)"
                                   " VALUES (?, 'session', ?, ?, ?, ?, 'on', ?)", (sid, r["session_id"], r["filed_by"], r["confidence"], r["reason"], r["updated_at"]))
            for r in db.execute("SELECT * FROM job_spaces").fetchall():
                if r["space_id"] in parents:
                    db.execute("INSERT OR IGNORE INTO space_links (space_id, target_type, target_id, by, state, at) VALUES (?, 'job', ?, 'person', 'on', ?)",
                               (r["space_id"], r["job_id"], now))
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('tags_migrated', '1')")

    # ---- summary -----------------------------------------------------------------------------

    def feed(self, *, now: Optional[float] = None, keep=None) -> dict:
        """Everything the home screen needs in one read; ``keep`` leaves out items it says no to (someone else's)."""
        now = now if now is not None else self._clock()
        self.mark_stalled(now)
        day_start = now - 24 * 3600
        mine = (lambda rows: [r for r in rows if keep(r)]) if keep else (lambda rows: rows)
        runs = mine(self.list("run", limit=50))
        active = [r for r in runs if r["status"] in ("running", "stalled")]
        recent = [r for r in runs if r["status"] in ("done", "failed") and r["updated_at"] >= day_start]
        decisions = mine(self.list("decision", status="open"))
        todos = mine(self.list("todo", status="open"))
        receipts = mine([r for r in self.list("receipt", limit=50) if r["created_at"] >= day_start])
        brief = self.latest_brief()
        return {
            "brief": brief if brief is None or not keep or keep(brief) else None,
            "decisions": decisions,
            "todos": todos,
            "runs": active + recent,
            "receipts": receipts,
            "counts": {
                "needs_you": len(decisions) + sum(1 for r in active if r["status"] == "stalled"),
                "running": sum(1 for r in active if r["status"] == "running"),
                "done_today": len(recent),
                "spent_today_usd": round(sum(float(r["data"].get("cost_usd") or 0) for r in recent + active), 2),
            },
            "cursor": self.cursor(),
        }

    def _require(self, item_id: str, kind: str) -> dict:
        item = self.get(item_id)
        if item["kind"] != kind:
            raise StoreError(f"{item_id} is a {item['kind']}, not a {kind}")
        return item


def _default_status(kind: str) -> str:
    return {"decision": "open", "todo": "open", "run": "running", "receipt": "done", "brief": "published", "suggestion": "open"}[kind]


def _fact_key(key: str) -> str:
    key = " ".join((key or "").lower().split())
    if not key:
        raise StoreError("key is required")
    return key


def _row(row: sqlite3.Row) -> dict:
    item = dict(row)
    item["data"] = json.loads(item["data"] or "{}")
    return item


def _space(row: sqlite3.Row) -> dict:
    return {**{k: row[k] for k in SPACE_FIELDS}, "keep_moving": bool(row["keep_moving"])}


def _slug(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")[:48] or "space"


def _touch(db: sqlite3.Connection, space_id: str, now: float) -> None:
    db.execute("INSERT INTO changes (item_id, at) VALUES (?, ?)", (f"space:{space_id}", now))


def _refile(db: sqlite3.Connection, old: str, new: str, now: float) -> None:
    """Move items and facts filed under the space name ``old`` to ``new``, logging the items."""
    ids = [r[0] for r in db.execute("SELECT id FROM items WHERE space = ?", (old,)).fetchall()]
    db.execute("UPDATE items SET space = ?, updated_at = ? WHERE space = ?", (new, now, old))
    db.execute("UPDATE facts SET space = ? WHERE space = ?", (new, old))
    db.executemany("INSERT INTO changes (item_id, at) VALUES (?, ?)", [(i, now) for i in ids])
