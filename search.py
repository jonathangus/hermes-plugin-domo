"""Search across every chat: Hermes's own full-text index for the words, a local vector index for the
meaning, fused by rank (reciprocal rank fusion, k=60).

Hermes keeps each profile's chats in ``<profile home>/state.db`` with FTS5 indexes kept by triggers, and
has no embeddings of its own. So the keyword half reads Hermes's ``messages_fts`` directly (read-only),
and the vector half is Domo's: ``plugin-data/domo/search.db`` per profile, filled by a background indexer
that follows ``messages.id`` and re-reads a chat when its ``transcript_revision`` moves (edits, rewinds).

Embeddings come from any OpenAI-compatible ``/embeddings`` endpoint: by default Ollama on the Mac with
EmbeddingGemma (multilingual, so Swedish and English chats meet), nothing leaves the house. Vectors are
searched with sqlite-vec when its extension is installed, otherwise by brute force in Python, which is
fine for a household's few thousand chats. With no embedder reachable, search is keyword-only and the
indexer catches up later.
"""

from __future__ import annotations

import array
import json
import logging
import math
import operator
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

import httpx

from . import bridge

log = logging.getLogger("domo.search")

RRF_K = 60
# A vector match counts only when it's near the best one and not plainly unrelated. Starting values,
# not measured on the household's chats yet: tune with /search/status and real queries.
MIN_SIMILARITY, SPREAD = 0.2, 0.2
MORE_HITS = 0.05
CHUNK_CHARS, CHUNK_OVERLAP = 1200, 150
BATCH = 32
HIDDEN_SOURCES = ("kanban", "subagent", "tool")   # what Hermes's own session_search leaves out
EVERY = 20.0

CRON_HINT = re.compile(r"\[IMPORTANT: You are running as a scheduled cron job.*?another job\.\]\s*", re.S)
RUN_TIME = re.compile(r"\[CURRENT RUN TIME: [^\]]*\]\s*")
STEER = re.compile(r"\[/?OUT-OF-BAND USER MESSAGE[^\]]*\]\s*")   # Hermes's frame around a message that steered a turn
ATTACH = re.compile(r"\[Domo attachments\](.*)", re.S)
MEDIA = re.compile(r"MEDIA:\s*(\S+)")
DATA_URL = re.compile(r"!?\[[^\]]*\]\(data:[^)]+\)")


# ---- configuration ----------------------------------------------------------------------------

def settings() -> dict:
    env = {**bridge.read_env(bridge.hermes_root()), **os.environ}
    return {
        "url": env.get("DOMO_EMBED_URL", "http://127.0.0.1:11434/v1").rstrip("/"),
        "model": env.get("DOMO_EMBED_MODEL", "embeddinggemma-2:270m"),
        "key": env.get("DOMO_EMBED_KEY", ""),
        "off": env.get("DOMO_SEARCH", "on") == "off",
        "vec": env.get("DOMO_SQLITE_VEC", ""),
    }


def _prompts(model: str) -> tuple[str, str]:
    """The task prefixes each embedding family was trained with (query, document)."""
    m = model.lower()
    if "gemma" in m:
        return "task: search result | query: ", "title: none | text: "
    if "e5" in m:
        return "query: ", "passage: "
    if "nomic" in m:
        return "search_query: ", "search_document: "
    return "", ""


class EmbedError(Exception):
    pass


def embed(texts: list[str], *, query: bool = False, cfg: Optional[dict] = None) -> list[list[float]]:
    cfg = cfg or settings()
    q, d = _prompts(cfg["model"])
    headers = {"Authorization": f"Bearer {cfg['key']}"} if cfg["key"] else {}
    try:
        r = httpx.post(f"{cfg['url']}/embeddings", json={"model": cfg["model"], "input": [(q if query else d) + t for t in texts]},
                       headers=headers, timeout=60)
        r.raise_for_status()
        rows = sorted(r.json()["data"], key=lambda x: x.get("index", 0))
    except Exception as exc:   # down, not pulled, wrong model: search falls back to keywords
        raise EmbedError(str(exc)[:200]) from exc
    out = []
    for row in rows:
        v = row["embedding"]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


# ---- text -------------------------------------------------------------------------------------

def clean(content: Optional[str]) -> str:
    """What a message says, without Hermes's cron preamble, image data, or the plugin's file list
    (file names stay: they're worth finding)."""
    if not content:
        return ""
    text = content
    if text.startswith("[") and text.rstrip().endswith("]"):
        try:
            parts = json.loads(text)
            if isinstance(parts, list):
                text = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict))
        except ValueError:
            pass
    text = CRON_HINT.sub("", text)
    text = RUN_TIME.sub("", text)
    text = STEER.sub("", text)
    files = []
    m = ATTACH.search(text)
    if m:
        files = [line[2:].split(" | ")[0] for line in m.group(1).splitlines() if line.startswith("- ")]
        text = text[: m.start()]
    text = DATA_URL.sub("", text)
    text = MEDIA.sub(lambda x: x.group(1).rsplit("/", 1)[-1], text)
    text = text.strip()
    if files:
        text += ("\n" if text else "") + "Files: " + ", ".join(files)
    return text


def chunks(text: str) -> list[str]:
    if len(text) <= CHUNK_CHARS:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + CHUNK_CHARS)
        if end < len(text):   # break at a paragraph or sentence when there's one near the end
            cut = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if cut > start + CHUNK_CHARS // 2:
                end = cut + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return [c for c in out if c]


def snippet(text: str, query: str, width: int = 220) -> str:
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 1]
    low = text.lower()
    at = min((low.find(w) for w in words if low.find(w) >= 0), default=-1)
    if at < 0 or len(text) <= width:
        return text[:width].strip() + ("…" if len(text) > width else "")
    start = max(0, at - width // 3)
    return ("…" if start else "") + text[start : start + width].strip() + ("…" if start + width < len(text) else "")


# ---- Hermes's database, read-only ------------------------------------------------------------

def state_db(home: Path) -> Path:
    return home / "state.db"


def _hermes(home: Path) -> Optional[sqlite3.Connection]:
    path = state_db(home)
    if not path.exists():
        return None
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    db.row_factory = sqlite3.Row
    return db


def _fts_query(q: str, any_word: bool) -> str:
    """Each word quoted (so FTS operators can't be injected), the last one as a prefix, as Hermes does."""
    words = re.findall(r"\w+", q)
    if not words:
        return ""
    terms = [f'"{w}"' for w in words[:-1]] + [f'"{words[-1]}"*']
    return (" OR " if any_word else " AND ").join(terms)


def keyword_hits(home: Path, q: str, limit: int = 60) -> list[dict]:
    db = _hermes(home)
    if db is None:
        return []
    try:
        hidden = ",".join("?" * len(HIDDEN_SOURCES))
        sql = (f"SELECT m.id, m.session_id, m.role, m.content FROM messages_fts f JOIN messages m ON m.id = f.rowid"
               f" JOIN sessions s ON s.id = m.session_id"
               f" WHERE messages_fts MATCH ? AND m.role IN ('user','assistant') AND m.active = 1"
               f" AND s.source NOT IN ({hidden}) ORDER BY bm25(messages_fts) LIMIT ?")
        rows = []
        for any_word in (False, True):   # all the words first; any of them if that finds little
            match = _fts_query(q, any_word)
            if not match:
                return []
            rows = db.execute(sql, (match, *HIDDEN_SOURCES, limit)).fetchall()
            if len(rows) >= 5:
                break
        return [{"message_id": r["id"], "session_id": r["session_id"], "role": r["role"], "text": clean(r["content"])} for r in rows]
    except sqlite3.Error as exc:
        log.warning("keyword search failed: %s", exc)
        return []
    finally:
        db.close()


def sessions_info(home: Path, ids: Iterable[str]) -> dict[str, dict]:
    ids = list(dict.fromkeys(ids))
    db = _hermes(home)
    if db is None or not ids:
        return {}
    try:
        marks = ",".join("?" * len(ids))
        rows = db.execute(f"SELECT id, title, source, started_at, COALESCE(last_activity_at, ended_at, started_at) AS last_active,"
                          f" message_count, archived, hidden FROM sessions WHERE id IN ({marks})", ids).fetchall()
        return {r["id"]: dict(r) for r in rows}
    finally:
        db.close()


# ---- Domo's vector index ----------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    session_id  TEXT NOT NULL,
    message_id  INTEGER NOT NULL,
    role        TEXT NOT NULL,
    text        TEXT NOT NULL,
    vector      BLOB
);
CREATE INDEX IF NOT EXISTS chunks_session ON chunks(session_id);
CREATE INDEX IF NOT EXISTS chunks_pending ON chunks(id) WHERE vector IS NULL;
CREATE TABLE IF NOT EXISTS indexed (
    session_id  TEXT PRIMARY KEY,
    revision    INTEGER NOT NULL,
    last_id     INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class Index:
    """One profile's chunks and their vectors."""

    def __init__(self, home: Path, cfg: dict):
        self.home = home
        path = home / "plugin-data" / "domo" / "search.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=10, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()
        self.vec = _load_vec(self.db, cfg["vec"])
        self._cache: Optional[tuple[list[int], list[array.array]]] = None
        model = self.meta("model")
        if model and model != cfg["model"]:   # another model's vectors don't compare: embed again
            with self.db:
                self.db.execute("UPDATE chunks SET vector = NULL")
                self.db.execute("DELETE FROM meta WHERE key = 'dim'")
                if self.vec:
                    self.db.execute("DROP TABLE IF EXISTS vec_chunks")
        self.set_meta("model", cfg["model"])

    def meta(self, key: str) -> Optional[str]:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    # -- following Hermes ------------------------------------------------------------------------

    def sync(self) -> int:
        """Bring the chunks up to date with Hermes's messages. Returns how many messages it read."""
        src = _hermes(self.home)
        if src is None:
            return 0
        read = 0
        try:
            hidden = ",".join("?" * len(HIDDEN_SOURCES))
            sessions = {r["id"]: r["transcript_revision"] for r in src.execute(
                f"SELECT id, transcript_revision FROM sessions WHERE source NOT IN ({hidden})", HIDDEN_SOURCES)}
            known = {r["session_id"]: (r["revision"], r["last_id"]) for r in self.db.execute("SELECT * FROM indexed")}
            with self.lock, self.db:
                for gone in set(known) - set(sessions):   # deleted in Hermes
                    self._drop(gone)
                    known.pop(gone)
            for sid, revision in sessions.items():
                rev, last = known.get(sid, (None, 0))
                if rev is not None and rev != revision:   # edited or rewound: read it again
                    with self.lock, self.db:
                        self._drop(sid)
                    last = 0
                rows = src.execute("SELECT id, role, content FROM messages WHERE session_id = ? AND id > ? AND active = 1"
                                   " AND role IN ('user','assistant') ORDER BY id", (sid, last)).fetchall()
                if not rows and rev == revision:
                    continue
                with self.lock, self.db:
                    for r in rows:
                        for piece in chunks(clean(r["content"])):
                            self.db.execute("INSERT INTO chunks (session_id, message_id, role, text) VALUES (?,?,?,?)",
                                            (sid, r["id"], r["role"], piece))
                    top = max([last] + [r["id"] for r in rows])
                    self.db.execute("INSERT OR REPLACE INTO indexed (session_id, revision, last_id) VALUES (?,?,?)",
                                    (sid, revision, top))
                read += len(rows)
        finally:
            src.close()
        if read:
            self._cache = None
        return read

    def _drop(self, sid: str) -> None:
        if self.vec:
            self.db.execute("DELETE FROM vec_chunks WHERE rowid IN (SELECT id FROM chunks WHERE session_id = ?)", (sid,))
        self.db.execute("DELETE FROM chunks WHERE session_id = ?", (sid,))
        self.db.execute("DELETE FROM indexed WHERE session_id = ?", (sid,))
        self._cache = None

    def pending(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM chunks WHERE vector IS NULL").fetchone()[0]

    def embed_pending(self, cfg: dict, limit: int = 512) -> int:
        """Embed chunks that have no vector yet, newest first (what you'll search for soonest)."""
        done = 0
        while done < limit:
            rows = self.db.execute("SELECT c.id, c.text, c.role FROM chunks c WHERE c.vector IS NULL ORDER BY c.id DESC LIMIT ?",
                                   (BATCH,)).fetchall()
            if not rows:
                break
            vectors = embed([r["text"] for r in rows], cfg=cfg)
            with self.lock, self.db:
                dim = len(vectors[0])
                if self.meta("dim") is None:
                    self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('dim', ?)", (str(dim),))
                    if self.vec:
                        self.db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[{dim}] distance_metric=cosine)")
                for r, v in zip(rows, vectors):
                    blob = array.array("f", v).tobytes()
                    self.db.execute("UPDATE chunks SET vector = ? WHERE id = ?", (blob, r["id"]))
                    if self.vec:
                        self.db.execute("INSERT OR REPLACE INTO vec_chunks (rowid, embedding) VALUES (?, ?)", (r["id"], blob))
            done += len(rows)
            self._cache = None
        return done

    # -- searching -------------------------------------------------------------------------------

    def nearest(self, qvec: list[float], k: int = 60) -> list[dict]:
        if self.meta("dim") is None:
            return []
        blob = array.array("f", qvec).tobytes()
        if self.vec:
            rows = self.db.execute("SELECT c.id, c.session_id, c.message_id, c.role, c.text, v.distance FROM vec_chunks v"
                                   " JOIN chunks c ON c.id = v.rowid WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
                                   (blob, k)).fetchall()
            # a vector with no direction (an empty chunk) has no cosine distance: count it as unrelated
            return [{**dict(r), "distance": 1.0 if r["distance"] is None else r["distance"]} for r in rows]
        ids, vecs = self._vectors()
        q = array.array("f", qvec)
        scored = sorted(((sum(map(operator.mul, q, v)), i) for i, v in zip(ids, vecs)), reverse=True)[:k]
        if not scored:
            return []
        by_id = {r["id"]: dict(r) for r in self.db.execute(
            f"SELECT id, session_id, message_id, role, text FROM chunks WHERE id IN ({','.join('?' * len(scored))})", [i for _, i in scored])}
        return [{**by_id[i], "distance": 1 - s} for s, i in scored if i in by_id]

    def _vectors(self) -> tuple[list[int], list[array.array]]:
        if self._cache is None:
            ids, vecs = [], []
            for r in self.db.execute("SELECT id, vector FROM chunks WHERE vector IS NOT NULL"):
                a = array.array("f")
                a.frombytes(r["vector"])
                ids.append(r["id"])
                vecs.append(a)
            self._cache = (ids, vecs)
        return self._cache

    def stats(self) -> dict:
        n = self.db.execute("SELECT COUNT(*), COUNT(vector), COUNT(DISTINCT session_id) FROM chunks").fetchone()
        return {"chunks": n[0], "embedded": n[1], "sessions": n[2], "vector": "sqlite-vec" if self.vec else "python"}


def _load_vec(db: sqlite3.Connection, path: str) -> bool:
    """sqlite-vec, from its Python package or the extension file setup put beside the plugin."""
    try:
        db.enable_load_extension(True)
        try:
            import sqlite_vec   # type: ignore
            sqlite_vec.load(db)
        except ImportError:
            candidates = [path] if path else []
            here = Path(__file__).resolve().parent / "vendor"
            candidates += [str(p) for p in here.glob("vec0.*")]
            for c in candidates:
                if c and Path(c).exists():
                    db.load_extension(c)
                    break
            else:
                return False
        db.enable_load_extension(False)
        db.execute("SELECT vec_version()")
        return True
    except Exception as exc:
        log.info("sqlite-vec not available, searching vectors in Python: %s", exc)
        return False


# ---- the indexer and search, per process --------------------------------------------------------

_indexes: dict[str, Index] = {}
_status: dict[str, dict] = {}
_thread: Optional[threading.Thread] = None
_wake = threading.Event()
_guard = threading.Lock()


def index_for(profile: str) -> Index:
    with _guard:
        if profile not in _indexes:
            _indexes[profile] = Index(bridge.profile_home(profile), settings())
        return _indexes[profile]


def ensure_running() -> None:
    """Start the indexer once per process (the dashboard's), and nudge it."""
    global _thread
    if settings()["off"]:
        return
    with _guard:
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(target=_loop, name="domo-search-indexer", daemon=True)
            _thread.start()
    _wake.set()


def _loop() -> None:
    backoff = 0.0
    while True:
        cfg = settings()
        embedder_down = False
        for profile in bridge.profile_names():
            try:
                idx = index_for(profile)
                read = idx.sync()
                if not embedder_down and idx.pending():
                    try:
                        idx.embed_pending(cfg)
                        _status[profile] = {"embedder": "ok", "at": time.time()}
                    except EmbedError as exc:
                        embedder_down = True
                        _status[profile] = {"embedder": "unreachable", "error": str(exc), "at": time.time()}
                elif not idx.pending():
                    _status.setdefault(profile, {"embedder": "ok", "at": time.time()})
                if read:
                    log.debug("indexed %d messages for %s", read, profile)
            except Exception as exc:   # one profile's trouble mustn't stop the rest
                log.warning("search indexer: %s: %s", profile, exc)
        backoff = min(300.0, backoff * 2 or 30.0) if embedder_down else 0.0
        _wake.wait(EVERY + backoff)
        _wake.clear()


def status(profile: str) -> dict:
    cfg = settings()
    idx = index_for(profile)
    return {**idx.stats(), "pending": idx.pending(), "model": cfg["model"], "url": cfg["url"],
            "embedder": _status.get(profile, {}).get("embedder", "starting"), "error": _status.get(profile, {}).get("error")}


def keyword_weight(q: str, has_keyword_hits: bool) -> float:
    """How much the words count against the meaning. A query of one or two words that the chats really
    contain is nearly always a name, a repo or an error code: the exact match should lead. A longer one
    describes what it was about, where meaning does better (vstash's finding: per-query weights beat fixed
    ones). Starting values, to tune on real queries."""
    if not has_keyword_hits:
        return 0.4
    return 0.7 if len(re.findall(r"\w+", q)) <= 2 else 0.45


def search(profile: str, q: str, *, limit: int = 20, sessions: Optional[set[str]] = None) -> dict:
    """Chats that match ``q``: by words (Hermes's FTS) and by meaning (vectors), fused by rank, one row per chat."""
    home = bridge.profile_home(profile)
    q = q.strip()
    if not q:
        return {"results": [], "mode": "none"}
    keyword = keyword_hits(home, q)
    mode, semantic = "keyword", []
    idx = index_for(profile)
    if idx.meta("dim") is not None:
        try:
            near = idx.nearest(embed([q], query=True)[0])
            best = max((1 - h["distance"] for h in near), default=0.0)
            semantic = [h for h in near if 1 - h["distance"] >= max(MIN_SIMILARITY, best - SPREAD)]
            mode = "hybrid"
        except EmbedError:
            pass

    wk = keyword_weight(q, bool(keyword))
    fused: dict[int, dict] = {}
    for weight, hits, kind in ((wk, keyword, "words"), (1 - wk, semantic, "meaning")):
        seen = set()
        for rank, h in enumerate(hits, 1):
            key = h["message_id"]
            if key in seen:   # a long message's other chunks don't vote twice
                continue
            seen.add(key)
            row = fused.setdefault(key, {"message_id": key, "session_id": h["session_id"], "role": h["role"],
                                         "text": h["text"], "score": 0.0, "matched": set()})
            row["score"] += weight / (RRF_K + rank)
            row["matched"].add(kind)
            if kind == "words":
                row["text"] = h["text"]

    by_session: dict[str, dict] = {}
    for hit in sorted(fused.values(), key=lambda h: -h["score"]):
        sid = hit["session_id"]
        if sessions is not None and sid not in sessions:
            continue
        best = by_session.get(sid)
        if best is None:
            by_session[sid] = {**hit, "hits": 1}
        else:   # more hits in one chat lift it a little, never past a better single match
            best["hits"] += 1
            best["score"] += hit["score"] * MORE_HITS
            best["matched"] |= hit["matched"]
    ranked = sorted(by_session.values(), key=lambda h: -h["score"])[:limit]
    info = sessions_info(home, [h["session_id"] for h in ranked])
    results = []
    for h in ranked:
        s = info.get(h["session_id"])
        if s is None or s.get("hidden"):
            continue
        results.append({
            "session_id": h["session_id"], "message_id": h["message_id"], "role": h["role"],
            "title": s["title"], "source": s["source"], "last_active": s["last_active"], "archived": bool(s["archived"]),
            "snippet": snippet(h["text"], q), "matched": sorted(h["matched"]), "hits": h["hits"], "score": round(h["score"], 5),
        })
    return {"results": results, "mode": mode}
