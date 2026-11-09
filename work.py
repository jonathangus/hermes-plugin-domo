"""Pieces of work: sessions from the other machines, stitched into threads and remembered.

A session is too small a unit (a median of three prompts, and `/clear` starts a new one), so sessions in
the same place (repo, or folder) on the same branch, less than GAP apart, are one thread. When a thread
goes quiet it gets a digest: a short account written by Haiku through the `claude` CLI on this Mac (the
subscription, not API billing), with facts taken from the transcript itself, never from the model: the
commits made (git's own "[branch sha] subject" lines) and the PRs Claude Code linked. Old threads, from a
first sync of months of history, get a plain digest without a model.

Each digest is also a line in Domo's memory trace (kind ``coding.work``), so the nightly routine learns from
it, and Hermes reads threads with the ``domo_work`` tool: as work Claude Code did on that machine, not as
its own chats.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from .local import LocalStore, _j

log = logging.getLogger("domo.work")

GAP = 2 * 3600              # a guess: work resumed within two hours is the same piece of work
QUIET = 20 * 60             # digest a thread once nothing happened for this long
MODEL_FOR = 14 * 86400      # older threads (a backfill) get a plain digest, without a model
DIGESTS_PER_PASS = 3
EVERY = 120
TEXT_BUDGET = 14000
DIGEST_TIMEOUT = 180
UNATTENDED_REASONS = ("repo_only_here", "login", "skill_or_tool", "gui", "steering", "judgement")

Summarizer = Callable[[str], Optional[dict]]


# ---- threads -----------------------------------------------------------------------------------

def stitch(local: LocalStore, *, limit: int = 2000) -> int:
    """Put every session that isn't in a thread yet (or grew) into one. Returns how many it placed."""
    rows = local._q("SELECT * FROM files WHERE stitched != size ORDER BY first_seen LIMIT ?", (limit,))
    placed = 0
    for row in sorted(rows, key=lambda r: local.info(r).get("started_at") or r["first_seen"]):
        info = local.info(row)
        if not info.get("prompts"):
            # a session nobody typed in (an aborted start) is no piece of work
            local._q("UPDATE files SET stitched = size WHERE source = ? AND session_id = ?", (row["source"], row["session_id"]))
            continue
        repo = _j(row["repo"], None) or {}
        place = repo.get("root") or row["cwd"] or info.get("cwd") or ""
        branch = repo.get("branch") or info.get("branch") or ""
        start = info.get("started_at") or row["first_seen"]
        end = info.get("last_at") or row["updated_at"]
        thread = local._q("SELECT * FROM threads WHERE id = ?", (row["thread_id"],)) if row["thread_id"] else []
        if not thread:
            thread = local._q("SELECT * FROM threads WHERE machine_id = ? AND place = ? AND branch = ?"
                              " AND last_active >= ? AND started_at <= ? ORDER BY last_active DESC LIMIT 1",
                              (row["machine_id"], place, branch, start - GAP, end + GAP))
        with local._lock:
            if thread:
                t = thread[0]
                sessions = _j(t["sessions"], [])
                if row["session_id"] not in sessions:
                    sessions.append(row["session_id"])
                size = sum(r["size"] for r in local._q(
                    f"SELECT size FROM files WHERE session_id IN ({','.join('?' * len(sessions))})", sessions))
                local._db.execute("UPDATE threads SET started_at = MIN(started_at, ?), last_active = MAX(last_active, ?),"
                                  " sessions = ?, size = ?, repo = CASE WHEN ? = 'null' THEN repo ELSE ? END WHERE id = ?",
                                  (start, end, json.dumps(sessions), size, json.dumps(repo or None), json.dumps(repo or None), t["id"]))
                thread_id = t["id"]
            else:
                thread_id = f"thr_{uuid.uuid4().hex[:12]}"
                local._db.execute("INSERT INTO threads (id, machine_id, place, branch, repo, started_at, last_active, sessions, size)"
                                  " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                  (thread_id, row["machine_id"], place, branch, json.dumps(repo or None), start, end,
                                   json.dumps([row["session_id"]]), row["size"]))
            local._db.execute("UPDATE files SET thread_id = ?, stitched = ? WHERE source = ? AND session_id = ?",
                              (thread_id, row["size"], row["source"], row["session_id"]))
        placed += 1
    return placed


def _sessions_of(local: LocalStore, thread: dict) -> list:
    ids = _j(thread["sessions"], [])
    if not ids:
        return []
    rows = local._q(f"SELECT * FROM files WHERE session_id IN ({','.join('?' * len(ids))})", ids)
    return sorted(rows, key=lambda r: local.info(r).get("started_at") or r["first_seen"])


def facts(local: LocalStore, thread) -> dict:
    """What the transcripts themselves prove: commits, PRs, how much was asked, which models."""
    commits, prs, prompts, models, cost = [], [], 0, [], 0.0
    for row in _sessions_of(local, thread):
        info = local.info(row)
        for c in info.get("commits") or []:
            if all(c["sha"] != x["sha"] for x in commits):
                commits.append(c)
        for url in info.get("prs") or []:
            if url not in prs:
                prs.append(url)
        prompts += info.get("prompts") or 0
        if info.get("model") and info["model"] not in models:
            models.append(info["model"])
        cost += info.get("cost_usd") or 0
    return {"commits": commits, "prs": prs, "prompts": prompts, "models": models, "cost_usd": round(cost, 4) or None}


def condensed(local: LocalStore, thread, budget: int = TEXT_BUDGET) -> str:
    """The thread as a reader needs it: what was asked, what Claude said, what it ran. Long sessions keep
    their start and their end, where the task and the outcome are."""
    parts = []
    rows = _sessions_of(local, thread)
    share = max(1500, budget // max(1, len(rows)))
    for row in rows:
        info = local.info(row)
        lines = []
        for m in local.parsed(row)["messages"]:
            if m["role"] == "user":
                lines.append("You: " + " ".join(m["text"].split())[:700])
            elif m["role"] == "assistant":
                lines.append(f"{'Cursor' if row['source'] == 'cursor' else 'Claude'}: " + " ".join(m["text"].split())[:500])
            else:
                a = m.get("toolArgs") or {}
                what = a.get("command") or a.get("path") or a.get("pattern") or a.get("url") or a.get("goal") or ""
                lines.append(f"  [{a.get('tool') or m.get('toolName')}{' failed' if m.get('toolFailed') else ''}] {str(what)[:160]}")
        text = "\n".join(lines)
        if len(text) > share:
            text = text[: share * 2 // 5] + "\n  […]\n" + text[-share * 3 // 5:]
        when = datetime.fromtimestamp(info.get("started_at") or row["first_seen"]).strftime("%a %d %b %H:%M")
        parts.append(f"## Session {when} ({info.get('prompts', 0)} prompts)\n{text}")
    return "\n\n".join(parts)[: budget + 2000]


DIGEST_PROMPT = """You summarise one piece of coding work for Jonathan's personal assistant, Domo. Below are his \
coding-agent sessions on {machine} in {place} (branch {branch}), from {start} to {end}. "You:" is Jonathan; \
"Claude:"/"Cursor:" is the coding agent; [Tool] lines are what it ran.

Facts already known (don't repeat them as prose, don't contradict them): {facts}

Answer with one JSON object only, no prose around it:
{{"title": "<the task, at most 8 words, like a commit subject>",
  "summary": "<at most 80 words: what he set out to do, what changed, where he stepped in or corrected the agent, how it ended>",
  "status": "<one of: shipped, in progress, stuck, explored, abandoned>",
  "left": ["<what's unfinished or next, at most 3 short items; [] if nothing>"],
  "corrections": ["<how he corrected or steered the agent, in his terms, at most 3; [] if none>"],
  "unattended": {{"could": <true if an agent could have done this alone on another always-on Mac with the code pushed, false otherwise>,
                 "why": [<when false, any of: "repo_only_here", "login", "skill_or_tool", "gui", "steering", "judgement">]}}}}

Sessions:
{text}"""


def claude_bin() -> Optional[str]:
    explicit = os.environ.get("DOMO_CLAUDE_BIN")
    if explicit:
        return explicit if os.path.exists(explicit) else None
    found = shutil.which("claude")
    if found:
        return found
    # launchd's PATH is short: look where the installer and Homebrew put it
    for p in ("~/.local/bin/claude", "~/.claude/local/claude", "/opt/homebrew/bin/claude", "/usr/local/bin/claude"):
        p = os.path.expanduser(p)
        if os.path.exists(p):
            return p
    return None


def _quiet_dir(local: LocalStore) -> Path:
    # Link skips any folder named .domo-quiet, so these summarising runs never come back as sessions
    d = local.path.parent / ".domo-quiet"
    d.mkdir(parents=True, exist_ok=True)
    return d


def claude_summarizer(local: LocalStore) -> Optional[Summarizer]:
    binary = claude_bin()
    if not binary:
        return None

    def run(prompt: str) -> Optional[dict]:
        cmd = [binary, "-p", "--model", os.environ.get("DOMO_DIGEST_MODEL", "haiku"), "--output-format", "json",
               "--no-session-persistence", "--tools", "", "--disable-slash-commands"]
        try:
            out = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=DIGEST_TIMEOUT, cwd=_quiet_dir(local))
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("domo: digest via claude failed: %s", exc)
            return None
        if out.returncode != 0:
            log.warning("domo: digest via claude exited %s: %s", out.returncode, out.stderr[-300:])
            return None
        try:
            result = json.loads(out.stdout).get("result", "")
        except ValueError:
            result = out.stdout
        return parse_digest(result)

    return run


def parse_digest(text: str) -> Optional[dict]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        d = json.loads(text[start:end + 1])
    except ValueError:
        return None
    if not isinstance(d, dict) or not str(d.get("summary") or "").strip():
        return None
    un = d.get("unattended") if isinstance(d.get("unattended"), dict) else {}
    return {
        "title": " ".join(str(d.get("title") or "").split())[:90],
        "summary": " ".join(str(d.get("summary")).split())[:700],
        "status": str(d.get("status") or "").strip().lower()[:20],
        "left": [str(x)[:160] for x in d.get("left") or [] if str(x).strip()][:3],
        "corrections": [str(x)[:160] for x in d.get("corrections") or [] if str(x).strip()][:3],
        "unattended": {"could": bool(un.get("could")), "why": [w for w in un.get("why") or [] if w in UNATTENDED_REASONS]},
    }


def plain_digest(local: LocalStore, thread, known: dict) -> dict:
    """Without a model: the first thing asked, and what git says happened."""
    rows = _sessions_of(local, thread)
    first = next((local.info(r).get("first_prompt") for r in rows if local.info(r).get("first_prompt")), "")
    title = next((local.info(r).get("title") for r in rows if local.info(r).get("title")), "") or " ".join(first.split())[:60]
    done = "; ".join(c["subject"] for c in known["commits"][:4])
    summary = " ".join(first.split())[:300] + (f" Commits: {done}." if done else "")
    return {"title": title[:90], "summary": summary.strip(), "status": "shipped" if known["commits"] else "",
            "left": [], "corrections": [], "unattended": None, "plain": True}


def digest(local: LocalStore, thread, summarize: Optional[Summarizer], *, now: Optional[float] = None) -> dict:
    now = now if now is not None else time.time()
    known = facts(local, thread)
    d = None
    if summarize and now - thread["last_active"] < MODEL_FOR:
        machine = local._q("SELECT name, hostname FROM machines WHERE id = ?", (thread["machine_id"],))
        prompt = DIGEST_PROMPT.format(
            machine=(machine[0]["name"] or machine[0]["hostname"]) if machine else "his Mac",
            place=thread["place"] or "an unknown folder", branch=thread["branch"] or "unknown",
            start=datetime.fromtimestamp(thread["started_at"]).strftime("%a %d %b %H:%M"),
            end=datetime.fromtimestamp(thread["last_active"]).strftime("%a %d %b %H:%M"),
            facts=json.dumps({k: known[k] for k in ("commits", "prs")}), text=condensed(local, thread))
        d = summarize(prompt)
    return {**(d or plain_digest(local, thread, known)), **known}


def digest_due(local: LocalStore, summarize: Optional[Summarizer], *, store=None, now: Optional[float] = None,
               limit: int = DIGESTS_PER_PASS) -> int:
    """Digest threads that went quiet since they last changed; model ones first, newest first."""
    now = now if now is not None else time.time()
    due = local._q("SELECT * FROM threads WHERE size != digest_size AND last_active < ? ORDER BY last_active DESC LIMIT 200",
                   (now - QUIET,))
    done = 0
    plain_budget = 200
    for t in due:
        recent = now - t["last_active"] < MODEL_FOR
        if recent and summarize and done >= limit:
            continue
        if not recent:
            if plain_budget <= 0:
                break
            plain_budget -= 1
        d = digest(local, t, summarize if recent else None, now=now)
        before = _j(t["digest"], {})
        with local._lock:
            local._db.execute("UPDATE threads SET digest = ?, digest_size = ?, digested_at = ? WHERE id = ?",
                              (json.dumps(d), t["size"], now, t["id"]))
        if recent:
            done += 1
            if store is not None and d.get("summary"):
                again = " (continued)" if before.get("summary") else ""
                where = (_j(t["repo"], None) or {}).get("name") or Path(t["place"]).name or "a folder"
                try:
                    store.note("coding.work", f"{where}{again}: {d.get('title')}. {d['summary']}", ref=f"thread:{t['id']}", space="Coding")
                except Exception:
                    log.exception("domo: noting thread %s", t["id"])
    return done


# ---- what Hermes and the app read ---------------------------------------------------------------

def describe(local: LocalStore, t, machines: Optional[dict] = None) -> dict:
    if machines is None:
        machines = {m["id"]: m for m in local.machines()}
    m = machines.get(t["machine_id"]) or {}
    repo = _j(t["repo"], None) or {}
    d = _j(t["digest"], {})
    return {"id": t["id"], "machine": m.get("name") or t["machine_id"], "repo": repo.get("name") or Path(t["place"]).name,
            "place": t["place"], "branch": t["branch"], "remote": repo.get("remote") or "",
            "started_at": t["started_at"], "last_active": t["last_active"], "sessions": _j(t["sessions"], []),
            "title": d.get("title") or "", "summary": d.get("summary") or "", "status": d.get("status") or "",
            "left": d.get("left") or [], "corrections": d.get("corrections") or [],
            "commits": d.get("commits") or [], "prs": d.get("prs") or [], "prompts": d.get("prompts"),
            "unattended": d.get("unattended"), "digested": t["digest_size"] == t["size"],
            "when": datetime.fromtimestamp(t["last_active"]).strftime("%a %d %b %H:%M")}


def recent(local: LocalStore, *, days: float = 3, limit: int = 30, now: Optional[float] = None) -> list[dict]:
    now = now if now is not None else time.time()
    machines = {m["id"]: m for m in local.machines()}
    rows = local._q("SELECT * FROM threads WHERE last_active >= ? ORDER BY last_active DESC LIMIT ?", (now - days * 86400, limit))
    return [describe(local, t, machines) for t in rows]


def day(local: LocalStore, date: str = "") -> dict:
    when = datetime.strptime(date, "%Y-%m-%d") if date else datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start, end = when.timestamp(), (when + timedelta(days=1)).timestamp()
    rows = local._q("SELECT * FROM threads WHERE last_active >= ? AND started_at < ? ORDER BY started_at", (start, end))
    machines = {m["id"]: m for m in local.machines()}
    threads = [describe(local, t, machines) for t in rows]
    by_repo: dict[str, list] = {}
    for t in threads:
        by_repo.setdefault(t["repo"] or "other", []).append(t)
    return {"date": when.strftime("%Y-%m-%d"), "repos": by_repo, "threads": len(threads)}


def thread(local: LocalStore, thread_id: str) -> Optional[dict]:
    rows = local._q("SELECT * FROM threads WHERE id = ?", (thread_id,))
    if not rows:
        return None
    return {**describe(local, rows[0]), "transcript": condensed(local, rows[0], budget=8000)}


# ---- the background pass ------------------------------------------------------------------------

_thread: Optional[threading.Thread] = None
_guard = threading.Lock()
_homes: dict[str, Path] = {}


def watch(home: Path) -> None:
    """Keep this profile's threads stitched and digested, in the background, from now on."""
    global _thread
    _homes[str(home)] = home
    with _guard:
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(target=_loop, name="domo-work", daemon=True)
            _thread.start()


def run_once(home: Path, *, summarize: Optional[Summarizer] = None, store=None) -> dict:
    from . import local as local_mod
    local = local_mod.store_for(home)
    for row in local.stale(limit=100):
        local.info(row)
    placed = stitch(local)
    if store is None:
        from .tools import store_for
        store = store_for(home)
    digested = digest_due(local, summarize if summarize is not None else claude_summarizer(local), store=store)
    return {"placed": placed, "digested": digested}


def _loop() -> None:
    time.sleep(20)
    while True:
        for home in list(_homes.values()):
            try:
                run_once(home)
            except Exception:
                log.exception("domo: work pass for %s", home)
        time.sleep(EVERY)
