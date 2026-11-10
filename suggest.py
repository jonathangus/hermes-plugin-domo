"""Next up: what the person is about to do, before they ask (docs/PRODUCT.md, "Now" 1).

Built from local signals only, so it's ready the moment the app opens and costs nothing: Hermes's own
chats (``state.db``, read-only), Domo's store (runs, to-dos, ideas the agent prepared), the routines, and
the git repos on the Mac. Each suggestion carries its reason and one action; feedback (done, dismiss,
more, less) hides it or tunes how much its kind counts.

The ladder is observe → prepare → propose: nothing here acts. Tapping a suggestion opens a chat with its
prompt, and the person sends it.
"""

from __future__ import annotations

import math
import re
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import memory, search

KIND_WEIGHT = {"habit": 0.86, "reply": 0.92, "promise": 0.9, "check-in": 0.84, "run": 0.88, "routine-failed": 0.82, "idea": 0.8, "repo": 0.72, "todo": 0.62, "make-routine": 0.55}
LIMIT = 8
STOP = set("the a an and or to of in on for with at by from is are was be it this that these those i you we me my our your please "
           "can could would will should do does did have has had make get let just about what which when where how why there here "
           "och att det som en ett är på för med till av den har vi jag du kan".split())
QUESTION = re.compile(r"([^.!?\n]{6,200}\?)\s*$")


def _ago(seconds: float) -> str:
    s = max(0, time.time() - seconds)
    if s < 3600:
        return f"{max(1, int(s // 60))} min ago"
    if s < 86400:
        return f"{int(s // 3600)} h ago"
    return f"{int(s // 86400)} d ago"


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[\wåäö]+", text.lower()) if len(w) > 2 and w not in STOP and not w.isdigit()}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _first_line(text: str, n: int = 90) -> str:
    line = " ".join(text.strip().split())
    return line if len(line) <= n else line[: n - 1].rsplit(" ", 1)[0] + "…"


# ---- signals ----------------------------------------------------------------------------------

def _chats(home: Path, days: int = 30) -> list[dict]:
    """Recent chats with their first and last words, from Hermes's own database."""
    db = search._hermes(home)
    if db is None:
        return []
    try:
        since = time.time() - days * 86400
        hidden = ",".join("?" * len(search.HIDDEN_SOURCES))
        rows = db.execute(
            f"SELECT id, title, source, started_at, COALESCE(last_activity_at, ended_at, started_at) AS last_active, archived"
            f" FROM sessions WHERE source NOT IN ({hidden}) AND hidden = 0 AND started_at > ? ORDER BY last_active DESC LIMIT 400",
            (*search.HIDDEN_SOURCES, since)).fetchall()
        out = []
        for r in rows:
            first = db.execute("SELECT content FROM messages WHERE session_id = ? AND role = 'user' AND active = 1 ORDER BY id LIMIT 1", (r["id"],)).fetchone()
            last = db.execute("SELECT id, role, content FROM messages WHERE session_id = ? AND role IN ('user','assistant') AND active = 1"
                              " AND COALESCE(content, '') != '' ORDER BY id DESC LIMIT 1", (r["id"],)).fetchone()
            out.append({**dict(r), "first": search.clean(first["content"]) if first else "",
                        "last_role": last["role"] if last else None, "last": search.clean(last["content"]) if last else "",
                        "last_id": last["id"] if last else None})
        return out
    finally:
        db.close()


def _replies(chats: list[dict]) -> list[dict]:
    """The agent asked something and the person never answered."""
    out = []
    for c in chats:
        if c["archived"] or c["source"] == "cron" or c["last_role"] != "assistant" or time.time() - (c["last_active"] or 0) > 3 * 86400:
            continue
        m = QUESTION.search(c["last"])
        if not m:
            continue
        q = _first_line(m.group(1).strip(), 110)
        out.append({"kind": "reply", "key": f"reply:{c['id']}:{c['last_id']}", "title": q, "at": c["last_active"],
                    "because": f"Domo asked in “{_first_line(c['title'] or 'a chat', 40)}” {_ago(c['last_active'])}",
                    "action": {"type": "chat", "session_id": c["id"], "prefill": ""}})
    return out


def _repeats(chats: list[dict], routines: list[dict]) -> list[dict]:
    """The same ask, three or more times this month, and no routine doing it."""
    asks = [(c, _words(c["first"])) for c in chats if c["source"] != "cron" and len(_words(c["first"])) >= 2]
    used: set[str] = set()
    out = []
    routine_words = [_words(str(r.get("prompt") or r.get("name") or "")) for r in routines]
    for c, w in asks:
        if c["id"] in used:
            continue
        group = [(d, v) for d, v in asks if d["id"] not in used and _jaccard(w, v) >= 0.5]
        if len(group) < 3 or any(_jaccard(w, rw) >= 0.4 for rw in routine_words):
            continue
        used.update(d["id"] for d, _ in group)
        times = [datetime.fromtimestamp(d["started_at"]) for d, _ in group]
        hour = int(statistics.median(t.hour for t in times))
        weekdays = Counter(t.weekday() for t in times)
        workdays = all(d < 5 for d in weekdays)
        when = f"weekdays around {hour:02d}:00" if workdays else f"around {hour:02d}:00"
        cron = f"0 {hour} * * {'1-5' if workdays else '*'}"
        text = _first_line(c["first"], 160)
        out.append({"kind": "make-routine", "key": f"make-routine:{sorted(w)[:6]}", "title": f"Make “{_first_line(c['first'], 48)}” a routine",
                    "at": max(d["last_active"] or 0 for d, _ in group),
                    "because": f"You asked this {len(group)} times since {min(times):%b %d}, usually {when}",
                    "action": {"type": "chat", "prefill": f"Create a routine with the schedule `{cron}` that does this, and tell me when it first runs: {text}"}})
    return out


def _runs(store) -> list[dict]:
    out = []
    for r in store.list("run", limit=50):
        if r["status"] not in ("stalled", "failed"):
            continue
        note = str(r["data"].get("note") or "")
        repo = str(r["data"].get("repo_name") or "")
        out.append({"kind": "run", "key": f"run:{r['id']}:{r['status']}", "title": f"{'Resume' if r['status'] == 'stalled' else 'Fix'} {r['title']}",
                    "at": r["updated_at"], "space": r["space"],
                    "because": " · ".join(x for x in (f"{'stopped' if r['status'] == 'stalled' else 'failed'} {_ago(r['updated_at'])}", note, repo) if x),
                    "action": {"type": "chat", "session_id": r["session_id"] or None,
                               "prefill": f"Pick up the run “{r['title']}”{f' ({note})' if note else ''}. What's needed to finish it?"}})
    return out


def due_at(due: object) -> Optional[float]:
    """A to-do's due moment from its ISO date or date-time; a bare date means the end of that day.
    None for anything else (older to-dos kept free text like "Friday")."""
    text = str(due or "").strip()
    if not text:
        return None
    try:
        if len(text) == 10:
            return datetime.fromisoformat(text).replace(hour=23, minute=59).timestamp()
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _verb(title: str) -> str:
    """'Send the deck' as it reads mid-sentence: 'send the deck' (a name or an acronym keeps its capital)."""
    return title[:1].lower() + title[1:] if title[1:2].islower() else title


def _whole_day(at: float) -> bool:
    d = datetime.fromtimestamp(at)
    return (d.hour, d.minute) == (23, 59)


def _when(at: float, now: float) -> str:
    d, today = datetime.fromtimestamp(at), datetime.fromtimestamp(now)
    days = (d.date() - today.date()).days
    day = "today" if days == 0 else "tomorrow" if days == 1 else "yesterday" if days == -1 else f"{d:%a %d %b}"
    return day if _whole_day(at) else f"{day} {d:%H:%M}"


def _todos(store) -> list[dict]:
    """What Domo may do, and promises (anyone's) coming due: from the start of the day, or three hours before a time."""
    now = time.time()
    out = []
    for t in store.list("todo", status="open"):
        owner, who = t["data"].get("owner"), str(t["data"].get("who") or "").strip()
        due = due_at(t["data"].get("due"))
        start = None if due is None else datetime.fromtimestamp(due).replace(hour=0, minute=0).timestamp() if _whole_day(due) else due - 3 * 3600
        coming = start is not None and now >= start
        by = f"by {_when(due, now)}" if due is not None else ""
        past = " · overdue" if coming and now > due else ""
        base = {"key": f"todo:{t['id']}", "at": max(t["created_at"], start or 0), "space": t["space"]}
        if who:
            if coming:
                out.append({**base, "kind": "check-in", "title": f"Check in with {who}: {t['title']}", "because": f"{who} promised it {by}{past}",
                            "action": {"type": "chat", "prefill": f"{who} said they'd {_verb(t['title'])} {by}. Find out whether it happened"
                                       " (our chats, shared notes); if it didn't, draft a friendly nudge for me to send."}})
        elif owner == "domo":
            out.append({**base, "kind": "todo", "title": f"Let Domo: {t['title']}",
                        "because": f"You asked for it {by}{past}" if coming else f"On your list {_ago(t['created_at'])}, marked for Domo to do",
                        "action": {"type": "chat", "prefill": f"Go ahead with the to-do “{t['title']}”. Ask me before anything goes out or costs money."}})
        elif coming:
            out.append({**base, "kind": "promise", "title": t["title"], "because": f"You promised it {by}{past}",
                        "action": {"type": "chat", "prefill": f"I said I'd {_verb(t['title'])} {by}. Help me get it done now: prepare what you can"
                                   " and tell me what's left for me."}})
    return out


def _ideas(store) -> list[dict]:
    now = time.time()
    out = []
    for s in store.list("suggestion", status="open"):
        if float(s["data"].get("expires_at") or now + 1) < now:
            continue
        out.append({"kind": "idea", "key": f"idea:{s['id']}", "title": s["title"], "at": s["created_at"], "space": s["space"],
                    "because": s["body"] or f"Prepared {_ago(s['created_at'])}",
                    "action": {"type": "chat", "session_id": None, "prefill": str(s["data"].get("prompt") or "")}, "item": s["id"]})
    return out


def _failed_routines(routines: list[dict]) -> list[dict]:
    out = []
    for r in routines:
        status = str(r.get("last_status") or "").lower()
        if not status or status in ("ok", "success", "completed", "complete", "silent"):
            continue
        name = r.get("name") or r.get("id")
        out.append({"kind": "routine-failed", "key": f"routine-failed:{r.get('id')}:{r.get('last_run_at')}", "title": f"Fix the routine “{name}”",
                    "at": time.time(), "because": f"Its last run ended {status}" + (f" at {r['last_run_at']}" if r.get("last_run_at") else ""),
                    "action": {"type": "chat", "prefill": f"The routine “{name}” failed ({status}). Find out why, fix it, and run it once to show me."}})
    return out


def _repos(repos: list[dict]) -> list[dict]:
    out = []
    for r in repos:
        if not r.get("dirty") or not r.get("last_commit_at") or time.time() - r["last_commit_at"] > 4 * 86400:
            continue
        out.append({"kind": "repo", "key": f"repo:{r['id']}:{r.get('last_commit_at')}", "title": f"Pick up {r['name']} on {r.get('branch') or 'main'}",
                    "at": r["last_commit_at"], "space": "Coding",
                    "because": f"Uncommitted changes · last commit “{_first_line(r.get('last_commit') or '', 40)}” {_ago(r['last_commit_at'])}",
                    "action": {"type": "chat", "prefill": f"In {r['path']} on {r.get('branch') or 'main'}: summarise the uncommitted changes and what's left, then propose the next step. Don't change anything yet."}})
    return out


# ---- ranking ----------------------------------------------------------------------------------

def rank(found: list[dict], weights: dict[str, float], hidden: set[str], limit: int = LIMIT) -> list[dict]:
    now = time.time()
    scored = []
    for s in found:
        if s["key"] in hidden:
            continue
        age_h = max(0.0, (now - (s.get("at") or now)) / 3600)
        fresh = math.exp(-age_h / 72)   # a three-day half-life-ish fade
        s["score"] = round(KIND_WEIGHT.get(s["kind"], 0.5) * weights.get(s["kind"], 1.0) * (0.55 + 0.45 * fresh), 4)
        scored.append(s)
    scored.sort(key=lambda s: -s["score"])
    out, per_kind = [], Counter()
    for s in scored:   # variety: at most three of one kind
        if per_kind[s["kind"]] >= 3:
            continue
        per_kind[s["kind"]] += 1
        out.append(s)
        if len(out) >= limit:
            break
    return out


def build(home: Path, store, *, routines: list[dict], repos: list[dict], limit: int = LIMIT) -> list[dict]:
    chats = _chats(home)
    found = (_replies(chats) + _runs(store) + _failed_routines(routines) + _ideas(store) + memory.habits_due(store) + _repos(repos)
             + _todos(store) + _repeats(chats, routines))
    return rank(found, store.kind_weights(), store.hidden_suggestions(), limit)


def vote(store, key: str, kind: str, verdict: str, item: Optional[str] = None, title: str = "", instinct: Optional[str] = None,
         until: Optional[float] = None) -> None:
    if verdict == "snooze":   # not a judgement on the suggestion: only "not now, then"
        store.snooze(key, float(until or 0))
        store.note("suggestion", f"snoozed the {kind} suggestion “{title or key}” until {datetime.fromtimestamp(float(until or 0)):%a %H:%M}", ref=key)
        return
    store.vote(key, kind, verdict)
    if item and verdict in ("done", "dismiss"):   # an idea the agent left is settled once acted on or turned down
        store.update(item, status="done" if verdict == "done" else "dismissed")
    if instinct:   # a habit that was right gets surer; one that wasn't, less so
        try:
            if verdict in ("done", "more"):
                pattern = store.instinct(instinct)["pattern"]
                store.learn(pattern)
            else:
                store.weaken(instinct, 0.25 if verdict == "less" else 0.1)
        except Exception:   # forgotten meanwhile
            pass
    words = {"done": "took", "dismiss": "set aside", "more": "wanted more like", "less": "wanted less like"}[verdict]
    store.note("suggestion", f"{words} the {kind} suggestion “{title or key}”", ref=key)
