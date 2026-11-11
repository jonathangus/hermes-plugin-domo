"""The tools Domo gives the agent. Each handler returns a JSON string and never raises."""

from __future__ import annotations

import time

import contextvars
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from . import shared, suggest
from .store import Store, StoreError, store_path

log = logging.getLogger(__name__)

_stores: dict[str, Store] = {}


def _home() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home()


home_resolver: Callable[[], Path] = _home
# The profile home each tool call writes to. A multiplexed gateway loads plugins once per profile
# and runs tool handlers on worker threads, so the home is bound at registration (see
# bound_tools) rather than looked up per call; this is the fallback for direct use and tests.
_bound_home: "contextvars.ContextVar[Optional[Path]]" = contextvars.ContextVar("domo_home", default=None)


def store_for(home: Path) -> Store:
    path = str(store_path(home))
    store = _stores.get(path)
    if store is None:
        store = _stores[path] = Store(Path(path))
    return store


def current_home() -> Path:
    return _bound_home.get() or home_resolver()


def current_store() -> Store:
    return store_for(current_home())


def bound_tools(home: Path, tools: Optional[list] = None) -> list[tuple[str, dict, Callable[..., str], str]]:
    """TOOLS (or the given tools) with every handler pinned to one profile's store."""
    def pin(handler: Callable[..., str]) -> Callable[..., str]:
        def pinned(args: dict, **kwargs: Any) -> str:
            token = _bound_home.set(home)
            try:
                return handler(args, **kwargs)
            finally:
                _bound_home.reset(token)
        pinned.__name__ = handler.__name__
        return pinned
    return [(name, schema, pin(handler), emoji) for name, schema, handler, emoji in (TOOLS if tools is None else tools)]


def _ok(**payload: Any) -> str:
    return json.dumps({"ok": True, **payload})


def _fail(message: str) -> str:
    return json.dumps({"ok": False, "error": message})


def _guard(fn: Callable[..., str]) -> Callable[..., str]:
    def handler(args: dict, **kwargs: Any) -> str:
        try:
            return fn(args or {}, kwargs.get("session_id") or "")
        except (StoreError, shared.SharedError) as exc:
            return _fail(str(exc))
        except Exception as exc:  # a tool must never take the turn down
            log.exception("domo tool %s failed", fn.__name__)
            return _fail(f"internal error: {exc}")
    handler.__name__ = fn.__name__
    return handler


def _brief_item(item: dict) -> dict:
    return {k: item[k] for k in ("id", "kind", "status", "title", "space")}


# ---- handlers ---------------------------------------------------------------------------------

@_guard
def domo_ask(args: dict, session_id: str) -> str:
    options = [str(o).strip() for o in (args.get("options") or []) if str(o).strip()]
    if len(options) < 2:
        return _fail("give at least two options, the safe one first")
    item = current_store().create(
        "decision", args.get("title", ""), body=args.get("body", ""), space=args.get("space", ""),
        session_id=session_id,
        data={"options": options, "risk": args.get("risk", "reversible"),
              "kind": args.get("kind", "approve"), "details": args.get("details", "")})
    return _ok(decision=_brief_item(item),
               next="Asked. Stop here and tell the person briefly what you asked; the answer comes back"
                    " as a new message later.")


@_guard
def domo_todo(args: dict, session_id: str) -> str:
    store = current_store()
    action = args.get("action", "add")
    if action == "done":
        item = store.update(args.get("id", ""), status="done")
        return _ok(todo=_brief_item(item))
    owner = args.get("owner", "me")
    if owner not in ("me", "domo"):
        return _fail("owner is 'me' or 'domo'")
    due = str(args.get("due") or "").strip()
    if due and suggest.due_at(due) is None:
        return _fail("due is a local ISO date or date-time, like 2026-11-13 or 2026-11-13T17:00: work 'Friday' out from today")
    item = store.create("todo", args.get("title", ""), body=args.get("body", ""), space=args.get("space", ""), session_id=session_id,
                        data={"owner": owner, "due": due, **({"who": str(args["who"]).strip()} if str(args.get("who") or "").strip() else {})})
    return _ok(todo=_brief_item(item))


@_guard
def domo_receipt(args: dict, session_id: str) -> str:
    data = {"proof": args.get("proof", ""), "undo": args.get("undo", "")}
    if args.get("cost_usd") is not None:
        data["cost_usd"] = float(args["cost_usd"])
    item = current_store().create("receipt", args.get("title", ""), body=args.get("detail", ""),
                                  space=args.get("space", ""), session_id=session_id, data=data)
    return _ok(receipt=_brief_item(item))


@_guard
def domo_run(args: dict, session_id: str) -> str:
    store = current_store()
    action = args.get("action", "")
    if action == "start":
        data = {"done_means": args.get("done_means", ""), "progress": 0.0, "repo": args.get("repo", ""),
                "expected_minutes": args.get("expected_minutes"), "budget_usd": args.get("budget_usd"),
                "stall_after_minutes": args.get("stall_after_minutes") or 30,
                "needs": args.get("needs", "")}
        item = store.create("run", args.get("title", ""), body=args.get("note", ""),
                            space=args.get("space", ""), session_id=session_id, data=data)
        return _ok(run=_brief_item(item), next="Send action='beat' while it works and 'finish' or 'fail' at the end.")
    run_id = args.get("id", "")
    if action == "beat":
        item = store.run_beat(run_id, progress=args.get("progress"), note=args.get("note", ""),
                              cost_usd=args.get("cost_usd"))
    elif action in ("finish", "fail"):
        item = store.run_finish(run_id, ok=action == "finish", note=args.get("note", ""),
                                proof=args.get("proof", ""), cost_usd=args.get("cost_usd"))
    else:
        return _fail("action is start, beat, finish or fail")
    return _ok(run=_brief_item(item))


@_guard
def domo_brief(args: dict, session_id: str) -> str:
    text = (args.get("text") or "").strip()
    if not text:
        return _fail("text is required")
    title = (args.get("title") or text.split(".")[0])[:120]
    item = current_store().create("brief", title, body=text, session_id=session_id,
                                  data={"highlights": args.get("highlights") or []})
    return _ok(brief=_brief_item(item))


@_guard
def domo_fact(args: dict, session_id: str) -> str:
    store = current_store()
    if args.get("action") == "set":
        fact = store.set_fact(args.get("key", ""), args.get("value", ""), source=args.get("source", ""),
                              space=args.get("space", ""), origin=args.get("origin") or "stated")
    else:
        fact = store.get_fact(args.get("key", ""))
    return _ok(fact=fact)


@_guard
def domo_suggest(args: dict, session_id: str) -> str:
    title = (args.get("title") or "").strip()
    prompt = (args.get("prompt") or "").strip()
    if not title or not prompt:
        return _fail("title and prompt are required")
    hours = float(args.get("expires_hours") or 36)
    item = current_store().create("suggestion", title[:140], body=(args.get("because") or "").strip()[:240], space=args.get("space", ""),
                                  session_id=session_id, data={"prompt": prompt, "expires_at": time.time() + hours * 3600})
    return _ok(suggestion=_brief_item(item))


@_guard
def domo_memory(args: dict, session_id: str) -> str:
    from . import memory
    store = current_store()
    action = args.get("action") or "recall"
    if action == "recall":
        return _ok(**memory.recall(store, since=float(args.get("since") or 0)))
    if action == "learn":
        return _ok(instinct=store.learn(args.get("pattern", ""), kind=args.get("kind") or "habit", origin=args.get("origin") or "inferred",
                                        space=args.get("space", ""), prompt=args.get("prompt", ""), hours=str(args.get("hours") or ""),
                                        days=str(args.get("days") or "")))
    if action == "weaken":
        return _ok(instinct=store.weaken(args.get("id", ""), float(args.get("by") or 0.2)))
    if action == "synthesize":
        for section, text in (args.get("sections") or {}).items():
            store.set_synthesis(section, str(text))
        return _ok(synthesis={k: v["text"] for k, v in store.synthesis().items()})
    if action == "note":
        return _ok(event=store.note("note", args.get("text", ""), ref=session_id or "", space=args.get("space", "")))
    return _fail("action is recall, learn, weaken, synthesize or note")


@_guard
def domo_open(args: dict, session_id: str) -> str:
    store = current_store()
    kinds = [args["kind"]] if args.get("kind") else ["decision", "todo", "run"]
    items = [_brief_item(i) for k in kinds for i in store.list(k, status="open", limit=50)]
    return _ok(items=items)


@_guard
def domo_shared(args: dict, session_id: str) -> str:
    home = current_home()
    if not shared.enabled(home):
        return _fail("shared notes are not enabled for this agent")
    action = args.get("action", "list")
    if action == "list":
        return _ok(topics=shared.topics(home))
    topic = args.get("topic", "")
    if action == "read":
        text = shared.read(home, topic)
        return _ok(topic=shared.slug(topic), text=text, exists=bool(text))
    if action == "append":
        return _ok(topic=shared.slug(topic), added=shared.append(home, topic, args.get("text", "")))
    if action == "write":
        shared.write(home, topic, args.get("text", ""))
        return _ok(topic=shared.slug(topic), written=True)
    return _fail("action must be list, read, append or write")


@_guard
def domo_work(args: dict, session_id: str) -> str:
    """Work done on the person's other machines, from Domo Link: by Claude Code or Cursor there, not by you."""
    from . import local, work
    home = current_home()
    if not local.db_path(home).exists():
        return _fail("no other machine is connected: Domo Link isn't set up on one yet")
    store = local.store_for(home)
    action = args.get("action") or "recent"
    if action == "recent":
        return _ok(threads=work.recent(store, days=float(args.get("days") or 3), limit=int(args.get("limit") or 20)),
                   machines=store.machines())
    if action == "day":
        return _ok(**work.day(store, str(args.get("date") or "")))
    if action == "search":
        query = str(args.get("query") or "").strip()
        if not query:
            return _fail("search needs a query")
        return _ok(sessions=[{k: s[k] for k in ("id", "title", "repo", "branch", "machine_name", "last_active", "snippet", "thread_id")}
                             for s in store.search(query, limit=int(args.get("limit") or 10))])
    if action == "thread":
        found = work.thread(store, str(args.get("id") or ""))
        return _ok(thread=found) if found else _fail("no thread with that id")
    if action == "running":
        return _ok(sessions=[s for s in store.sessions(limit=50) if s["status"] in ("working", "waiting", "idle")],
                   machines=store.machines())
    return _fail("action is recent, day, search, thread or running")


@_guard
def domo_minne(args: dict, session_id: str) -> str:
    """The organisation's Minne meetings, for a profile with its key (minne.py)."""
    from . import minne
    token = minne.key(current_home())
    if not token:
        return _fail("no MINNE_KEY in this profile's .env")
    try:
        return _ok(**minne.run(token, args))
    except minne.MinneError as exc:
        return _fail(str(exc))


# ---- schemas ----------------------------------------------------------------------------------

def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}


_S = {"type": "string"}
_SPACE = {"type": "string", "description": "Space this belongs to, e.g. Work, Coding, Personal, Apartment hunt"}

TOOLS: list[tuple[str, dict, Callable[..., str], str]] = [
    ("domo_ask", _schema(
        "domo_ask",
        "Ask the person for a decision instead of acting: anything outbound (sending, booking), paying, deleting,"
        " or a choice only they can make. It shows as a card in the Domo app and on their phone. Then stop the turn.",
        {"title": {"type": "string", "description": "Plain-language question, e.g. 'Book Ekstedt Sat 19:30 for 4?'"},
         "body": {"type": "string", "description": "One or two sentences with the facts that matter"},
         "options": {"type": "array", "items": _S, "description": "Two to four answers, the safe one first"},
         "risk": {"type": "string", "enum": ["reversible", "outbound", "money"]},
         "kind": {"type": "string", "enum": ["approve", "pick", "send", "credential", "delegate", "routine"]},
         "details": {"type": "string", "description": "The exact action or payload, shown behind a disclosure"},
         "space": _SPACE},
        ["title", "options"]), domo_ask, "🙋"),
    ("domo_todo", _schema(
        "domo_todo", "Add a to-do (owner 'me' for the person, 'domo' for work you will do once approved) or mark one done."
        " A promise with a time ('I'll send it Friday') is a to-do with due: it comes up in Next up when due.",
        {"action": {"type": "string", "enum": ["add", "done"]},
         "title": {"type": "string", "description": "What's to be done, as an action: 'Send Anna the deck'"}, "body": _S,
         "owner": {"type": "string", "enum": ["me", "domo"]},
         "due": {"type": "string", "description": "Local ISO date or date-time it's promised by: 2026-11-13 or 2026-11-13T17:00"},
         "who": {"type": "string", "description": "When someone else promised it (Frida, a plumber): their name, to check in with them when due"},
         "id": {"type": "string", "description": "For action 'done'"}, "space": _SPACE},
        []), domo_todo, "☑️"),
    ("domo_receipt", _schema(
        "domo_receipt", "Record something you did, with proof (a confirmation number, link, file) and how to undo it.",
        {"title": _S, "detail": _S, "proof": _S, "undo": _S, "cost_usd": {"type": "number"}, "space": _SPACE},
        ["title"]), domo_receipt, "🧾"),
    ("domo_run", _schema(
        "domo_run", "Track a long job so the person is told when it stalls or finishes instead of asking.",
        {"action": {"type": "string", "enum": ["start", "beat", "finish", "fail"]},
         "id": {"type": "string", "description": "Run id, for beat/finish/fail"},
         "title": _S, "done_means": {"type": "string", "description": "What finished looks like"},
         "expected_minutes": {"type": "number"}, "budget_usd": {"type": "number"},
         "stall_after_minutes": {"type": "number"}, "needs": {"type": "string", "description": "Credentials it depends on"},
         "progress": {"type": "number", "description": "0 to 1"}, "note": _S, "proof": _S,
         "cost_usd": {"type": "number"}, "space": _SPACE,
         "repo": {"type": "string", "description": "For coding runs: the repo's path on this Mac"}},
        ["action"]), domo_run, "🏃"),
    ("domo_brief", _schema(
        "domo_brief", "Publish the brief shown on the Domo home screen: prose, under 200 words, decisions first.",
        {"title": _S, "text": _S, "highlights": {"type": "array", "items": _S}},
        ["text"]), domo_brief, "📰"),
    ("domo_fact", _schema(
        "domo_fact", "Save a stable fact (VAT number, address, contact) or look one up, so it is answered once.",
        {"action": {"type": "string", "enum": ["set", "get"]}, "key": _S, "value": _S, "source": _S, "space": _SPACE,
         "origin": {"type": "string", "enum": ["stated", "inferred"], "description": "stated: they said it; inferred: you worked it out"}},
        ["action", "key"]), domo_fact, "📌"),
    ("domo_memory", _schema(
        "domo_memory",
        "Domo's memory of this person, in layers. recall: the synthesis, instincts (patterns with confidence), facts and"
        " recent events. learn: record a pattern you noticed (seen again, it gets surer; origin 'stated' when they told you);"
        " a habit with prompt plus hours/days becomes a suggestion at that time. weaken: counter-evidence."
        " synthesize: rewrite the short picture (sections who, now, preferences; under 80 words each), as facts, not commands"
        " ('Prefers short answers', not 'Always be short'). note: add to the trace.",
        {"action": {"type": "string", "enum": ["recall", "learn", "weaken", "synthesize", "note"]},
         "pattern": {"type": "string", "description": "learn: 'Reviews merged PRs on Monday mornings', 'Prefers pnpm over npm'"},
         "kind": {"type": "string", "enum": ["habit", "preference", "avoid"]},
         "origin": {"type": "string", "enum": ["stated", "inferred"]},
         "prompt": {"type": "string", "description": "learn, habits: the message that starts it"},
         "hours": {"type": "string", "description": "learn, habits: local hours like '8-10'"},
         "days": {"type": "string", "description": "learn, habits: weekdays like '1-5' (0 is Sunday)"},
         "id": {"type": "string", "description": "weaken: the instinct's id"}, "by": {"type": "number"},
         "sections": {"type": "object", "description": "synthesize: {who, now, preferences}"},
         "text": _S, "since": {"type": "number", "description": "recall: only events after this unix time"}, "space": _SPACE},
        ["action"]), domo_memory, "🧠"),
    ("domo_suggest", _schema(
        "domo_suggest",
        "Leave the person a suggestion for the next time they open Domo: something prepared and ready to go"
        " (a drafted reply, a next step on a repo, a plan for tomorrow). It appears on their home screen with your"
        " reason; tapping it opens a chat with the prompt filled in. Read and prepare only; never act for them here.",
        {"title": {"type": "string", "description": "What to do, as a short imperative: 'Reply to Anna about Thursday'"},
         "because": {"type": "string", "description": "Why now, in one line, naming the source: 'She asked yesterday; you're free 14:00'"},
         "prompt": {"type": "string", "description": "The message that starts the chat when they tap it"},
         "expires_hours": {"type": "number", "description": "When it stops being worth showing (default 36)"},
         "space": _SPACE},
        ["title", "prompt"]), domo_suggest, "💡"),
    ("domo_work", _schema(
        "domo_work",
        "What the person did on their other machines (Claude Code and Cursor on the MacBook), as pieces of work with"
        " a summary, commits and PRs. This was their coding agent there, not you: say so ('on your MacBook, Claude Code…')."
        " recent: the last days' work. day: one day by repo (date YYYY-MM-DD, default today). search: sessions by words."
        " thread: one piece of work in full, with its transcript. running: sessions open there right now.",
        {"action": {"type": "string", "enum": ["recent", "day", "search", "thread", "running"]},
         "days": {"type": "number"}, "date": _S, "query": _S, "id": {"type": "string", "description": "thread: its id"},
         "limit": {"type": "number"}},
        ["action"]), domo_work, "💻"),
    ("domo_open", _schema(
        "domo_open", "List open decisions, to-dos and runs, to avoid asking twice.",
        {"kind": {"type": "string", "enum": ["decision", "todo", "run"]}}, []), domo_open, "📋"),
]

PROMPT = (
    "You are Domo, this person's chief of staff. Prepare and do the work; bring them decisions, not questions.\n"
    "- Before anything outbound, paying, booking or deleting, call domo_ask and stop the turn.\n"
    "- Long jobs: domo_run start, beat while working, finish with proof.\n"
    "- After you act, domo_receipt with proof and how to undo.\n"
    "- Facts people ask for again: domo_fact.\n"
    "- Check domo_open before asking something you may already have asked.\n"
    "- When someone commits to something by a time ('I'll send the deck Friday', 'Frida books the dentist this week'),"
    " domo_todo add it with due (and who, if it's someone else) so it comes back when due.\n"
    "- Their work on other machines (Claude Code, Cursor on the MacBook): domo_work.\n"
    "- When they tell you how they like things, or you see the same pattern twice, domo_memory learn it.\n"
    "- Reports, digests and replies with posts, numbers or screenshots: follow the domo-rich-reports skill.\n"
    "- When you see something they will want to do next (a reply owed, a next step, a fix), prepare it and"
    " leave it with domo_suggest instead of saying 'let me know'.\n"
    "- When they ask for a file (or you made one for them), attach it: put MEDIA:/absolute/path on its own line"
    " in your reply, for any file type. The app turns it into a download. Never give only the path as text.\n"
    "  This overrides any note that non-image files are not delivered: the Domo app handles MEDIA: itself."
)

SHARED_TOOLS: list[tuple[str, dict, Callable[..., str], str]] = [
    ("domo_shared", _schema(
        "domo_shared",
        "The couple's shared notes, read and written by both partners' agents: one markdown note per topic"
        " (Apartment hunt, Daughter, Trips, Home, Money). 'list' topics, 'read' one, 'append' a dated line"
        " (decisions, facts, next steps), or 'write' to tidy a whole note (keep everything still true).",
        {"action": {"type": "string", "enum": ["list", "read", "append", "write"]},
         "topic": {"type": "string", "description": "e.g. 'Apartment hunt'"},
         "text": {"type": "string", "description": "For append: the new line. For write: the whole note"}},
        ["action"]), domo_shared, "🤝"),
]


def shared_prompt(partners: list[str]) -> str:
    who = " and ".join(partners) if partners else "the couple"
    return (
        f"Shared notes (domo_shared) are the memory {who} share; the other partner's agent reads them too.\n"
        "- Joint matters (apartment hunt, their daughter, trips, home, household money) go there, not in your own"
        " memory or USER.md.\n"
        "- Before answering about one, read its note. After a decision, fact or next step, append it.\n"
        "- Keep private matters (work, health, one partner's own plans) out of shared notes."
    )
