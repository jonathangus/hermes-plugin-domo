"""Domo app backend, mounted by the Hermes dashboard at /api/plugins/domo/.

The dashboard's login gate guards every route. Domo's own data (brief, decisions, to-dos, runs,
receipts, facts, spaces) is read from the profile's store; chat, sessions and routines are passed
through to that profile's API server over loopback, with the space each is filed under joined in.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the plugins dir, so `domo` imports as a package

from domo import access, bridge, filing, files as attach, memory, repos as repo_scan, rtk, search as chat_search, stitch, suggest  # noqa: E402
from domo.store import Store, StoreError, store_path  # noqa: E402

log = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(access.guard)])
VERSION = "0.1.0"
INBOX_TITLE = "Domo inbox"
DOMO_SOURCE = "api_server"   # Hermes's source for sessions started through its API server, i.e. by Domo
RUN_TITLE = re.compile(r"^(?P<job>.+) · [A-Z][a-z]{2} \d{1,2} \d{2}:\d{2}$")  # "<job name> · Nov 08 00:00"

_stores: dict[str, Store] = {}
transport = None  # tests inject an httpx transport for the API server


def _store(profile: str) -> Store:
    try:
        home = bridge.profile_home(profile)
    except bridge.ProfileError as exc:
        raise HTTPException(404, str(exc))
    path = str(store_path(home))
    if path not in _stores:
        _stores[path] = Store(Path(path))
        _stores[path].seed_spaces(_seed_names(profile, home))
    return _stores[path]


def _seed_names(profile: str, home: Path) -> list[str]:
    """DOMO_SPACES (the profile's .env, then the process's), else defaults for the kind of profile."""
    listed = bridge.read_env(home).get("DOMO_SPACES") or os.environ.get("DOMO_SPACES", "")
    if listed.strip():
        return [n.strip() for n in listed.split(",") if n.strip()]
    profile = (profile or bridge.DEFAULT).lower()
    kind = bridge.profile_info(profile).get("kind")
    if profile in (bridge.DEFAULT, "family") or kind == "group":
        return ["Family"]
    if profile == "ours" or kind == "shared":
        return ["Apartment hunt", "Daughter", "Trips", "Home"]
    return ["Work", "Coding", "Personal"]


def _api(profile: str) -> bridge.ApiServer:
    try:
        return bridge.ApiServer(profile, transport=transport)
    except bridge.ProfileError as exc:
        raise HTTPException(404, str(exc))


def _store_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except StoreError as exc:
        raise HTTPException(409 if "already" in str(exc) else 400, str(exc))


async def _passthrough(profile: str, method: str, path: str, *, body: Optional[dict] = None,
                       params: Optional[dict] = None):
    api = _api(profile)
    try:
        response = await api.request(method, path, json_body=body, params=params)
    except bridge.ProfileError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Hermes API server unreachable for {profile}: {exc}")
    if response.status_code >= 400:
        raise HTTPException(response.status_code, response.text[:500])
    return response.json() if response.content else {}


# ---- meta -------------------------------------------------------------------------------------

@router.get("/health")
async def health(request: Request):
    allowed = access.allowed_profiles(request)
    profiles = []
    for name in bridge.profile_names():
        if allowed is not None and name not in allowed:
            continue
        entry = {**bridge.profile_info(name), "api_server": "unknown", "rtk": "on" if rtk.available() else "off"}
        try:
            response = await _api(name).request("GET", "/health", timeout=3.0)
            entry["api_server"] = "ok" if response.status_code == 200 else f"http {response.status_code}"
        except Exception as exc:
            entry["api_server"] = f"unreachable: {type(exc).__name__}"
        profiles.append(entry)
    return {"domo": VERSION, "profiles": profiles}


def _git_stamp(path) -> dict:
    """Last commit of a checkout: its date (ISO) and short hash. Empty when it isn't a git checkout."""
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(path), "log", "-1", "--format=%cI|%h"], capture_output=True, text=True, timeout=3).stdout.strip()
        date, sha = out.split("|")
        return {"updated": date, "sha": sha}
    except Exception:
        return {}


@router.get("/versions")
def versions():
    """The Hermes agent (its code and gateway) as installed on this Mac: version and when its code last changed."""
    agent: dict = {}
    try:
        import hermes_cli
        agent = {"version": str(getattr(hermes_cli, "__version__", "")), **_git_stamp(Path(hermes_cli.__file__).resolve().parent.parent)}
    except Exception:
        pass
    return {"agent": agent}


@router.get("/profiles")
def profiles(request: Request):
    allowed = access.allowed_profiles(request)
    return {"profiles": [bridge.profile_info(n) for n in bridge.profile_names() if allowed is None or n in allowed]}


# ---- Domo items -------------------------------------------------------------------------------

@router.get("/feed")
def feed(profile: str = Query(bridge.DEFAULT, alias="agent")):
    _stitching()
    chat_search.ensure_running()
    store = _store(profile)
    return {"profile": bridge.profile_info(profile), **store.feed()}


@router.get("/changes")
def changes(profile: str = Query(bridge.DEFAULT, alias="agent"), since: int = Query(0, ge=0)):
    store = _store(profile)
    items, cursor = store.changed_since(since)
    return {"items": items, "spaces": store.spaces_changed_since(since), "cursor": cursor}


@router.get("/items")
def items(profile: str = Query(bridge.DEFAULT, alias="agent"), kind: Optional[str] = None, status: Optional[str] = None,
          space: Optional[str] = None, limit: int = Query(100, ge=1, le=500)):
    return {"items": _store(profile).list(kind, status=status, space=space, limit=limit)}


class TodoIn(BaseModel):
    title: str
    body: str = ""
    space: str = ""
    owner: str = Field("me", pattern="^(me|domo)$")


@router.post("/todos")
def add_todo(todo: TodoIn, profile: str = Query(bridge.DEFAULT, alias="agent")):
    return _store_call(_store(profile).create, "todo", todo.title, body=todo.body, space=todo.space,
                       data={"owner": todo.owner})


class ItemPatch(BaseModel):
    status: str = Field(..., pattern="^(open|done|dismissed)$")


@router.patch("/items/{item_id}")
def patch_item(item_id: str, patch: ItemPatch, profile: str = Query(bridge.DEFAULT, alias="agent")):
    store = _store(profile)
    item = _store_call(store.get, item_id)
    if item["kind"] not in ("todo", "receipt"):
        raise HTTPException(400, "only to-dos and receipts change status here; answer decisions instead")
    updated = _store_call(store.update, item_id, status=patch.status)
    if item["kind"] == "todo" and patch.status == "done":
        store.note("todo", f"did “{item['title']}”", ref=item_id, space=item["space"])
    return updated


class Answer(BaseModel):
    choice: str
    note: str = ""
    by: str = ""


@router.post("/decisions/{decision_id}/answer")
def answer(decision_id: str, body: Answer, background: BackgroundTasks, profile: str = Query(bridge.DEFAULT, alias="agent")):
    store = _store(profile)
    item = _store_call(store.answer, decision_id, body.choice, note=body.note, by=body.by)
    store.note("decision", f"answered “{item['title']}” with {body.choice}" + (f" ({body.note})" if body.note else ""), ref=decision_id, space=item["space"])
    item = store.update(decision_id, data={"resume": "queued"})
    background.add_task(_resume_agent, profile, decision_id)
    return item


async def _resume_agent(profile: str, decision_id: str) -> None:
    """Tell the agent the answer in its Domo inbox session, so parked work continues."""
    store = _store(profile)
    item = store.get(decision_id)
    message = (f"[Domo] Decision answered: \"{item['title']}\" → {item['data'].get('choice')}."
               + (f" Note: {item['data']['note']}." if item["data"].get("note") else "")
               + (f" It was asked in session {item['session_id']}." if item["session_id"] else "")
               + " Continue the work accordingly and record a receipt when done.")
    try:
        session_id = await _inbox_session(profile, store)
        response = await _api(profile).request("POST", f"/api/sessions/{session_id}/chat",
                                               json_body={"message": message}, timeout=600.0)
        state = "sent" if response.status_code < 400 else f"failed: http {response.status_code}"
    except Exception as exc:
        log.warning("domo: resuming %s for %s failed: %s", decision_id, profile, exc)
        state = f"failed: {type(exc).__name__}"
    store.update(decision_id, data={"resume": state})


async def _inbox_session(profile: str, store: Store) -> str:
    marker = store.path.parent / "inbox_session"
    if marker.exists():
        return marker.read_text().strip()
    api = _api(profile)
    response = await api.request("POST", "/api/sessions", json_body={"title": INBOX_TITLE})
    if response.status_code < 400:
        data = response.json()
        session_id = data.get("id") or data.get("session_id") or (data.get("session") or {}).get("id")
    else:  # titles are unique per profile: the inbox already exists (the marker file was lost)
        listing = (await api.request("GET", "/api/sessions", params={"limit": 200})).json()
        rows = listing.get("data") or listing.get("sessions") or []
        session_id = next((r["id"] for r in rows if r.get("title") == INBOX_TITLE), None)
    if not session_id:
        raise RuntimeError(f"could not create or find the {INBOX_TITLE!r} session")
    marker.write_text(session_id)
    return session_id


@router.get("/facts")
def facts(profile: str = Query(bridge.DEFAULT, alias="agent")):
    return {"facts": _store(profile).facts()}


# ---- spaces -----------------------------------------------------------------------------------

def _space_or_404(store: Store, space_id: str) -> dict:
    space = store.space(space_id)
    if not space:
        raise HTTPException(404, f"no space {space_id!r}")
    return space


_COLOR = r"^(#[0-9A-Fa-f]{6})?$"


class SpaceIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    icon: str = Field("", max_length=16)
    color: str = Field("", pattern=_COLOR)
    description: str = ""
    prompt: str = ""
    parent_id: str | None = None   # a topic's home area
    kind: str | None = None        # area (the person's few big parts of life) or topic


class SpacePatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=80)
    icon: str | None = Field(None, max_length=16)
    color: str | None = Field(None, pattern=_COLOR)
    description: str | None = None
    prompt: str | None = None


class SpaceTag(BaseModel):
    space_id: str | None = None


def _stitching() -> None:
    stitch.ensure_running(bridge.profile_names, _home, _store)


@router.get("/spaces")
def spaces(profile: str = Query(bridge.DEFAULT, alias="agent")):
    """Areas, then topics by recent activity; quiet topics come too, marked, for the app to fold away."""
    _stitching()
    return {"spaces": _store(profile).spaces()}


@router.post("/spaces", status_code=201)
def create_space(body: SpaceIn, profile: str = Query(bridge.DEFAULT, alias="agent")):
    return _store_call(_store(profile).create_space, **body.model_dump())


@router.patch("/spaces/{space_id}")
def patch_space(space_id: str, body: SpacePatch, profile: str = Query(bridge.DEFAULT, alias="agent")):
    store = _store(profile)
    _space_or_404(store, space_id)
    return _store_call(store.update_space, space_id, **body.model_dump(exclude_unset=True))


@router.delete("/spaces/{space_id}")
def delete_space(space_id: str, profile: str = Query(bridge.DEFAULT, alias="agent")):
    store = _store(profile)
    _space_or_404(store, space_id)
    store.delete_space(space_id)
    return {"ok": True}


@router.get("/spaces/{space_id}")
async def space_detail(space_id: str, profile: str = Query(bridge.DEFAULT, alias="agent")):
    """A space's chats, files, routines, recent routine runs and open items. Chats and routines come
    from the API server; when it is down the space still shows, with tagged chats bare and no routines."""
    store = _store(profile)
    space = _space_or_404(store, space_id)
    in_space = set(store.tagged(space_id, "session"))
    jobs_in = set(store.tagged(space_id, "job"))
    chat_tags = store.tag_map("session")
    try:
        sessions = _rows(await _passthrough(profile, "GET", "/api/sessions", params={"limit": 200}), "data", "sessions")
    except HTTPException:
        sessions = [{"id": sid} for sid in in_space]
    try:
        jobs = _rows(await _passthrough(profile, "GET", "/api/jobs", params={"include_disabled": "true"}), "jobs")
    except HTTPException:
        jobs = []
    routines = [{**j, "space_id": space_id} for j in jobs if str(j.get("id")) in jobs_in]
    job_ids = {j["name"]: j.get("id") for j in routines if j.get("name")}
    runs = []
    for s in sessions:
        match = RUN_TITLE.match(s.get("title") or "")
        if match and match["job"] in job_ids:
            runs.append({**_chat(s), "job_id": job_ids[match["job"]]})
    run_ids = {r["id"] for r in runs}
    chats = [{**_chat(s), **_tagged(chat_tags.get(s.get("id"), []))} for s in sessions
             if s.get("id") in in_space or (s.get("id") in run_ids and s.get("id") not in chat_tags)]
    # everything about it in one place: its chats, files, routines and their runs, open items, and
    # for an area its topics; for any space, the spaces it's most often with
    return {"space": space, "lineage": store.lineage(space_id), "children": store.children(space_id) if space["kind"] == "area" else [],
            "related": store.related(space_id), "chats": _newest(chats), "files": store.files(space_id), "routines": routines,
            "runs": _newest(runs)[:10], "items": store.space_items(space_id)}


def _tagged(tags: list[dict]) -> dict:
    """A chat's tags for the app: its main space (area first) and every tag, who put it there and how sure."""
    if not tags:
        return {"space_id": None, "tags": []}
    main = tags[0]
    return {"space_id": main["space_id"], "tags": [{"space_id": t["space_id"], "by": t["by"], "confidence": t["confidence"]} for t in tags],
            "filed_by": main["by"], "filing_confidence": main["confidence"], "filing_reason": main["reason"]}


def _filed(f: Optional[dict]) -> dict:
    """Who filed a chat where it is: the person, or Domo (how sure, and why)."""
    if not f or not f["space_id"]:
        return {}
    return {"filed_by": f["filed_by"], "filing_confidence": f["confidence"], "filing_reason": f["reason"]}


def _rows(payload, *keys: str) -> list[dict]:
    """The list inside an API server listing, whichever key it uses."""
    if isinstance(payload, list):
        return payload
    return next((payload[k] for k in keys if isinstance(payload.get(k), list)), [])


def _chat(session: dict) -> dict:
    return {"id": session.get("id"), "title": session.get("title") or "",
            "last_active": session.get("last_active") or session.get("updated_at") or session.get("started_at"),
            "preview": session.get("preview") or "", "message_count": session.get("message_count") or 0}


def _newest(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: _when(r.get("last_active")), reverse=True)


def _when(value) -> float:
    """Hermes times may be epoch seconds or ISO strings."""
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0


PLAN_MODE = ("[Plan mode] Investigate and propose; change nothing. Read files, search and run read-only commands as"
             " you need, then answer with a short numbered plan: what you'd change, where, and how you'll know it"
             " worked. Don't edit files, run anything that writes, commit, push or message anyone until they say go.")


CONTEXT_LIMIT = 1600


def _tag_context(store, session_id: str) -> str:
    """What the chat's spaces tell Domo: context only, never orders. Each space's description and notes, area
    first, the surest topics after, kept short so it costs little on every turn."""
    lines = []
    for t in store.tags("session", session_id)[:4]:
        s = store.space(t["space_id"])
        if not s:
            continue
        bits = [b for b in (s["description"], s["prompt"] and f"Notes: {s['prompt']}") if b]
        lines.append(f"- {s['name']} ({s['kind']}){': ' + ' '.join(bits) if bits else ''}")
    if not lines:
        return ""
    text = "[Domo context] This chat is about:\n" + "\n".join(lines)
    return text if len(text) <= CONTEXT_LIMIT else text[: CONTEXT_LIMIT - 1].rsplit(" ", 1)[0] + "…"


# ---- repos ------------------------------------------------------------------------------------

def _repo_env() -> dict[str, str]:
    return bridge.read_env(bridge.hermes_root())


@router.get("/repos")
def repos(fresh: bool = False):
    """The Mac's git repos, most recent first. Runs name theirs in data.repo."""
    return {"repos": repo_scan.scan(_repo_env(), fresh=fresh)}


class StartRun(BaseModel):
    prompt: str = Field(..., min_length=1)
    title: str = ""
    space: str = "Coding"


@router.post("/repos/{repo_id}/start")
async def start_in_repo(repo_id: str, body: StartRun, background: BackgroundTasks,
                        profile: str = Query(bridge.DEFAULT, alias="agent")):
    """Start an agent on a task in a repo: a tracked run now, and a session where the agent does the work."""
    repo = repo_scan.find(repo_id, _repo_env())
    if not repo:
        raise HTTPException(404, f"no repo {repo_id!r} under the repo roots")
    store = _store(profile)
    title = (body.title or body.prompt.strip().splitlines()[0])[:80]
    run = store.create("run", title, body=body.prompt, space=body.space,
                       data={"repo": repo["path"], "repo_name": repo["name"], "branch": repo["branch"],
                             "progress": 0.0, "note": "Starting", "stall_after_minutes": 30,
                             "done_means": "The task is done and committed on a branch"})
    session = await _passthrough(profile, "POST", "/api/sessions", body={"title": f"{repo['name']}: {title} ({run['id'][-4:]})"})
    session_id = session.get("id") or (session.get("session") or {}).get("id") or ""
    run = store.update(run["id"], data={"session": session_id})
    background.add_task(_kick_off, profile, run["id"], session_id, repo, body.prompt)
    return {"run": run, "session_id": session_id}


async def _kick_off(profile: str, run_id: str, session_id: str, repo: dict, prompt: str) -> None:
    message = (f"[Domo] Start a coding run in the repo at {repo['path']} ({repo['name']}, branch {repo['branch'] or 'unknown'}).\n"
               f"Task: {prompt}\n"
               f"It is tracked as Domo run {run_id}: work in that directory, call domo_run with action 'beat' and id "
               f"'{run_id}' as you go (progress, note, cost_usd), ask with domo_ask before pushing, merging or "
               f"anything outbound, and finish or fail the run with proof (the branch and commits).")
    store = _store(profile)
    try:
        response = await _api(profile).request("POST", f"/api/sessions/{session_id}/chat",
                                               json_body={"message": message}, timeout=3600.0)
        if response.status_code >= 400:
            store.run_finish(run_id, ok=False, note=f"The agent didn't start: http {response.status_code}")
    except Exception as exc:
        log.warning("domo: starting run %s for %s failed: %s", run_id, profile, exc)
        store.run_finish(run_id, ok=False, note=f"The agent didn't start: {type(exc).__name__}")


# ---- chat, sessions and routines (passed through to the API server) ----------------------------

@router.get("/search")
async def search(q: str = "", profile: str = Query(bridge.DEFAULT, alias="agent"), limit: int = Query(20, ge=1, le=50),
                 space_id: Optional[str] = None):
    """Every chat of this agent, by words and by meaning: one row per chat, its best matching message."""
    chat_search.ensure_running()
    _home(profile)
    store = _store(profile)
    only = set(store.tagged(space_id, "session")) if space_id else None
    found = await asyncio.to_thread(chat_search.search, profile, q, limit=limit, sessions=only)
    chat_tags = store.tag_map("session")
    for row in found["results"]:
        row.update({k: v for k, v in _tagged(chat_tags.get(row["session_id"], [])).items() if k in ("space_id", "tags")})
    return found


@router.get("/suggestions")
async def suggestions(profile: str = Query(bridge.DEFAULT, alias="agent"), limit: int = Query(8, ge=1, le=20)):
    """Next up: what this person is about to do, each with its reason and one action (domo/suggest.py)."""
    home = _home(profile)
    store = _store(profile)
    try:
        jobs = _rows(await _passthrough(profile, "GET", "/api/jobs"), "jobs", "data")
    except HTTPException:
        jobs = []   # Hermes away: suggest from what's local
    info = bridge.profile_info(profile)
    repos = repo_scan.scan(_repo_env()) if info.get("kind", "person") == "person" else []
    found = await asyncio.to_thread(suggest.build, home, store, routines=jobs, repos=repos, limit=limit)
    tags = store.session_tags()
    spaces = {s["name"].lower(): s["id"] for s in store.spaces()}
    for s in found:
        sid = (s.get("action") or {}).get("session_id")
        s["space_id"] = tags.get(sid) if sid else spaces.get(str(s.get("space") or "").lower())
        s["agent"] = profile
    return {"suggestions": found}


class VoteIn(BaseModel):
    key: str
    kind: str
    verdict: str
    item: str | None = None
    title: str = ""
    instinct: str | None = None


@router.post("/suggestions/vote")
def vote(body: VoteIn, profile: str = Query(bridge.DEFAULT, alias="agent")):
    try:
        suggest.vote(_store(profile), body.key, body.kind, body.verdict, body.item, body.title, body.instinct)
    except StoreError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True}


@router.get("/memory")
def get_memory(profile: str = Query(bridge.DEFAULT, alias="agent")):
    """Everything Domo remembers about this person, for the Memory screen."""
    return memory.recall(_store(profile), events=60)


class SynthesisIn(BaseModel):
    text: str


@router.put("/memory/synthesis/{section}")
def put_synthesis(section: str, body: SynthesisIn, profile: str = Query(bridge.DEFAULT, alias="agent")):
    store = _store(profile)
    _store_call(store.set_synthesis, section, body.text)
    return {k: v["text"] for k, v in store.synthesis().items()}


class InstinctIn(BaseModel):
    pattern: str
    kind: str = "preference"


@router.post("/memory/instincts", status_code=201)
def teach(body: InstinctIn, profile: str = Query(bridge.DEFAULT, alias="agent")):
    """Told by the person: it starts strong."""
    return _store_call(_store(profile).learn, body.pattern, kind=body.kind, origin="stated")


@router.post("/memory/instincts/{instinct_id}/weaken")
def weaken(instinct_id: str, profile: str = Query(bridge.DEFAULT, alias="agent")):
    return _store_call(_store(profile).weaken, instinct_id, 0.2)


@router.delete("/memory/instincts/{instinct_id}", status_code=204)
def forget_instinct(instinct_id: str, profile: str = Query(bridge.DEFAULT, alias="agent")):
    _store(profile).forget_instinct(instinct_id)


@router.delete("/memory/facts/{key}", status_code=204)
def forget_fact(key: str, profile: str = Query(bridge.DEFAULT, alias="agent")):
    _store_call(_store(profile).forget_fact, key)


@router.get("/search/status")
def search_status(profile: str = Query(bridge.DEFAULT, alias="agent")):
    chat_search.ensure_running()
    _home(profile)
    return chat_search.status(profile)


@router.get("/sessions")
async def sessions(profile: str = Query(bridge.DEFAULT, alias="agent"), limit: int = Query(30, ge=1, le=200),
                   archived: bool = False, all_sources: bool = Query(False, alias="all")):
    """Domo's chats: the ones started through the API server, not the agent's Slack, cron or
    terminal sessions. ``all=true`` lists every source."""
    store = _store(profile)
    if archived:
        listing = {"object": "list", "data": await _archived_sessions(profile, store)}
    else:
        params = {"limit": limit} if all_sources else {"limit": limit, "source": DOMO_SOURCE}
        listing = await _passthrough(profile, "GET", "/api/sessions", params=params)
    chat_tags = store.tag_map("session")
    for row in _rows(listing, "data", "sessions"):
        row.update(_tagged(chat_tags.get(row.get("id"), [])))
    return listing


class TagsIn(BaseModel):
    add: list[str] = []
    remove: list[str] = []


@router.put("/sessions/{session_id}/tags")
def set_tags(session_id: str, body: TagsIn, profile: str = Query(bridge.DEFAULT, alias="agent")):
    """The person adds or takes off a chat's spaces; what they take off stays off."""
    store = _store(profile)
    for sid in [*body.add, *body.remove]:
        _space_or_404(store, sid)
    for sid in body.remove:
        store.untag("session", session_id, sid)
    for sid in body.add:
        store.tag("session", session_id, sid)
    return _tagged(store.tags("session", session_id))


class FileIn(BaseModel):
    session_id: str | None = None   # one chat; without it, a sweep of everything unfiled


@router.post("/filing")
async def file_now(body: FileIn, profile: str = Query(bridge.DEFAULT, alias="agent")):
    """Tag chats now (the stitcher does this on its own every half hour): one, or all with something new."""
    home, store = _home(profile), _store(profile)
    if not filing.jev_key():
        raise HTTPException(503, "Filing needs TYPESAFE_API_KEY in the agent's environment")
    if body.session_id:
        done = await asyncio.to_thread(filing.file_session, home, store, body.session_id)
        return {"filed": [done] if done else []}
    return {"filed": await asyncio.to_thread(filing.sweep, home, store)}


async def _archived_sessions(profile: str, store) -> list[dict]:
    """Chats archived from the app, newest first, read one by one: Hermes lists no archived chats."""
    async def one(sid: str):
        try:
            got = await _passthrough(profile, "GET", f"/api/sessions/{sid}")
        except HTTPException as exc:
            if exc.status_code == 404:
                store.set_archived(sid, False)   # deleted since
            return None
        row = got.get("session", got)
        if row.get("archived") is False:   # unarchived elsewhere (Hermes's own desktop app)
            store.set_archived(sid, False)
            return None
        return row
    rows = await asyncio.gather(*(one(sid) for sid in store.archived()[:200]))
    return [r for r in rows if r]


class ArchiveIn(BaseModel):
    archived: bool = True


@router.put("/sessions/{session_id}/archive")
async def archive_session(session_id: str, body: ArchiveIn, profile: str = Query(bridge.DEFAULT, alias="agent")):
    """Archive a chat (or bring it back) with Hermes's own flag, the one its desktop sidebar uses."""
    await _passthrough(profile, "PATCH", f"/api/sessions/{session_id}", body={"archived": body.archived})
    _store(profile).set_archived(session_id, body.archived)
    return {"session_id": session_id, "archived": body.archived}


class NewSession(BaseModel):
    title: str = ""
    model: str | None = None
    provider: str | None = None


@router.post("/sessions")
async def new_session(body: NewSession, profile: str = Query(bridge.DEFAULT, alias="agent")):
    payload = {"title": body.title} if body.title else {}
    if body.model:   # a chat started on a chosen model keeps it (Hermes stores it on the session)
        payload.update({"model": body.model, **({"provider": body.provider} if body.provider else {})})
    return await _passthrough(profile, "POST", "/api/sessions", body=payload)


_FAMILY = re.compile(r"(opus|sonnet|haiku)", re.I)


def _version(model_id: str) -> tuple:
    """claude-opus-5-5 > claude-opus-5 > claude-opus-4-8; dated snapshots after their family's newest."""
    nums = [int(n) for n in re.findall(r"(?<![0-9])(\d{1,2})(?![0-9])", model_id)]
    return (tuple(nums), -len(model_id))


@router.get("/models")
async def models(profile: str = Query(bridge.DEFAULT, alias="agent")):
    """The models this agent's provider offers, and the newest Opus, Sonnet and Haiku among them, for
    the one-click switch on a new chat."""
    options = await _passthrough(profile, "GET", "/api/model/options")
    providers = options.get("providers") or []
    current = next((p for p in providers if p.get("is_current")), providers[0] if providers else {})
    ids = [m if isinstance(m, str) else m.get("id", "") for m in current.get("models") or []]
    families: dict[str, str] = {}
    for fam in ("opus", "sonnet", "haiku"):
        pick = sorted((i for i in ids if fam in i.lower()), key=_version, reverse=True)
        if pick:
            families[fam] = pick[0]
    return {"provider": current.get("slug") or options.get("provider"), "model": options.get("model"), "families": families, "models": ids}


@router.get("/sessions/{session_id}/messages")
async def messages(session_id: str, profile: str = Query(bridge.DEFAULT, alias="agent")):
    return await _passthrough(profile, "GET", f"/api/sessions/{session_id}/messages",
                              params={"inline_images": "false"})


class ChatIn(BaseModel):
    message: str = ""
    attachments: list[str] = []   # paths returned by POST /files
    space_id: str | None = None  # files the chat under this space first
    mode: str | None = None   # "plan": investigate and propose, change nothing
    model: str | None = None   # this turn and the ones after it, on this model (Hermes's model lock)
    provider: str | None = None


@router.put("/sessions/{session_id}/space")
def tag_session(session_id: str, body: SpaceTag, profile: str = Query(bridge.DEFAULT, alias="agent")):
    store = _store(profile)
    if body.space_id:
        _space_or_404(store, body.space_id)
    return {"session_id": session_id, "space_id": store.tag_session(session_id, body.space_id)}


# chats with a turn streaming through Domo right now, per profile: {session_id: turns in flight}
_WORKING: dict[str, dict[str, int]] = {}


@router.get("/working")
def working(request: Request, profile: str = Query(bridge.DEFAULT, alias="agent")):
    """Session ids with a turn running now, from any device (cheap: meant to be polled every second or two)."""
    allowed = access.allowed_profiles(request)
    if allowed is not None and profile not in allowed:
        raise HTTPException(403, "not your agent")
    return {"sessions": sorted(_WORKING.get(profile, {}))}


@router.post("/sessions/{session_id}/chat/stream")
async def chat_stream(session_id: str, body: ChatIn, profile: str = Query(bridge.DEFAULT, alias="agent")):
    home = _home(profile)
    found = [attach.describe(home, p) for p in body.attachments]
    if any(f is None for f in found):
        raise HTTPException(400, "an attachment isn't one of this agent's uploads")
    if not body.message.strip() and not found:
        raise HTTPException(422, "say something or attach a file")
    store = _store(profile)
    if body.space_id:
        _space_or_404(store, body.space_id)
        store.tag_session(session_id, body.space_id)
    api = _api(profile)
    content = attach.compose(body.message, found)
    tag = store.session_space(session_id)
    space = store.space(tag["space_id"]) if tag and tag["space_id"] else None
    payload = {"message": content}
    if body.model:   # locked, so it holds: a bare model only counts on a session's first turn
        payload.update({"model": body.model, "require_model_lock": True, **({"provider": body.provider} if body.provider else {})})
    context = _tag_context(store, session_id)
    instructions = [context] if context else []
    known = memory.for_prompt(store)
    if known:
        instructions.append(known)
    if body.mode == "plan":
        instructions.append(PLAN_MODE)
    if body.message.strip():
        store.note("asked", body.message.strip()[:160], ref=session_id, space=space["name"] if space else "")
    if instructions:
        payload["instructions"] = "\n\n".join(instructions)
    for f in found:
        store.link_file(f["path"], session_id=session_id, space_id=space and space["id"])
    async def then_file():
        mine = _WORKING.setdefault(profile, {})
        mine[session_id] = mine.get(session_id, 0) + 1
        try:
            async for chunk in api.stream(f"/api/sessions/{session_id}/chat/stream", payload):
                yield chunk
        finally:
            left = mine.get(session_id, 1) - 1
            if left > 0:
                mine[session_id] = left
            else:
                mine.pop(session_id, None)
        # just started chatting, with no space: Domo works out where it belongs, off the request
        if not store.session_space(session_id):
            threading.Thread(target=_file_quietly, args=(home, store, session_id), daemon=True).start()

    return StreamingResponse(then_file(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def _file_quietly(home: Path, store: Store, session_id: str) -> None:
    try:
        filing.after_turn(home, store, session_id)
    except Exception:   # filing is a nicety: a chat never fails for it
        log.exception("filing %s", session_id)


# ---- files ------------------------------------------------------------------------------------

def _home(profile: str) -> Path:
    try:
        return bridge.profile_home(profile)
    except bridge.ProfileError as exc:
        raise HTTPException(404, str(exc))


@router.post("/files")
async def upload(file: UploadFile = File(...), space_id: Optional[str] = Form(None),
                 session_id: Optional[str] = Form(None), profile: str = Query(bridge.DEFAULT, alias="agent")):
    """Save an attachment on the Mac; send its path with the next chat message. A file sent from a
    chat in a space is filed under that space unless another is named."""
    store = _store(profile)
    if space_id:
        _space_or_404(store, space_id)
    elif session_id:
        space_id = (store.session_space(session_id) or {}).get("space_id")
    data = await file.read(attach.MAX_BYTES + 1)
    try:
        return attach.save(_home(profile), file.filename or "file", data, file.content_type or "",
                           store=store, space_id=space_id, session_id=session_id)
    except ValueError as exc:
        raise HTTPException(413, str(exc))


@router.get("/files")
def download(path: Optional[str] = None, space_id: Optional[str] = None,
             profile: str = Query(bridge.DEFAULT, alias="agent")):
    """With ``path``: an attachment, or a file an agent returned as MEDIA:<path>. Without: the
    upload records, newest first, optionally only one space's."""
    if not path:
        return {"files": _store(profile).files(space_id)}
    found = attach.deliverable(path, _home(profile))
    if not found:
        raise HTTPException(404, "not a file Domo can hand out")
    return FileResponse(found, headers={"Cache-Control": "private, max-age=86400"})


class ApprovalIn(BaseModel):
    request_id: str
    execution_generation: int | None = None
    choice: str


@router.post("/runs/{run_id}/approval")
async def run_approval(run_id: str, body: ApprovalIn, profile: str = Query(bridge.DEFAULT, alias="agent")):
    return await _passthrough(profile, "POST", f"/v1/runs/{run_id}/approval", body=body.model_dump(exclude_none=True))


@router.get("/jobs")
async def jobs(profile: str = Query(bridge.DEFAULT, alias="agent")):
    tags = _store(profile).job_tags()
    listing = await _passthrough(profile, "GET", "/api/jobs", params={"include_disabled": "true"})
    for job in _rows(listing, "jobs"):
        job["space_id"] = tags.get(str(job.get("id")))
    return listing


@router.put("/jobs/{job_id}/space")
def tag_job(job_id: str, body: SpaceTag, profile: str = Query(bridge.DEFAULT, alias="agent")):
    store = _store(profile)
    if body.space_id:
        _space_or_404(store, body.space_id)
    return {"job_id": job_id, "space_id": store.tag_job(job_id, body.space_id)}


_JOB_ID = re.compile(r"^[a-f0-9]{12}$")
_SCHEDULE_HELP = "Hermes doesn't read that schedule: try 0 8 * * *, every 2h, 30m, weekdays at 9am or every monday 9am"


class JobIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    schedule: str = Field(..., min_length=1, max_length=120)
    prompt: str = Field(..., min_length=1, max_length=20000)
    deliver: str = Field("local", max_length=200)
    skills: list[str] = []


class JobPatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=120)
    schedule: str | None = Field(None, min_length=1, max_length=120)
    prompt: str | None = Field(None, min_length=1, max_length=20000)
    deliver: str | None = Field(None, max_length=200)
    skills: list[str] | None = None
    enabled: bool | None = None


def _job_id(job_id: str) -> str:
    if not _JOB_ID.match(job_id):
        raise HTTPException(404, "no such routine")
    return job_id


async def _job_write(profile: str, method: str, path: str, body: dict):
    try:
        return await _passthrough(profile, method, path, body=body)
    except HTTPException as exc:
        if exc.status_code in (400, 500) and "schedule" in body:
            raise HTTPException(400, _SCHEDULE_HELP)   # Hermes answers a bad schedule on PATCH with a 500
        raise


@router.post("/jobs", status_code=201)
async def create_job(body: JobIn, profile: str = Query(bridge.DEFAULT, alias="agent")):
    return await _job_write(profile, "POST", "/api/jobs", body.model_dump())


@router.patch("/jobs/{job_id}")
async def update_job(job_id: str, body: JobPatch, profile: str = Query(bridge.DEFAULT, alias="agent")):
    """Edit a routine: name, schedule, prompt, delivery, skills, or on/off."""
    patch = body.model_dump(exclude_none=True)
    if not patch:
        raise HTTPException(400, "nothing to change")
    return await _job_write(profile, "PATCH", f"/api/jobs/{_job_id(job_id)}", patch)


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str, profile: str = Query(bridge.DEFAULT, alias="agent")):
    await _passthrough(profile, "DELETE", f"/api/jobs/{_job_id(job_id)}")
    _store(profile).tag_job(job_id, None)


_RUN_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})\.md$")


def _read_run(path: Path) -> dict:
    """One run's saved output (Hermes writes one per run, delivered or not): when, how it went, what it said."""
    m = _RUN_FILE.match(path.name)
    at = f"{m[1]}T{m[2]}:{m[3]}:{m[4]}" if m else ""
    doc = path.read_text(errors="replace")
    status_line = re.search(r"^\*\*Status:\*\*\s*(.+)$", doc, re.M)
    response = doc.split("## Response", 1)[1].strip() if "## Response" in doc else re.sub(r"^#.*\n|^\*\*[^*]+:\*\*.*\n", "", doc, flags=re.M).strip()
    if status_line and not status_line[1].upper().startswith("OK"):
        status = "failed"
    elif "[CRON_FAILURE]" in response or response.startswith("Error:"):
        status = "failed"
    elif response.strip() == "[SILENT]" or not response.strip():
        status = "silent"
    else:
        status = "ok"
    return {"at": at, "status": status, "response": response[:12000], "file": path.name}


@router.get("/jobs/{job_id}/runs")
async def job_runs(job_id: str, profile: str = Query(bridge.DEFAULT, alias="agent"), limit: int = Query(20, ge=1, le=50)):
    """A routine's recent runs, newest first, with what each one said and the chat it ran in."""
    folder = _home(profile) / "cron" / "output" / _job_id(job_id)
    files = sorted((p for p in folder.glob("*.md") if _RUN_FILE.match(p.name)), reverse=True)[:limit] if folder.is_dir() else []
    runs = [_read_run(p) for p in files]
    try:   # the chat each agent run happened in, matched by its title's stamp ("<job> · Nov 08 16:20")
        job = await _passthrough(profile, "GET", f"/api/jobs/{job_id}")
        name = (job.get("job") or job).get("name", "")
        sessions = _rows(await _passthrough(profile, "GET", "/api/sessions", params={"limit": 200}), "data", "sessions")
        by_stamp = {}
        for s in sessions:
            t = RUN_TITLE.match(s.get("title") or "")
            if t and t["job"] == name:
                by_stamp[(s.get("title") or "").rsplit(" · ", 1)[1]] = s.get("id")
        for r in runs:
            if r["at"]:
                stamp = datetime.fromisoformat(r["at"]).strftime("%b %d %H:%M")
                r["session_id"] = by_stamp.get(stamp)
    except (HTTPException, ValueError):
        pass
    return {"runs": runs}


@router.post("/jobs/{job_id}/{action}")
async def job_action(job_id: str, action: str, profile: str = Query(bridge.DEFAULT, alias="agent")):
    if action not in ("run", "pause", "resume"):
        raise HTTPException(404, "unknown action")
    return await _passthrough(profile, "POST", f"/api/jobs/{job_id}/{action}")
