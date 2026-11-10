"""Other Hermes installs (the work computer) whose agents Domo shows beside this Mac's own.

The apps only ever talk to this dashboard. A request naming an agent as `<profile>@<hermes>` is passed as it is
to that Hermes's own dashboard (over Tailscale), signed in with the username and password given when it was
connected; the answer streams back unchanged. Each connection belongs to one person here (its owner): only a
sign-in that may open the owner's profile sees its agents. Connections live in the default profile's
plugin-data, beside nothing else, readable by this user only.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from domo import access, bridge

SEP = "@"
REFRESH_AFTER = 600  # seconds before an agent's name and spaces are read from its Hermes again
# Over Tailscale a Hermes that's up answers in milliseconds; one that's away mustn't hold a page up for long
CONNECT_TIMEOUT = 3.0
PASS_HEADERS = ("content-type", "cache-control", "content-disposition")
transport: Optional[httpx.AsyncBaseTransport] = None  # tests inject the other dashboard

_clients: dict[str, httpx.AsyncClient] = {}
_refreshing: set[str] = set()


class RemoteError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


# ---- the connections file ---------------------------------------------------------------------

def _path() -> Path:
    return bridge.hermes_root() / "plugin-data" / "domo" / "remotes.json"


def load() -> list[dict]:
    path = _path()
    return json.loads(path.read_text()).get("remotes", []) if path.exists() else []


def save(remotes: list[dict]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"remotes": remotes}, indent=2))
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def find(remote_id: str) -> Optional[dict]:
    return next((r for r in load() if r["id"] == remote_id), None)


def split(agent: str) -> Optional[tuple[str, str]]:
    """`jonathan@work` → ("jonathan", "work"); a profile on this Mac → None."""
    profile, sep, remote_id = (agent or "").lower().rpartition(SEP)
    return (profile, remote_id) if sep and profile and remote_id else None


def slug(name: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32] or "hermes"
    out, n = base, 2
    while out in taken:
        out, n = f"{base}-{n}", n + 1
    return out


def may_see(remote: dict, allowed: Optional[set[str]]) -> bool:
    return allowed is None or remote["owner"] in allowed


def public(remote: dict) -> dict:
    """What the apps see of a connection: never its password."""
    return {k: remote.get(k) for k in ("id", "name", "url", "username", "owner", "enabled", "refreshed_at")} | {
        "agents": agents_of(remote)}


def agents_of(remote: dict) -> list[dict]:
    """The connection's agents as this Mac lists its own, named so every request finds its way back."""
    return [{**a, "id": f"{a['id']}{SEP}{remote['id']}", "hermes": {"id": remote["id"], "name": remote["name"]},
             "owner": remote["owner"]} for a in remote.get("agents", [])]


def listed(allowed: Optional[set[str]]) -> list[dict]:
    """Every agent of the connections switched on that this sign-in may see; stale names refresh behind."""
    out = []
    for r in load():
        if not r.get("enabled", True) or not may_see(r, allowed):
            continue
        out += agents_of(r)
        if time.time() - (r.get("refreshed_at") or 0) > REFRESH_AFTER and r["id"] not in _refreshing:
            _refreshing.add(r["id"])
            try:
                asyncio.get_running_loop().create_task(_refresh(r["id"]))
            except RuntimeError:  # no loop (a sync caller): the next listing tries
                _refreshing.discard(r["id"])
    return out


# ---- signing in to the other dashboard --------------------------------------------------------

def _new_client(url: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=url, transport=transport, timeout=httpx.Timeout(30.0, connect=CONNECT_TIMEOUT), follow_redirects=False)


async def _sign_in(client: httpx.AsyncClient, name: str, username: str, password: str) -> None:
    try:
        r = await client.post("/auth/password-login", json={"provider": "basic", "username": username, "password": password})
    except httpx.HTTPError as exc:
        raise RemoteError(502, f"Can't reach {name}. Is it on, and on Tailscale?")
    if r.status_code in (401, 403):
        raise RemoteError(401, f"{name} didn't take that username and password")
    if r.status_code >= 400:
        raise RemoteError(502, f"{name}'s sign-in said {r.status_code}")


async def _client(remote: dict) -> httpx.AsyncClient:
    client = _clients.get(remote["id"])
    if client is None:
        client = _new_client(remote["url"])
        await _sign_in(client, remote["name"], remote["username"], remote["password"])
        _clients[remote["id"]] = client
    return client


def forget_client(remote_id: str) -> None:
    client = _clients.pop(remote_id, None)
    if client is not None:
        try:
            asyncio.get_running_loop().create_task(client.aclose())
        except RuntimeError:
            pass


async def probe(name: str, url: str, username: str, password: str) -> list[dict]:
    """Sign in to a Hermes dashboard and read its Domo profiles: proof the address and password work."""
    async with _new_client(url) as client:
        await _sign_in(client, name, username, password)
        try:
            r = await client.get("/api/plugins/domo/profiles")
        except httpx.HTTPError as exc:
            raise RemoteError(502, f"Can't reach {name}. Is it on, and on Tailscale?")
        if r.status_code == 404:
            raise RemoteError(502, f"{name} answers, but the Domo plugin isn't on it")
        if r.status_code >= 400:
            raise RemoteError(502, f"{name} said {r.status_code} for its profiles")
        return [{**{k: p.get(k) for k in ("id", "display_name", "kind", "members", "spaces")}, "features": p.get("features") or []}
                for p in r.json().get("profiles", [])]


async def _refresh(remote_id: str) -> None:
    try:
        remote = find(remote_id)
        if not remote:
            return
        found = {p["id"]: p for p in await probe(remote["name"], remote["url"], remote["username"], remote["password"])}
        remotes = load()
        for r in remotes:
            if r["id"] == remote_id:
                r["agents"] = [found.get(a["id"], a) for a in r.get("agents", [])]
                r["refreshed_at"] = time.time()
        save(remotes)
    except Exception:
        pass  # away for now: the names it had stay, and the next listing tries again
    finally:
        _refreshing.discard(remote_id)


# ---- passing a request on ---------------------------------------------------------------------

def _requalify(data, profile: str, agent: str):
    """The other Hermes names its own profile; the apps know it by its name here."""
    if isinstance(data, list):
        return [_requalify(x, profile, agent) for x in data]
    if isinstance(data, dict):
        out = {k: _requalify(v, profile, agent) for k, v in data.items()}
        if out.get("agent") == profile:
            out["agent"] = agent
        if isinstance(out.get("profile"), dict) and out["profile"].get("id") == profile:
            out["profile"] = {**out["profile"], "id": agent}
        return out
    return data


def _error(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


async def relay(request: Request, agent: str) -> Response:
    """Any Domo route, for an agent on another Hermes: the same path and body there, its answer back here."""
    profile, remote_id = split(agent)
    remote = find(remote_id)
    if not remote or not remote.get("enabled", True):
        return _error(404, f"No Hermes called {remote_id!r} is connected")
    if not may_see(remote, access.allowed_profiles(request)):
        return _error(403, "This sign-in doesn't have access to that Domo profile")
    if profile not in {a["id"] for a in remote.get("agents", [])}:
        return _error(404, f"{profile!r} isn't one of the agents connected on {remote['name']}")

    params = [(k, profile if k == "agent" else v) for k, v in request.query_params.multi_items()]
    headers = {k: request.headers[k] for k in ("content-type", "accept") if k in request.headers}
    body = await request.body()
    streaming = request.url.path.endswith("/stream")
    timeout = httpx.Timeout(30.0, connect=CONNECT_TIMEOUT, read=None if streaming else 60.0)
    try:
        client = await _client(remote)
        send = lambda: client.send(client.build_request(request.method, request.url.path, params=params, content=body,
                                                        headers=headers, timeout=timeout), stream=True)
        response = await send()
        if response.status_code == 401:  # its session ended: sign in again, once
            await response.aclose()
            await _sign_in(client, remote["name"], remote["username"], remote["password"])
            response = await send()
    except RemoteError as exc:
        forget_client(remote_id)
        return _error(exc.status if exc.status != 401 else 502, exc.detail)
    except httpx.HTTPError as exc:
        return _error(502, f"Can't reach {remote['name']}. Is it on, and on Tailscale?")
    if response.status_code == 401:  # never a 401 to the app: that would sign it out of this Mac
        await response.aclose()
        forget_client(remote_id)
        return _error(502, f"{remote['name']} won't let Domo in: connect it again in Settings")

    out = {k: response.headers[k] for k in PASS_HEADERS if k in response.headers}
    if response.headers.get("content-type", "").startswith("application/json") and request.url.path.endswith(("/suggestions", "/feed")):
        data = json.loads(await response.aread() or b"null")
        await response.aclose()
        return JSONResponse(_requalify(data, profile, f"{profile}{SEP}{remote_id}"), status_code=response.status_code)
    return StreamingResponse(response.aiter_bytes(), status_code=response.status_code, headers=out,
                             background=BackgroundTask(response.aclose))
