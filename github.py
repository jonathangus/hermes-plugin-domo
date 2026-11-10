"""Where a pull request stands, for the card under a reply that links one: open, draft, merged or
closed, and how its checks went. Read with the Mac's own `gh` (its sign-in reaches the private
repos), cached briefly so a chat full of links costs one call each.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Optional

PR_URL = re.compile(r"^https://github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)(?:[/?#].*)?$")
CANDIDATES = ("/opt/homebrew/bin/gh", "/usr/local/bin/gh", "~/.local/bin/gh")
FIELDS = "number,title,state,isDraft,url,additions,deletions,headRefName,statusCheckRollup"
TIMEOUT = 8

_cache: dict[str, tuple[float, Optional[dict]]] = {}
_lock = threading.Lock()


def binary() -> Optional[str]:
    for path in filter(None, (os.environ.get("DOMO_GH_BIN"), shutil.which("gh"), *CANDIDATES)):
        full = os.path.expanduser(path)
        if os.path.isfile(full) and os.access(full, os.X_OK):
            return full
    return None


def checks(rollup: list[dict]) -> Optional[str]:
    """passed, failed or pending over every check and status; None when there are none."""
    seen = set()
    for c in rollup or []:
        if c.get("__typename") == "StatusContext" or "state" in c:
            state = str(c.get("state") or "").upper()
            seen.add("pass" if state == "SUCCESS" else "pending" if state in ("PENDING", "EXPECTED", "") else "fail")
        elif str(c.get("status") or "").upper() != "COMPLETED":
            seen.add("pending")
        else:
            seen.add("pass" if str(c.get("conclusion") or "").upper() in ("SUCCESS", "NEUTRAL", "SKIPPED") else "fail")
    return None if not seen else "failed" if "fail" in seen else "pending" if "pending" in seen else "passed"


def _shape(raw: dict, owner: str, repo: str) -> dict:
    state = str(raw.get("state") or "").lower()
    return {"url": raw.get("url"), "repo": f"{owner}/{repo}", "number": raw.get("number"), "title": raw.get("title") or "",
            "state": "draft" if state == "open" and raw.get("isDraft") else state, "checks": checks(raw.get("statusCheckRollup") or []),
            "additions": raw.get("additions") or 0, "deletions": raw.get("deletions") or 0, "branch": raw.get("headRefName") or ""}


def _ttl(pr: Optional[dict]) -> float:
    if pr is None:
        return 300
    if pr["state"] in ("merged", "closed"):
        return 3600
    return 20 if pr["checks"] == "pending" else 90


def pull_request(url: str, *, run=subprocess.run) -> Optional[dict]:
    """The pull request at a github.com URL, or None when it isn't one or can't be read here."""
    m = PR_URL.match(url.strip())
    if not m:
        return None
    owner, repo, number = m.groups()
    key = f"{owner}/{repo}#{number}".lower()
    with _lock:
        hit = _cache.get(key)
    if hit and time.time() < hit[0]:
        return hit[1]
    exe = binary()
    pr = None
    if exe:
        try:
            out = run([exe, "pr", "view", number, "--repo", f"{owner}/{repo}", "--json", FIELDS],
                      stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=TIMEOUT)
            if out.returncode == 0:
                pr = _shape(json.loads(out.stdout), owner, repo)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pr = None
    with _lock:
        _cache[key] = (time.time() + _ttl(pr), pr)
    return pr
