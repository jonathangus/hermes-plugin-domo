"""Who may open which profile.

A person who signed in through auth.ljungek.com carries a `domo_profiles` claim in the ID token the dashboard
already verified (it keeps that token as the session's access token). Their requests may only name those
profiles. The password sign-in is the household's admin and the loopback/no-gate setups carry no such
session: they are not restricted.
"""

from __future__ import annotations

import base64
import json
from typing import Optional

from fastapi import HTTPException, Request

CLAIM = "domo_profiles"
SSO_PROVIDER = "self-hosted"
FILTERED = ("/profiles", "/health")
OWN_CHECKS = "/connections"  # each connection is checked against its owner


def _claims(token: str) -> dict:
    """The payload of a JWT the dashboard has already verified on this request: read, not re-verified."""
    try:
        body = token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except Exception:
        return {}


def allowed_profiles(request: Request) -> Optional[set[str]]:
    """The profiles this request may use, or None when it isn't restricted."""
    session = getattr(request.state, "session", None)
    if session is None or getattr(session, "provider", "") != SSO_PROVIDER:
        return None
    listed = _claims(getattr(session, "access_token", "") or "").get(CLAIM)
    if isinstance(listed, str):
        listed = [listed]
    return {str(p).strip().lower() for p in listed or [] if str(p).strip()}


def guard(request: Request) -> None:
    """Router dependency: every route names its profile as ?agent=, defaulting to the family's."""
    allowed = allowed_profiles(request)
    path = request.url.path.rstrip("/")
    if allowed is None or path.endswith(FILTERED) or OWN_CHECKS in path:  # those list only what you may open
        return
    from domo import bridge  # late: this module is imported by the plugin's package init order

    wanted = (request.query_params.get("agent") or bridge.DEFAULT).lower()
    if wanted not in allowed:
        raise HTTPException(403, "This sign-in doesn't have access to that Domo profile")
