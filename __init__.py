"""Domo: chief-of-staff tools and the Domo app's backend, as one Hermes plugin."""

from __future__ import annotations

from . import minne, rtk, shared, tools
from .tools import PROMPT, SHARED_TOOLS, bound_tools, shared_prompt


def register(ctx):
    # register() runs once per profile scope; pin this scope's home so tool calls on worker
    # threads write to the right profile's store.
    home = tools.home_resolver()
    with_shared = shared.enabled(home)
    for name, schema, handler, emoji in bound_tools(home, tools.TOOLS + (SHARED_TOOLS if with_shared else [])):
        ctx.register_tool(name=name, toolset="domo", schema=schema, handler=handler,
                          description=schema["description"], emoji=emoji)
    ctx.register_system_prompt_section("domo.chief-of-staff", PROMPT, max_chars=1200)
    # an organisation's Minne key in this profile's .env (the work computer's): its shared meetings, as a tool
    if minne.key(home):
        (name, schema, handler, _), = bound_tools(home, [("minne", minne.SCHEMA, tools.domo_minne, "🎙️")])
        ctx.register_tool(name=name, toolset="domo", schema=schema, handler=handler, description=schema["description"], emoji="🎙️")
        ctx.register_system_prompt_section("domo.minne", minne.PROMPT, max_chars=400)
    if rtk.available():
        ctx.register_hook("pre_tool_call", rtk.pre_tool_call)
    if with_shared:
        partners = shared.profile_json(home).get("shared_with", [])
        ctx.register_system_prompt_section("domo.shared-notes", shared_prompt(partners), max_chars=800)
