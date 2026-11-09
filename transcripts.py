"""Claude Code and Cursor transcripts, read into the shapes Domo's chat already shows.

A transcript is JSONL: one record per line, appended as the session runs. Claude Code's format "changes
between versions", so everything here is lenient: unknown record types are skipped, missing fields are
fine. Cursor's agent transcripts are written "Claude Code-compatible", with `role` where Claude has `type`.

Tool steps get the names the web app already draws (terminal, read_file, patch…), so a Claude Code
session shows its commands, paths and diffs exactly like a Hermes chat does.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Iterable, Optional

OUTPUT_LIMIT = 4000
TEXT_LIMIT = 20000

# Claude Code's tool → (Domo's tool name, how to read its input)
_TOOLS = {
    "Bash": "terminal", "BashOutput": "process_manage", "KillShell": "process_manage",
    "Read": "read_file", "Write": "write_file", "Edit": "patch", "MultiEdit": "patch", "NotebookEdit": "patch",
    "Grep": "search_files", "Glob": "search_files", "LS": "search_files",
    "WebSearch": "web_search", "WebFetch": "web_extract",
    "Task": "delegate_task", "Agent": "delegate_task",
}
_COMMIT = re.compile(r"^\[(?P<branch>[^\s\]]+)(?: \(root-commit\))? (?P<sha>[0-9a-f]{7,40})\] (?P<subject>.+)$", re.M)
_COMMIT_CMD = re.compile(r"\bgit\b[^\n|;&]*\bcommit\b")
_COMMIT_MSG = re.compile(r"""(?<!\S)-[a-zA-Z]*m\s+(?:"\$\(cat <<'?(?P<tag>\w+)'?\n(?P<heredoc>.*?)\n\s*(?P=tag)|(?P<q>["'])(?P<quoted>.*?)(?P=q))""", re.S)
_PR = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/pull/\d+")
_COMMAND = re.compile(r"<command-name>(?P<name>[^<]*)</command-name>(?:.*?<command-args>(?P<args>[^<]*)</command-args>)?", re.S)
_NOISE = ("<system-reminder>", "<local-command-stdout>", "<local-command-stderr>", "Caveat: ", "<command-message>")


def records(text: str) -> Iterable[dict]:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            yield rec


def _ts(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return value / 1000 if value > 1e12 else float(value)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _role(rec: dict) -> str:
    return rec.get("type") if rec.get("type") in ("user", "assistant") else rec.get("role") or ""


def _content(rec: dict):
    msg = rec.get("message")
    if isinstance(msg, dict) and "content" in msg:
        return msg["content"]
    return rec.get("content")


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, dict) and block.get("type") == "image":
                parts.append("[image]")
        return "\n".join(parts)
    return ""


def prompt_text(text: str) -> Optional[str]:
    """What the person typed, or None for what Claude Code injects into the user's turn."""
    text = text.strip()
    if not text:
        return None
    command = _COMMAND.search(text)
    if command:
        args = (command.group("args") or "").strip()
        return f"{command.group('name').strip()} {args}".strip()
    if text.startswith(_NOISE):
        return None
    return text


def _diff(path: str, old: str, new: str) -> str:
    lines = [f"--- a/{path}", f"+++ b/{path}", "@@"]
    lines += [f"-{l}" for l in old.splitlines()] + [f"+{l}" for l in new.splitlines()]
    return "\n".join(lines)


def commits(command: str, output: str, *, failed: bool = False) -> list[dict]:
    """Commits a shell step made: git's own "[branch sha] subject" lines, or, when it ran quietly (`-q`) or
    the message came in a heredoc, the subject from the command and its sha from any `git log` line shown."""
    found = [{"sha": m.group("sha"), "branch": m.group("branch"), "subject": m.group("subject").strip()}
             for m in _COMMIT.finditer(output)]
    if found or failed or not _COMMIT_CMD.search(command):
        return found
    m = _COMMIT_MSG.search(command)
    if not m:
        return []
    subject = (m.group("heredoc") or m.group("quoted") or "").strip().splitlines()[0].strip() if (m.group("heredoc") or m.group("quoted")) else ""
    if not subject:
        return []
    shown = re.search(rf"^\s*([0-9a-f]{{7,40}})\s+{re.escape(subject)}\s*$", output, re.M)
    return [{"sha": shown.group(1) if shown else "", "branch": "", "subject": subject}]


def tool_step(name: str, args: dict) -> tuple[str, dict]:
    """A Claude Code tool call as Domo's step: its name, and the arguments the step line reads."""
    args = args if isinstance(args, dict) else {}
    domo = _TOOLS.get(name, name)
    out: dict = {"tool": name}
    path = args.get("file_path") or args.get("notebook_path") or args.get("path") or ""
    if domo == "terminal":
        out["command"] = args.get("command", "")
    elif domo in ("read_file", "write_file"):
        out["path"] = path
    elif domo == "patch":
        out["path"] = path
        if name == "MultiEdit" and isinstance(args.get("edits"), list):
            out["diff"] = "\n".join(_diff(path, str(e.get("old_string", "")), str(e.get("new_string", "")))
                                    for e in args["edits"] if isinstance(e, dict))
        elif "old_string" in args or "new_string" in args:
            out["diff"] = _diff(path, str(args.get("old_string", "")), str(args.get("new_string", "")))
    elif domo == "search_files":
        out["pattern"] = args.get("pattern") or args.get("path") or ""
    elif domo == "web_search":
        out["query"] = args.get("query", "")
    elif domo == "web_extract":
        out["url"] = args.get("url", "")
    elif domo == "delegate_task":
        out["goal"] = args.get("description") or str(args.get("prompt", ""))[:200]
    else:
        out.update({k: v for k, v in args.items() if isinstance(v, (str, int, float, bool)) and len(str(v)) < 400})
    return domo, out


def parse(text: str) -> dict:
    """The whole transcript: its messages (Domo's Message shape) and what it says about itself."""
    messages: list[dict] = []
    tools: dict[str, dict] = {}
    info: dict = {"title": "", "first_prompt": "", "last_prompt": "", "prompts": 0, "model": "", "cwd": "",
                  "branch": "", "started_at": None, "last_at": None, "commits": [], "prs": [], "cost_usd": None,
                  "last_text": ""}
    last_text_id = ""
    for rec in records(text):
        kind = rec.get("type")
        at = _ts(rec.get("timestamp"))
        if at:
            info["started_at"] = info["started_at"] or at
            info["last_at"] = at
        info["cwd"] = rec.get("cwd") or info["cwd"]
        if rec.get("gitBranch") and rec.get("gitBranch") != "HEAD":
            info["branch"] = rec["gitBranch"]
        if kind in ("custom-title", "ai-title"):
            title = rec.get("customTitle") or rec.get("aiTitle") or rec.get("title")
            if isinstance(title, str) and title.strip() and (kind == "custom-title" or not info.get("custom")):
                info["title"] = title.strip()
                info["custom"] = info.get("custom") or kind == "custom-title"
        elif kind == "summary" and isinstance(rec.get("summary"), str) and not info["title"]:
            info["title"] = rec["summary"].strip()
        elif kind == "cost-state" and isinstance(rec.get("totalCostUSD"), (int, float)):
            info["cost_usd"] = rec["totalCostUSD"]
        elif kind == "pr-link":
            for url in _PR.findall(json.dumps(rec)):
                if url not in info["prs"]:
                    info["prs"].append(url)
        role = _role(rec)
        if role not in ("user", "assistant") or rec.get("isSidechain") or rec.get("isMeta"):
            continue
        content = _content(rec)
        uid = str(rec.get("uuid") or len(messages))
        if role == "assistant":
            msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
            info["model"] = msg.get("model") or rec.get("requestedModel") or info["model"]
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}] if isinstance(content, str) else []
            for i, block in enumerate(blocks):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and str(block.get("text", "")).strip():
                    body = str(block["text"])
                    # one API message is split over several lines, a block each: keep its text as one reply
                    if messages and messages[-1]["role"] == "assistant" and last_text_id and last_text_id == msg.get("id"):
                        messages[-1]["text"] = _clip(messages[-1]["text"] + "\n\n" + body, TEXT_LIMIT)
                    else:
                        messages.append({"id": f"{uid}:{i}", "role": "assistant", "text": _clip(body, TEXT_LIMIT), "at": at})
                    last_text_id = msg.get("id") or ""
                    info["last_text"] = body
                elif block.get("type") == "tool_use":
                    name, args = tool_step(str(block.get("name", "tool")), block.get("input") or {})
                    step = {"id": str(block.get("id") or f"{uid}:{i}"), "role": "tool", "text": "", "toolName": name,
                            "toolArgs": args, "at": at}
                    tools[step["id"]] = step
                    messages.append(step)
                    last_text_id = ""
            continue
        # a user record: what was typed, or the results of tool calls
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content if isinstance(content, list) else []
        for i, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                step = tools.get(str(block.get("tool_use_id")))
                if step is None:
                    continue
                output = _result_text(block.get("content"))
                step["toolOutput"] = _clip(output, OUTPUT_LIMIT)
                step["toolFailed"] = bool(block.get("is_error"))
                if step["toolName"] == "terminal":
                    for commit in commits(str(step["toolArgs"].get("command") or ""), output, failed=step["toolFailed"]):
                        if all(c["sha"] != commit["sha"] or c["subject"] != commit["subject"] for c in info["commits"]):
                            info["commits"].append(commit)
                    for url in _PR.findall(output):
                        if url not in info["prs"]:
                            info["prs"].append(url)
            elif block.get("type") == "text":
                typed = prompt_text(str(block.get("text", "")))
                if typed is None:
                    continue
                messages.append({"id": f"{uid}:{i}", "role": "user", "text": _clip(typed, TEXT_LIMIT), "at": at})
                info["prompts"] += 1
                info["first_prompt"] = info["first_prompt"] or typed
                info["last_prompt"] = typed
                last_text_id = ""
            elif block.get("type") == "image":
                messages.append({"id": f"{uid}:{i}", "role": "user", "text": "[image]", "at": at})
    info.pop("custom", None)
    if not info["title"]:
        info["title"] = _clip(" ".join(info["first_prompt"].split()), 90)
    for m in messages:
        m.setdefault("attachments", [])
        m.setdefault("inlineImages", [])
    return {"messages": messages, "info": info}


def searchable(line_text: str) -> list[tuple[str, str]]:
    """The words worth finding in new transcript lines, without reading the whole session: what was typed,
    what the agent said, and the commands it ran. Tool output bodies are left out on purpose."""
    found: list[tuple[str, str]] = []
    for rec in records(line_text):
        role = _role(rec)
        if role not in ("user", "assistant") or rec.get("isSidechain") or rec.get("isMeta"):
            continue
        content = _content(rec)
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content if isinstance(content, list) else []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = str(block.get("text", ""))
                text = prompt_text(text) if role == "user" else text.strip()
                if text:
                    found.append((role, text[:8000]))
            elif block.get("type") == "tool_use":
                args = block.get("input") or {}
                if isinstance(args, dict):
                    what = args.get("command") or args.get("file_path") or args.get("pattern") or args.get("url") or ""
                    if what:
                        found.append(("tool", f"{block.get('name', '')} {what}"[:2000]))
    return found
