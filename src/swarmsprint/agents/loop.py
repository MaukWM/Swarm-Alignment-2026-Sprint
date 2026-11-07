"""One agent: a model with a bash tool, driving its own worker container turn by turn."""

import http.client
import itertools
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from swarmsprint.agents import responses
from swarmsprint.agents.prompts import NUDGE, status
from swarmsprint.config import RunConfig
from swarmsprint.infra.docker import CAP, read_capped

Log = Callable[..., None]
ATTEMPTS = 3  # model calls per turn before the agent gives up (transient errors only)
BASH = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command in your container and get its output.",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
    },
}


def chat(cfg: RunConfig, messages: list[dict[str, Any]]) -> dict[str, Any]:
    if cfg.api == "responses":
        return responses.chat(cfg.model, messages, BASH)
    return chat_completions(cfg, messages)


def chat_completions(cfg: RunConfig, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Call the chat endpoint with the bash tool, streamed so long reasoning cannot hit a gateway timeout.

    The streamed deltas are put back together into the usual non-streamed response shape.
    """
    stream = {"stream": True, "stream_options": {"include_usage": True}}
    body = json.dumps({"model": cfg.model, "messages": messages, "tools": [BASH], **stream}).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}"}
    req = urllib.request.Request(os.environ["LLM_BASE_URL"], body, headers)
    msg: dict[str, Any] = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []}
    choice: dict[str, Any] = {"message": msg, "finish_reason": None}
    out: dict[str, Any] = {"choices": [choice]}
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            out["model"], out["usage"] = chunk.get("model", out.get("model")), chunk.get("usage") or out.get("usage")
            for c in chunk.get("choices") or []:
                delta = c.get("delta") or {}
                msg["content"] += delta.get("content") or ""
                msg["reasoning_content"] += delta.get("reasoning_content") or delta.get("reasoning") or ""
                for tc in delta.get("tool_calls") or []:
                    while len(msg["tool_calls"]) <= tc.get("index", 0):
                        msg["tool_calls"].append(
                            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                        )
                    slot, fn = msg["tool_calls"][tc.get("index", 0)], tc.get("function") or {}
                    slot["id"] = tc.get("id") or slot["id"]
                    slot["function"]["name"] += fn.get("name") or ""
                    slot["function"]["arguments"] += fn.get("arguments") or ""
                choice["finish_reason"] = c.get("finish_reason") or choice["finish_reason"]
    for i, call in enumerate(msg["tool_calls"]):
        call["id"] = call["id"] or f"call_{i}"  # some servers stream no id; tool results must refer to one
    return out


def bash(cfg: RunConfig, worker: str, cmd: str) -> dict[str, Any]:
    """Run one command in the worker container under a hard timeout; keep at most CAP bytes per stream.

    The cap is applied inside the container (head -c) and again on the host, which keeps at most CAP bytes per pipe.
    """
    start = time.time()
    wrap = f'timeout -k 5 "$0" sh -c "$1" > >(head -c {CAP}) 2> >(head -c {CAP} >&2); rc=$?; wait; exit $rc'
    argv = ["docker", "exec", worker, "bash", "-c", wrap, str(cfg.cmd_timeout), cmd]
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, process_group=0)
    got: dict[str, bytes] = {}
    pipes = {"stdout": p.stdout, "stderr": p.stderr}
    readers = [threading.Thread(target=read_capped, args=(s, k, got)) for k, s in pipes.items()]
    for r in readers:
        r.start()
    try:
        p.wait(timeout=cfg.cmd_timeout + 30)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
    for r in readers:
        r.join()
    secs = round(time.time() - start, 3)
    text = {k: v.decode(errors="replace") for k, v in got.items()}
    return {"cmd": cmd, "exit": p.returncode, **text, "secs": secs, "timed_out": secs >= cfg.cmd_timeout}


def run_agent(
    cfg: RunConfig,
    worker: str,
    system: str,
    task: str,
    best: Callable[[], tuple[float, int] | None],
    log: Log,
    stop: threading.Event,
) -> None:
    """Loop model → tool calls → container until stopped or out of turns, logging every call and command.

    After a turn with tool calls the agent sees its best score; a turn without one gets a nudge instead.
    Model events log only the messages added since the previous call; concatenating them rebuilds each request.
    """
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    sent = 0
    for turn in itertools.count():
        if stop.is_set() or turn == cfg.turns:
            log("stop", turn=turn, reason="turn limit" if turn == cfg.turns else "stopped")
            return
        new, sent = messages[sent:], len(messages)
        for attempt in range(1, ATTEMPTS + 1):
            start = time.time()
            try:
                resp = chat(cfg, messages)
                break
            except (OSError, http.client.HTTPException) as e:  # URLError and HTTPError are OSErrors
                body = e.read().decode(errors="replace") if isinstance(e, urllib.error.HTTPError) else None
                transient = not isinstance(e, urllib.error.HTTPError) or e.code in (429, 500, 502, 503, 504)
                final = attempt == ATTEMPTS or not transient
                request = {"new_messages": new} if final else {}  # logged once, when the turn is given up
                log("model_error", turn=turn, attempt=attempt, error=repr(e), body=body, **request)
                if final:
                    raise
                if stop.wait(10 * attempt):
                    log("stop", turn=turn, reason="stopped")
                    return
        log("model", turn=turn, new_messages=new, response=resp, secs=round(time.time() - start, 3))
        msg = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        assistant = {"role": "assistant", "content": msg.get("content")}
        if "_responses_output" in msg:  # the Responses API needs its own output items back, incl. reasoning
            assistant["_responses_output"] = msg["_responses_output"]
        messages.append({**assistant, "tool_calls": calls} if calls else assistant)  # strict APIs reject []
        for call in calls:
            log("exec_start", turn=turn, call_id=call["id"], arguments=call["function"]["arguments"])
            try:
                result = bash(cfg, worker, json.loads(call["function"]["arguments"])["cmd"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                result = {"cmd": None, "error": f"bad tool arguments: {e}", "raw": call["function"]["arguments"]}
            seen = f"exit={result.get('exit')}\n{result.get('stdout', '')}{result.get('stderr', result.get('error'))}"
            truncated = len(seen) > cfg.max_output
            log("exec", turn=turn, call_id=call["id"], truncated=truncated, chars=len(seen), **result)
            seen = seen[-cfg.max_output :] if truncated else seen
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": seen})
        if not calls:
            log("nudge", turn=turn)
        messages.append({"role": "user", "content": status(best()) if calls else NUDGE})
