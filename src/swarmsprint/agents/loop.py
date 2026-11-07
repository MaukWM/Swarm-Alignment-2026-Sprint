"""One agent: a model with a bash tool, driving its own worker container turn by turn."""

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from swarmsprint.agents.prompts import NUDGE, status
from swarmsprint.config import RunConfig
from swarmsprint.infra.docker import CAP, read_capped

Log = Callable[..., None]
BASH = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command in your container and get its output.",
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
    },
}


def chat(cfg: RunConfig, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Call the OpenAI-compatible chat completions endpoint with the bash tool."""
    body = json.dumps({"model": cfg.model, "messages": messages, "tools": [BASH]}).encode()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}"}
    req = urllib.request.Request(os.environ["LLM_BASE_URL"], body, headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        out: dict[str, Any] = json.load(r)
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
    best: Callable[[], tuple[int, int] | None],
    log: Log,
    stop: threading.Event,
) -> None:
    """Loop model → tool calls → container for cfg.turns turns, logging every call and command.

    After a turn with tool calls the agent sees its best score; a turn without one gets a nudge instead.
    Model events log only the messages added since the previous call; concatenating them rebuilds each request.
    """
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    sent = 0
    for turn in range(cfg.turns):
        if stop.is_set():
            log("stop", turn=turn, reason="interrupted")
            return
        start, new, sent = time.time(), messages[sent:], len(messages)
        try:
            resp = chat(cfg, messages)
        except Exception as e:
            body = e.read().decode(errors="replace") if isinstance(e, urllib.error.HTTPError) else None
            log("model_error", turn=turn, new_messages=new, error=repr(e), body=body)
            raise
        log("model", turn=turn, new_messages=new, response=resp, secs=round(time.time() - start, 3))
        msg = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        assistant = {"role": "assistant", "content": msg.get("content")}
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
    log("stop", turn=cfg.turns - 1, reason="turn limit")
