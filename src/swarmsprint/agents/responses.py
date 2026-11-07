"""Adapter for OpenAI's Responses API, which OpenAI requires for reasoning models with function tools.

The loop keeps a chat-completions message list; this module translates it to Responses `input` items and turns the
response back into a chat-completions-shaped message. Reasoning stays on: the model's encrypted reasoning items are
kept on the assistant message (`_responses_output`) and sent back every turn, so it keeps its reasoning across tool
calls without OpenAI storing anything (`store: false`).
"""

import json
import urllib.request
from typing import Any


def to_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat-completions messages as Responses input items."""
    items: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "assistant" and "_responses_output" in m:
            items += m["_responses_output"]  # reasoning, function_call and message items exactly as returned
        elif m["role"] == "tool":
            items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]})
        else:
            items.append({"role": "developer" if m["role"] == "system" else m["role"], "content": m["content"] or ""})
    return items


def chat(model: str, messages: list[dict[str, Any]], tool: dict[str, Any], url: str, key: str) -> dict[str, Any]:
    """One Responses API call, returned in the chat-completions response shape (plus the raw response)."""
    fn = tool["function"]
    body = {
        "model": model,
        "input": to_input(messages),
        "tools": [
            {"type": "function", "name": fn["name"], "description": fn["description"], "parameters": fn["parameters"]}
        ],
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        raw: dict[str, Any] = json.load(r)
    output = raw.get("output") or []
    text = "".join(c.get("text", "") for o in output if o["type"] == "message" for c in o.get("content") or [])
    summary = "".join(s.get("text", "") for o in output if o["type"] == "reasoning" for s in o.get("summary") or [])
    calls = [
        {"id": o["call_id"], "type": "function", "function": {"name": o["name"], "arguments": o["arguments"]}}
        for o in output
        if o["type"] == "function_call"
    ]
    msg = {"role": "assistant", "content": text, "reasoning_content": summary, "tool_calls": calls}
    msg["_responses_output"] = output
    choice = {"message": msg, "finish_reason": raw.get("status")}
    return {"model": raw.get("model"), "usage": raw.get("usage"), "choices": [choice], "raw": raw}
