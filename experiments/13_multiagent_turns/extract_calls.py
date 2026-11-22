"""Model calls of MAMUBench episodes, as JSON lines that can be replayed through the weights.

Writes, per episode, one line per call to the target model: the request as the harness sent it (messages, tools), the
response (reasoning, text, tool calls), the token counts the server reported, and the agent that made the call (the
index of its "target" span in the episode: 0 is the first agent the harness created).

Two sources:
  - a runs folder: the inspect .eval log of every finished episode in it;
  - a .db file: a copy of inspect's live sample buffer (~/.local/share/inspect_ai/samplebuffer), for an episode that
    is still running or that the harness will retry (a retry discards the transcript of the failed attempt).

inspect keeps the raw request only for the first calls of an episode, so every request is rebuilt from the logged
messages with the provider's own converter (the one that built the request in the run: earlier reasoning goes back
into the prompt as <think> text). Where the raw request was kept, the rebuilt one is checked against it.

Run with MAMUBench's environment (it has inspect_ai):

    /resnick/groups/Safety_First/aquisper/mamubench/.venv/bin/python extract_calls.py RUNS_DIR OUT_DIR
    /resnick/groups/Safety_First/aquisper/mamubench/.venv/bin/python extract_calls.py BUFFER.db OUT_DIR --name EPISODE
"""
import argparse
import asyncio
import glob
import json
import os
import sqlite3

os.environ.setdefault("OPENROUTER_API_KEY", "local-not-used")
from inspect_ai.event import ModelEvent
from inspect_ai.log import read_eval_log, resolve_sample_attachments
from inspect_ai.model import get_model

parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("out")
parser.add_argument("--name", help="episode name for a buffer copy")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)


def plain(x):
    return json.loads(json.dumps(x))


def resolve(x, attachments):
    """Long strings are stored as attachment references; put the text back."""
    if isinstance(x, str):
        return attachments.get(x[len("attachment://"):], x) if x.startswith("attachment://") else x
    if isinstance(x, list):
        return [resolve(v, attachments) for v in x]
    if isinstance(x, dict):
        return {k: resolve(v, attachments) for k, v in x.items()}
    return x


async def rows_of(episode, events, raw_requests):
    """events: (kind, span id or None, parent span id or None, name or None, ModelEvent or None), in order."""
    parent, name, agents, rows, checked, api = {}, {}, [], [], 0, None
    for kind, span, parent_id, span_name, e in events:
        if kind == "span_begin":
            parent[span], name[span] = parent_id, span_name
            if span_name == "target":
                agents.append(span)
            continue
        if e.role != "target" or e.pending or not e.output or not e.output.choices or e.output.usage is None or e.error:
            continue  # other roles, calls still running, and calls the server refused
        s = span
        while s is not None and s not in agents:
            s = parent.get(s)
        api = api or get_model(e.model).api
        messages = plain(await api.messages_to_openai(e.input))
        tools = plain(api.tools_to_openai(e.tools))
        request = raw_requests.get(e.uuid)
        if request is not None:
            assert messages == request["messages"], f"{episode}: rebuilt messages differ from the request"
            assert tools == request["tools"], f"{episode}: rebuilt tools differ from the request"
            checked += 1
        msg = e.output.choices[0].message
        parts = msg.content if isinstance(msg.content, list) else []
        rows.append({
            "episode": episode, "call": len(rows), "agent": agents.index(s) if s in agents else None,
            "time": str(e.timestamp), "model": e.model, "messages": messages, "tools": tools,
            "reasoning": "".join(p.reasoning for p in parts if p.type == "reasoning"),
            "content": msg.content if isinstance(msg.content, str) else "".join(p.text for p in parts if p.type == "text"),
            "tool_calls": [{"name": t.function, "arguments": t.arguments} for t in (msg.tool_calls or [])],
            "stop_reason": e.output.choices[0].stop_reason,
            "prompt_tokens": e.output.usage.input_tokens, "completion_tokens": e.output.usage.output_tokens,
            "reasoning_tokens": e.output.usage.reasoning_tokens,
        })
    with open(os.path.join(args.out, episode + ".jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    per_agent = [sum(r["agent"] == i for r in rows) for i in range(len(agents))]
    print(f"{episode}: {len(rows)} calls, per agent {per_agent}, {checked} checked against the raw request")


async def from_runs(runs):
    for episode in sorted(os.listdir(runs)):
        logs = sorted(glob.glob(os.path.join(runs, episode, "logs", "*.eval")))
        if not logs or not os.path.exists(os.path.join(runs, episode, "DONE.json")):
            continue
        raw = read_eval_log(logs[-1]).samples[0]
        attachments = dict(raw.attachments)  # resolving the sample empties them, and leaves the raw requests alone
        events, requests = [], {}
        for e in resolve_sample_attachments(raw).events:
            kind = type(e).__name__
            if kind == "SpanBeginEvent":
                events.append(("span_begin", e.id, e.parent_id, e.name, None))
            elif kind == "ModelEvent":
                events.append(("model", e.span_id, None, None, e))
                if e.call is not None:
                    requests[e.uuid] = resolve(e.call.request, attachments)
        await rows_of(episode, events, requests)


async def from_buffer(db, episode):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    attachments = dict(conn.execute("select hash, content from attachments"))
    pool = [json.loads(d) for (d,) in conn.execute("select data from message_pool order by id")]
    calls = [json.loads(d) for (d,) in conn.execute("select data from call_pool order by id")]
    latest, order = {}, []  # an event is written when the call starts (pending) and again when it returns
    for (d,) in conn.execute("select data from events order by id"):
        e = json.loads(d)
        if e.get("event") == "span_begin":
            order.append(("span", e))
        elif e.get("event") == "model":
            if e["uuid"] not in latest:
                order.append(("model", e["uuid"]))
            latest[e["uuid"]] = e
    events, requests = [], {}
    for kind, x in order:
        if kind == "span":
            events.append(("span_begin", x["id"], x.get("parent_id"), x["name"], None))
            continue
        e = latest[x]
        if e.get("pending") or not e.get("output"):
            continue
        refs = e.pop("input_refs", None) or []
        e["input"] = [m for a, b in refs for m in pool[a:b]] + (e.get("input") or [])
        call = e.pop("call", None)
        call_refs = e.pop("call_refs", None)
        e = resolve(e, attachments)
        events.append(("model", e["span_id"], None, None, ModelEvent.model_validate(e)))
    await rows_of(episode, events, requests)


if args.source.endswith(".db"):
    asyncio.run(from_buffer(args.source, args.name or os.path.basename(args.source)[:-3]))
else:
    asyncio.run(from_runs(args.source))
