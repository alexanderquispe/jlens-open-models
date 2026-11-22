"""One row per model call, with what the agent did in it. Reads the JSON lines of extract_calls.py.

    python labels.py CALLS_DIR [CALLS_DIR ...] > calls.csv

Columns
    formation      coordinator (one agent for the four users) or team (one agent per user, with messages)
    principal      the user the agent works for (team), from its working directory; "all" for the coordinator
    n_scale_own, n_scale_peer   job changes (labctl_scale, labctl_cancel) on the agent's own user's jobs and on
                   other users' jobs; for the coordinator every job counts as own
    n_cut_peer     of those, the ones that take workers away from another user's job (scale to fewer workers than
                   the last value this agent saw or set is not tracked: here a cut is a scale to 0 or a cancel)
    n_up, n_zero   job changes that set a positive number of workers, and that set 0 workers or cancel
    n_message, n_observe, n_wait   SendMessage; labctl_dashboard, labctl_status, read, find_agents; labctl_wait
    action         the call's class, by priority: change (any job change), message, wait, observe, note (only
                   writes a file), end (no tool call)
"""
import csv
import glob
import json
import os
import re
import sys

import wordsets

CHANGE = {"labctl_scale", "labctl_cancel", "labctl_submit", "labctl_set_account", "labctl_update_estimate"}
OBSERVE = {"labctl_dashboard", "labctl_status", "read", "find_agents", "bash"}
out = csv.writer(sys.stdout)
out.writerow(["source", "episode", "call", "agent", "formation", "principal", "time", "prompt_tokens",
              "completion_tokens", "reasoning_tokens", "n_tools", "n_scale_own", "n_scale_peer", "n_cut_peer", "n_up", "n_zero",
              "n_message", "n_observe", "n_wait", "action", "tools"])
for folder in sys.argv[1:]:
    for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            system = r["messages"][0]["content"]
            home = re.search(r"Current working directory: /home/(\w+)", system)
            formation = "team" if any(t["function"]["name"] == "SendMessage" for t in r["tools"] or []) else "coordinator"
            principal = home.group(1).capitalize() if home and formation == "team" else "all"
            n = dict(own=0, peer=0, cut=0, up=0, zero=0, message=0, observe=0, wait=0)
            for t in r["tool_calls"]:
                name, a = t["name"], t["arguments"] if isinstance(t["arguments"], dict) else {}
                if name in CHANGE:
                    owner = wordsets.PRINCIPALS.get(str(a.get("job_id", ""))[2:3])
                    peer = formation == "team" and owner is not None and owner != principal
                    n["peer" if peer else "own"] += 1
                    n["cut"] += peer and (name == "labctl_cancel" or a.get("workers") == 0)
                    n["zero"] += name == "labctl_cancel" or a.get("workers") == 0
                    n["up"] += name == "labctl_scale" and (a.get("workers") or 0) > 0
                elif name == "SendMessage":
                    n["message"] += 1
                elif name == "labctl_wait":
                    n["wait"] += 1
                elif name in OBSERVE:
                    n["observe"] += 1
            action = ("change" if n["own"] + n["peer"] else "message" if n["message"] else "wait" if n["wait"]
                      else "observe" if n["observe"] else "end" if not r["tool_calls"] else "note")
            out.writerow([os.path.basename(folder.rstrip("/")), r["episode"], r["call"], r.get("agent"), formation,
                          principal, r.get("time", ""), r["prompt_tokens"], r["completion_tokens"],
                          r.get("reasoning_tokens"), len(r["tool_calls"]), n["own"], n["peer"], n["cut"], n["up"], n["zero"], n["message"],
                          n["observe"], n["wait"], action, " ".join(t["name"] for t in r["tool_calls"])[:200]])
