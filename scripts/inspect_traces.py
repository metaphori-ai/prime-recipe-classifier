#!/usr/bin/env python3
"""Print, per rollout, the model's final assistant message and the reward/metric breakdown
from a vf-run traces.jsonl.

Usage:
    uv run python -I scripts/inspect_traces.py env/outputs/<run-dir>/traces.jsonl [max_rows]
"""
import json
import sys


def find_assistant_messages(obj, out):
    if isinstance(obj, dict):
        if obj.get("role") == "assistant" and "content" in obj:
            out.append(obj["content"])
        for v in obj.values():
            find_assistant_messages(v, out)
    elif isinstance(obj, list):
        for v in obj:
            find_assistant_messages(v, out)


def find_keys(obj, names, out, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if k in names:
                out.append((p, v))
            find_keys(v, names, out, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_keys(v, names, out, f"{path}[{i}]")


def main(path, max_rows=10):
    for n, line in enumerate(open(path, encoding="utf-8")):
        if n >= max_rows:
            break
        r = json.loads(line)
        data = r.get("task", {}).get("data", {})
        print("=" * 100)
        print(f"task_id={data.get('id')}  ok={r.get('ok')}  errors={r.get('errors')}  out_tokens={r.get('num_output_tokens')}")
        print("prompt:", (data.get("prompt") or "")[:160].replace("\n", " / "))
        if "answer" in data or "labels" in data or "gold" in data:
            print("gold:", json.dumps(data.get("answer") or data.get("labels") or data.get("gold"), ensure_ascii=False)[:400])
        found = []
        find_keys(r.get("run", r), {"reward", "metrics", "scores", "rubric", "reward_breakdown", "info", "stop"}, found)
        for p, v in found:
            s = json.dumps(v, ensure_ascii=False, default=str)
            print(f"{p}: {s[:500]}")
        msgs = []
        find_assistant_messages(r.get("traces"), msgs)
        if msgs:
            last = msgs[-1]
            if not isinstance(last, str):
                last = json.dumps(last, ensure_ascii=False)
            print("--- last assistant message ---")
            print(last[:1500])
        else:
            print("--- no assistant message found; top-level keys:", list(r.keys()))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
