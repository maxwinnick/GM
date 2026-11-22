"""Offline stand-in for ``ecr_topics.llm_caller.call_llm`` used by the tests.

Returns canned JSON for each TopicGPT / LLooM prompt and raises the same "finish_reason='length'"
error a real endpoint gives when a reply would exceed ``LIMIT`` characters.
"""
import json
import threading

_lock = threading.Lock()
calls = {"n": 0, "too_long": 0}
LIMIT = 3100


def _quote(d, a, b):
    return " ".join(d["text"].split()[a:b])


def _response(prompt, data):
    if "new_topics" in prompt:
        return {"new_topics": [{"axis": "why", "name": "Cost", "definition": "d"},
                               {"axis": "what", "name": "Seat", "definition": "d"}],
                "ecrs": [{"id": d["id"], "why": ["Cost"], "what": "Seat"} for d in data]}
    if "names only" in prompt and "topics" in prompt:
        name = "Cost" if "WHY" in prompt else "Seat"
        return {"topics": [{"name": name, "merged_from": [name]}]}
    if "exclude_rule" in prompt:
        return {"topics": [{"given_name": d["given_name"], "name": d["given_name"] + " X", "definition": "x" * 120,
                            "include_rule": "y" * 120, "exclude_rule": "z" * 120} for d in data]}
    if "Merge only true duplicates" in prompt:
        return {"concepts": [{"name": "Cost saving", "merged_from": ["Cost saving"]}]}
    if "Propose 1-3" in prompt:
        return {"concepts": [{"name": "Cost saving", "definition": "d", "criterion": "c"}]}
    if "criterion: a yes/no" in prompt:
        return {"concepts": [{"given_name": d["given_name"], "name": "Cost saving", "definition": "d" * 100,
                              "criterion": "c" * 100} for d in data]}
    if "bullets" in prompt:
        return {"records": [{"id": d["id"], "bullets": ["cost save on part", "second reason here"]} for d in data]}
    if "Score each ECR" in prompt:
        return {"records": [{"id": d["id"], "concepts": [{"name": "Cost saving", "quote": _quote(d, 1, 8)}]}
                            for d in data]}
    return {"records": [{"id": d["id"], "why": [{"topic": "Cost X", "quote": _quote(d, 1, 9)}],
                         "what": {"topic": "Seat X", "quote": _quote(d, 1, 6)}} for d in data]}


def call_llm(*, model, instructions, prompt, data):
    with _lock:
        calls["n"] += 1
    r = _response(prompt, data)
    if len(json.dumps(r)) > LIMIT:
        with _lock:
            calls["too_long"] += 1
        raise ValueError("The model response did not contain valid JSON. finish_reason='length'; preview")
    return r
