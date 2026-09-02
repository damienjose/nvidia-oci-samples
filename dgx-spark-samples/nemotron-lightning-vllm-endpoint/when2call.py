# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Scoring for the nvidia/When2Call agentic tool-calling benchmark.

When2Call asks a harder question than most function-calling benchmarks: not
"was the tool call correct" but "should the model have called anything at all".
Three labels:

    tool_call         every required argument is available -> call the tool
    request_for_info  a required argument was withheld     -> ask a follow-up
    cannot_answer     no available tool covers the request -> decline

Scoring is deterministic. We observe what the server actually did -- no LLM
judge -- so the numbers are reproducible on your own hardware.

Metrics:
    decision accuracy             did it call exactly when it should have?
    actionable decision accuracy  correct decision AND it either called a tool
                                  or produced real text. Guards against a model
                                  scoring well by saying nothing.
    tool-selection accuracy       when it correctly called, was it the right tool?
    over-call rate                how often it called when it should not have.
                                  Lower is better. This is the number that
                                  predicts agent misbehaviour in production.

Dataset: https://huggingface.co/datasets/nvidia/When2Call (CC-BY-4.0)
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, asdict
from typing import Any

LABELS = ("tool_call", "request_for_info", "cannot_answer")

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the tools listed below. "
    "Use a tool only when the request needs it and every required argument is "
    "available. If a required argument is missing, ask the user for it. If no "
    "available tool covers the request, say so plainly. Always reply with "
    "something useful -- never return an empty response."
)


# --------------------------------------------------------------------------
# Wilson score interval
# --------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (point, lo, hi) as proportions. Wilson interval, not normal
    approximation -- at n=120 the difference matters."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

def load_examples(n_per_label: int = 40, seed: int = 0,
                  split: str = "test", config: str = "mcq") -> list[dict]:
    """Load When2Call, stratified equally across the three labels.

    Field names on the Hub have moved between revisions; we normalise here and
    fail loudly rather than silently mis-scoring.
    """
    from datasets import load_dataset

    ds = load_dataset("nvidia/When2Call", config, split=split)

    def norm(row: dict) -> dict | None:
        label = row.get("label") or row.get("answer") or row.get("category")
        tools = row.get("tools") or row.get("functions") or row.get("available_tools")
        query = row.get("query") or row.get("question") or row.get("user_query")
        gold = row.get("gold_tool") or row.get("target_function") or row.get("function_name")
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except json.JSONDecodeError:
                return None
        if label not in LABELS or not query or tools is None:
            return None
        return {"label": label, "tools": tools, "query": query, "gold_tool": gold}

    rows = [r for r in (norm(dict(x)) for x in ds) if r]
    if not rows:
        raise RuntimeError(
            "No usable rows after normalisation. The dataset schema has changed -- "
            "inspect load_dataset('nvidia/When2Call', 'mcq', split='test')[0] and "
            "extend norm() in when2call.py."
        )

    rng = random.Random(seed)
    out: list[dict] = []
    for lab in LABELS:
        pool = [r for r in rows if r["label"] == lab]
        rng.shuffle(pool)
        out.extend(pool[:n_per_label])
    rng.shuffle(out)
    return out


def to_openai_tools(tools: Any) -> list[dict]:
    """Convert the dataset's tool spec into OpenAI tools format."""
    converted = []
    for t in tools or []:
        if isinstance(t, str):
            try:
                t = json.loads(t)
            except json.JSONDecodeError:
                continue
        if t.get("type") == "function" and "function" in t:
            converted.append(t)
            continue
        converted.append({
            "type": "function",
            "function": {
                "name": t.get("name", "unknown"),
                "description": t.get("description", ""),
                "parameters": t.get("parameters") or t.get("input_schema")
                              or {"type": "object", "properties": {}},
            },
        })
    return converted


def build_messages(example: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["query"]}]


# --------------------------------------------------------------------------
# Observing what the server did
# --------------------------------------------------------------------------

@dataclass
class Record:
    label: str
    called: bool
    tool_name: str | None
    gold_tool: str | None
    has_text: bool
    finish_reason: str | None
    reasoned: bool = False
    error: str | None = None

    @property
    def predicted(self) -> str:
        return "tool_call" if self.called else "no_call"

    @property
    def decision_correct(self) -> bool:
        want_call = self.label == "tool_call"
        return self.called == want_call

    @property
    def actionable(self) -> bool:
        """Correct decision AND it actually did something."""
        return self.decision_correct and (self.called or self.has_text)


def observe(message: Any, finish_reason: str | None = None) -> Record | None:
    """Turn one chat completion message into a Record. Label is attached by
    the caller; this only reads what the server returned."""
    tool_calls = getattr(message, "tool_calls", None) or []
    content = (getattr(message, "content", None) or "").strip()

    # Reasoning models return their thinking on a separate field -- `reasoning`
    # on vLLM 0.26+, `reasoning_content` on the NVIDIA API catalog and older
    # builds. It is not an answer, so it must not count as one for
    # actionable accuracy. But a model that produced a long trace and no
    # content did not "stay silent" either; it ran out of budget mid-thought.
    # Record that separately so the two are distinguishable in the raw data.
    reasoning = (getattr(message, "reasoning", None)
                 or getattr(message, "reasoning_content", None) or "").strip()

    name = tool_calls[0].function.name if tool_calls else None
    return Record(label="", called=bool(tool_calls), tool_name=name,
                  gold_tool=None, has_text=bool(content),
                  reasoned=bool(reasoning),
                  finish_reason=finish_reason)


# --------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------

def score(records: list[Record]) -> dict:
    ok = [r for r in records if r.error is None]
    n = len(ok)

    dec_k = sum(r.decision_correct for r in ok)
    act_k = sum(r.actionable for r in ok)

    # tool selection: only cases it correctly decided to call are eligible
    sel_pool = [r for r in ok if r.label == "tool_call" and r.called and r.gold_tool]
    sel_k = sum(r.tool_name == r.gold_tool for r in sel_pool)

    # over-call: fired a tool when it should not have
    over_pool = [r for r in ok if r.label != "tool_call"]
    over_k = sum(r.called for r in over_pool)

    def block(k, m):
        p, lo, hi = wilson(k, m)
        return {"k": k, "n": m, "rate": p, "lo": lo, "hi": hi}

    per_label = {}
    for lab in LABELS:
        pool = [r for r in ok if r.label == lab]
        per_label[lab] = block(sum(r.decision_correct for r in pool), len(pool))

    return {
        "n_scored": n,
        "n_errors": len(records) - n,
        "decision_accuracy": block(dec_k, n),
        "actionable_decision_accuracy": block(act_k, n),
        "tool_selection_accuracy": block(sel_k, len(sel_pool)),
        "over_call_rate": block(over_k, len(over_pool)),
        "per_label": per_label,
    }


def summarise(name: str, s: dict) -> str:
    d, a = s["decision_accuracy"], s["actionable_decision_accuracy"]
    t, o = s["tool_selection_accuracy"], s["over_call_rate"]
    return (
        f"  {name}\n"
        f"    decision accuracy      {d['k']:>3}/{d['n']:<3} {pct(d['rate']):>7}"
        f"   [{pct(d['lo'])} – {pct(d['hi'])}]\n"
        f"    actionable             {a['k']:>3}/{a['n']:<3} {pct(a['rate']):>7}\n"
        f"    tool selection         {t['k']:>3}/{t['n']:<3} {pct(t['rate']):>7}\n"
        f"    over-call rate         {o['k']:>3}/{o['n']:<3} {pct(o['rate']):>7}"
        f"   (lower is better)\n"
    )


def overlaps(a: dict, b: dict) -> bool:
    """True if two Wilson intervals overlap -- i.e. the difference is not
    separable at this sample size. Used to state ties honestly."""
    return not (a["hi"] < b["lo"] or b["hi"] < a["lo"])
