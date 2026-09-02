#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Check every endpoint in endpoints.json before spending ten minutes on a sweep.

    ./preflight.py                  # all endpoints
    ./preflight.py --only kimi-k3   # one of them

Why this exists
---------------
A When2Call sweep is 600 requests. Finding out afterwards that one endpoint
401'd, or served a different model than you configured, or cannot emit
structured tool calls at all, costs the whole run. Worse, two of those three
failures produce a *plausible-looking number* rather than an error:

  * A model that describes a tool call in prose instead of returning
    `tool_calls` scores as "decided not to call" on every single example. That
    is not a low score, it is a broken measurement, and it looks the same.

  * A budget too small for a reasoning model is spent thinking, so the call
    never arrives. Recall is understated and tool-selection accuracy is
    flattered, because fewer calls were attempted.

This script sends exactly one request per endpoint and reports what came back.
It never prints an API key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

# A prompt where calling is unambiguously correct: the tool exists and every
# required argument is present. Anything other than a tool call is a finding.
PROBE_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string",
                                    "description": "City name"}},
            "required": ["city"],
        },
    },
}]
PROBE_PROMPT = "What is the weather in Austin, Texas right now?"

OK, WARN, FAIL, SKIP = "ok", "warn", "FAIL", "skip"


# A sweep is 3 x n examples per endpoint, run `concurrency` at a time. If one
# request takes t seconds, that endpoint alone costs roughly
# (3n / concurrency) * t. At n=40 and concurrency 2 a 30 s request is a
# half-hour; a 264 s request is most of a day.
SLOW_REQUEST_S = 25.0
SWEEP_N_PER_LABEL = 40


def check(spec: dict, timeout: int | None = None) -> dict:
    """One endpoint, one request. Returns a verdict dict."""
    from openai import OpenAI

    name = spec["name"]
    r = {"name": name, "location": spec.get("location", ""),
         "verdict": OK, "notes": []}

    key_env = spec.get("api_key_env")
    if key_env:
        api_key = os.environ.get(key_env, "")
        if not api_key:
            r["verdict"] = SKIP
            r["notes"].append(f"{key_env} is not set in this environment")
            return r
    else:
        api_key = "not-needed"

    timeout = timeout if timeout is not None else spec.get("timeout", 120)
    client = OpenAI(base_url=spec["base_url"], api_key=api_key, timeout=timeout)
    configured = spec.get("model")

    # --- 1. can we reach it, and is the configured model actually there? ----
    try:
        served = [m.id for m in client.models.list().data]
    except Exception as e:                                      # noqa: BLE001
        r["verdict"] = FAIL
        msg = str(e)
        if "401" in msg or "Unauthorized" in msg:
            r["notes"].append(
                f"401 Unauthorized. Is {key_env} a key for {spec['base_url']}? "
                "Keys are not portable between the public catalog and the "
                "internal Inference Hub.")
        else:
            r["notes"].append(f"cannot list models: {type(e).__name__}: {msg[:110]}")
        return r

    r["catalogue"] = len(served)
    if configured and configured not in served:
        if len(served) == 1:
            r["notes"].append(f"serves one model, '{served[0]}' — using that")
            configured = served[0]
        else:
            r["verdict"] = FAIL
            near = [m for m in served if configured.split("/")[-1] in m]
            hint = f" Closest: {', '.join(near[:3])}." if near else ""
            r["notes"].append(
                f"'{configured}' not offered ({len(served)} models).{hint}")
            return r

    # --- 2. does it return a *structured* tool call? ------------------------
    budget = spec.get("max_tokens", 3072)
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=configured,
            messages=[{"role": "user", "content": PROBE_PROMPT}],
            tools=PROBE_TOOL, tool_choice="auto",
            max_tokens=budget, temperature=0.0,
            **({"extra_body": spec["extra_body"]} if spec.get("extra_body") else {}),
        )
    except Exception as e:                                      # noqa: BLE001
        r["verdict"] = FAIL
        msg = str(e)
        if "429" in msg:
            r["notes"].append("429 rate limited on a single request — lower "
                              "concurrency will not save the sweep")
        elif "Timeout" in type(e).__name__:
            r["notes"].append(
                f"timed out after {timeout}s. Raise \"timeout\" for this "
                f"endpoint in endpoints.json -- free-tier endpoints queue, and "
                f"a timeout here is a queue length, not a broken model.")
        elif "tool" in msg.lower():
            r["notes"].append(f"rejected the tools parameter: {msg[:110]}")
        else:
            r["notes"].append(f"{type(e).__name__}: {msg[:110]}")
        return r

    r["latency_s"] = round(time.perf_counter() - t0, 2)
    choice = resp.choices[0]
    msg = choice.message
    used = getattr(resp.usage, "completion_tokens", None)
    r["model"] = resp.model
    r["tokens"] = used
    r["finish"] = choice.finish_reason

    reasoning = (getattr(msg, "reasoning", None)
                 or getattr(msg, "reasoning_content", None) or "")
    r["reasons"] = bool(reasoning)

    if msg.tool_calls:
        r["tool"] = msg.tool_calls[0].function.name
        try:
            json.loads(msg.tool_calls[0].function.arguments)
        except Exception:                                       # noqa: BLE001
            r["verdict"] = WARN
            r["notes"].append("tool_calls present but arguments are not valid JSON")
    else:
        r["verdict"] = FAIL
        if choice.finish_reason == "length":
            suggest = budget * 2
            r["notes"].append(
                f"budget of {budget} exhausted before any tool call "
                f"({used} tokens, all reasoning). Try max_tokens {suggest} "
                f"for this endpoint in endpoints.json.")
        else:
            body = (msg.content or "").strip().replace("\n", " ")[:70]
            r["notes"].append(
                "no structured tool_calls — every When2Call example would "
                f"score as 'decided not to call'. Returned text instead: {body!r}")

    # --- 3. latency, projected onto the full sweep --------------------------
    # A slow endpoint is not a broken one, but it can make the sweep
    # impractical, and that is worth knowing before you start rather than
    # forty minutes in.
    lat = r["latency_s"]
    if lat >= SLOW_REQUEST_S:
        rounds = (3 * SWEEP_N_PER_LABEL) / max(1, spec.get("concurrency", 2))
        mins = rounds * lat / 60
        r["verdict"] = WARN if r["verdict"] == OK else r["verdict"]
        unit = f"{mins/60:.1f} hours" if mins > 90 else f"{mins:.0f} minutes"
        r["notes"].append(
            f"{lat:.0f}s for a trivial request. At n={SWEEP_N_PER_LABEL} and "
            f"concurrency {spec.get('concurrency', 2)} this endpoint alone "
            f"would take about {unit}.")
        if used and used < 60:
            r["notes"].append(
                f"only {used} tokens generated, so that time is queueing on a "
                f"shared endpoint, not compute. Raising concurrency overlaps "
                f"the waits; lowering max_tokens will not help.")

    # --- 4. budget headroom -------------------------------------------------
    if r["verdict"] == OK and used and used > 0.8 * budget:
        r["verdict"] = WARN
        r["notes"].append(
            f"used {used} of {budget} tokens on a trivial prompt — little "
            f"headroom for the harder When2Call examples")

    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "endpoints.json"))
    ap.add_argument("--only", action="append", help="check only these (repeatable)")
    args = ap.parse_args()

    specs = json.loads(Path(args.config).read_text())["endpoints"]
    if args.only:
        specs = [s for s in specs if s["name"] in args.only]
    if not specs:
        print("No endpoints selected.", file=sys.stderr)
        return 2

    print(f"Preflight: {len(specs)} endpoint(s), one request each.\n")
    results = []
    for spec in specs:
        print(f"  {spec['name']:<18} ...", end=" ", flush=True)
        r = check(spec)
        results.append(r)
        bits = []
        if r.get("tool"):
            bits.append(f"tool_calls -> {r['tool']}")
        if r.get("tokens") is not None:
            bits.append(f"{r['tokens']} tok")
        if r.get("reasons"):
            bits.append("reasons")
        if r.get("latency_s"):
            bits.append(f"{r['latency_s']}s")
        print(f"{r['verdict']:<5} {'  '.join(bits)}")
        for n in r["notes"]:
            print(f"      {n}")

    print()
    bad = [r for r in results if r["verdict"] == FAIL]
    skipped = [r for r in results if r["verdict"] == SKIP]
    warned = [r for r in results if r["verdict"] == WARN]

    if bad:
        print(f"  {len(bad)} endpoint(s) would produce meaningless data: "
              f"{', '.join(r['name'] for r in bad)}")
        print("  Fix these before sweeping — a broken measurement looks like a low score.")
    if skipped:
        print(f"  {len(skipped)} skipped for missing credentials: "
              f"{', '.join(r['name'] for r in skipped)}")
    if warned:
        print(f"  {len(warned)} with warnings — usable, but read the notes.")
    if not bad and not skipped:
        print("  All endpoints ready. Run:  ./run_benchmark.py --config "
              f"{Path(args.config).name} --n 40 --save-raw")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
