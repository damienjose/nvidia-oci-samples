#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Run the When2Call agentic tool-calling benchmark against one or more
OpenAI-compatible endpoints and write a summary you can chart.

    ./run_benchmark.py --config endpoints.json --n 40
    ./run_benchmark.py --only spark --n 12          # quick local slice
    ./run_benchmark.py --config endpoints.json --n 40 --concurrency 1

Why the backoff matters
-----------------------
An earlier sweep issued 960 requests back-to-back at concurrency 4 and tripped
the hosted gateway's rate limit: four of eight models returned HTTP 429 on
*every* call and produced no data at all. Retry-with-exponential-backoff plus a
low default concurrency for remote endpoints is the fix, and it is the reason
this script defaults to 2 workers rather than something faster.

Endpoints are configured in endpoints.json. API keys come from the environment
so nothing secret is ever committed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import when2call as w2c

HERE = Path(__file__).parent
RESULTS = HERE / "results"


# --------------------------------------------------------------------------
# Retrying client call
# --------------------------------------------------------------------------

def call_with_backoff(client, *, model, messages, tools, max_tokens,
                      max_attempts=6, base=2.0, cap=45.0, timeout=180):
    """One chat completion, retrying 429 and 5xx with exponential backoff and
    full jitter. Returns (message, finish_reason, error_str)."""
    last = "unknown"
    for attempt in range(max_attempts):
        try:
            kwargs = dict(model=model, messages=messages,
                          max_tokens=max_tokens, temperature=0.0,
                          timeout=timeout)
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            return choice.message, choice.finish_reason, None
        except Exception as e:                                  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            status = getattr(e, "status_code", None)
            retryable = status in (408, 409, 429, 500, 502, 503, 504) or \
                        "429" in str(e) or "rate" in str(e).lower() or \
                        "timeout" in str(e).lower()
            if not retryable or attempt == max_attempts - 1:
                break
            sleep = min(cap, base * (2 ** attempt))
            sleep = random.uniform(0, sleep)          # full jitter
            time.sleep(sleep)
    return None, None, last


# --------------------------------------------------------------------------
# One endpoint
# --------------------------------------------------------------------------

def run_endpoint(spec: dict, examples: list[dict], max_tokens: int,
                 concurrency: int | None, verbose: bool = True):
    from openai import OpenAI

    name = spec["name"]
    key_env = spec.get("api_key_env")
    api_key = os.environ.get(key_env, "") if key_env else "not-needed"
    if key_env and not api_key:
        return None, f"{key_env} is not set"

    client = OpenAI(base_url=spec["base_url"], api_key=api_key or "not-needed")
    workers = concurrency if concurrency is not None else spec.get("concurrency", 2)

    def one(ex):
        msg, finish, err = call_with_backoff(
            client, model=spec["model"], messages=w2c.build_messages(ex),
            tools=w2c.to_openai_tools(ex["tools"]), max_tokens=max_tokens)
        if err:
            return w2c.Record(label=ex["label"], called=False, tool_name=None,
                              gold_tool=ex.get("gold_tool"), has_text=False,
                              finish_reason=None, error=err)
        rec = w2c.observe(msg, finish)
        rec.label = ex["label"]
        rec.gold_tool = ex.get("gold_tool")
        return rec

    t0 = time.perf_counter()
    records: list[w2c.Record] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, ex): ex for ex in examples}
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            if verbose and i % 10 == 0:
                print(f"    {name}: {i}/{len(examples)}", flush=True)
    elapsed = time.perf_counter() - t0

    errs = [r for r in records if r.error]
    if len(errs) == len(records):
        return None, f"all {len(records)} requests failed — first: {errs[0].error[:120]}"

    summary = w2c.score(records)
    summary.update({"name": name, "model": spec["model"],
                    "location": spec.get("location", ""),
                    "elapsed_s": round(elapsed, 1),
                    "concurrency": workers})
    return (summary, records), None


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "endpoints.json"))
    ap.add_argument("--n", type=int, default=40,
                    help="examples per label (3 labels, so total = 3n). Default 40 -> 120.")
    ap.add_argument("--only", action="append",
                    help="run only these endpoint names (repeatable)")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="override per-endpoint concurrency")
    ap.add_argument("--max-tokens", type=int, default=3072,
                    help="Nemotron's median completion is ~768 tokens with a tail to 4096; "
                         "too low and it exhausts the budget while reasoning and never emits "
                         "a call, which scores identically to deciding not to call.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "summary.json"))
    ap.add_argument("--save-raw", action="store_true")
    args = ap.parse_args()

    specs = json.loads(Path(args.config).read_text())["endpoints"]
    if args.only:
        specs = [s for s in specs if s["name"] in args.only]
    if not specs:
        print("No endpoints selected.", file=sys.stderr)
        return 2

    print(f"Loading When2Call ({args.n} per label = {3 * args.n} examples)...")
    examples = w2c.load_examples(n_per_label=args.n, seed=args.seed)
    print(f"  {len(examples)} examples\n")

    RESULTS.mkdir(exist_ok=True)
    out = {"n_examples": len(examples), "max_tokens": args.max_tokens,
           "seed": args.seed, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "models": [], "failed": []}

    for spec in specs:
        print(f"  {spec['name']} ({spec.get('location','')}) ...", flush=True)
        result, err = run_endpoint(spec, examples, args.max_tokens,
                                   args.concurrency)
        if err:
            print(f"    SKIPPED — {err}\n")
            out["failed"].append({"name": spec["name"], "reason": err})
            continue
        summary, records = result
        out["models"].append(summary)
        print(w2c.summarise(spec["name"], summary))
        if args.save_raw:
            raw = RESULTS / f"raw-{spec['name']}.jsonl"
            raw.write_text("\n".join(json.dumps(asdict(r)) for r in records))

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")
    if out["failed"]:
        print("\nEndpoints that produced no data:")
        for f in out["failed"]:
            print(f"  {f['name']}: {f['reason'][:100]}")
        print("\nA model with no data is reported as 'no data', never as a low score.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
