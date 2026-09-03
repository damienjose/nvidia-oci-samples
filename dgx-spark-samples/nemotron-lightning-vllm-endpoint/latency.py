#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Measure end-to-end latency to every endpoint in endpoints.json.

    ./latency.py                    # all endpoints, 7 samples each
    ./latency.py --only spark       # just the local one, no key needed
    ./latency.py --samples 11       # tighter medians, longer run

What this measures, precisely
-----------------------------
**Time to first token** and **decode rate**, one request at a time, from the
machine you run it on. Not throughput: concurrency is 1 throughout, because a
single stream is what an interactive tool feels like.

Those two metrics are chosen because they compare across models fairly and
total wall time does not. Total depends on how many tokens a model decided to
emit, and a reasoning model that thinks for 900 tokens is not "slower" than one
that answers in 40 -- it did more work. TTFT is unaffected by output length,
and decode rate is per-token by construction.

What it does NOT isolate
------------------------
For a hosted endpoint this is **end-to-end from where you are sitting**: your
network path, the gateway's routing and admission control, any queueing behind
other tenants, and only then the model. Those cannot be separated from outside,
so this script does not pretend to. A hosted figure here is what a developer on
this network would actually experience at this time of day -- which is a useful
number, and an honest one, as long as it is not relabelled "model speed".

Two consequences worth taking seriously before quoting a result:

  * **It is a snapshot.** Shared free-tier gateways vary by hour. Re-run it
    close to when the number will be used, and record the timestamp -- which
    is why results/latency.json carries one.

  * **A slow third-party model here is probably a statement about the free
    gateway, not about the model.** The pair worth trusting is the same model
    served in two places, because everything except the serving path is held
    constant. Treat the rest as context.

The local endpoint needs no API key. Hosted endpoints read NVIDIA_API_KEY from
the environment, and are skipped with a reason if it is not set.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"

# Short, fixed, and deliberately dull. A prompt that invites a long answer would
# make total time depend on the model's verbosity rather than the serving path,
# and we cap output anyway.
PROMPT = "In one sentence, what is a GPU?"
MAX_TOKENS = 128


def one_request(client, model: str, extra_body: dict | None, timeout: int):
    """Stream one completion. Returns (ttft_s, decode_tok_s, n_tokens, total_s)."""
    t0 = time.perf_counter()
    ttft = None
    n = 0
    stream = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": PROMPT}],
        max_tokens=MAX_TOKENS, temperature=0.0, stream=True, timeout=timeout,
        **({"extra_body": extra_body} if extra_body else {}),
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # Count reasoning tokens too. They are tokens the server generated and
        # streamed; excluding them would understate a thinking model's decode
        # rate, which is the opposite of what this measures.
        piece = (getattr(delta, "content", None)
                 or getattr(delta, "reasoning", None)
                 or getattr(delta, "reasoning_content", None))
        if piece:
            if ttft is None:
                ttft = time.perf_counter() - t0
            n += 1
    total = time.perf_counter() - t0
    if ttft is None or n < 2:
        return None
    return ttft, (n - 1) / (total - ttft), n, total


def probe(spec: dict, samples: int, verbose: bool = True) -> dict | None:
    from openai import OpenAI

    name = spec["name"]
    key_env = spec.get("api_key_env")
    if key_env and not os.environ.get(key_env):
        return {"name": name, "skipped": f"{key_env} is not set"}

    client = OpenAI(base_url=spec["base_url"],
                    api_key=os.environ.get(key_env, "not-needed") if key_env else "not-needed",
                    timeout=spec.get("timeout", 120))
    model = spec["model"]
    extra = spec.get("extra_body")

    # One throwaway request first. A cold connection pays TLS setup, and a cold
    # server pays cache and graph warmup; neither is what anyone means by
    # latency, and both land entirely on the first sample.
    try:
        one_request(client, model, extra, spec.get("timeout", 120))
    except Exception as e:                                      # noqa: BLE001
        return {"name": name, "skipped": f"{type(e).__name__}: {str(e)[:90]}"}

    ttfts, decodes, errors = [], [], 0
    for i in range(samples):
        try:
            got = one_request(client, model, extra, spec.get("timeout", 120))
        except Exception as e:                                  # noqa: BLE001
            errors += 1
            if verbose:
                print(f"    {name}: sample {i+1} failed — {type(e).__name__}", flush=True)
            continue
        if got is None:
            errors += 1
            continue
        ttfts.append(got[0] * 1000)
        decodes.append(got[1])
        if verbose:
            print(f"    {name}: {i+1}/{samples}  ttft {got[0]*1000:6.0f} ms  "
                  f"decode {got[1]:5.1f} tok/s", flush=True)

    if not ttfts:
        return {"name": name, "skipped": f"all {samples} samples failed"}

    q = lambda xs: {"median": round(statistics.median(xs), 1),
                    "min": round(min(xs), 1), "max": round(max(xs), 1)}
    return {"name": name, "location": spec.get("location", ""), "model": model,
            "samples": len(ttfts), "errors": errors,
            "ttft_ms": q(ttfts), "decode_tok_s": q(decodes)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "endpoints.json"))
    ap.add_argument("--only", action="append", help="endpoint name; repeatable")
    ap.add_argument("--samples", type=int, default=7)
    ap.add_argument("--out", default=str(RESULTS / "latency.json"))
    args = ap.parse_args()

    specs = json.loads(Path(args.config).read_text())["endpoints"]
    if args.only:
        specs = [s for s in specs if s["name"] in args.only]
        if not specs:
            print(f"no endpoint matched {args.only}", file=sys.stderr)
            return 2

    print(f"Latency probe — {args.samples} sequential samples, "
          f"max_tokens {MAX_TOKENS}, concurrency 1\n")
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "prompt": PROMPT, "max_tokens": MAX_TOKENS,
           "samples_requested": args.samples, "endpoints": []}

    for spec in specs:
        print(f"  {spec['name']} ...", flush=True)
        out["endpoints"].append(probe(spec, args.samples))

    print(f"\n  {'endpoint':<18}{'TTFT median':>14}{'range':>18}{'decode':>16}")
    print("  " + "-" * 66)
    for r in out["endpoints"]:
        if r.get("skipped"):
            print(f"  {r['name']:<18}  skipped — {r['skipped'][:44]}")
            continue
        t, d = r["ttft_ms"], r["decode_tok_s"]
        rng = f"{t['min']:.0f}–{t['max']:.0f} ms"
        print(f"  {r['name']:<18}{t['median']:>11.0f} ms{rng:>18}"
              f"{d['median']:>11.1f} tok/s")

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {args.out}")
    print("\n  Hosted figures are end-to-end from this machine and include network and\n"
          "  gateway queueing, which cannot be separated from model time. The pair worth\n"
          "  trusting is the same model in two places; read the rest as context.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
