#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Chart the latency probe from results/latency.json.

    ./make_latency_chart.py                    # -> results/latency.png
    ./make_latency_chart.py --compact          # shorter, for slides

Two panels:
  left   time to first token, median with the full min-max range drawn
  right  decode rate, median with range

The range is drawn deliberately and at the same weight as the median. On a
shared gateway the median can look healthy while the spread runs to seconds,
and a chart showing only medians would hide exactly the thing a reader needs
to see before trusting the number. Where the whisker is long, the honest
reading is "this endpoint is queueing", not "this model is slow".

Endpoints that were skipped -- usually a missing API key -- are drawn as an
explicit row saying so, never as a zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
GREEN, GREY, RED, INK = "#76B900", "#B9BEC4", "#C4453B", "#141618"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", default=str(HERE / "results" / "latency.json"))
    ap.add_argument("--out", default=str(HERE / "results" / "latency.png"))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--compact", action="store_true",
                    help="shorter figure with no footnote — for embedding in slides")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(Path(args.latency).read_text())
    rows = data["endpoints"]
    ok = [r for r in rows if not r.get("skipped")]
    skipped = [r for r in rows if r.get("skipped")]

    is_local = lambda r: "local" in (r.get("location") or "").lower()
    ok.sort(key=lambda r: (0 if is_local(r) else 1, r["ttft_ms"]["median"]))
    names = [r["name"] for r in ok] + [r["name"] for r in skipped]

    height = 3.4 if args.compact else 4.2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, height), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    y = list(range(len(names)))[::-1]

    def panel(ax, key, fmt, label, title):
        # Axis range first, so the value label can be offset by a fixed fraction
        # of the axis rather than a fraction of its own value. A local TTFT of
        # 70 ms on an axis running to 5 s would otherwise land on top of its own
        # whisker cap, which is precisely the row a reader looks at first.
        top = max([r[key]["max"] for r in ok] + [1]) * 1.30
        pad = top * 0.015
        for i, r in enumerate(ok):
            v, c, yy = r[key], GREEN if is_local(r) else GREY, y[i]
            ax.barh(yy, v["median"], height=0.55, color=c, zorder=2)
            if v["max"] > v["min"]:
                ax.plot([v["min"], v["max"]], [yy, yy], color=INK, lw=1.4,
                        zorder=3, solid_capstyle="butt")
                ax.plot([v["min"], v["max"]], [yy, yy], "|", color=INK, ms=7, zorder=3)
            ax.text(max(v["max"], v["median"]) + pad, yy, fmt.format(v["median"]),
                    va="center", fontsize=9.5, color=INK, zorder=4)
        for j, r in enumerate(skipped):
            ax.text(0.02, y[len(ok) + j], f"skipped — {r['skipped'][:34]}",
                    va="center", fontsize=9, style="italic", color=RED,
                    transform=ax.get_yaxis_transform())
        ax.set_yticks(y)
        ax.set_xlabel(label, fontsize=10)
        ax.set_title(title, fontsize=11.5, color=INK, pad=10, loc="left")
        ax.grid(axis="x", color="#E6E8EA", zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(0, top)

    panel(ax1, "ttft_ms", "{:.0f} ms", "Time to first token (ms) — lower is better",
          "How long before anything comes back?")
    panel(ax2, "decode_tok_s", "{:.0f} tok/s", "Decode rate (tokens/s) — higher is better",
          "How fast does it write after that?")
    ax1.set_yticklabels(names, fontsize=10)
    ax2.set_yticklabels([])

    for ax in (ax1, ax2):
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#D8DBDE")
        ax.tick_params(length=0)

    if not args.compact:
        n = data.get("samples_requested", "?")
        fig.text(0.008, -0.02,
                 f"{n} sequential samples per endpoint, concurrency 1, max_tokens "
                 f"{data.get('max_tokens','?')}, one warmup discarded. Bars are medians; "
                 f"whiskers are the full min-max range. Green = local on one DGX Spark. "
                 f"Hosted figures are end-to-end and include network and gateway queueing, "
                 f"which cannot be separated from model time.  Run: {data.get('generated','')}",
                 fontsize=8.2, color="#7A8188")

    fig.tight_layout()
    Path(args.out).parent.mkdir(exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
