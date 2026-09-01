#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
Chart When2Call results from results/summary.json.

    ./make_chart.py                       # -> results/when2call.png
    ./make_chart.py --out /tmp/chart.png

Two panels:
  left   decision accuracy with 95% Wilson intervals
  right  over-call rate (lower is better) -- the metric that actually separates

Models that returned no data are drawn as an explicit "no data" row, never as
a zero score. A rate-limited model is a missing measurement, not a bad one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
GREEN, GREY, RED, INK = "#76B900", "#B9BEC4", "#C4453B", "#141618"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(HERE / "results" / "summary.json"))
    ap.add_argument("--out", default=str(HERE / "results" / "when2call.png"))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--compact", action="store_true",
                    help="shorter figure with no footnote — for embedding in slides, "
                         "where the caption lives on the slide instead")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(Path(args.summary).read_text())
    models = data["models"]
    failed = data.get("failed", [])

    # local model first, then by accuracy
    models.sort(key=lambda m: (0 if "local" in m.get("location", "").lower() else 1,
                               -m["decision_accuracy"]["rate"]))
    names = [m["name"] for m in models] + [f["name"] for f in failed]

    height = 3.6 if args.compact else 4.4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, height), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    y = list(range(len(names)))[::-1]

    def is_local(m):
        return "local" in m.get("location", "").lower()

    # ---- left: decision accuracy -----------------------------------------
    for i, m in enumerate(models):
        d = m["decision_accuracy"]
        c = GREEN if is_local(m) else GREY
        yy = y[i]
        ax1.barh(yy, 100 * d["rate"], height=0.55, color=c, zorder=2)
        ax1.plot([100 * d["lo"], 100 * d["hi"]], [yy, yy],
                 color=INK, lw=1.4, zorder=3, solid_capstyle="butt")
        ax1.plot([100 * d["lo"], 100 * d["hi"]], [yy, yy], "|",
                 color=INK, ms=7, zorder=3)
        # place the value clear of the upper CI whisker, not on top of it
        ax1.text(100 * d["hi"] + 1.8, yy, f"{100*d['rate']:.1f}%",
                 va="center", fontsize=9.5, color=INK, zorder=4)
    for j, f in enumerate(failed):
        yy = y[len(models) + j]
        ax1.text(2, yy, "no data — rate limited", va="center", fontsize=9.5,
                 style="italic", color=RED)

    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlim(0, 100); ax1.set_xlabel("Decision accuracy (%)", fontsize=10)
    ax1.set_title("Did it call a tool exactly when it should have?",
                  fontsize=11.5, color=INK, pad=10, loc="left")
    ax1.grid(axis="x", color="#E6E8EA", zorder=0)
    ax1.set_axisbelow(True)

    # ---- right: over-call rate -------------------------------------------
    for i, m in enumerate(models):
        o = m["over_call_rate"]
        c = GREEN if is_local(m) else GREY
        yy = y[i]
        ax2.barh(yy, 100 * o["rate"], height=0.55, color=c, zorder=2)
        ax2.text(100 * o["rate"] + 0.6, yy, f"{100*o['rate']:.1f}%",
                 va="center", fontsize=9.5, color=INK, zorder=4)
    for j, f in enumerate(failed):
        yy = y[len(models) + j]
        ax2.text(0.6, yy, "no data", va="center", fontsize=9.5,
                 style="italic", color=RED)

    ax2.set_yticks(y); ax2.set_yticklabels([])
    top = max([100 * m["over_call_rate"]["rate"] for m in models] + [10]) * 1.35
    ax2.set_xlim(0, top)
    ax2.set_xlabel("Over-call rate (%) — lower is better", fontsize=10)
    ax2.set_title("How often did it fire a tool it shouldn't have?",
                  fontsize=11.5, color=INK, pad=10, loc="left")
    ax2.grid(axis="x", color="#E6E8EA", zorder=0)
    ax2.set_axisbelow(True)

    for ax in (ax1, ax2):
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#D8DBDE")
        ax.tick_params(length=0)

    n = data.get("n_examples", "?")
    if not args.compact:
        fig.text(0.008, -0.02,
                 f"nvidia/When2Call, {n} examples per model, identical prompts and tool "
                 f"schemas. Error bars are 95% Wilson intervals. Green = running locally "
                 f"on one DGX Spark.  Run: {data.get('generated','')}",
                 fontsize=8.2, color="#7A8188")

    fig.tight_layout()
    Path(args.out).parent.mkdir(exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
