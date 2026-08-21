#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validate a completed run before anyone quotes it.

Checks that the manifest records what a reader needs to trust or reproduce the
result: which GPU, which library versions, which model revisions, and whether
the stages that matter actually succeeded.

It does not check image quality or whether the recipe is correct. Those need a
human looking at outputs, and this deliberately does not pretend otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_ENVIRONMENT = ("hostname", "cpu_arch", "python", "gpu", "packages")
CRITICAL_PACKAGES = ("torch", "diffusers")
EXPECTED_STAGES = ("preflight", "download", "export", "verify")


def check(manifest_path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not manifest_path.exists():
        return [f"No manifest at {manifest_path}"], []

    data = json.loads(manifest_path.read_text())

    environment = data.get("environment", {})
    for field in REQUIRED_ENVIRONMENT:
        if not environment.get(field):
            errors.append(f"environment.{field} is missing")

    gpu = environment.get("gpu", {})
    if not gpu.get("available"):
        errors.append("No GPU recorded. Results cannot be attributed to hardware.")
    else:
        architectures = gpu.get("architecture") or []
        if not any(str(a).startswith("10.") for a in architectures):
            warnings.append(
                f"GPU architecture {architectures} is not Blackwell. NVFP4 results are "
                "only meaningful on Blackwell."
            )
        if len(architectures) > 1:
            warnings.append(f"Mixed GPU architectures in one run: {architectures}")

    packages = environment.get("packages", {})
    for package in CRITICAL_PACKAGES:
        if not packages.get(package):
            errors.append(f"No version recorded for {package}")

    stages = {entry["stage"]: entry for entry in data.get("stages", [])}
    for name in EXPECTED_STAGES:
        entry = stages.get(name)
        if entry is None:
            warnings.append(f"Stage '{name}' was never run")
        elif entry["status"] != "ok":
            errors.append(f"Stage '{name}' finished with status '{entry['status']}'")

    download = stages.get("download", {}).get("outputs", {})
    for record in download.get("downloads", []):
        if not record.get("revision"):
            warnings.append(
                f"{record.get('repo')} has no pinned revision. A floating branch makes "
                "this run unreproducible later."
            )

    export = stages.get("export", {}).get("outputs", {})
    if export and not export.get("hf_export"):
        errors.append("Export stage recorded no Diffusers export path")

    preflight = stages.get("preflight", {}).get("outputs", {})
    if preflight.get("ephemeral_workspace"):
        warnings.append(
            "Workspace was node-local. Confirm the export was copied somewhere "
            "persistent before the allocation ended."
        )

    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        help="Path to run_manifest.json. Defaults to $FLUX_QUANT_WORKSPACE/results/run_manifest.json",
    )
    args = parser.parse_args(argv)

    if args.manifest:
        path = Path(args.manifest)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from common import paths

        path = paths.resolve(create=False).results / "run_manifest.json"

    errors, warnings = check(path)

    for warning in warnings:
        print(f"warning: {warning}")
    for error in errors:
        print(f"error: {error}", file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} error(s). This run is not ready to quote.", file=sys.stderr)
        return 1
    print(f"\nManifest OK{f', {len(warnings)} warning(s)' if warnings else ''}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
