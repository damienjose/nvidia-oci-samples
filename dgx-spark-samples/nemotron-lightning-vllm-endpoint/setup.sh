#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# One-time setup: verify the machine, install client deps, pre-download weights.
#
# Run this once. Then ./serve.sh to start the endpoint.
# Downloads ~21.6 GB of model weights.

set -euo pipefail

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.19.0-cu130-ubuntu2404}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

# --- 1. machine check --------------------------------------------------------

say "Checking the machine"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found. Is this a DGX Spark with the NVIDIA stack installed?"
else
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

ARCH="$(uname -m)"
echo "Architecture: $ARCH"
if [[ "$ARCH" != "aarch64" ]]; then
  echo "NOTE: expected aarch64 (DGX Spark). Continuing anyway."
fi

if ! docker info >/dev/null 2>&1; then
  cat <<'EOF'

ERROR: cannot talk to the Docker daemon. Fix with:

  sudo gpasswd -a $(whoami) docker

Then log out and back in. In JupyterLab, restart the *server*, not the kernel --
group membership is inherited at login. Use $(whoami), not $USER.
EOF
  exit 1
fi
echo "Docker: OK"

# --- 2. client dependencies --------------------------------------------------

say "Installing client dependencies"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r "${SCRIPT_DIR}/requirements.txt"
echo "Done."

# --- 3. container image ------------------------------------------------------

say "Pulling the vLLM container ($IMAGE)"
docker pull "$IMAGE"

# --- 4. model weights --------------------------------------------------------

say "Model weights (~21.6 GB)"
echo "Model:    $MODEL"
echo "Revision: ${MODEL_REVISION:-main}"
echo

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "NOTE: HF_TOKEN is not set. Public downloads usually work, but export a"
  echo "      token if you hit rate limits:  export HF_TOKEN=hf_..."
  echo
fi

# Safe to re-run. We look in the local cache first with local_files_only=True,
# which needs no network at all: if the weights are already there, this returns
# immediately and nothing is re-downloaded. Only a cache miss reaches the Hub.
python3 - "$MODEL" "${MODEL_REVISION:-}" <<'PY'
import sys, os
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError

model = sys.argv[1]
revision = sys.argv[2] or None
kwargs = {"repo_id": model}
if revision:
    kwargs["revision"] = revision

def report(path, cached):
    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(path, followlinks=True)
               for f in fs if os.path.exists(os.path.join(r, f)))
    print(f"\n  {'Already cached' if cached else 'Downloaded'}: {size/2**30:.1f} GB")
    print(f"  Path:     {path}")
    print(f"  Snapshot: {os.path.basename(path)}")

try:
    path = snapshot_download(**kwargs, local_files_only=True)
    report(path, cached=True)
    print("\n  Nothing to download. Re-running this script is free.")
except LocalEntryNotFoundError:
    print("  Not in the local cache — downloading. This takes 20-30 minutes")
    print("  on a first run, and is resumable if interrupted.\n")
    path = snapshot_download(**kwargs)
    report(path, cached=False)

print("\n  Record that snapshot id alongside any benchmark numbers you publish —")
print("  it is what makes them reproducible.")
PY

# --- done --------------------------------------------------------------------

cat <<EOF

$(printf '\033[1m== Setup complete\033[0m')

Next:

  ./serve.sh                  start the endpoint (leave running, ~5 min cold)
  jupyter lab demo.ipynb      open the walkthrough

Verify the endpoint once it is up:

  curl -s http://localhost:8000/v1/models | python3 -m json.tool

EOF
