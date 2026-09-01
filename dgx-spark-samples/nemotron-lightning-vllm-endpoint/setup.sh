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
IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:latest}"
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

say "Pre-downloading model weights (~21.6 GB)"
echo "Model: $MODEL"
echo "Doing this now means ./serve.sh starts in ~5 min instead of 20+."
echo

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "NOTE: HF_TOKEN is not set. Public downloads usually work, but export a"
  echo "      token if you hit rate limits:  export HF_TOKEN=hf_..."
  echo
fi

python3 - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
model = sys.argv[1]
path = snapshot_download(repo_id=model, resume_download=True)
print(f"\nWeights cached at: {path}")
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
