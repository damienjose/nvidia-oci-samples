#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Serve Nemotron 3.5 Lightning 30B-A3B (NVFP4) on DGX Spark via vLLM.
#
# Cold start is ~5 minutes. Leave this running in its own terminal; the
# notebook talks to it over HTTP.
#
# Usage:
#   ./serve.sh              # foreground
#   ./serve.sh --detach     # background container, logs via: docker logs -f vllm-nemotron

set -euo pipefail

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-nemotron}"
# vLLM publishes arm64 builds under arch-suffixed tags (`*-aarch64`), but this
# versioned tag is a multi-arch manifest list containing linux/arm64, so Docker
# resolves the right platform on GB10 automatically. Pinned and stable rather
# than a nightly, which can be pruned from the registry without warning.
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.19.0-cu130-ubuntu2404}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

DETACH_FLAG=""
if [[ "${1:-}" == "--detach" ]]; then
  DETACH_FLAG="-d"
fi

# --- preflight ---------------------------------------------------------------

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: cannot talk to the Docker daemon.

  sudo gpasswd -a $(whoami) docker

Then log out and back in. If you are in a JupyterLab terminal, restart the
JupyterLab *server* -- a kernel restart does not pick up new group membership.
Use $(whoami), not $USER: in a Jupyter-spawned shell $USER can be 'root'.
EOF
  exit 1
fi

# IMPORTANT: mount the whole HF cache, not just the snapshot directory.
# Snapshot dirs are symlink farms into blobs/ -- mounting only the snapshot
# leaves every link dangling and the model fails to load.
mkdir -p "$HF_CACHE"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Removing existing container '$CONTAINER_NAME'..."
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

echo "Model:        $MODEL"
echo "Port:         $PORT"
echo "Context:      $MAX_MODEL_LEN"
echo "HF cache:     $HF_CACHE"
echo "Image:        $IMAGE"
echo
echo "Cold start takes ~5 minutes. Endpoint will be at http://localhost:$PORT/v1"
echo

# --- serve -------------------------------------------------------------------
#
# --reasoning-parser nemotron_v3
#     returns the reasoning trace as a structured field instead of inline text
# --enable-auto-tool-choice --tool-call-parser qwen3_coder
#     returns tool calls as structured tool_calls instead of text to scrape
#
# The fp8 KV cache is pinned by the checkpoint's own quantisation config, so
# vLLM selects fp8_e4m3 regardless of --kv-cache-dtype. Don't bother passing it.

# shellcheck disable=SC2086
exec docker run $DETACH_FLAG --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  -p "${PORT}:8000" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --host 0.0.0.0 \
    --port 8000
