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

# Is the port already taken by something we did not start? Docker's own error
# for this ("address already in use") does not say what is holding it, which
# leaves you guessing. Check first and say something useful.
port_holder() {
  docker ps --format '{{.Names}}\t{{.Ports}}' 2>/dev/null \
    | awk -v p=":${PORT}->" '$0 ~ p {print "docker container: " $1}'
}

if command -v ss >/dev/null 2>&1 && ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  HOLDER="$(port_holder)"
  cat >&2 <<EOF

ERROR: port ${PORT} is already in use.

$( [[ -n "$HOLDER" ]] && echo "  Held by ${HOLDER}" || echo "  Not a container this script manages. Identify it with:
    docker ps --format '{{.Names}}\t{{.Ports}}'
    sudo ss -tlnp | grep :${PORT}" )

If it is already serving this model, you do not need to start another one:

    curl -s http://localhost:${PORT}/v1/models | python3 -m json.tool

Otherwise either stop it, or run on a different port:

    PORT=8001 ./serve.sh

If you change the port, update BASE_URL in demo.ipynb to match.
EOF
  exit 1
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

# --entrypoint vllm is deliberate, and the failure it prevents is worth knowing.
#
# This image ships ENTRYPOINT ["vllm", "serve"]. Passing "vllm serve <model>" as
# the container command therefore runs:
#
#     vllm serve vllm serve <model> --flags
#            ^^^^ entrypoint    ^^^^ our command
#
# argparse consumes the literal string "vllm" as the model_tag positional and
# then reports the real model as junk:
#
#     vllm: error: unrecognized arguments: serve nvidia/NVIDIA-Nemotron-...
#
# which reads like a bad model name and is not. Overriding the entrypoint makes
# the final command exactly "vllm serve <model> --flags" regardless of what the
# image's default entrypoint is in this or a future tag. Confirm yours with:
#
#     docker inspect "$IMAGE" --format '{{json .Config.Entrypoint}}'

# shellcheck disable=SC2086
exec docker run $DETACH_FLAG --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  -p "${PORT}:8000" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  --entrypoint vllm \
  "$IMAGE" \
  serve "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --host 0.0.0.0 \
    --port 8000
