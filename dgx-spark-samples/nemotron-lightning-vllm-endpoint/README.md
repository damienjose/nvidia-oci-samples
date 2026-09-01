<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# Nemotron 3.5 Lightning vLLM Endpoint on DGX Spark

This sample shows how developers can serve [NVIDIA Nemotron 3.5 Lightning 30B-A3B (NVFP4)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) on a single NVIDIA DGX Spark as a private, OpenAI-compatible endpoint, then drive it from the standard OpenAI Python SDK.

It is intentionally small and external-safe:

- No API keys, credentials, customer data, or internal NVIDIA content.
- No OCI account required — this sample runs entirely on local DGX Spark hardware.
- All model and inference components are publicly available open-source or open-weights releases.
- Agent tool implementations are deterministic local functions with synthetic data. Nothing in the demo reaches the network at runtime.

The point of the sample is a single line:

```python
client = OpenAI(
    base_url="http://localhost:8000/v1",   # <- the only line that changes
    api_key="not-needed",                  #    no key: it is your hardware
)
```

Any tool built on the OpenAI API — LangChain, LlamaIndex, Continue, or your own services — points at a model running on hardware you own by changing configuration, not code.

## What the Sample Shows

1. Brings up an open-weights 30B model on one DGX Spark with a single command.
2. Confirms the endpoint is healthy and reports the served model and context length.
3. Drives it from the OpenAI Python SDK by changing `base_url`.
4. Streams a response while reporting live time-to-first-token and decode rate.
5. Runs a multi-step tool-calling agent that chains two unrelated tools without being told to.
6. Reports remaining memory headroom on the machine.
7. Points at the fine-tuning and quantization workflows that use the same box.

## Why This Goes Beyond A Chat Demo

- **Structured tool calling, not text scraping.** The model is served with `--enable-auto-tool-choice --tool-call-parser qwen3_coder`, so tool calls arrive as structured `tool_calls`.
- **Structured reasoning traces.** `--reasoning-parser nemotron_v3` returns the reasoning trace as a field rather than inline prose.
- **Multi-step planning.** Given a flight-search tool and a currency-conversion tool and a task needing both, the model works out that it must search first and convert second.
- **Honest performance characterisation.** The notebook reports measured TTFT and decode rate rather than asserting them, and states plainly where a single DGX Spark is the wrong tool.

## Everything Here Is Open

| Component | License |
| --- | --- |
| [vLLM](https://github.com/vllm-project/vllm) | Apache 2.0 |
| [Nemotron 3.5 Lightning 30B-A3B NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) | OpenMDW-1.1 |
| [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) | CC-BY-4.0 |
| This sample | Apache 2.0 |

NVIDIA released the **pre-training and post-training data** alongside the Nemotron weights, not only the weights. For teams evaluating whether to build on a model, that is a materially different position from a weights-only release.

No NVIDIA licence, NGC subscription, or support contract is required to run this sample.

## Requirements

- NVIDIA DGX Spark (GB10, `sm_121`, 128 GB unified memory) running DGX OS.
- Docker available to your user.
- Python 3.10 or newer.
- Approximately 25 GB of free disk for model weights.
- A Hugging Face token (`HF_TOKEN`) only if you hit download rate limits.

**No OCI account, cloud GPU shape, or network egress is required.**

## Connecting to Your DGX Spark

You can work directly on the device with a keyboard and monitor, but most people drive it from a laptop. There are two ways to do that, and the first is considerably easier.

### Option A: NVIDIA Sync (recommended)

[NVIDIA Sync](https://docs.nvidia.com/dgx/dgx-spark/nvidia-sync.html) is a free utility for macOS, Windows, and Ubuntu that manages SSH connections, port forwarding, and tunnels for you. It is the fastest way to get from a laptop to a working JupyterLab on the device.

1. **Install it** from [build.nvidia.com/spark/connect-to-your-spark/sync](https://build.nvidia.com/spark/connect-to-your-spark/sync).
2. **Add your device.** On the same network, Sync discovers DGX Spark systems automatically over mDNS. Otherwise add it by hostname or IP, using the account credentials you created during first boot. Sync configures SSH key-based authentication for you, so subsequent connections need no password. See [Direct Connections](https://docs.nvidia.com/sync/latest/direct-connections.html).
3. **Connect**, then launch **DGX Dashboard**, **Terminal**, or **VS Code** from the app list. Each opens against the device with tunnels already in place.

**If your laptop is not on the same network as the device**, Sync supports [Tailscale connections](https://docs.nvidia.com/sync/latest/tailscale.html) — useful for a Spark sitting on an office network while you work remotely.

Full walkthrough: [NVIDIA Sync Getting Started](https://docs.nvidia.com/sync/latest/getting-started.html).

### Running this notebook through the DGX Dashboard

The [DGX Dashboard](https://docs.nvidia.com/dgx/dgx-spark/dgx-dashboard.html) has an integrated JupyterLab, and NVIDIA Sync tunnels it automatically. Connect in Sync, click **DGX Dashboard** (it opens at `http://localhost:11000`), set the working directory to this sample, and start JupyterLab. Then open `demo.ipynb`.

> **Important if you take this route.** The Dashboard's JupyterLab creates a **fresh virtual environment per working directory** and installs its own recommended packages. That environment is *not* the system Python that `setup.sh` installed into, so the sample's client dependencies will be missing. Install them once from a cell inside the notebook, or from a JupyterLab terminal:
>
> ```
> %pip install -r requirements.txt
> ```
>
> Changing the working directory creates another new environment, so you would need to repeat this.

### Option B: plain SSH

```bash
ssh <username>@<device-name>.local
```

**The `.local` suffix is required** when connecting by device name — DGX Spark advertises itself over mDNS as `spark-xxxx.local`, and `ssh user@spark-xxxx` will fail to resolve. Use the IP address instead if mDNS is blocked on your network.

To reach JupyterLab or the Dashboard this way you must tunnel the ports yourself:

```bash
# DGX Dashboard
ssh -L 11000:localhost:11000 <username>@<device-name>.local

# JupyterLab started manually by this sample
ssh -L 8888:localhost:8888 <username>@<device-name>.local
```

The Dashboard's integrated JupyterLab assigns a **per-user port**, listed in `/opt/nvidia/dgx-dashboard-service/jupyterlab_ports.yaml` on the device — check there before tunnelling. NVIDIA Sync handles all of this for you, which is why Option A is recommended.

## Quickstart

Verify local prerequisites:

```bash
nvidia-smi
docker info >/dev/null && echo "docker OK"
python3 --version
```

If `docker info` fails, add yourself to the docker group and start a new login shell:

```bash
sudo gpasswd -a $(whoami) docker
```

Clone the sample:

```bash
git clone -b feature/nemotron-lightning-vllm-endpoint \
  https://github.com/damienjose/nvidia-oci-samples.git
```

> **Why a branch, and why that fork?** This sample is under review for
> [NVIDIA/nvidia-oci-samples](https://github.com/NVIDIA/nvidia-oci-samples). Until it merges, the
> code lives on the feature branch above. Once merged, the clone command becomes simply
> `git clone https://github.com/NVIDIA/nvidia-oci-samples.git` with no branch flag — check the
> upstream repository first, and use the fork only if the `dgx-spark-samples/` directory isn't
> there yet.

Run the sample:

```bash
cd nvidia-oci-samples/dgx-spark-samples/nemotron-lightning-vllm-endpoint
./setup.sh          # pull container, install client deps, fetch weights
./serve.sh          # start vLLM (leave running; ~5 min cold start)
```

`setup.sh` is **safe to re-run**. It checks the local Hugging Face cache first with no network call, so weights you already have are never re-downloaded — a second run takes seconds. Expect 20–30 minutes the first time, and the download resumes if interrupted.

In a second terminal:

```bash
jupyter lab demo.ipynb
```

Or open `demo.ipynb` from the DGX Dashboard's integrated JupyterLab — see [Connecting to Your DGX Spark](#connecting-to-your-dgx-spark), and note the virtual-environment caveat there.

Verify the endpoint from the shell at any time:

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

## Files

| File | Purpose |
| --- | --- |
| `setup.sh` | Machine check, client dependencies, model pre-download |
| `serve.sh` | Launch vLLM with the flags this model requires |
| `demo.ipynb` | Walkthrough: environment, endpoint, SDK switch, streaming, agent loop, headroom |
| `demo_tools.py` | Tool implementations and JSON schemas for the agent section |
| `when2call.py` | Benchmark scoring — dataset loading, metrics, Wilson intervals |
| `run_benchmark.py` | Run When2Call against one or more endpoints, with 429 backoff |
| `make_chart.py` | Chart `results/summary.json` |
| `endpoints.json` | Endpoint config. Keys come from the environment, never committed |
| `results/` | Benchmark output — `summary.json` and generated charts |
| `requirements.txt` | Client-side Python dependencies only |

## Serving Configuration

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --max-model-len 65536 \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 --port 8000
```

| Flag | Why |
| --- | --- |
| `--reasoning-parser nemotron_v3` | Returns the reasoning trace as a structured field |
| `--enable-auto-tool-choice` | Enables model-initiated tool calls |
| `--tool-call-parser qwen3_coder` | Returns tool calls as structured `tool_calls` |
| `--max-model-len 65536` | 64k context; weights use ~17.9 GiB, leaving ~84 GiB for KV cache |

The fp8 KV cache is pinned by the checkpoint's own quantisation config, so vLLM selects `fp8_e4m3` whether or not `--kv-cache-dtype` is passed.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL` | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | Model to serve |
| `MODEL_REVISION` | unset (`main`) | Pin a specific Hugging Face revision. Set this if you need byte-identical weights across machines |
| `PORT` | `8000` | Host port for the endpoint |
| `MAX_MODEL_LEN` | `65536` | Context length |
| `VLLM_IMAGE` | `vllm/vllm-openai:v0.19.0-cu130-ubuntu2404` | Container image. Multi-arch, includes `linux/arm64` — see Known Issues |
| `CONTAINER_NAME` | `vllm-nemotron` | Docker container name |
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face cache location |
| `HF_TOKEN` | unset | Only needed if you hit download rate limits |

## Benchmarking with When2Call

Serving a model is easy to demo and hard to trust. This sample includes a reproducible
evaluation so the claim is checkable on your own hardware.

[`nvidia/When2Call`](https://huggingface.co/datasets/nvidia/When2Call) (CC-BY-4.0) asks a
harder question than most function-calling benchmarks: not "was the call correct" but
**"should it have called anything at all"**. Three labels — call the tool, ask for the
missing argument, or decline because no tool fits.

Scoring is deterministic. We observe what the server did; there is no LLM judge, so the
numbers reproduce.

### Run it

```bash
# quick local slice — about a minute
./run_benchmark.py --only spark --n 4

# the full local run — roughly 9 minutes on one GB10
./run_benchmark.py --only spark --n 40

# all five endpoints (needs NVIDIA_API_KEY for the hosted ones)
export NVIDIA_API_KEY=nvapi-...
./run_benchmark.py --n 40
./make_chart.py
```

### Metrics

| Metric | What it tells you |
| --- | --- |
| Decision accuracy | Did it call exactly when it should have? |
| Actionable decision accuracy | Correct decision *and* it either called or produced real text. Stops a model scoring well by staying silent. |
| Tool-selection accuracy | When it correctly called, was it the right tool? |
| **Over-call rate** | How often it fired a tool it shouldn't have. **Lower is better** — this is the number that predicts agent misbehaviour in production. |

### Reading results honestly

Two things the harness will not do for you:

- **At n=120 the interval is roughly ±6.5 points.** Models within about 10 points of each
  other are a statistical tie, not a ranking. `when2call.overlaps()` tells you when two
  confidence intervals overlap; the notebook uses it to say so out loud.
- **A model that returns no data is reported as "no data", never as a low score.** A
  rate-limited endpoint is a missing measurement.

### Rate limits are the main failure mode

An earlier sweep issued 960 requests at concurrency 4 and tripped the hosted gateway:
four of eight models returned HTTP 429 on *every* call, including control requests with
no tools attached, and produced nothing. `run_benchmark.py` retries 429 and 5xx with
exponential backoff and full jitter, and defaults remote endpoints to low concurrency.
If a hosted model still returns no data, lower it further:

```bash
./run_benchmark.py --concurrency 1 --n 40
```

### The local-vs-hosted control

`endpoints.json` runs Nemotron 3.5 Lightning in **two places** — on the Spark and on the
Inference Hub. That pair is the control for the whole comparison: if the same model
scored differently in the two locations, the cross-model numbers would mean nothing.

## Measured On One DGX Spark

| Metric | Value |
| --- | --- |
| Weights on disk | 21.6 GB across 69 files |
| Weights resident | 17.86 GiB |
| KV cache available | 84.78 GiB (~23.4M tokens) |
| Cold start | ~5 minutes |
| TTFT, short prompt | 67.1 ms |
| TTFT, ~8k prompt | 87.2 ms |
| Decode, batch 1 | 76.2 tok/s |

Throughput is flat between a short prompt and an 8k prompt: prefill is cheap relative to memory-bandwidth-bound decode on GB10.

## Demo Flow

For a 5 to 10 minute presentation:

1. Show the clone command and invite the audience to run it on their own DGX Spark.
2. Run the environment cell to confirm GB10, 128 GB unified memory, `sm_121`.
3. Confirm the endpoint is healthy.
4. Change `base_url` and run the same OpenAI SDK code against local hardware.
5. Stream a response with live TTFT and decode rate visible.
6. Take a prompt from the audience and run it.
7. Run the agent loop and point out that nobody told it to chain the two tools.
8. Show remaining memory headroom.
9. Close on the fine-tuning and quantization workflows that use the same machine.

## Scope And Limitations

Stated plainly, because sizing this box correctly matters more than overselling it:

- **A single GB10 is a development and prototyping endpoint, not a shared production one.** With 273 GB/s of memory bandwidth, concurrent requests contend and largely serialise. One developer or one CI job is an excellent fit; dozens of simultaneous users are not.
- **GB10 has no native FP4 compute.** vLLM logs `Your GPU does not have native support for FP4 computation` and falls back to the Marlin kernel. NVFP4 here is a *memory* optimisation — weights stored 4-bit and decompressed to compute. A large win on a bandwidth-bound part, but it is not FP4 tensor-core acceleration.
- **Do not select this hardware for throughput.** Its defensible advantages are capacity, data residency, zero marginal cost per token, and deterministic capacity with no rate limits.

## Known Issues

### Picking a vLLM container image for aarch64

This costs people an afternoon, so it is worth stating plainly.

- **`nvcr.io/nvidia/vllm:latest` does not exist.** Pulling it fails with `manifest unknown`.
- **vLLM's `latest` tag is not multi-arch.** Its arm64 builds are published under separate
  arch-suffixed tags — `latest-aarch64`, `nightly-aarch64`, `cu130-nightly-aarch64` — which
  are single-architecture manifests.
- **Versioned tags are multi-arch.** `vllm/vllm-openai:v0.19.0-cu130-ubuntu2404` is a manifest
  list containing both `linux/arm64` and `linux/amd64`, so Docker resolves the correct platform
  on GB10 with no suffix to get wrong. That is why it is the default here: stable, pinned to a
  version, and CUDA 13 to match DGX Spark.

Check any candidate before committing to it:

```bash
docker buildx imagetools inspect vllm/vllm-openai:v0.19.0-cu130-ubuntu2404 \
  | grep -iE "platform|mediatype"
```

A manifest list shows `manifest.list.v2+json` and one `Platform:` line per architecture. A
single-arch image shows only `manifest.v2+json`. Avoid pinning a `nightly-<sha>` tag in anything
you expect others to run — nightly tags are pruned from the registry over time.

- **Hugging Face snapshot directories are symlink farms** into `blobs/`. Mount the whole cache into the container, not just the snapshot directory, or every link dangles. `serve.sh` handles this.
- **JupyterLab inherits group membership at login.** After `sudo gpasswd -a $(whoami) docker` you must restart the JupyterLab *server*; a kernel restart is not sufficient. Use `$(whoami)` rather than `$USER`, which can be `root` in a Jupyter-spawned shell.
- **vLLM 0.26 and later expose the reasoning trace as `reasoning`, not `reasoning_content`.** The notebook handles both.

## Tests

This sample has no automated test suite. Verify manually:

```bash
# 1. Endpoint health
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# 1b. Benchmark harness on a small slice
./run_benchmark.py --only spark --n 2

# 2. Tool implementations
python3 -c "
from demo_tools import search_flights, convert_currency
f = search_flights('SFO','AUS','2026-09-14')
cheapest = min(f['results'], key=lambda x: x['fare_usd'])
print(cheapest)
print(convert_currency(cheapest['fare_usd'],'USD','EUR'))
"

# 3. Notebook, top to bottom
jupyter nbconvert --to notebook --execute demo.ipynb --output /tmp/demo-executed.ipynb
```

All three should complete without error. Step 3 is the meaningful one: it exercises the endpoint, the SDK path, and the full agent loop.

## Going Further

Same machine, same 128 GB:

- [Quantize models to NVFP4 with NVIDIA Model Optimizer](https://build.nvidia.com/spark/nvfp4-quantization) — approximately 3.5x memory reduction versus FP16
- [Fine-tune with PyTorch](https://build.nvidia.com/spark/pytorch-fine-tune) — FSDP and LoRA, up to 70B across two DGX Spark systems
- [Unsloth](https://build.nvidia.com/spark/unsloth) · [LLaMA-Factory](https://build.nvidia.com/spark/llama-factory)
- [Connect two DGX Spark systems](https://build.nvidia.com/spark/connect-two-sparks) for 256 GB of pooled memory

Fine-tuning is memory-bound. Full fine-tuning of Llama 3.2 3B, LoRA on Llama 3.1 8B, and QLoRA on Llama 3.3 70B all run here, and none of them fit on a 32 GB consumer GPU.

## Contributing This Sample

Follow the repository contribution flow:

```bash
git checkout main
git pull origin main
git checkout -b feature/nemotron-lightning-vllm-endpoint
git add dgx-spark-samples README.md
git commit -s -m "Add DGX Spark Nemotron 3.5 Lightning vLLM endpoint sample"
git push origin feature/nemotron-lightning-vllm-endpoint
```

Open a pull request to `NVIDIA/nvidia-oci-samples:main` and tag the maintainers listed in the root `MAINTAINERS.md`.

## References

- [Model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) · [build.nvidia.com](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b)
- [vLLM on DGX Spark playbook](https://build.nvidia.com/spark/vllm)
- [DGX Spark playbooks](https://build.nvidia.com/spark)
- [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) — agentic tool-calling benchmark (CC-BY-4.0)
