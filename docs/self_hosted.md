# Self-hosted runs: vLLM on vast.ai

This runbook covers the batch tooling under `tools/self_hosted/` used to benchmark open-weight
models on rented vast.ai GPUs, and how anyone can submit a self-hosted run without vast.ai.

## How a self-hosted run works

`sensebench run --hosting-kind self_hosted` points the runner at an OpenAI-compatible endpoint
(vLLM). When the endpoint is on localhost, machine details (GPU model and count, CPU, RAM) are
collected automatically and the vLLM engine version is read from the server's `/version` endpoint.
LiteLLM's `hosted_vllm` route requires a dummy API key, so export `HOSTED_VLLM_API_KEY=dummy`
(the host env script does this).

**Machine-time metric:** passing `--hourly-rate-usd` records the machine's hourly rate, and run
cost is estimated from the wall-clock time the benchmark occupied the box. `--warmup-calls 8`
issues unrecorded completions before the timed loop so engine warmup (compilation, caches) does
not pollute the timing.

## Tooling layout

| File | Runs | Purpose |
| --- | --- | --- |
| `tools/self_hosted/manifest.json` | n/a | Dataset, prompts, sampling, GPU presets, ordered job list |
| `tools/self_hosted/provision.py` | locally | Rent + verify a vast.ai box, write `work/<gpu>/instance.json` |
| `tools/self_hosted/setup_host.sh` | on the box | Bootstrap repo, venv, dataset, `env.sh` (idempotent) |
| `tools/self_hosted/run_leg.sh` | on the box | Serve one job with vLLM, run + verify each prompt |
| `tools/self_hosted/fetch_runs.sh` | locally | scp run artifacts back |
| `tools/self_hosted/destroy.py` | locally | Destroy the instance and confirm it is gone |

## Prerequisites

* vast.ai account with billing and the CLI key set: `uvx vastai@0.5.0 set api-key <key>`
* SSH keypair at `~/.ssh/id_ed25519`, registered with vast.ai (or attached per instance, below)
* Hugging Face token with access to the gated repos in the manifest (`meta-llama/*`, `google/*`)
  saved to a local file, e.g. `~/.secrets/hf_token`
* This repository checked out with `uv sync` done, and `gh` authenticated for the submission PR

## Per-GPU batch sequence

Run one batch per GPU preset (`a100`, `h100`, `h200`). All local commands from the repo root.

### 1. Provision

```bash
uv run python tools/self_hosted/provision.py --gpu h100
```

This searches offers for the preset, filters by the preset's price cap, rents the cheapest offer,
polls until the instance is running, SSH-verifies `nvidia-smi`, writes `work/h100/instance.json`,
and prints the SSH command. Failed candidates are destroyed and the next offer is tried (up to
`--max-attempts`, default 3). If SSH authentication fails, attach your key and re-run:

```bash
uvx vastai@0.5.0 attach ssh <instance-id> "$(cat ~/.ssh/id_ed25519.pub)"
```

Pull the coordinates for the later steps:

```bash
HOST=$(python3 -c "import json; print(json.load(open('work/h100/instance.json'))['ssh_host'])")
PORT=$(python3 -c "import json; print(json.load(open('work/h100/instance.json'))['ssh_port'])")
RATE=$(python3 -c "import json; print(json.load(open('work/h100/instance.json'))['hourly_rate_usd'])")
ID=$(python3 -c "import json; print(json.load(open('work/h100/instance.json'))['instance_id'])")
SSH=(ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 -p "$PORT" "root@$HOST")
```

### 2. Bootstrap the box

`setup_host.sh` clones this repository on the box (branch `self-hosted-vllm-support`, override
with `SENSEBENCH_BRANCH`), so only the bootstrap script and the HF token need copying:

```bash
"${SSH[@]}" "mkdir -p /workspace/sensebench"
scp -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 -P "$PORT" \
  tools/self_hosted/setup_host.sh "root@$HOST:/workspace/"
scp -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 -P "$PORT" \
  ~/.secrets/hf_token "root@$HOST:/workspace/sensebench/hf_token"
"${SSH[@]}" "bash /workspace/setup_host.sh"
```

The script installs uv, clones the repo, installs sensebench into a venv, writes
`/workspace/sensebench/env.sh` (PATH, `HF_HOME`, dummy API keys, `HF_TOKEN`), installs the HF
token, prefetches the dataset and the NLTK WordNet corpus, prints the box's UTC time
(sanity-check it against your clock), and ends with `DONE`. It is idempotent — re-run it freely.

### 3. Run the legs

The jobs for a preset are the manifest `jobs` whose `gpus` list contains the preset, executed in
list order (smallest models first, `mistral-small-3.2-fp8` last). Run legs sequentially — each
leg owns the GPU. Launch each leg inside tmux so it survives SSH disconnects:

```bash
JOB=granite-4.1-8b-fp8
"${SSH[@]}" "tmux new-session -d -s leg-$JOB \
  \"bash /workspace/sensebench/repo/tools/self_hosted/run_leg.sh \
    --job $JOB --gpu h100 --hourly-rate-usd $RATE --instance-id $ID \
    --github-handle <your-handle> \
    >> /workspace/sensebench/logs/leg-$JOB.log 2>&1\""
```

Monitor:

```bash
"${SSH[@]}" "cat /workspace/sensebench/status/$JOB"        # DONE or FAILED:<reason>
"${SSH[@]}" "tail -f /workspace/sensebench/logs/leg-$JOB.log"
"${SSH[@]}" "tail -f /workspace/sensebench/logs/server-$JOB.log"
```

Each leg downloads the checkpoint, serves it in the `vllm` tmux session, waits for `/health`,
then runs every manifest prompt with run id `vllm-<job>-<gpu>-<prompt>-<dataset>-<YYYYMMDD>`
(e.g. `vllm-qwen3.6-27b-fp8-h100-p003-lexen-v0.1.0-20260613`) and verifies it on the box. Runs
that already have a `run.json` are skipped, so re-launching a failed leg resumes where it left
off. Add `--limit N` for a smoke leg: the run id gets a `-smoke` suffix, and verification is
skipped (partial runs can never pass full-dataset verification and are not leaderboard-eligible).
Check the job's `notes` in the manifest before starting it — the Gemma 4 jobs have a fallback
image and the Mistral job may need vLLM 0.9.x.

### 4. Fetch and verify locally

```bash
bash tools/self_hosted/fetch_runs.sh work/h100/instance.json runs
for run_dir in runs/vllm-*-h100-p003-*; do
  uv run sensebench verify "$run_dir" --dataset lexen-v0.1.0 --prompt p003
done
```

Match the `--prompt` to the prompt id embedded in each run id when the manifest lists several.

### 5. Destroy the instance

```bash
uv run python tools/self_hosted/destroy.py work/h100/instance.json
uvx vastai@0.5.0 show instances --raw   # must print an empty list
```

`destroy.py` re-checks that the instance is actually gone and exits non-zero with a loud warning
if it still appears alive — never end a batch with that command failing, the box keeps billing.

## Submitting the runs

For each verified full run, follow the standard submission flow (see the README):

1. Copy the run directory into the repo: `cp -R runs/<run-id> results/<run-id>`
2. Open a pull request titled `submit-<run-id>` (or one PR for the whole batch).

CI re-verifies every submitted run from the raw artifacts; the leaderboard updates on merge.

## Self-hosted runs without vast.ai

Anyone can submit a self-hosted run from any machine with a GPU — vast.ai and the batch tooling
are not required:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --max-model-len 8192
export HOSTED_VLLM_API_KEY=dummy
sensebench run \
  --prompt p003 --model meta-llama/Llama-3.1-8B-Instruct \
  --hosting-kind self_hosted --endpoint-base-url http://localhost:8000/v1 \
  --quantization bf16 --source-kind open_source --vendor Meta --license llama3.1 \
  --model-url https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct \
  --temperature 0 --max-tokens 64 --warmup-calls 8 \
  --hourly-rate-usd <your-rate> --github-handle <your-handle>
```

Because the endpoint is on localhost, machine info is collected automatically. If the runner is a
different machine from the GPU host, run `sensebench machine-info` on the GPU host, save the JSON
to a file, and pass `--machine-info-json <file>` to `sensebench run` — self-hosted submissions
without GPU details fail verification. Record provenance with `--provider`, `--instance-id`, and
`--hourly-rate-usd` where applicable, then verify and submit as above.
