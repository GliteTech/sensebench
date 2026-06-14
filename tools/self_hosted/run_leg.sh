#!/usr/bin/env bash
# Run one manifest job on this box: download the checkpoint, serve it with vLLM
# in a tmux session, then run + verify `sensebench run` for every manifest prompt.
#
# Runs ON the box (launch it inside tmux so it survives SSH disconnects):
#   bash run_leg.sh --job <job_id> --gpu <preset> --hourly-rate-usd <rate> \
#     --instance-id <id> --github-handle <handle> [--runner-name <name>] \
#     [--runner-contact <contact>] [--date YYYYMMDD] [--limit N] [--port 8000]
#
# Status is written to /workspace/sensebench/status/<job_id>: DONE or FAILED:<reason>.
set -euo pipefail

WORKDIR=/workspace/sensebench
MANIFEST=$WORKDIR/repo/tools/self_hosted/manifest.json
STATUS_DIR=$WORKDIR/status
LOGS_DIR=$WORKDIR/logs
RUNS_DIR=$WORKDIR/runs
VLLM_SESSION=vllm
HEALTH_POLL_SECONDS=15
HEALTH_TIMEOUT_SECONDS=1200
GPU_DRAIN_TIMEOUT_SECONDS=120
GPU_DRAIN_MAX_MIB=2048
GPU_DRAIN_POLL_SECONDS=5

JOB=""
GPU_PRESET=""
HOURLY_RATE_USD=""
INSTANCE_ID=""
GITHUB_HANDLE=""
RUNNER_NAME=""
RUNNER_CONTACT=""
RUN_DATE="$(date -u +%Y%m%d)"
LIMIT=""
PORT=8000
VOTES=""
TEMPERATURE_OVERRIDE=""
SHUFFLE_SENSES=0
RUN_SUFFIX=""

while [ $# -gt 0 ]; do
  case "$1" in
    --job) JOB="$2"; shift 2 ;;
    --gpu) GPU_PRESET="$2"; shift 2 ;;
    --hourly-rate-usd) HOURLY_RATE_USD="$2"; shift 2 ;;
    --instance-id) INSTANCE_ID="$2"; shift 2 ;;
    --github-handle) GITHUB_HANDLE="$2"; shift 2 ;;
    --runner-name) RUNNER_NAME="$2"; shift 2 ;;
    --runner-contact) RUNNER_CONTACT="$2"; shift 2 ;;
    --date) RUN_DATE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --votes) VOTES="$2"; shift 2 ;;
    --temperature) TEMPERATURE_OVERRIDE="$2"; shift 2 ;;
    --shuffle-senses) SHUFFLE_SENSES=1; shift 1 ;;
    --run-suffix) RUN_SUFFIX="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for required in JOB GPU_PRESET HOURLY_RATE_USD INSTANCE_ID GITHUB_HANDLE; do
  if [ -z "${!required}" ]; then
    echo "missing required argument: --$(echo "$required" | tr '[:upper:]_' '[:lower:]-')" >&2
    exit 2
  fi
done

# shellcheck disable=SC1091
source "$WORKDIR/env.sh"
mkdir -p "$STATUS_DIR" "$LOGS_DIR" "$RUNS_DIR"
STATUS_FILE=$STATUS_DIR/$JOB
SERVER_LOG=$LOGS_DIR/server-$JOB.log

# vLLM engine workers outlive their tmux session and rename themselves to
# VLLM::EngineCore, so kill every vLLM process pattern and wait until the GPU
# memory is actually released before the next server starts.
stop_vllm() {
  tmux kill-session -t "$VLLM_SESSION" 2>/dev/null || true
  pkill -9 -f "vllm serve" 2>/dev/null || true
  pkill -9 -f "VLLM::" 2>/dev/null || true
  pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
  for _ in $(seq 1 24); do
    USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | sort -rn | head -1 || echo 0)
    if [ "${USED_MIB:-0}" -lt "$GPU_DRAIN_MAX_MIB" ] \
      && ! curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "warning: GPU memory still in use after stop_vllm" >&2
}

fail() {
  echo "FAILED:$1" > "$STATUS_FILE"
  echo "FAILED:$1" >&2
  stop_vllm
  exit 1
}

job_field() {
  jq -r --arg job "$JOB" ".jobs[] | select(.job_id == \$job) | $1" "$MANIFEST"
}

if [ "$(jq -r --arg job "$JOB" \
  '[.jobs[] | select(.job_id == $job)] | length' "$MANIFEST")" != "1" ]; then
  echo "job $JOB not found in $MANIFEST" >&2
  exit 2
fi
if [ "$(jq -r --arg job "$JOB" --arg gpu "$GPU_PRESET" \
  '[.jobs[] | select(.job_id == $job) | .gpus[] | select(. == $gpu)] | length' \
  "$MANIFEST")" != "1" ]; then
  echo "job $JOB is not configured for gpu preset $GPU_PRESET in $MANIFEST" >&2
  exit 2
fi

MODEL=$(job_field '.model')
CHECKPOINT=$(job_field '.served_checkpoint // .model')
HF_REVISION=$(job_field '.hf_revision // empty')
QUANTIZATION=$(job_field '.quantization')
# Read serve args one-per-line into an array so values containing spaces or
# JSON (e.g. --default-chat-template-kwargs '{"enable_thinking": false}')
# survive intact instead of being word-split.
mapfile -t SERVE_ARGS < <(job_field '.serve_args[]')
VENDOR=$(job_field '.vendor')
LICENSE=$(job_field '.license')
MODEL_URL=$(job_field '.model_url')
NOTES=$(job_field '.notes // empty')

CONTAINER_IMAGE=$(job_field '.image_override // empty')
if [ -z "$CONTAINER_IMAGE" ]; then
  CONTAINER_IMAGE=$(jq -r '.default_image' "$MANIFEST")
fi
CONCURRENCY=$(jq -r --arg gpu "$GPU_PRESET" '.gpu_presets[$gpu].concurrency' "$MANIFEST")
DATASET=$(jq -r '.dataset' "$MANIFEST")
TEMPERATURE=$(jq -r '.sampling.temperature' "$MANIFEST")
[ -n "$TEMPERATURE_OVERRIDE" ] && TEMPERATURE="$TEMPERATURE_OVERRIDE"
WARMUP_CALLS=$(jq -r '.warmup_calls' "$MANIFEST")
PROMPTS=$(jq -r '.prompts[]' "$MANIFEST")

echo "job: $JOB  model: $MODEL  checkpoint: $CHECKPOINT  gpu: $GPU_PRESET" >&2
if [ -n "$NOTES" ]; then
  echo "job notes: $NOTES" >&2
fi

echo "downloading $CHECKPOINT${HF_REVISION:+ @ $HF_REVISION}" >&2
if command -v hf >/dev/null 2>&1; then
  hf download "$CHECKPOINT" ${HF_REVISION:+--revision "$HF_REVISION"}
elif command -v huggingface-cli >/dev/null 2>&1; then
  # Older vLLM images ship huggingface-cli instead of the hf entrypoint.
  huggingface-cli download "$CHECKPOINT" ${HF_REVISION:+--revision "$HF_REVISION"}
else
  fail "no_hf_downloader"
fi

stop_vllm
echo "starting vLLM server (log: $SERVER_LOG)" >&2
# Build the launch command in a script file with printf %q so every argument
# (including JSON values) is faithfully quoted; tmux then just runs the file,
# avoiding a second round of word-splitting on the serve args.
SERVE_SCRIPT=$LOGS_DIR/serve-$JOB.cmd.sh
{
  printf '#!/usr/bin/env bash\n'
  # Serve the (possibly third-party quantized) checkpoint under the canonical
  # model id so the OpenAI model id, the runner's --model, and the recorded
  # display name all match the model identity rather than the checkpoint repo.
  printf 'exec vllm serve %q --port %q --served-model-name %q' \
    "$CHECKPOINT" "$PORT" "$MODEL"
  printf ' --max-model-len 8192 --gpu-memory-utilization 0.90'
  for serve_arg in "${SERVE_ARGS[@]}"; do
    printf ' %q' "$serve_arg"
  done
  if [ -n "$HF_REVISION" ]; then
    printf ' --revision %q' "$HF_REVISION"
  fi
  printf '\n'
} > "$SERVE_SCRIPT"
SERVE_COMMAND=$(tail -1 "$SERVE_SCRIPT")
tmux new-session -d -s "$VLLM_SESSION" "bash $SERVE_SCRIPT > $SERVER_LOG 2>&1"

# Readiness = the server lists THIS checkpoint; a bare /health check could be
# answered by a stale server from a previous leg.
echo "waiting for $MODEL at http://localhost:$PORT/v1/models" >&2
HEALTH_DEADLINE=$(( $(date +%s) + HEALTH_TIMEOUT_SECONDS ))
until curl -sf "http://localhost:$PORT/v1/models" 2>/dev/null | grep -qF "\"$MODEL\""; do
  if [ "$(date +%s)" -ge "$HEALTH_DEADLINE" ]; then
    echo "vLLM did not serve $MODEL within ${HEALTH_TIMEOUT_SECONDS}s; last server log:" >&2
    tail -100 "$SERVER_LOG" >&2 || true
    fail "server_health_timeout"
  fi
  sleep "$HEALTH_POLL_SECONDS"
done
echo "server healthy and serving $MODEL" >&2

RUNNER_ARGS=()
if [ -n "$RUNNER_NAME" ]; then
  RUNNER_ARGS+=(--runner-name "$RUNNER_NAME")
fi
if [ -n "$RUNNER_CONTACT" ]; then
  RUNNER_ARGS+=(--runner-contact "$RUNNER_CONTACT")
fi

# Optional self-consistency knobs (votes / per-vote sense shuffle).
EXTRA_RUN_ARGS=()
if [ -n "$VOTES" ]; then
  EXTRA_RUN_ARGS+=(--votes "$VOTES")
fi
if [ "$SHUFFLE_SENSES" = "1" ]; then
  EXTRA_RUN_ARGS+=(--shuffle-senses)
fi

for PROMPT in $PROMPTS; do
  MAX_TOKENS=$(jq -r --arg prompt "$PROMPT" '.prompt_max_tokens[$prompt]' "$MANIFEST")
  RUN_ID="vllm-$JOB-$GPU_PRESET-$PROMPT-$DATASET-$RUN_DATE${RUN_SUFFIX:+-$RUN_SUFFIX}"
  HF_REVISION_ARGS=()
  if [ -n "$HF_REVISION" ]; then
    HF_REVISION_ARGS=(--hf-revision "$HF_REVISION")
  fi
  LIMIT_ARGS=()
  if [ -n "$LIMIT" ]; then
    # Smoke runs get a distinct id so they never collide with full runs.
    RUN_ID="$RUN_ID-smoke"
    LIMIT_ARGS=(--limit "$LIMIT")
  fi
  if [ -f "$RUNS_DIR/$RUN_ID/run.json" ]; then
    echo "skipping $RUN_ID: run.json already exists" >&2
    continue
  fi
  echo "running $RUN_ID" >&2
  if ! sensebench run \
    --prompt "$PROMPT" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --hosting-kind self_hosted \
    --endpoint-base-url "http://localhost:$PORT/v1" \
    --quantization "$QUANTIZATION" \
    "${HF_REVISION_ARGS[@]}" \
    --container-image "$CONTAINER_IMAGE" \
    --serve-command "$SERVE_COMMAND" \
    --source-kind open_source \
    --vendor "$VENDOR" \
    --license "$LICENSE" \
    --model-url "$MODEL_URL" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --concurrency "$CONCURRENCY" \
    --warmup-calls "$WARMUP_CALLS" \
    --hourly-rate-usd "$HOURLY_RATE_USD" \
    --provider vast.ai \
    --instance-id "$INSTANCE_ID" \
    --run-id "$RUN_ID" \
    --github-handle "$GITHUB_HANDLE" \
    "${RUNNER_ARGS[@]}" \
    "${EXTRA_RUN_ARGS[@]}" \
    --output-root "$RUNS_DIR" \
    "${LIMIT_ARGS[@]}"; then
    fail "run:$PROMPT"
  fi
  if [ -n "$LIMIT" ]; then
    # Partial runs never cover the full dataset, so full verification cannot pass.
    echo "smoke run: skipping verification of $RUN_ID" >&2
    continue
  fi
  if ! sensebench verify "$RUNS_DIR/$RUN_ID" --dataset "$DATASET" --prompt "$PROMPT"; then
    fail "verify:$PROMPT"
  fi
done

stop_vllm

# Evict this leg's checkpoint from the HF cache so a batch of large models does
# not exhaust the disk (each leg keeps only its own weights resident).
if [ "${EVICT_MODEL:-0}" = "1" ]; then
  CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${CHECKPOINT//\//--}"
  if [ -d "$CACHE_DIR" ]; then
    echo "evicting $CHECKPOINT from HF cache ($CACHE_DIR)" >&2
    rm -rf "$CACHE_DIR"
  fi
fi

echo "waiting for GPU memory to drain" >&2
DRAIN_DEADLINE=$(( $(date +%s) + GPU_DRAIN_TIMEOUT_SECONDS ))
while [ "$(date +%s)" -lt "$DRAIN_DEADLINE" ]; do
  USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | sort -rn | head -1 || echo 0)
  if [ "${USED_MIB:-0}" -lt "$GPU_DRAIN_MAX_MIB" ]; then
    break
  fi
  sleep "$GPU_DRAIN_POLL_SECONDS"
done

echo "DONE" > "$STATUS_FILE"
echo "DONE: job $JOB on $GPU_PRESET (runs in $RUNS_DIR)" >&2
