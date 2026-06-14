#!/usr/bin/env bash
# Re-run the OpenAI cloud-API benchmark set on the current registered dataset.
#
# Reproduces the OpenAI models previously benchmarked, mapped onto the two
# registered prompts (p001 = detokenized 5+1 JSON, p002 = detokenized minimal):
#   - gpt-5.5            reasoning=medium                 on p001, p002
#   - gpt-5-mini         reasoning sweep on p001          (minimal/low/medium/high); medium on p002
#   - gpt-5-nano         reasoning sweep on p001          (minimal/low/medium/high); medium on p002
#   - gpt-4.1-mini       non-reasoning, temperature 0     on p001, p002
#   - gpt-4.1-nano       non-reasoning, temperature 0     on p001, p002
#   - gpt-4o-mini        non-reasoning, temperature 0     on p001, p002
#
# Requires OPENAI_API_KEY (read from the environment or a .env in the repo root).
# Runs sequentially; a single model failure is logged and does not abort the batch.
#
# Usage:
#   OPENAI_API_KEY=... bash tools/cloud/run_openai.sh [YYYYMMDD]
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATE="${1:-$(date -u +%Y%m%d)}"
DATASET=lexen-v1
GITHUB_HANDLE=vassiliphilippov
RUNNER_NAME="Vassili Philippov"

# OpenAI models are proprietary; vendor + api-provider are OpenAI.
OPENAI_COMMON=(
  --dataset "$DATASET"
  --hosting-kind cloud_api
  --api-provider OpenAI
  --vendor OpenAI
  --source-kind proprietary
  --github-handle "$GITHUB_HANDLE"
  --runner-name "$RUNNER_NAME"
)

run() {
  local label="$1"; shift
  echo "=== $(date -u +%H:%M:%S) RUN $label"
  if uv run sensebench run "$@"; then
    echo "=== OK $label"
  else
    echo "=== FAIL $label"
  fi
}

# Reasoning models (gpt-5.x): pass --reasoning-effort, no temperature, generous token budget.
reasoning_run() {
  local model="$1" effort="$2" prompt="$3" maxtok="$4"
  local rid="${model}-${effort}-reasoning-${prompt}-${DATASET}-${DATE}"
  run "$rid" --model "$model" --prompt "$prompt" \
    --reasoning-effort "$effort" --max-tokens "$maxtok" \
    "${OPENAI_COMMON[@]}" --run-id "$rid"
}

# Non-reasoning models: deterministic (temperature 0), small token budget.
plain_run() {
  local model="$1" prompt="$2" maxtok="$3"
  local rid="${model}-${prompt}-${DATASET}-${DATE}"
  run "$rid" --model "$model" --prompt "$prompt" \
    --temperature 0 --max-tokens "$maxtok" \
    "${OPENAI_COMMON[@]}" --run-id "$rid"
}

# gpt-5.5 — medium reasoning on both prompts.
for prompt in p001 p002; do
  reasoning_run gpt-5.5 medium "$prompt" 8192
done

# gpt-5-mini / gpt-5-nano — reasoning-effort sweep on the richer p001; medium on p002.
for model in gpt-5-mini gpt-5-nano; do
  for effort in minimal low medium high; do
    reasoning_run "$model" "$effort" p001 8192
  done
  reasoning_run "$model" medium p002 8192
done

# gpt-4.1-mini / gpt-4.1-nano / gpt-4o-mini — non-reasoning on both prompts.
for model in gpt-4.1-mini gpt-4.1-nano gpt-4o-mini; do
  plain_run "$model" p001 2048
  plain_run "$model" p002 256
done

echo "OPENAI_RUNS_DONE $(date -u +%H:%M:%S)"
