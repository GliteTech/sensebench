#!/usr/bin/env bash
# Evaluate OpenRouter models on p001 at their lowest and highest reasoning effort.
# Usage: bash tools/cloud/run_openrouter_sweep.sh [YYYYMMDD]
# Optional environment: ONLY (substring filter on run ids), CONCURRENCY, MAX_CONCURRENCY,
# MAX_TOKENS, LIMIT, DRY_RUN=1. LIMIT runs are written under runs/smoke-* and never copied
# into results/. Full runs are verified before copying to results/. Existing runs are preserved.
# Formatting errors are benchmark outcomes: retain retries and failed votes.
# Reject transport errors and token truncation, which can invalidate the comparison.
# max effort truncated GLM-5.3, GLM-5.3-Flash and DeepSeek V4.1 Flash at 32k tokens, so it
# gets 128k and a token suffix in the run id.
# Efforts follow each model's OpenRouter listing: Mistral Large 4 has no low (none instead),
# Nemotron 3 Ultra's lowest is medium, and MiMo-V2.6-Pro exposes no effort levels at all.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_DATE="${1:-$(date -u +%Y%m%d)}"
PROMPT="p001"

# slug|openrouter id|efforts ("-" for none set)|vendor|source kind|license|concurrency
# Mistral Large 4 returned 429s for 458 calls at concurrency 32.
MODELS=(
  "mimo-v2.6-pro|xiaomi/mimo-v2.6-pro|-|Xiaomi|open_source|MIT"
  "qwen3.8-max-0902|qwen/qwen3.8-max-0902|low xhigh|Alibaba|proprietary|"
  "glm-5.3|z-ai/glm-5.3|low max|Z.ai|open_source|other"
  "step-5-preview|stepfun/step-5-preview|low high|StepFun|proprietary|"
  "glm-5.3-flash|z-ai/glm-5.3-flash|low max|Z.ai|open_source|MIT"
  "deepseek-v4.1-flash|deepseek/deepseek-v4.1-flash|low max|DeepSeek|open_source|MIT"
  "mistral-large-4|mistralai/mistral-large-4-0|none high|Mistral AI|proprietary||8"
  "qwen3.8-27b|qwen/qwen3.8-27b|low xhigh|Alibaba|open_source|Apache-2.0"
  "inkling|thinkingmachines/inkling|low max|Thinking Machines|open_source|Apache-2.0"
  "nemotron-3-ultra|nvidia/nemotron-3-ultra-550b-a55b|medium high|NVIDIA|open_source|other"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r slug model_id efforts vendor source_kind license model_concurrency <<< "$entry"
  read -r -a effort_list <<< "$efforts"
  for effort in "${effort_list[@]}"; do
    if [[ "$effort" == - ]]; then
      rid="${slug}-${PROMPT}-lexen-v1-${RUN_DATE}"
      effort_args=()
    else
      rid="${slug}-${effort}-reasoning-${PROMPT}-lexen-v1-${RUN_DATE}"
      effort_args=(--reasoning-effort "$effort")
    fi
    run_max_tokens="${MAX_TOKENS:-32768}"
    run_concurrency="${CONCURRENCY:-${model_concurrency:-32}}"
    if [[ "$effort" == max ]]; then
      run_max_tokens="${MAX_TOKENS:-128000}"
      run_concurrency="${MAX_CONCURRENCY:-${model_concurrency:-32}}"
      rid="${rid}-${run_max_tokens}tokens"
    fi
    [[ -n "${ONLY:-}" && "$rid" != *"$ONLY"* ]] && continue
    limit_args=()
    if [[ -n "${LIMIT:-}" ]]; then
      rid="smoke-${rid}-${LIMIT}"
      limit_args=(--limit "$LIMIT")
    fi
    license_args=()
    [[ -n "$license" ]] && license_args=(--license "$license")
    command=(uv run sensebench run
      --model "openrouter/${model_id}" --prompt "$PROMPT" "${effort_args[@]}"
      --max-tokens "$run_max_tokens" --concurrency "$run_concurrency" --dataset lexen-v1
      --hosting-kind cloud_api --api-provider OpenRouter --vendor "$vendor"
      --source-kind "$source_kind" "${license_args[@]}"
      --model-url "https://openrouter.ai/${model_id}"
      --github-handle emilkayumov --runner-name "Emil Kaiumov"
      --run-id "$rid" --no-progress "${limit_args[@]}")
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
      printf '%q ' "${command[@]}"
      printf '\n'
      continue
    fi
    if [[ -d "runs/$rid" || -d "results/$rid" ]]; then
      echo "skip existing $rid" >&2
    else
      "${command[@]}"
    fi
    [[ -n "${LIMIT:-}" ]] && continue
    run_dir="runs/$rid"
    [[ -d "results/$rid" ]] && run_dir="results/$rid"
    if ! uv run python tools/verify_run_quality.py "$run_dir" --dataset lexen-v1 \
      --allow-invalid-attempts 9722 --allow-invalid-votes 4861 --allow-final-no-valid 4861; then
      echo "not copying $rid: quality check failed" >&2
      continue
    fi
    if [[ ! -d "results/$rid" ]]; then
      cp -R "$run_dir" "results/$rid"
    fi
  done
done
