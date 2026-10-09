#!/usr/bin/env bash
# Evaluate GPT-6 Luna on all registered prompts and benchmark reasoning modes.
# Usage: bash tools/cloud/run_luna_6.sh [YYYYMMDD]
# Optional environment: PROMPTS, EFFORTS, OPENAI_CONCURRENCY, MAX_CONCURRENCY,
# MAX_TOKENS, LIMIT, DRY_RUN=1. LIMIT runs are never copied into results/.
# Full runs are verified before copying to results/. Existing runs are preserved.
# Formatting errors are benchmark outcomes: retain retries and failed votes.
# Reject transport errors and token truncation, which can invalidate the comparison.
# https://developers.openai.com/api/docs/models/gpt-6-luna
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_DATE="${1:-$(date -u +%Y%m%d)}"
read -r -a PROMPT_IDS <<< "${PROMPTS:-p001 p002 p003 p004}"
read -r -a REASONING_MODES <<< "${EFFORTS:-none low medium high xhigh max}"
CONCURRENCY="${OPENAI_CONCURRENCY:-64}"

for prompt in "${PROMPT_IDS[@]}"; do
  for effort in "${REASONING_MODES[@]}"; do
    rid="gpt-6-luna-${effort}-reasoning-${prompt}-lexen-v1-${RUN_DATE}"
    run_max_tokens="${MAX_TOKENS:-32768}"
    run_concurrency="$CONCURRENCY"
    if [[ "$effort" == max ]]; then
      run_max_tokens="${MAX_TOKENS:-128000}"
      run_concurrency="${MAX_CONCURRENCY:-32}"
    fi
    limit_args=()
    if [[ -n "${LIMIT:-}" ]]; then
      rid="smoke-${rid}-${LIMIT}"
      limit_args=(--limit "$LIMIT")
    fi
    command=(uv run sensebench run
      --model gpt-6-luna --prompt "$prompt" --reasoning-effort "$effort"
      --max-tokens "$run_max_tokens" --concurrency "$run_concurrency" --dataset lexen-v1
      --hosting-kind cloud_api --api-provider OpenAI --vendor OpenAI
      --source-kind proprietary --github-handle vassiliphilippov
      --runner-name "Vassili Philippov" --run-id "$rid" "${limit_args[@]}")
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
      printf '%q ' "${command[@]}"
      printf '\n'
      continue
    fi
    if [[ ! -d "runs/$rid" && ! -d "results/$rid" ]]; then
      "${command[@]}"
    fi
    if [[ -n "${LIMIT:-}" ]]; then
      continue
    fi
    run_dir="runs/$rid"
    [[ -d "results/$rid" ]] && run_dir="results/$rid"
    uv run python tools/verify_run_quality.py "$run_dir" --dataset lexen-v1 \
      --allow-invalid-attempts 9722 --allow-invalid-votes 4861 --allow-final-no-valid 4861
    if [[ ! -d "results/$rid" ]]; then
      cp -R "$run_dir" "results/$rid"
    fi
  done
done
