#!/usr/bin/env bash
# Run selected p003 cloud-API benchmarks and submit each clean run as its own PR.
#
# Usage:
#   bash tools/cloud/run_p003_and_submit.sh <openai|gemini|anthropic|openrouter|all>
#
# Provider streams may be run in parallel. PR creation and merge are serialized
# with a filesystem lock because they mutate the shared git worktree.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

set -a
[ -f .env ] && . ./.env
set +a

DATE="${DATE:-$(date -u +%Y%m%d)}"
DATASET=lexen-v1
PROMPT="${PROMPT:-p003}"
GH=vassiliphilippov
RUNNER="Vassili Philippov"
LOG="${LOG:-/tmp/sensebench_p003_submit.log}"
COSTFILE="${COSTFILE:-/tmp/sensebench_p003_spend.txt}"
LOCKDIR="${LOCKDIR:-/tmp/sensebench_submit.lock}"
[ -f "$COSTFILE" ] || echo 0 > "$COSTFILE"

note() { echo "$(date -u +%H:%M:%S) [${STREAM:-?}] $*" | tee -a "$LOG"; }

acquire() {
  local n=0
  until mkdir "$LOCKDIR" 2>/dev/null; do
    sleep 2
    n=$((n+1))
    [ "$n" -gt 900 ] && { note "LOCK TIMEOUT"; return 1; }
  done
}

release() { rmdir "$LOCKDIR" 2>/dev/null; }

already_done() {
  git fetch origin main -q 2>/dev/null
  git ls-tree -r --name-only origin/main 2>/dev/null | grep -q "^results/$1/run.json$"
  if [ "$?" = "0" ]; then
    return 0
  fi
  gh pr list --state open --head "submit-$1" --json number --jq 'length > 0' 2>/dev/null \
    | grep -q true
}

run_quality_ok() {
  local rid="$1"
  uv run python tools/verify_run_quality.py "runs/$rid" --dataset "$DATASET" >>"$LOG" 2>&1
}

submit() {
  local rid="$1"
  if ! uv run sensebench verify "runs/$rid" --dataset "$DATASET" --prompt "$PROMPT" >>"$LOG" 2>&1; then
    note "FAIL_VERIFY $rid"
    return 1
  fi
  if ! run_quality_ok "$rid"; then
    note "FAIL_QUALITY $rid"
    return 1
  fi

  local acc cost csrc
  acc=$(python3 -c "import json;print(f\"{json.load(open('runs/$rid/run.json'))['totals']['accuracy']:.4f}\")")
  cost=$(python3 -c "import json;print(f\"{json.load(open('runs/$rid/run.json'))['totals']['cost']['total_usd']:.2f}\")")
  csrc=$(python3 -c "import json;print(json.load(open('runs/$rid/run.json'))['totals']['cost']['source'])")
  rm -rf "results/$rid"
  cp -r "runs/$rid" "results/$rid"

  acquire || return 1
  git fetch origin main -q
  git checkout -q -B "submit-$rid" origin/main
  git add -f "results/$rid"
  git commit -q -m "Submit $rid (acc $acc)

Cost: \$$cost ($csrc). Verified locally; CI re-verifies from raw artifacts."
  git push -q -u origin "submit-$rid" 2>/dev/null
  gh pr create --base main --head "submit-$rid" --title "submit-$rid" \
    --body "Automated SenseBench p003 submission. Accuracy **$acc**, cost \$$cost ($csrc). Verified locally with \`sensebench verify\` and \`tools/verify_run_quality.py\`; CI re-verifies from raw artifacts." >>"$LOG" 2>&1
  local newtot
  newtot=$(python3 -c "print(f\"{$(cat "$COSTFILE")+$cost:.2f}\")")
  echo "$newtot" > "$COSTFILE"
  note "OPENED_PR submit-$rid acc=$acc cost=\$$cost [total \$$newtot]"
  release
}

reason_run() {
  local slug="$1" model="$2" effort="$3" maxtok="$4" conc="$5" vendor="$6" source="$7" provider="$8"
  local rid="${slug}-${effort}-reasoning-${PROMPT}-${DATASET}-${DATE}"
  if already_done "$rid"; then note "SKIP $rid"; return 0; fi
  rm -rf "runs/$rid"
  note "RUN $rid conc=$conc max_tokens=$maxtok"
  if uv run sensebench run --model "$model" --prompt "$PROMPT" \
      --reasoning-effort "$effort" --max-tokens "$maxtok" --concurrency "$conc" \
      --dataset "$DATASET" --hosting-kind cloud_api --api-provider "$provider" \
      --vendor "$vendor" --source-kind "$source" \
      --github-handle "$GH" --runner-name "$RUNNER" --no-progress --run-id "$rid" >>"$LOG" 2>&1; then
    submit "$rid"
  else
    note "FAIL_RUN $rid"
  fi
}

plain_run() {
  local slug="$1" model="$2" maxtok="$3" conc="$4" vendor="$5" source="$6" provider="$7"
  local rid="${slug}-${PROMPT}-${DATASET}-${DATE}"
  if already_done "$rid"; then note "SKIP $rid"; return 0; fi
  rm -rf "runs/$rid"
  note "RUN $rid conc=$conc max_tokens=$maxtok"
  if uv run sensebench run --model "$model" --prompt "$PROMPT" \
      --temperature 0 --max-tokens "$maxtok" --concurrency "$conc" \
      --dataset "$DATASET" --hosting-kind cloud_api --api-provider "$provider" \
      --vendor "$vendor" --source-kind "$source" \
      --github-handle "$GH" --runner-name "$RUNNER" --no-progress --run-id "$rid" >>"$LOG" 2>&1; then
    submit "$rid"
  else
    note "FAIL_RUN $rid"
  fi
}

openai_runs() {
  export STREAM=openai
  local co="${OPENAI_CONCURRENCY:-192}"
  plain_run  gpt-4.1      gpt-4.1      2048  "$co" OpenAI proprietary OpenAI
  plain_run  gpt-4.1-mini gpt-4.1-mini 2048  "$co" OpenAI proprietary OpenAI
  plain_run  gpt-4.1-nano gpt-4.1-nano 2048  "$co" OpenAI proprietary OpenAI
  plain_run  gpt-4o-mini  gpt-4o-mini  2048  "$co" OpenAI proprietary OpenAI
  reason_run gpt-5.4-mini gpt-5.4-mini low     8192  "$co" OpenAI proprietary OpenAI
  reason_run gpt-5.4-nano gpt-5.4-nano low     8192  "$co" OpenAI proprietary OpenAI
  reason_run gpt-5-mini   gpt-5-mini   minimal 8192  "$co" OpenAI proprietary OpenAI
  reason_run gpt-5-mini   gpt-5-mini   low     8192  "$co" OpenAI proprietary OpenAI
  reason_run gpt-5-mini   gpt-5-mini   medium  8192  "$co" OpenAI proprietary OpenAI
  reason_run gpt-5-mini   gpt-5-mini   high    8192  "$co" OpenAI proprietary OpenAI
  reason_run gpt-5-nano   gpt-5-nano   minimal 8192  "$co" OpenAI proprietary OpenAI
  reason_run gpt-5-nano   gpt-5-nano   low     8192  "$co" OpenAI proprietary OpenAI
  reason_run gpt-5-nano   gpt-5-nano   medium  8192  "$co" OpenAI proprietary OpenAI
  reason_run gpt-5-nano   gpt-5-nano   high    8192  "$co" OpenAI proprietary OpenAI
  reason_run gpt-5.5      gpt-5.5      high    16384 "$co" OpenAI proprietary OpenAI
}

gemini_runs() {
  export STREAM=gemini
  local cg="${GEMINI_CONCURRENCY:-16}"
  plain_run  gemini-3.1-flash-lite gemini/gemini-3.1-flash-lite 4096  "$cg" Google proprietary Google
  plain_run  gemini-2.5-flash      gemini/gemini-2.5-flash      4096  "$cg" Google proprietary Google
  plain_run  gemini-3-flash        gemini/gemini-3-flash-preview 4096 "$cg" Google proprietary Google
  plain_run  gemini-3.5-flash      gemini/gemini-3.5-flash      4096  "$cg" Google proprietary Google
  reason_run gemini-3.1-pro        gemini/gemini-3.1-pro-preview medium 12288 "$cg" Google proprietary Google
  reason_run gemini-3.1-pro        gemini/gemini-3.1-pro-preview high   12288 "$cg" Google proprietary Google
}

anthropic_runs() {
  export STREAM=anthropic
  local ca="${ANTHROPIC_CONCURRENCY:-16}"
  reason_run claude-haiku-4.5  claude-haiku-4-5  low    8192  "$ca" Anthropic proprietary Anthropic
  reason_run claude-sonnet-4.6 claude-sonnet-4-6 low    8192  "$ca" Anthropic proprietary Anthropic
  reason_run claude-opus-4.6   claude-opus-4-6   medium 12288 "$ca" Anthropic proprietary Anthropic
  reason_run claude-opus-4.7   claude-opus-4-7   medium 12288 "$ca" Anthropic proprietary Anthropic
  reason_run claude-opus-4.8   claude-opus-4-8   low    16384 "$ca" Anthropic proprietary Anthropic
  reason_run claude-opus-4.8   claude-opus-4-8   medium 16384 "$ca" Anthropic proprietary Anthropic
  reason_run claude-opus-4.8   claude-opus-4-8   high   16384 "$ca" Anthropic proprietary Anthropic
  reason_run claude-opus-4.8   claude-opus-4-8   xhigh  16384 "$ca" Anthropic proprietary Anthropic
  claude_sonnet_5_runs
}

claude_sonnet_5_runs() {
  export STREAM=anthropic
  local ca="${ANTHROPIC_CONCURRENCY:-16}"
  reason_run claude-sonnet-5 claude-sonnet-5 low    16384 "$ca" Anthropic proprietary Anthropic
  reason_run claude-sonnet-5 claude-sonnet-5 medium 16384 "$ca" Anthropic proprietary Anthropic
  reason_run claude-sonnet-5 claude-sonnet-5 high   16384 "$ca" Anthropic proprietary Anthropic
  reason_run claude-sonnet-5 claude-sonnet-5 xhigh  16384 "$ca" Anthropic proprietary Anthropic
}

claude_fable_5_runs() {
  export STREAM=anthropic
  local ca="${ANTHROPIC_CONCURRENCY:-16}"
  reason_run claude-fable-5 claude-fable-5 xhigh  16384 "$ca" Anthropic proprietary Anthropic
}

openrouter_runs() {
  export STREAM=openrouter
  local cr="${OPENROUTER_CONCURRENCY:-32}"
  reason_run deepseek-v4-flash openrouter/deepseek/deepseek-v4-flash high   8192  "$cr" DeepSeek open_source OpenRouter
  reason_run deepseek-v4-pro   openrouter/deepseek/deepseek-v4-pro   high   8192  "$cr" DeepSeek open_source OpenRouter
  plain_run  kimi-k2.5         openrouter/moonshotai/kimi-k2.5             8192  "$cr" Moonshot open_source OpenRouter
  plain_run  qwen3.7-plus      openrouter/qwen/qwen3.7-plus                8192  "$cr" Alibaba open_source OpenRouter
  reason_run qwen3.7-max       openrouter/qwen/qwen3.7-max       medium 8192  "$cr" Alibaba open_source OpenRouter
  reason_run glm-5             openrouter/z-ai/glm-5             low    8192  "$cr" Z.ai open_source OpenRouter
  reason_run grok-4.3          openrouter/x-ai/grok-4.3          low    12288 "$cr" xAI proprietary OpenRouter
  reason_run grok-4.3          openrouter/x-ai/grok-4.3          medium 12288 "$cr" xAI proprietary OpenRouter
  reason_run grok-4.3          openrouter/x-ai/grok-4.3          high   12288 "$cr" xAI proprietary OpenRouter
}

case "${1:-}" in
  openai) openai_runs ;;
  gemini) gemini_runs ;;
  anthropic) anthropic_runs ;;
  claude-sonnet-5) claude_sonnet_5_runs ;;
  claude-fable-5) claude_fable_5_runs ;;
  openrouter) openrouter_runs ;;
  all) openai_runs; gemini_runs; anthropic_runs; openrouter_runs ;;
  *) echo "usage: $0 <openai|gemini|anthropic|openrouter|claude-sonnet-5|claude-fable-5|all>"; exit 2 ;;
esac

note "STREAM ${1:-?} DONE. Running total: \$$(cat "$COSTFILE")"
