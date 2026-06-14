#!/usr/bin/env bash
# Run cloud-API models on the registered dataset and submit each completed run as
# its own individual PR (verified locally, then merged to main).
#
# One run == one PR == one results/<run-id>/ directory. Designed to run as up to
# four PARALLEL provider streams (openai|gemini|anthropic|openrouter) — the slow,
# rate-limited API runs proceed concurrently across providers, while the fast
# git/PR/merge step is serialised by a mkdir lock so the shared working tree is
# only ever mutated by one stream at a time. Within a stream, runs are ordered
# cheapest first.
#
# Usage:
#   bash tools/cloud/run_and_submit.sh <openai|gemini|anthropic|openrouter|all>
#
# Requires OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY
# (environment or .env), and an authenticated gh CLI with push rights to origin.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATE="${DATE:-20260614}"
DATASET=lexen-v1
GH=vassiliphilippov
RUNNER="Vassili Philippov"
LOG=/tmp/sensebench_submit.log
COSTFILE=/tmp/sensebench_spend.txt
LOCKDIR=/tmp/sensebench_submit.lock
[ -f "$COSTFILE" ] || echo 0 > "$COSTFILE"

note() { echo "$(date -u +%H:%M:%S) [${STREAM:-?}] $*" | tee -a "$LOG"; }

acquire() { local n=0; until mkdir "$LOCKDIR" 2>/dev/null; do sleep 2; n=$((n+1)); [ $n -gt 900 ] && { note "LOCK TIMEOUT"; return 1; }; done; }
release() { rmdir "$LOCKDIR" 2>/dev/null; }

# already_done <run-id> : true if results/<run-id> already merged to origin/main.
already_done() { git fetch origin main -q 2>/dev/null; git ls-tree -r --name-only origin/main 2>/dev/null | grep -q "^results/$1/run.json$"; }

# submit <run-id> <prompt> : verify, copy to results/, then (locked) branch/commit/push/PR/merge.
submit() {
  local rid="$1" prompt="$2"
  if ! uv run sensebench verify "runs/$rid" --dataset "$DATASET" --prompt "$prompt" >/dev/null 2>&1; then
    note "FAIL_VERIFY $rid"; return 1
  fi
  local acc cost csrc
  acc=$(python3 -c "import json;print(f\"{json.load(open('runs/$rid/run.json'))['totals']['accuracy']:.4f}\")")
  cost=$(python3 -c "import json;print(f\"{json.load(open('runs/$rid/run.json'))['totals']['cost']['total_usd']:.2f}\")")
  csrc=$(python3 -c "import json;print(json.load(open('runs/$rid/run.json'))['totals']['cost']['source'])")
  rm -rf "results/$rid"; cp -r "runs/$rid" "results/$rid"
  acquire || return 1
  git fetch origin main -q
  git checkout -q -B "submit-$rid" origin/main
  git add -f "results/$rid"
  git commit -q -m "Submit $rid (acc $acc)

Cost: \$$cost ($csrc). Verified locally; CI re-verifies from raw artifacts.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  git push -q -u origin "submit-$rid" 2>/dev/null
  gh pr create --base main --head "submit-$rid" --title "submit-$rid" \
    --body "Automated SenseBench submission. Accuracy **$acc**, cost \$$cost ($csrc). Verified locally with \`sensebench verify\`; CI re-verifies from the raw \`calls.jsonl.gz\`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)" >/dev/null 2>&1
  if gh pr merge "submit-$rid" --squash >/dev/null 2>&1; then
    local newtot; newtot=$(python3 -c "print(f\"{$(cat $COSTFILE)+$cost:.2f}\")"); echo "$newtot" > "$COSTFILE"
    note "MERGED submit-$rid  acc=$acc  cost=\$$cost  [total \$$newtot]"; release
  else
    note "FAIL_MERGE $rid"; release; return 1
  fi
}

# reason_run <slug> <model> <prompt> <effort> <maxtok> <conc> <vendor> <source> <provider>
reason_run() {
  local slug="$1" model="$2" prompt="$3" effort="$4" maxtok="$5" conc="$6" vendor="$7" source="$8" provider="$9"
  local rid="${slug}-${effort}-reasoning-${prompt}-${DATASET}-${DATE}"
  if already_done "$rid"; then note "SKIP $rid"; return 0; fi
  rm -rf "runs/$rid"
  note "RUN $rid (conc=$conc)"
  if uv run sensebench run --model "$model" --prompt "$prompt" \
      --reasoning-effort "$effort" --max-tokens "$maxtok" --concurrency "$conc" \
      --dataset "$DATASET" --hosting-kind cloud_api --api-provider "$provider" \
      --vendor "$vendor" --source-kind "$source" \
      --github-handle "$GH" --runner-name "$RUNNER" --no-progress --run-id "$rid" >>"$LOG" 2>&1; then
    submit "$rid" "$prompt"
  else
    note "FAIL_RUN $rid"
  fi
}

# plain_run <slug> <model> <prompt> <maxtok> <conc> <vendor> <source> <provider>
plain_run() {
  local slug="$1" model="$2" prompt="$3" maxtok="$4" conc="$5" vendor="$6" source="$7" provider="$8"
  local rid="${slug}-${prompt}-${DATASET}-${DATE}"
  if already_done "$rid"; then note "SKIP $rid"; return 0; fi
  rm -rf "runs/$rid"
  note "RUN $rid (conc=$conc)"
  if uv run sensebench run --model "$model" --prompt "$prompt" \
      --temperature 0 --max-tokens "$maxtok" --concurrency "$conc" \
      --dataset "$DATASET" --hosting-kind cloud_api --api-provider "$provider" \
      --vendor "$vendor" --source-kind "$source" \
      --github-handle "$GH" --runner-name "$RUNNER" --no-progress --run-id "$rid" >>"$LOG" 2>&1; then
    submit "$rid" "$prompt"
  else
    note "FAIL_RUN $rid"
  fi
}

CO=256; CG=32; CGR=24; CA=16; CR=48

openai_runs() {
  export STREAM=openai
  reason_run gpt-5.4-nano gpt-5.4-nano p001 low 8192 $CO OpenAI proprietary OpenAI
  reason_run gpt-5.4-nano gpt-5.4-nano p002 low 8192 $CO OpenAI proprietary OpenAI
  plain_run  gpt-4.1      gpt-4.1      p001 2048 $CO OpenAI proprietary OpenAI
  plain_run  gpt-4.1      gpt-4.1      p002 256  $CO OpenAI proprietary OpenAI
  reason_run gpt-5.4-mini gpt-5.4-mini p001 low 8192 $CO OpenAI proprietary OpenAI
  reason_run gpt-5.4-mini gpt-5.4-mini p002 low 8192 $CO OpenAI proprietary OpenAI
  reason_run gpt-5.5      gpt-5.5      p001 low  12288 $CO OpenAI proprietary OpenAI
  reason_run gpt-5.5      gpt-5.5      p001 high 16384 $CO OpenAI proprietary OpenAI
  reason_run gpt-5.5      gpt-5.5      p002 high 16384 $CO OpenAI proprietary OpenAI
}

gemini_runs() {
  export STREAM=gemini
  plain_run gemini-3.1-flash-lite gemini/gemini-3.1-flash-lite p001 2048 $CG Google proprietary Google
  plain_run gemini-3.1-flash-lite gemini/gemini-3.1-flash-lite p002 256  $CG Google proprietary Google
  plain_run gemini-2.5-flash      gemini/gemini-2.5-flash      p001 2048 $CG Google proprietary Google
  plain_run gemini-2.5-flash      gemini/gemini-2.5-flash      p002 256  $CG Google proprietary Google
  plain_run gemini-3-flash        gemini/gemini-3-flash-preview p001 2048 $CG Google proprietary Google
  plain_run gemini-3-flash        gemini/gemini-3-flash-preview p002 256  $CG Google proprietary Google
  plain_run gemini-3.5-flash      gemini/gemini-3.5-flash      p001 2048 $CG Google proprietary Google
  plain_run gemini-3.5-flash      gemini/gemini-3.5-flash      p002 256  $CG Google proprietary Google
  for e in low medium high; do
    reason_run gemini-3.1-pro gemini/gemini-3.1-pro-preview p001 $e 12288 $CGR Google proprietary Google
  done
}

anthropic_runs() {
  export STREAM=anthropic
  reason_run claude-haiku-4.5  claude-haiku-4-5  p001 low 8192 $CA Anthropic proprietary Anthropic
  reason_run claude-haiku-4.5  claude-haiku-4-5  p002 low 8192 $CA Anthropic proprietary Anthropic
  reason_run claude-sonnet-4.6 claude-sonnet-4-6 p001 low 8192 $CA Anthropic proprietary Anthropic
  reason_run claude-sonnet-4.6 claude-sonnet-4-6 p002 low 8192 $CA Anthropic proprietary Anthropic
  reason_run claude-opus-4.7   claude-opus-4-7   p001 medium 12288 $CA Anthropic proprietary Anthropic
  reason_run claude-opus-4.6   claude-opus-4-6   p001 medium 12288 $CA Anthropic proprietary Anthropic
  for e in low medium high xhigh; do
    reason_run claude-opus-4.8 claude-opus-4-8 p001 $e 16384 $CA Anthropic proprietary Anthropic
  done
  reason_run claude-opus-4.8 claude-opus-4-8 p002 high 16384 $CA Anthropic proprietary Anthropic
}

openrouter_runs() {
  export STREAM=openrouter
  reason_run deepseek-v4-flash openrouter/deepseek/deepseek-v4-flash p001 high 8192 $CR DeepSeek open_source OpenRouter
  reason_run deepseek-v4-flash openrouter/deepseek/deepseek-v4-flash p002 high 8192 $CR DeepSeek open_source OpenRouter
  # Kimi/MiniMax/Qwen-Plus reason by default on OpenRouter, so give the answer room
  # past the thinking trace (2048 truncated ~14% of Kimi outputs -> verify rejects).
  plain_run  kimi-k2.5       openrouter/moonshotai/kimi-k2.5 p001 8192 $CR Moonshot open_source OpenRouter
  plain_run  kimi-k2.5       openrouter/moonshotai/kimi-k2.5 p002 8192 $CR Moonshot open_source OpenRouter
  plain_run  minimax-m3      openrouter/minimax/minimax-m3   p001 8192 $CR MiniMax  open_source OpenRouter
  plain_run  minimax-m3      openrouter/minimax/minimax-m3   p002 8192 $CR MiniMax  open_source OpenRouter
  plain_run  qwen3.7-plus    openrouter/qwen/qwen3.7-plus    p001 8192 $CR Alibaba  open_source OpenRouter
  plain_run  qwen3.7-plus    openrouter/qwen/qwen3.7-plus    p002 8192 $CR Alibaba  open_source OpenRouter
  # grok-4.1-fast dropped: deprecated on OpenRouter (404 -> xAI recommends Grok 4.3). xAI is covered by the grok-4.3 sweep below.
  reason_run deepseek-v4-pro openrouter/deepseek/deepseek-v4-pro p001 high 8192 $CR DeepSeek open_source OpenRouter
  reason_run deepseek-v4-pro openrouter/deepseek/deepseek-v4-pro p002 high 8192 $CR DeepSeek open_source OpenRouter
  reason_run glm-5           openrouter/z-ai/glm-5           p001 low 8192 $CR Z.ai open_source OpenRouter
  reason_run glm-5           openrouter/z-ai/glm-5           p002 low 8192 $CR Z.ai open_source OpenRouter
  reason_run qwen3.7-max     openrouter/qwen/qwen3.7-max     p001 medium 8192 $CR Alibaba open_source OpenRouter
  reason_run qwen3.7-max     openrouter/qwen/qwen3.7-max     p002 medium 8192 $CR Alibaba open_source OpenRouter
  for e in low medium high; do
    reason_run grok-4.3 openrouter/x-ai/grok-4.3 p001 $e 12288 $CR xAI proprietary OpenRouter
  done
}

case "${1:-}" in
  openai)     openai_runs ;;
  gemini)     gemini_runs ;;
  anthropic)  anthropic_runs ;;
  openrouter) openrouter_runs ;;
  all)        openai_runs; gemini_runs; anthropic_runs; openrouter_runs ;;
  *) echo "usage: $0 <openai|gemini|anthropic|openrouter|all>"; exit 2 ;;
esac
note "STREAM ${1:-?} DONE. Running total: \$$(cat $COSTFILE)"
