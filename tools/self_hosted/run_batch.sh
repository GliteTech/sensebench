#!/usr/bin/env bash
# Orchestrate one full self-hosted GPU batch end to end: provision a vast.ai box
# for a GPU preset, bootstrap it from a git bundle (the repo is private), run every
# manifest job for that preset (each prompt), fetch the run artifacts, and destroy
# the box. Always destroys the box it provisioned, even on failure.
#
# Only ever acts on the single instance it provisions (tracked in work/<gpu>/
# instance.json). It never touches any other vast.ai instance on the account.
#
# Runs locally from the repo root:
#   bash tools/self_hosted/run_batch.sh --gpu h100 [--date YYYYMMDD]
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
# provision.py / destroy.py import the `tools` package; ensure the repo root is importable.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

GPU=""
RUN_DATE="$(date -u +%Y%m%d)"
JOBS_OVERRIDE=""
BRANCH_OVERRIDE=""
VOTES=""
TEMPERATURE=""
SHUFFLE_SENSES=0
RUN_SUFFIX=""
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --date) RUN_DATE="$2"; shift 2 ;;
    --jobs) JOBS_OVERRIDE="$2"; shift 2 ;;
    --branch) BRANCH_OVERRIDE="$2"; shift 2 ;;
    --votes) VOTES="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --shuffle-senses) SHUFFLE_SENSES=1; shift 1 ;;
    --run-suffix) RUN_SUFFIX="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$GPU" ] || { echo "missing --gpu" >&2; exit 2; }

BRANCH=lexen-v1-rebuild
[ -n "$BRANCH_OVERRIDE" ] && BRANCH="$BRANCH_OVERRIDE"
GITHUB_HANDLE=vassiliphilippov
RUNNER_NAME="Vassili Philippov"
HF_TOKEN_SRC="$HOME/.ssh/hf_token"
[ -f "$HF_TOKEN_SRC" ] || HF_TOKEN_SRC="$HOME/.cache/huggingface/token"
SSH_KEY="$HOME/.ssh/id_ed25519"
MANIFEST=tools/self_hosted/manifest.json
INSTANCE_JSON="work/$GPU/instance.json"
BUNDLE="work/$GPU/repo.bundle"
LAUNCHER="work/$GPU/launch_leg.sh"
JOB_POLL_SECONDS=30
JOB_TIMEOUT_SECONDS=5400          # 90 min per job (download + serve + both prompts)
LOG_PREFIX="[batch:$GPU]"

log() { echo "$LOG_PREFIX $(date -u +%H:%M:%S) $*"; }

# ---- provision -------------------------------------------------------------
log "provisioning $GPU ..."
if ! uv run python tools/self_hosted/provision.py --gpu "$GPU"; then
  log "provision failed; nothing to clean up"
  exit 1
fi
[ -f "$INSTANCE_JSON" ] || { log "no $INSTANCE_JSON written"; exit 1; }

HOST=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['ssh_host'])" "$INSTANCE_JSON")
PORT=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['ssh_port'])" "$INSTANCE_JSON")
ID=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['instance_id'])" "$INSTANCE_JSON")
RATE=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['hourly_rate_usd'])" "$INSTANCE_JSON")
log "instance $ID at $HOST:$PORT (\$$RATE/h)"

SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i "$SSH_KEY" -p "$PORT" "root@$HOST")
SCP=(scp -o StrictHostKeyChecking=no -i "$SSH_KEY" -P "$PORT")

# Always destroy the box we provisioned, whatever happens next.
DESTROYED=0
cleanup() {
  if [ "$DESTROYED" = "0" ]; then
    log "destroying instance $ID ..."
    uv run python tools/self_hosted/destroy.py "$INSTANCE_JSON" || \
      log "WARNING: destroy failed; destroy manually: uvx vastai@0.5.0 destroy instance $ID"
    DESTROYED=1
  fi
}
trap cleanup EXIT INT TERM

# ---- bootstrap -------------------------------------------------------------
log "building git bundle of $BRANCH"
git bundle create "$BUNDLE" "$BRANCH" >/dev/null 2>&1 || { log "git bundle failed"; exit 1; }

log "copying bootstrap files to the box"
"${SSH[@]}" "mkdir -p /workspace/sensebench" || { log "ssh mkdir failed"; exit 1; }
"${SCP[@]}" tools/self_hosted/setup_host.sh "root@$HOST:/workspace/" || { log "scp setup_host failed"; exit 1; }
"${SCP[@]}" "$BUNDLE" "root@$HOST:/workspace/sensebench/repo.bundle" || { log "scp bundle failed"; exit 1; }
if [ -f "$HF_TOKEN_SRC" ]; then
  "${SCP[@]}" "$HF_TOKEN_SRC" "root@$HOST:/workspace/sensebench/hf_token" || log "WARNING: scp hf_token failed (gated models may fail)"
else
  log "WARNING: no HF token at $HF_TOKEN_SRC; gated models (llama/gemma) will fail"
fi

log "running setup_host.sh on the box (clone, venv, dataset prefetch)"
if ! "${SSH[@]}" "SENSEBENCH_BRANCH=$BRANCH bash /workspace/setup_host.sh"; then
  log "setup_host.sh failed; aborting batch"
  exit 1
fi

# ---- per-job launcher (baked args, avoids nested-quoting in tmux) -----------
EXTRA_LEG_ARGS=""
[ -n "$VOTES" ] && EXTRA_LEG_ARGS="$EXTRA_LEG_ARGS --votes $VOTES"
[ -n "$TEMPERATURE" ] && EXTRA_LEG_ARGS="$EXTRA_LEG_ARGS --temperature $TEMPERATURE"
[ "$SHUFFLE_SENSES" = "1" ] && EXTRA_LEG_ARGS="$EXTRA_LEG_ARGS --shuffle-senses"
[ -n "$RUN_SUFFIX" ] && EXTRA_LEG_ARGS="$EXTRA_LEG_ARGS --run-suffix $RUN_SUFFIX"
cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
mkdir -p /workspace/sensebench/logs
EVICT_MODEL=1 HF_HUB_DISABLE_XET=1 bash /workspace/sensebench/repo/tools/self_hosted/run_leg.sh \\
  --job "\$1" --gpu $GPU --hourly-rate-usd $RATE --instance-id $ID \\
  --github-handle $GITHUB_HANDLE --runner-name "$RUNNER_NAME"$EXTRA_LEG_ARGS \\
  >> /workspace/sensebench/logs/leg-"\$1".log 2>&1
EOF
"${SCP[@]}" "$LAUNCHER" "root@$HOST:/workspace/sensebench/launch_leg.sh" || { log "scp launcher failed"; exit 1; }

# ---- jobs ------------------------------------------------------------------
if [ -n "$JOBS_OVERRIDE" ]; then
  JOBS="$JOBS_OVERRIDE"
else
  JOBS=$(python3 -c "
import json,sys
m=json.load(open('$MANIFEST'))
print(' '.join(j['job_id'] for j in m['jobs'] if '$GPU' in j['gpus']))
")
fi
log "jobs for $GPU: $JOBS"

for JOB in $JOBS; do
  log "starting job $JOB"
  "${SSH[@]}" "rm -f /workspace/sensebench/status/$JOB; tmux kill-session -t leg-$JOB 2>/dev/null; tmux new-session -d -s leg-$JOB 'bash /workspace/sensebench/launch_leg.sh $JOB'" \
    || { log "failed to launch $JOB; skipping"; continue; }
  deadline=$(( $(date +%s) + JOB_TIMEOUT_SECONDS ))
  while :; do
    STATUS=$("${SSH[@]}" "cat /workspace/sensebench/status/$JOB 2>/dev/null" 2>/dev/null || echo "")
    case "$STATUS" in
      DONE) log "job $JOB: DONE"; break ;;
      FAILED:*) log "job $JOB: $STATUS (continuing)"; break ;;
    esac
    if [ "$(date +%s)" -ge "$deadline" ]; then
      log "job $JOB: TIMEOUT after ${JOB_TIMEOUT_SECONDS}s (continuing)"
      "${SSH[@]}" "tmux kill-session -t leg-$JOB 2>/dev/null" || true
      break
    fi
    sleep "$JOB_POLL_SECONDS"
  done
done

# ---- fetch + destroy -------------------------------------------------------
log "fetching run artifacts"
bash tools/self_hosted/fetch_runs.sh "$INSTANCE_JSON" runs || log "WARNING: fetch_runs failed"

log "fetching box logs (server + leg) for diagnostics"
mkdir -p "work/$GPU/box-logs"
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" -P "$PORT" -r \
  "root@$HOST:/workspace/sensebench/logs/." "work/$GPU/box-logs/" 2>/dev/null \
  || log "WARNING: box-log fetch failed"

cleanup
trap - EXIT INT TERM
log "batch complete for $GPU"
