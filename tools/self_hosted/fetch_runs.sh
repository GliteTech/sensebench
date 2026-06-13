#!/usr/bin/env bash
# Copy run artifacts from a provisioned vast.ai box to a local directory.
#
# Runs locally: bash tools/self_hosted/fetch_runs.sh <instance.json> [dest=runs]
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: bash tools/self_hosted/fetch_runs.sh <instance.json> [dest=runs]" >&2
  exit 2
fi

INSTANCE_JSON=$1
DEST=${2:-runs}
SSH_KEY=$HOME/.ssh/id_ed25519
REMOTE_RUNS_DIR=/workspace/sensebench/runs

SSH_HOST=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['ssh_host'])" \
  "$INSTANCE_JSON")
SSH_PORT=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['ssh_port'])" \
  "$INSTANCE_JSON")

mkdir -p "$DEST"
scp -o StrictHostKeyChecking=no -i "$SSH_KEY" -P "$SSH_PORT" -r \
  "root@$SSH_HOST:$REMOTE_RUNS_DIR/." "$DEST/"

echo "run directories now in $DEST:"
find "$DEST" -mindepth 2 -maxdepth 2 -name run.json | sort | while read -r run_json; do
  dirname "$run_json"
done
