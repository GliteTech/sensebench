#!/usr/bin/env bash
# Bootstrap a fresh vast.ai box for SenseBench self-hosted runs. Idempotent.
#
# Usage (on the box): bash setup_host.sh
# Optional: SENSEBENCH_BRANCH=<branch> bash setup_host.sh
#
# If the box needs a Hugging Face token (gated repos), scp it to
# /workspace/sensebench/hf_token BEFORE running this script.
set -euo pipefail

WORKDIR=/workspace/sensebench
REPO_URL=https://github.com/GliteTech/sensebench.git
BRANCH="${SENSEBENCH_BRANCH:-self-hosted-vllm-support}"
DATASET=lexen-v0.1.0
PREWARM_PROMPT=p003

mkdir -p "$WORKDIR"

# The vLLM image usually already has these; ignore apt failures.
apt-get install -y tmux curl git jq 2>/dev/null || true

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv" >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# A private repo cannot be cloned anonymously; ship it as a git bundle instead
# (scp it to $WORKDIR/repo.bundle) so no credentials ever land on the box.
if [ -f "$WORKDIR/repo.bundle" ]; then
  echo "installing repo from bundle" >&2
  rm -rf "$WORKDIR/repo"
  git clone --branch "$BRANCH" "$WORKDIR/repo.bundle" "$WORKDIR/repo"
elif [ -d "$WORKDIR/repo/.git" ]; then
  echo "updating existing checkout on branch $BRANCH" >&2
  git -C "$WORKDIR/repo" fetch origin "$BRANCH"
  git -C "$WORKDIR/repo" checkout "$BRANCH"
  git -C "$WORKDIR/repo" pull --ff-only origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$WORKDIR/repo"
fi

if [ ! -x "$WORKDIR/venv/bin/python" ]; then
  # The serving image may ship an older system Python; sensebench needs >=3.12,
  # and uv downloads a managed interpreter when none is available.
  uv venv --python 3.12 "$WORKDIR/venv"
fi
uv pip install --python "$WORKDIR/venv/bin/python" -e "$WORKDIR/repo"

# env.sh is sourced by run_leg.sh and by interactive shells on the box.
cat > "$WORKDIR/env.sh" <<EOF
export PATH="$WORKDIR/venv/bin:\$PATH"
export HF_HOME=/workspace/hf
export HOSTED_VLLM_API_KEY=dummy
export OPENAI_API_KEY=dummy
if [ -f "$WORKDIR/hf_token" ]; then
  export HF_TOKEN="\$(cat "$WORKDIR/hf_token")"
fi
EOF

if [ -f "$WORKDIR/hf_token" ]; then
  mkdir -p "$HOME/.cache/huggingface"
  cp "$WORKDIR/hf_token" "$HOME/.cache/huggingface/token"
  echo "installed Hugging Face token" >&2
else
  echo "no $WORKDIR/hf_token found; gated repos will not be downloadable" >&2
fi

# shellcheck disable=SC1091
source "$WORKDIR/env.sh"
mkdir -p "$HF_HOME"

# Prewarm: dataset release and the NLTK WordNet corpus (pulled in by render).
sensebench fetch-dataset "$DATASET"
sensebench render --prompt "$PREWARM_PROMPT" --limit 1 >/dev/null

echo "box UTC time (sanity-check against your clock):" >&2
date -u
echo "DONE"
