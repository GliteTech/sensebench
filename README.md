# SenseBench

[![PyPI](https://img.shields.io/pypi/v/sensebench)](https://pypi.org/project/sensebench/)
[![Python](https://img.shields.io/pypi/pyversions/sensebench)](https://pypi.org/project/sensebench/)
[![CI](https://github.com/GliteTech/sensebench/actions/workflows/ci.yml/badge.svg)](https://github.com/GliteTech/sensebench/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/sensebench)](LICENSE)

SenseBench is a benchmark and leaderboard for evaluating English word sense disambiguation (WSD) on
[lexEN](https://github.com/GliteTech/lexen) and related datasets. A model is shown a target word in
context plus its candidate WordNet senses and must answer with the index of the correct sense.
Prompts are immutable, registered definitions; runs produce fully auditable artifacts that anyone
can re-verify down to the raw API responses.

**Live leaderboard:** <https://glitetech.github.io/sensebench/>

## Install

Requires Python 3.12+.

```bash
uv tool install sensebench   # or: pipx install sensebench / pip install sensebench
```

For development, clone this repository and run `uv sync`.

## Configure API keys

SenseBench calls models through [LiteLLM](https://docs.litellm.ai/), so any supported provider
works. Export the provider key, or put it in a `.env` file in your working directory (see
`.env.example`):

```bash
export OPENAI_API_KEY=...
```

## Run a benchmark

```bash
sensebench run \
  --model gpt-5.5 \
  --reasoning-effort medium \
  --max-tokens 4096 \
  --prompt p001 \
  --github-handle your-github-handle
```

For reasoning models, always pass `--reasoning-effort` explicitly (it is recorded in the run
metadata and shown on the leaderboard) and give the thinking budget headroom with `--max-tokens`.
Pass `--github-handle` so the run is eligible for leaderboard submission (you can also stamp it
later with `sensebench set-runner`).

On first use this downloads the NLTK WordNet corpus and the registered dataset release
(`lexen-v0.1.0`, cached under `~/.cache/sensebench/`, integrity-checked against a pinned SHA-256
hash). The run preflights the model with a single call, evaluates every item with a live progress
bar, writes `runs/<run-id>/`, and ends with an accuracy summary and submission instructions:

```text
run:     gpt-5.5-medium-reasoning-p001-lexen-v0.1.0-20260612
model:   gpt-5.5 (reasoning effort: medium)
prompt:  p001 — 5+1 Context, Frequency-Ordered WordNet Glosses
dataset: lexen-v0.1.0 (4,917 items)
policy:  votes_per_item=1 concurrency=512 max_tokens=4096
runner:  github:your-github-handle
preflight OK: gpt-5.5
Evaluating items: 100%|██████████| 4917/4917 [01:03<00:00, 78.0item/s, acc 0.941, cost $31.47]
run_id: gpt-5.5-medium-reasoning-p001-lexen-v0.1.0-20260612
accuracy: 0.9406 (4625/4917 correct)
items: success=4904 monosemous=0 no_valid_vote=13 no_candidates=0
votes: invalid_output=13 transport_error=0
calls: 5057  cost: $31.47  elapsed: 62.8s
artifacts: runs/gpt-5.5-medium-reasoning-p001-lexen-v0.1.0-20260612

Submit this run to the public leaderboard:
  1. Verify it: sensebench verify runs/<run-id> --dataset lexen-v0.1.0 --prompt p001
  2. Fork https://github.com/GliteTech/sensebench and copy the run directory to results/<run-id>/
  3. Open a pull request titled submit-<run-id>
CI re-verifies every submitted run from the raw artifacts; a maintainer reviews each submission,
and the leaderboard updates when the pull request is merged.
```

Useful flags:

* `--limit 25` for a cheap smoke run (not leaderboard-eligible)
* `--run-id my-run` to name the run yourself (otherwise generated from model, prompt, and dataset)
* `--temperature`, `--max-tokens`, `--seed`, `--reasoning-effort` for sampling control
* `--source-kind open_source`, `--license`, `--model-url`, `--vendor` to record model metadata
  shown on the leaderboard
* `--hosting-kind self_hosted --endpoint-base-url http://...` for self-hosted models

Each run directory contains:

* `run.json` — run metadata, policy, and totals
* `predictions.jsonl` — one record per item with candidates, votes, and correctness
* `calls.jsonl.gz` — every raw API request and response

### Running OpenRouter models

SenseBench calls providers through LiteLLM, so any model available on
[OpenRouter](https://openrouter.ai) works with the `openrouter/` model prefix and an
`OPENROUTER_API_KEY`:

```bash
export OPENROUTER_API_KEY=...
sensebench run \
  --model openrouter/qwen/qwen3-235b-a22b \
  --vendor Alibaba \
  --api-provider OpenRouter \
  --source-kind open_source \
  --max-tokens 4096 \
  --prompt p001 \
  --github-handle your-github-handle
```

The same prefix pattern works for every other LiteLLM provider (`gemini/`, `anthropic/`,
`mistral/`, ...). If LiteLLM has no pricing entry for a model, the runner warns you up front and
records the run cost as unavailable.

## Verify a run

Verification replays the full chain — prompt rendering, answer extraction, vote decisions, and
correctness against gold — from the stored artifacts:

```bash
sensebench verify runs/<run-id> --dataset lexen-v0.1.0 --prompt p001
```

## Dataset

[lexEN](https://github.com/GliteTech/lexen) is Glite's verified English WSD evaluation set.
`lexen-v0.1.0` contains 4,917 polysemous items derived from the Senseval-2, Senseval-3,
SemEval-2013, and SemEval-2015 all-words tasks, with candidate senses drawn from WordNet 3.0 and
gold senses re-verified by Glite to correct annotation errors in the original test sets.

Dataset releases are immutable JSONL exports published as
[GitHub release assets](https://github.com/GliteTech/lexen/releases) and downloaded automatically on
first use. Every release is pinned inside the package by URL, SHA-256 content hash, and item count,
and the loader rejects any file that does not match.

```bash
sensebench fetch-dataset lexen-v0.1.0
```

## Prompts

Registered prompts are immutable JSON definitions under `src/sensebench/prompts/registered/`. Any
benchmark-relevant change requires a new prompt ID. See `docs/prompts.md`.

## Submitting results

Anyone can submit a run to the public leaderboard:

1. Run the benchmark with the released package, your own API key, and your `--github-handle`.
2. Verify it locally: `sensebench verify runs/<run-id> --dataset lexen-v0.1.0 --prompt p001`.
3. Open a pull request that places the complete run directory (`run.json`, `predictions.jsonl`,
   `calls.jsonl.gz`) under `results/<run-id>/`.

Runner identity is required for submissions: `run.json` must carry the submitter's GitHub handle.
Stamp an existing run with `sensebench set-runner runs/<run-id> --github-handle <handle>`.

CI re-verifies every submitted run from the raw API responses and builds a site preview. Every pull
request is then reviewed by a maintainer: a run appears on the leaderboard only after the pull
request is accepted and merged, which deploys the updated site automatically. Partial runs and runs
that fail verification are rejected.

## Leaderboard

`sensebench leaderboard` aggregates verified run directories from `results/` into
`leaderboard.json`. Every run is re-verified before inclusion, and accuracy is recomputed from the
predictions rather than trusted from metadata.

## Website

The public leaderboard site is generated as static GitHub Pages output:

```bash
sensebench site build --results-dir results --output-dir _site --strict
```

The generated site includes an interactive leaderboard with bootstrap confidence intervals and
rank ranges, Pareto charts, reference baselines (MFS, BEM, ESCHER, ConSeC) scored on the same
items, paired statistical run comparisons, static run-detail pages, dataset and prompt reference
pages, submission instructions, a sitemap, and static JSON under `_site/data/`.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src tests tools
uv run mypy src
uv run python tools/verify_prompt.py --all
uv run sensebench site build --results-dir results --output-dir _site --strict
```

## About Glite

SenseBench is built and maintained by [Glite](https://glite.ai), the company behind the Glite
language-learning app. Showing readers the right dictionary sense for a word in context is at the
core of the Glite product, which is why Glite develops and publishes
[lexEN](https://github.com/GliteTech/lexen), SenseBench, and the public leaderboard as open
resources for the WSD community.

## License

Apache-2.0.
