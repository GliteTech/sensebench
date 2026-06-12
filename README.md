# SenseBench

[![PyPI](https://img.shields.io/pypi/v/sensebench)](https://pypi.org/project/sensebench/)
[![Python](https://img.shields.io/pypi/pyversions/sensebench)](https://pypi.org/project/sensebench/)
[![CI](https://github.com/GliteTech/sensebench/actions/workflows/ci.yml/badge.svg)](https://github.com/GliteTech/sensebench/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/sensebench)](LICENSE)

SenseBench is a benchmark and leaderboard for evaluating English word sense disambiguation (WSD) on
lexEN and related datasets. A model is shown a target word in context plus its candidate WordNet
senses and must answer with the index of the correct sense. Prompts are immutable, registered
definitions; runs produce fully auditable artifacts that anyone can re-verify down to the raw API
responses.

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
sensebench run --model gpt-5.5 --prompt p001
```

On first use this downloads the NLTK WordNet corpus and the registered dataset release
(`lexen-v0.1.0`, cached under `~/.cache/sensebench/`, integrity-checked against a pinned SHA-256
hash). The run preflights the model with a single call, evaluates every item, writes
`runs/<run-id>/`, and prints an accuracy summary. Useful flags:

* `--limit 25` for a cheap smoke run (not leaderboard-eligible)
* `--run-id my-run` to name the run yourself (otherwise generated from model, prompt, and dataset)
* `--temperature`, `--max-tokens`, `--seed`, `--reasoning-effort` for sampling control
* `--hosting-kind self_hosted --endpoint-base-url http://...` for self-hosted models

Each run directory contains:

* `run.json` — run metadata, policy, and totals
* `predictions.jsonl` — one record per item with candidates, votes, and correctness
* `calls.jsonl.gz` — every raw API request and response

## Verify a run

Verification replays the full chain — prompt rendering, answer extraction, vote decisions, and
correctness against gold — from the stored artifacts:

```bash
sensebench verify runs/<run-id> --dataset lexen-v0.1.0 --prompt p001
```

## Dataset

lexEN dataset releases are immutable JSONL exports published as
[GitHub release assets](https://github.com/GliteTech/lexen/releases) and downloaded automatically on
first use. Every release is pinned inside the package by URL, SHA-256 content hash, and item count,
and the loader rejects any file that does not match. `lexen-v0.1.0` contains 4,917 items derived
from Senseval-2, Senseval-3, SemEval-2013, and SemEval-2015, with candidate senses drawn from
WordNet 3.0.

```bash
sensebench fetch-dataset lexen-v0.1.0
```

## Prompts

Registered prompts are immutable JSON definitions under `src/sensebench/prompts/registered/`. Any
benchmark-relevant change requires a new prompt ID. See `docs/prompts.md`.

## Submitting results

Anyone can submit a run to the public leaderboard:

1. Run the benchmark with the released package and your own API key.
2. Verify it locally: `sensebench verify runs/<run-id> --dataset lexen-v0.1.0 --prompt p001`.
3. Open a pull request that places the complete run directory (`run.json`, `predictions.jsonl`,
   `calls.jsonl.gz`) under `results/<run-id>/`.

CI re-verifies every submitted run from the raw API responses and builds a site preview; merging to
`main` deploys the updated leaderboard automatically. Partial runs and runs that fail verification
are rejected.

## Leaderboard

`sensebench leaderboard` aggregates verified run directories from `results/` into
`leaderboard.json`. Every run is re-verified before inclusion, and accuracy is recomputed from the
predictions rather than trusted from metadata.

## Website

The public leaderboard site is generated as static GitHub Pages output:

```bash
sensebench site build --results-dir results --output-dir _site --strict
```

The generated site includes an interactive leaderboard, Pareto charts, static run-detail pages,
dataset and prompt reference pages, submission instructions, a sitemap, and static JSON under
`_site/data/`.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src tests tools
uv run mypy src
uv run python tools/verify_prompt.py --all
uv run sensebench site build --results-dir results --output-dir _site --strict
```

## License

Apache-2.0.
