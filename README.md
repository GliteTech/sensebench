# SenseBench

SenseBench is benchmark and leaderboard tooling for evaluating English word sense disambiguation
(WSD) on lexEN and related datasets. A model is shown a target word in context plus its candidate
WordNet senses and must answer with the index of the correct sense. Prompts are immutable,
registered definitions; runs produce fully auditable artifacts that anyone can re-verify down to
the raw API responses.

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

## Prompts

Registered prompts are immutable JSON definitions under `src/sensebench/prompts/registered/`. Any
benchmark-relevant change requires a new prompt ID. See `docs/prompts.md`.

## Leaderboard

`sensebench leaderboard` aggregates verified run directories from `results/` into
`leaderboard.json`. Every run is re-verified before inclusion, and accuracy is recomputed from the
predictions rather than trusted from metadata.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src tests tools
uv run mypy src
uv run python tools/verify_prompt.py --all
```

## License

Apache-2.0.
