---
name: "write-homepage-answer"
description: "Rewrite the hand-written answer block on the SenseBench homepage from current leaderboard data. Use when tools/check_homepage_answer.py reports it stale, or after runs that change the top of the board."
---
# Write the Homepage Answer Block

**Version**: 1

## Goal

Rewrite the `<section class="answer-block">` in `src/sensebench/site/templates/index.html.j2` so
that it answers "which LLM is best at word sense disambiguation" truthfully from current data.

This block is deliberately hand-written rather than template-generated. Its interesting sentences —
which models are statistically tied, what separates ConSeC from LENS — are judgements a format
string cannot make. That is also why it can go stale, and why every figure in it is machine-checked.

## Inputs

* `$ARGUMENTS` — optional. A results directory; defaults to `results/`.

## Context

Read before starting:

* `src/sensebench/site/templates/index.html.j2` — the current block, and the `.intro` / `.note`
  classes it reuses
* `src/sensebench/leaderboard/baselines.py` — baseline labels and, critically, the `source_note`
  provenance caveats
* `src/sensebench/leaderboard/schemes.py` — the nine schemes and `DEFAULT_SCHEME_ID`
* `tools/check_homepage_answer.py` — the checks the result must pass

## Steps

1. Rebuild the data. Do not read figures from `_site/` without rebuilding it first; a checked-out
   `_site/` can be months stale and may predate whole baselines.

   ```bash
   uv run sensebench site build --results-dir results --output-dir _site --strict
   ```

2. Read `_site/data/leaderboard.json`. Take the top entry's accuracy, both confidence bounds,
   `correct_count`, `item_count`, `display_label`, `reasoning_effort` and `prompt_id`; the baseline
   accuracies; and `summary.verified_run_count` / `summary.model_count`.

3. **Compute the statistical ties.** Load `results/<run-id>/predictions.jsonl` for the top run and
   for every run within a few points of it, and run an exact two-sided McNemar test over the
   discordant pairs. Report which models are indistinguishable from the leader at p >= 0.05, and
   name the notable models that are *not*. Accuracy proximity is not a tie — runs 0.4 points apart
   have tested both ways.

4. Take the latest run date as `max(created_at)` across entries, for both the prose and the
   `<time datetime="...">` attribute.

5. Write the block: an `<h2>` phrased as the question, one `<p class="intro">` paragraph, and a
   `<p class="note">` counts line. Keep the five links and their targets.

6. Verify:

   ```bash
   uv run sensebench site build --results-dir results --output-dir _site --strict
   uv run python tools/check_homepage_answer.py
   ```

## Rules

* **Recompute every figure.** Never carry a number over from the previous version of the block.
* **Always name the scheme.** Accuracy here is meaningless without gold source and granularity. The
  same run swings 95.60% to 85.39% between `lexen_fine` and `raganato_fine`, and the top model
  changes with it. The block states default-scheme figures, so it must say so.
* **Disclose ownership, and give the mechanism.** Glite LENS outscores the third-party supervised
  baselines. Say so — but say *why*: it is retrained on model-relabelled SemCor, where ConSeC is
  trained on the original human labels. Note that `baselines.py` records LENS's lexEN score as
  "confirmatory under the paper's Section 6.4 rule", because its training labels share a model
  family with the lexEN triage. It is not a clean independent comparison and must not be framed as
  one.
* **Prefer the number that goes down.** The residual error, the hard-subset score and the
  supervised-baseline gap are more informative, and more defensible, than a headline near ceiling.
* Keep it one paragraph, roughly 110–130 words: dated, numeric, source-attributed, self-contained.
  It should stand alone if lifted verbatim, because it will be.
* Match the surrounding prose: British spelling, no marketing register, no superlatives that are not
  measurements.

## Forbidden

* NEVER state a figure that `tools/check_homepage_answer.py` cannot verify against built site data.
* NEVER repeat a claim from the paper without re-testing it. The paper's "the top three families are
  statistically indistinguishable" was true when written and is false on current data — as of
  2026-07-25 only Claude Fable 5 ties with GPT-5.5 (p = 0.20), while Gemini 3.1 Pro (p = 0.031),
  GPT-5.6 (p = 0.019) and Claude Opus 5 (p = 0.004) all differ significantly.
* NEVER call any Glite artifact "best" or "strongest" without the training-data qualifier.
* NEVER add a sixth link, and never link the same target twice.
* NEVER rewrite the rest of the page "for AI readability". The one end-to-end study on this measured
  body-only optimisation as worse: -6% citations and -16% top-10 presence after reranking. Add the
  block; leave the intro, the tiles and the table alone.
* NEVER edit `_site/` — it is build output.
