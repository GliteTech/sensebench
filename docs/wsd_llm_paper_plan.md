# WSD LLM Paper Plan

## Core Positioning

**Working title**: lexEN and SenseBench: Verified English WSD Evaluation in the LLM Era.

**Sharper alternate title**: When 2% Label Noise Matters: Verified English WSD for Frontier LLMs.

**One-sentence claim**: English WSD evaluation has entered a high-accuracy regime where residual
gold-label errors materially distort leaderboard results, so the field needs a traceable corrected
benchmark, public raw-artifact evaluation, and current LLM baselines on the same protocol.

**Best framing**: This should be a benchmark and measurement paper, not only a leaderboard paper.
The durable contribution is the corrected verification set plus the public evaluation protocol. The
headline SOTA numbers make the paper visible, but the reason it can be cited is that future systems
can be compared against a better maintained benchmark.

## Main Contributions

* **lexEN correction layer**: A conservative correction of Maru2022 ALL_NEW with model-assisted
  triage, independent professional lexicographer review, provenance, and release metadata.
* **SenseBench**: A public English WSD benchmark and leaderboard for LLMs with fixed prompts,
  dataset canaries, raw call artifacts, cost tracking, and strict artifact validation.
* **Current LLM results**: A leaderboard over public and proprietary LLMs using WordNet candidate
  senses, with prompt variants and cost/accuracy comparisons.
* **Label-noise impact analysis**: A direct comparison of Maru2022 labels, lexEN corrected labels,
  and previous WSD systems showing that corrections affect frontier LLM scores more than classic
  supervised WSD systems.
* **Human ambiguity analysis**: Evidence that remaining hard English WSD items contain real
  fine-grained ambiguity, translation defects, and source-text defects, not just model errors.

## Evidence Base

### lexEN Dataset

Use `/Users/vassiliphilippov/lexen/` as the source for dataset facts.

Key facts for the paper:

* The release is derived from Maru2022 ALL_NEW (**4,917** source instances) and contains **4,861**
  English WSD instances after the consensus review removes 56 unresolvable items.
* The published release for the paper is `lexen-v1`, with all three professional lexicographers (RF,
  PW, PH) incorporated and the adjudication rule frozen.
* `lexen-v1` is the published benchmark release; `lexen-v0.2.0` (interim two-reviewer evidence) and
  `lexen-v0.1.0` (mixed-protocol draft) are deprecated.
* The audit selected **363** suspicious items for review and kept **4,554** unreviewed Maru2022
  labels.
* The three-annotator gold rule retains a reviewed item when at least two of RF/PW/PH select the
  same non-empty sense set, and removes it when at least two mark it cannot-answer or no
  fine-grained sense receives two-reviewer support.
* Relative to Maru2022, `lexen-v1` changes **211** gold labels and removes **56** items. Reviewed
  decisions: **124** three-way exact agreement, **183** two-of-three sense agreement (**307**
  retained); **29** three-way no-consensus and **27** two-of-three cannot-answer removed.
* The pairwise lexicographer-agreement statistics later in this document (the RF/PW2 figures) come
  from the interim two-reviewer `lexen-v0.2.0` analysis; refresh them from
  `reports/rf-pw-ph-2026-06-13/metrics.json` for the final three-annotator (RF/PW/PH) numbers before
  publishing.
* Coarse Glite mapping raises agreement substantially, with coarse kappa **0.748** (interim).

Recommended dataset table:

| Quantity | Count | Paper use |
| --- | ---: | --- |
| Maru2022 ALL_NEW source instances | 4,917 | Source corpus size |
| lexEN v1 benchmark items | 4,861 | Main benchmark size (after 56 removed) |
| Model-triaged suspicious items | 363 | Human review subset |
| Unreviewed labels kept from Maru2022 | 4,554 | Conservative coverage statement |
| Reviewed items retained | 307 | Three-annotator consensus kept |
| Three-way exact agreement | 124 | Strongest consensus subset |
| Two-of-three sense agreement | 183 | Majority consensus subset |
| Reviewed items removed | 56 | No consensus or cannot-answer |
| lexEN gold labels changed from Maru2022 | 211 | Label-quality headline |

Correction yield by selection bucket:

| Bucket | Reviewed | Primary-label changes | Yield |
| --- | ---: | ---: | ---: |
| S1 | 55 | 21 | 38.2% |
| S2 | 138 | 59 | 42.8% |
| S3 | 41 | 10 | 24.4% |
| S4 | 75 | 20 | 26.7% |
| S5 | 48 | 11 | 22.9% |
| S6 | 6 | 1 | 16.7% |

Interpretation:

* The highest-yield buckets are the cases where all GPT-5.5 variants agreed against the original
  gold label.
* This is strong evidence that model-assisted triage was useful, but it must be reported as a
  possible selection bias.
* The final gold decision must be presented as lexicographer-driven, not model-driven.

### SenseBench Benchmark

Use this repository as the source for benchmark and leaderboard facts.

Key facts for the paper:

* SenseBench evaluates WSD as candidate selection: a target token in context plus candidate WordNet
  senses.
* The main prompt family returns a `sense_index`; alternative prompts test shorter context and
  detokenized context.
* Each run records `run.json`, `predictions.jsonl`, and `calls.jsonl.gz`.
* The leaderboard aggregator supports strict validation and confidence intervals.
* Public leaderboard artifacts can be hosted through GitHub Pages.
* The repository registers `lexen-v1` as the default dataset release; official runs use `lexen-v1`.

Paper protocol to specify:

* Freeze dataset release, prompt version, model identifier, provider, decoding settings, and run
  date.
* Report accuracy, confidence intervals, total cost, call count, failed-call rate, and retry policy.
* Publish raw artifacts for every leaderboard row that is included in the main table.
* Treat any model that cannot be re-run or whose API semantics changed as a historical result.
* Use paired significance tests for close model comparisons.

### Literature Context

Use `/Users/vassiliphilippov/research-wsd/` as the source for related-work summaries.

Key narrative:

* Raganato et al. 2017 unified WSD evaluation by standardizing five English all-words test sets and
  WordNet 3.0 scoring.
* Maru et al. 2022 showed that the unified benchmark contained many annotation errors and created
  corrected subsets, including ALL_NEW and hardEN.
* GlossBERT, LMMS, BEM, ESCHER, and ConSeC moved WSD from feature-rich and embedding methods toward
  gloss-aware transformer systems.
* Recent systems such as SANDWiCH and fine-tuned Llama variants show that compact supervised or
  distilled systems remain competitive.
* Recent LLM papers show that general-purpose LLMs are strong, but still struggle on rare,
  fine-grained, domain-shifted, or defective examples.

Related-work table to build in the paper:

| Work | Main role in this paper |
| --- | --- |
| Raganato et al. 2017 | Unified English WSD evaluation baseline |
| Maru et al. 2022 | Hard-core analysis and corrected ALL_NEW source |
| Huang et al. 2019, GlossBERT | Context-gloss transformer framing |
| Loureiro and Jorge 2019, LMMS | Sense embeddings and full-coverage WSD |
| Blevins and Zettlemoyer 2020, BEM | Bi-encoder gloss-informed baseline |
| Barba et al. 2021, ESCHER | Extractive sense comprehension baseline |
| Barba et al. 2021, ConSeC | Continuous sense comprehension baseline |
| GuzmanOlivares et al. 2025, SANDWiCH | Recent public supervised SOTA reference |
| Meconi et al. 2025 | Broad LLM evaluation on WSD |
| Navigli 2026 | Positioning WSD as still useful in the LLM era |

## Provisional Results To Replace With Final Runs

These numbers are useful for paper planning, but the final paper should use the official `lexen-v1`
runs (now being produced for the public leaderboard). The numbers below were produced for the
deprecated `lexen-v0.1.0` draft and retrospectively rescored against Maru2022 and the interim
`lexen-v0.2.0` release.

| System | Prompt | Maru2022 | lexEN v0.2 primary | Delta |
| --- | --- | ---: | ---: | ---: |
| `gpt-5.5/xhigh` | `p001` | 90.787 | 92.963 | +2.176 |
| `gpt-5.5/high` | `p001` | 90.380 | 92.617 | +2.237 |
| `gpt-5.5/low` | `p001` | 90.360 | 92.516 | +2.156 |
| `gpt-5.5/medium` | `p001` | 90.279 | 92.495 | +2.217 |
| `gpt-5-mini/medium` | `p001` | 87.411 | 89.018 | +1.607 |
| `openrouter:google/gemma-4-26b-a4b-it` | `p001` | 86.292 | 87.675 | +1.383 |
| `gpt-4o-mini` | `p002` | 81.086 | 81.757 | +0.671 |
| ConSeC | baseline | 80.659 | 80.903 | +0.244 |
| ESCHER | baseline | 80.212 | 80.374 | +0.163 |
| BEM | baseline | 78.707 | 78.747 | +0.041 |
| MFS | baseline | 60.911 | 60.931 | +0.020 |

Interpretation for the results section:

* Correcting Maru2022 labels raises frontier LLM scores by about **2.2** points in the current
  retrospective analysis.
* The same correction barely moves older supervised baselines, with deltas from **0.02** to **0.24**
  points.
* This supports the central measurement claim: a few percent of label noise becomes material once
  systems exceed **90%** accuracy.
* `lexen-v0.1.0` gave even higher LLM scores, but `lexen-v0.2.0` is more defensible because it
  removes mixed-protocol review. Final headline claims should use `lexen-v1`.

## Correction Examples

Use examples that show different kinds of corrections rather than only easy wins.

| Item | Target | Maru2022 | lexEN v0.2 | Why it matters |
| --- | --- | --- | --- | --- |
| `senseval2.d000.s003.t009` | fields | `field%1:17:00::`, an extensive tract of level open land | `field%1:15:00::`, a cleared and usually enclosed piece of land | The church stands amid rural fields, not a plain. All models agreed with the lexEN label. |
| `senseval2.d000.s003.t014` | calling | `call%2:41:04::`, call a meeting | `call%2:32:05::`, order, request, or command to come | Bells are summoning the faithful to evensong, not convening a meeting. |
| `senseval2.d000.s031.t003` | solemn | `solemn%5:00:01:serious:00`, earnest or humorless belief | `solemn%5:00:00:serious:00`, dignified and somber | Shows fine-grained adjective confusion where the corrected label better matches ritual tone. |
| `senseval2.d002.s042.t002` | need | `need%2:34:01::`, have or feel a need for | `need%2:42:00::`, require as useful, just, or proper | "All we need to know" is about what is required, not a felt need. |
| `semeval2013.d011.s023.t002` | points | `point%1:10:03::` | Empty `lexen_gold`; primary scorer falls back to Maru2022 | Both lexicographers found the translated phrase unclear enough to mark it unanswerable. |
| `semeval2013.d006.s010.t003` | spirit | Maru2022 kept | Maru2022 kept after no consensus | Shows that the correction process is conservative when the source text is defective or disputed. |

## Paper Outline

### Abstract

State the high-accuracy measurement problem, the lexEN correction method, the SenseBench public
benchmark, the headline final LLM result, and the score-shift finding.

### Introduction

Main points:

* WSD has been treated as nearly mature, but LLMs make old evaluation noise visible again.
* Maru2022 improved on the Raganato benchmark, but frontier LLMs now expose residual errors in
  Maru2022 ALL_NEW.
* A leaderboard without a verified gold set risks ranking systems by annotation artifacts.
* The paper contributes both a corrected dataset layer and an open evaluation harness.

Suggested introduction arc:

1. WSD remains a useful diagnostic task because it requires lexical semantics, context use, and
   inventory-grounded choice.
2. Evaluation has improved from Raganato to Maru2022, but the target accuracy range has changed.
3. Once models reach or exceed **90%**, **2-4%** label noise is no longer a nuisance.
4. lexEN and SenseBench address this with reviewed labels and auditable public runs.

### Related Work

Organize by role rather than chronology:

* Unified WSD benchmarks and Maru2022 corrections.
* Gloss-aware transformer WSD systems.
* Recent supervised and distilled WSD systems.
* LLM-as-WSD systems and prompt sensitivity.
* Dataset verification and benchmark governance.

### Dataset Correction Method

Required details:

* Starting point: Maru2022 ALL_NEW, **4,917** items.
* Candidate senses: WordNet sense-key inventory inherited from the benchmark.
* Suspicious-item selection: 10-model panel with eight GPT-5.5 variants plus supervised systems.
* Bucket definitions S1-S6 and the waterfall counts.
* Lexicographer protocol, evidence shown, allowed decisions, and note handling.
* Final adjudication rule in `lexen-v1`, with the interim `lexen-v0.2.0` consensus rule described
  only as background if needed.
* Difference between `lexen_gold` and `lexen_primary_scorable`.

Important caveat:

* Because triage used frontier LLMs, the reviewed subset is not a random sample. The paper should
  state this openly and avoid claiming a corpus-wide Maru2022 error rate from the **363** reviewed
  items.

### Benchmark Protocol

Required details:

* Prompt variants and the main selected prompt.
* Exact model settings, retry policy, response parsing, and failure handling.
* Dataset canary and artifact verification.
* Confidence intervals and paired significance testing.
* Public leaderboard policy and model-version freeze date.

External model-list references to freeze near submission:

* [LMArena Text Leaderboard](https://arena.ai/leaderboard/text)
* [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/leaderboards/models)
* [OpenRouter Rankings](https://openrouter.ai/rankings)

### Results

Primary tables:

* Final `lexen-v1` leaderboard with all rerun top public and proprietary models.
* Classic WSD baselines on the same release.
* Prompt and cost ablations.
* Maru2022 versus lexEN score deltas.

Primary figures:

* Pipeline figure: Maru2022 to model triage to lexicographers to lexEN to SenseBench.
* Accuracy/cost Pareto plot.
* Score-change plot comparing classic systems and LLMs.
* Human agreement plot with exact and coarse agreement.

### Error And Agreement Analysis

Useful subsections:

* Clean Maru2022 label corrections.
* Fine-grained WordNet distinctions where humans disagree.
* Unanswerable or defective source text.
* Cases where models agree with each other but lexicographers reject the modal prediction.
* Cases where the corrected label helps one model family more than another.

### Limitations

Must include:

* Until `lexen-v1` is frozen, all `lexen-v0.2.0` numbers are interim because the third lexicographer
  result is pending.
* The audited subset was selected by a model panel and is therefore enriched for model-Maru2022
  disagreements.
* Unreviewed items inherit Maru2022 labels.
* WordNet fine-grained senses have known granularity and inter-annotator agreement limits.
* Proprietary model APIs can change; final results need run dates, model IDs, and artifact hashes.
* Public leaderboard benchmarks can be contaminated over time, so the paper should include canary
  and release-governance details.

## Work Required Before Submission

Implementation and release work:

* Register `lexen-v1` in SenseBench.
* Decide whether the main metric is `lexen_primary_scorable`, strict non-empty `lexen_gold`, or
  both.
* Rerun all headline LLMs on the final `lexen-v1` release.
* Add SANDWiCH predictions as a baseline if licensing and provenance allow.
* Add paired significance testing for top model comparisons.
* Publish release hashes for dataset, prompt files, and result artifacts.
* Add an artifact manifest that maps every paper table row to a result directory.

Writing work:

* Draft the dataset correction section first because it anchors the paper's credibility.
* Draft the benchmark protocol second so the leaderboard is reproducible.
* Keep the introduction focused on measurement, not on declaring WSD solved.
* Use four to six correction examples with glosses and lexicographer rationale.
* Add a short benchmark-governance section explaining how future model submissions will be handled.

Decision points:

* How the third-lexicographer adjudication rule should be represented in `lexen-v1`.
* Whether to keep the `lexen-v0.2.0` score-shift analysis as an appendix or replace it entirely with
  `lexen-v1` results.
* Whether unanswerable items should be excluded from a secondary strict metric.
* Which public leaderboard sources define "top models" at the final model-freeze date.
* Whether to submit as an ACL/EMNLP main paper, findings paper, or benchmark/resource paper.

## Recommended Claim Boundaries

Strong claims:

* lexEN provides a more defensible evaluation layer for high-accuracy English WSD than raw Maru2022
  ALL_NEW.
* SenseBench makes LLM WSD evaluation auditable through released prompts, raw artifacts, and strict
  validation.
* Residual label noise has a larger measured effect on frontier LLMs than on older supervised
  baselines in the current analysis.

Avoid or qualify:

* Do not claim a full corpus-wide Maru2022 error rate from the reviewed subset.
* Do not claim WSD is solved; the agreement and defective-text analysis argues the opposite.
* Do not call `v0.1.0` or `v0.2.0` headline SOTA; `lexen-v1` supersedes them.
* Do not treat model-assisted triage as independent evidence of gold correctness.
* Do not claim final model rankings until official `lexen-v1` runs are complete.
