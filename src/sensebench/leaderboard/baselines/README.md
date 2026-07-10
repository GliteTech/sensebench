# Baseline prediction files

Per-item predictions of reference WSD systems on the Raganato et al. (2017) unified evaluation
framework instances, in `instance_id sense_key` format (one prediction per line, 7,253 instances
covering Senseval-2, Senseval-3, SemEval-2007, SemEval-2013, and SemEval-2015). lexEN items keep the
same instance IDs, so the site build scores these predictions on exactly the lexEN item subset,
using the same correctness rule as LLM runs.

Provenance:

* `bem.key.txt` — BEM (Blevins & Zettlemoyer 2020), predictions as released by Maru et al. 2022
  ("Nibbling at the Hard Core of Word Sense Disambiguation", ACL 2022,
  <https://github.com/SapienzaNLP/wsd-hard-benchmark>).
* `escher.key.txt` — ESCHER (Barba et al. 2021), reproduced by Glite (SemCor training; 79.6 F1 on
  Raganato ALL, −1.1 of the published 80.7 F1).
* `consec.key.txt` — ConSeC (Barba et al. 2021), reproduced by Glite (SemCor + WordNet
  Gloss+Examples training; 82.9 F1 on Raganato ALL, −0.3 of the published 83.2 F1).
* `glite_lens.key.txt` — Glite LENS, published by Glite (ModernBERT bi-encoder trained on
  SemCor-GPT5.5; shipped seed-42 predictions score 83.7 F1 on Raganato ALL, with a 3-seed mean of
  83.6 F1).

The MFS (most frequent sense) baseline is not a file: it is computed at build time as WordNet 3.0's
first (frequency-ranked) sense for each item's lemma and part of speech.

**Training regimes differ.** Each baseline uses its strongest available faithful Glite reproduction,
but they were not all trained on the same data: ESCHER is SemCor-only while ConSeC is SemCor +
WordNet Gloss+Examples (its paper-best configuration). The ConSeC−ESCHER margin on the leaderboard
therefore reflects this training-data difference on top of the architecture; in the original papers
the matched-training gap is ~1.3 F1 (SemCor) to ~1.6 F1 (+WNGE) on Raganato ALL. Glite LENS is
different again: it is trained on SemCor-GPT5.5, the GPT-5.5-relabeled SemCor layer, so its row
demonstrates the relabel-and-retrain result rather than a classic system trained on the original
corpus. Because those training labels share a model family with the lexEN triage, its lexEN score is
confirmatory under the paper's Section 6.4 rule; Raganato and Maru carry the claim.
