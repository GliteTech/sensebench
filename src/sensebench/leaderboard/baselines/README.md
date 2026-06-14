# Baseline prediction files

Per-item predictions of classic (pre-LLM) WSD systems on the Raganato et al. (2017) unified
evaluation framework instances, in `instance_id sense_key` format (one prediction per line, 7,253
instances covering Senseval-2, Senseval-3, SemEval-2007, SemEval-2013, and SemEval-2015). lexEN
items keep the same instance IDs, so the site build scores these predictions on exactly the lexEN
item subset, using the same correctness rule as LLM runs.

Provenance:

* `bem.key.txt` — BEM (Blevins & Zettlemoyer 2020), predictions as released by Maru et al. 2022
  ("Nibbling at the Hard Core of Word Sense Disambiguation", ACL 2022,
  <https://github.com/SapienzaNLP/wsd-hard-benchmark>).
* `escher.key.txt` — ESCHER (Barba et al. 2021), reproduced by Glite from the official checkpoint.
* `consec.key.txt` — ConSeC (Barba et al. 2021), reproduced by Glite (SemCor + WordNet
  Gloss+Examples training; 82.9 F1 on Raganato ALL, −0.3 of the published 83.2 F1).

The MFS (most frequent sense) baseline is not a file: it is computed at build time as WordNet 3.0's
first (frequency-ranked) sense for each item's lemma and part of speech.
