# Glite coarse-sense mapping (vendored)

Maps WordNet 3.0 sense keys to Glite coarse concept ids. Used to score the "Glite coarse-grained"
leaderboard schemes: a prediction is coarse-correct when its mapped concept matches a gold key's
concept (or when it is already fine-grained correct).

Vendored verbatim from the lexEN release source (`lexen/sources/glite-coarsening/files/`), which is
also what produced the dataset's `lexen_gold_glite_concept_ids` metadata. Research-use; the mapping
rows contain no contamination canary.

| File | rows | sha256 |
| --- | ---: | --- |
| `wordnet_sense_key_to_glite_concept.jsonl` | 10,412 | `4fd14626838520eaa1a3168ab30166a708ab181b073a92df8d8bcb4cda2e4ba6` |
| `lexen_report_aliases.json` | 5 | `5ce96f30a772cb4753607e81700f45474ed3e3a1f9b24e363083b68c9c259428` |

`wordnet_sense_key_to_glite_concept.jsonl` rows:
`{"sense_key", "concept_id", "lemma", "wordnet_pos"}`. `lexen_report_aliases.json` maps
legacy/British-spelling sense keys to a canonical concept. Sense keys absent from both are reported
as `unmapped:<sense_key>` and match only an identical `unmapped:` token (per the lexEN coarsening
policy).
