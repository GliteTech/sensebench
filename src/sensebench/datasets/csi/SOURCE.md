# CSI coarse-sense mapping (vendored)

Maps WordNet 3.0 sense keys to **CSI** (Coarse Sense Inventory) composite concept ids. Used to score
the "CSI coarse-grained" leaderboard schemes: a prediction is coarse-correct when its mapped concept
matches a gold key's concept (or when it is already fine-grained correct) — identical scoring contract
to the Glite coarse schemes, with the CSI map swapped in.

CSI is the 45-domain inventory of **Lacerra, Bevilacqua, Pasini & Navigli (AAAI 2020)**,
<https://sapienzanlp.github.io/csi/>, **licensed CC-BY-NC-SA 4.0**. This vendored subset is derived
from the released `wn_synset2csi.txt` (WordNet 3.0 synset offsets): each synset is mapped to a single
*composite* concept `csi:<sorted+joined domain labels>` (multi-domain synsets reduced to one class, a
conservative partition). Distributed under the same CC-BY-NC-SA 4.0 terms with attribution. The map is
byte-identical to the lexEN add-on layer (`coarsenings/csi/files/...`), same SHA-256.

| File | rows | sha256 |
| --- | ---: | --- |
| `wordnet_sense_key_to_csi_concept.jsonl` | 7,783 (413 concepts) | `b84ee722e7cb6df6ad86e2fe55985e736b7d28e5871d6baef23e9f754f2b5e2a` |
| `csi_aliases.json` | 5 | `fb4996937b97e866a6741f71a1814874f79838819f961ae70e2667ccce02fd3a` |

`wordnet_sense_key_to_csi_concept.jsonl` rows: `{"sense_key", "concept_id", "lemma", "wordnet_pos"}`.
`csi_aliases.json` maps legacy/British-spelling sense keys to a canonical concept. Sense keys absent
from both are reported as `unmapped:<sense_key>` and match only an identical `unmapped:` token. CSI
covers 79.2% of the referenced keys; the remainder are `unmapped:` and never silently merged.
