# SenseBench Prompt Registry

SenseBench prompts are immutable JSON files that define how dataset items are rendered into LLM
calls. A prompt includes both the template text and the rendering parameters, such as context window
size, sense ordering, sense fields, and expected output format.

Registered prompt files live in:

```text
src/sensebench/prompts/
  registry.py
  render.py
  registered/
    p001.json
    p002.json
    p003.json
```

The `registered/` directory is the immutable prompt registry. The surrounding
`src/sensebench/prompts/` package contains code for loading, validating, and rendering those prompt
definitions.

## Design Principles

Prompts are part of the benchmark definition. Once a prompt is accepted into the registry, its
content must not change.

If any benchmark-relevant prompt content changes, create a new prompt ID.

Examples of benchmark-relevant changes:

* changing system or user message text
* changing output instructions
* changing the number of previous or next sentences
* changing sense order
* adding or removing WordNet IDs, definitions, examples, POS, synonyms, or other sense fields
* changing how candidate senses are rendered

Examples of non-benchmark-relevant changes:

* fixing documentation outside the prompt file
* changing dashboard display logic
* adding external commentary about an already registered prompt

Prompt identity is intentionally simple:

```text
p001
p002
p003
```

SenseBench does not use prompt versions. If a prompt changes, it receives a new ID. A prompt may
optionally point to an older prompt with `supersedes`.

## Prompt File Format

Each prompt is stored as one JSON file:

```text
src/sensebench/prompts/registered/p003.json
```

The filename must match the prompt ID.

Minimal example:

```json
{
  "schema_version": "sensebench-prompt-v1",
  "id": "p003",
  "name": "Definition + Examples + POS, Frequency Order",
  "description": "Shows WordNet ID, POS, definition, examples; frequency-ordered senses.",
  "template_type": "chat_messages",
  "template": {
    "messages": [
      {
        "role": "system",
        "content": "Choose exactly one candidate sense for the marked target word."
      },
      {
        "role": "user",
        "content": "Context: {{context}}\nSenses: {{candidate_senses}}\nJSON {\"sense_index\": 1}."
      }
    ]
  },
  "params": {
    "previous_sentences": 1,
    "next_sentences": 1,
    "target_marker": "xml_t",
    "sense_order": "frequency",
    "candidate_format": "sensebench_multiline",
    "include_wordnet_id": true,
    "wordnet_id_kind": "sense_key",
    "include_definition": true,
    "include_examples": true,
    "examples_max_per_sense": 2,
    "include_pos": true,
    "include_synonyms": false,
    "synonyms_max_per_sense": 0
  },
  "output": {
    "mode": "json_sense_index"
  }
}
```

## Required Fields

`schema_version`

Must be `"sensebench-prompt-v1"`.

`id`

Stable immutable prompt ID. It must match:

```text
^p[0-9]{3,}$
```

Examples:

```text
p001
p002
p125
```

`name`

Human-readable prompt name for documentation and dashboards. It does not define prompt identity. Use
names that expose the main comparable prompt factors, such as context window, sense evidence,
candidate order, target marking, or output mode. Avoid vague labels such as "rich" or names that
only describe project history.

`description`

Short explanation of what the prompt renders in NLP terms. It should be understandable without
knowing SenseBench implementation history or earlier experiments.

`template_type`

For v1, must be:

```json
"chat_messages"
```

The runner may serialize chat messages into a single text prompt for non-chat models, but the
registered prompt itself is stored as chat messages.

`template`

The prompt template. For `chat_messages`, this contains a `messages` array with ordered chat
messages.

Each message has:

* `role`: one of `system`, `user`, or `assistant`
* `content`: template text

Most prompts should use only `system` and `user`.

`params`

Rendering parameters that affect the prompt shown to the model.

`output`

Expected model output mode. SenseBench v1 supports exactly two modes:

```json
{"mode": "json_sense_index"}
```

or:

```json
{"mode": "plain_sense_index"}
```

## Optional Fields

`supersedes`

An older prompt ID that this prompt replaces or revises.

Example:

```json
"supersedes": "p003"
```

This is metadata only. It does not make two prompts equivalent.

`notes`

Short maintainer-facing notes about known limitations or intended comparisons.

## Template Variables

Template variables use double braces:

```text
{{context}}
{{target_lemma}}
{{candidate_senses}}
```

The v1 renderer supports these variables:

`{{context}}`

Rendered context around the target sentence. The number of included previous and next sentences is
controlled by `params.previous_sentences` and `params.next_sentences`. The target span inside the
context is marked according to `params.target_marker`.

`{{target_text}}`

Surface form of the target word or phrase as it appears in context.

`{{target_lemma}}`

Lemma of the target word.

`{{target_pos}}`

Part of speech of the target item.

`{{candidate_senses}}`

Rendered list of candidate senses. The content and order are controlled by `params`.

`{{item_id}}`

Dataset item ID. This is mainly useful for debugging prompts and should usually be omitted from
official prompts.

## Rendering Parameters

The `params` object defines how the renderer prepares item data before filling the template.

Required v1 parameters:

```json
{
  "previous_sentences": 1,
  "next_sentences": 1,
  "target_marker": "xml_t",
  "sense_order": "frequency",
  "candidate_format": "sensebench_multiline",
  "include_wordnet_id": true,
  "wordnet_id_kind": "sense_key",
  "include_definition": true,
  "include_examples": true,
  "examples_max_per_sense": 2,
  "include_pos": true,
  "include_synonyms": false,
  "synonyms_max_per_sense": 0
}
```

`previous_sentences`

Number of sentences before the target sentence to include. Must be an integer greater than or equal
to `0`.

`next_sentences`

Number of sentences after the target sentence to include. Must be an integer greater than or equal
to `0`.

`target_marker`

How the target span is marked inside `{{context}}`. Allowed values:

```text
none
xml_t
xml_wsd
xml_target
square_brackets
double_square_brackets
double_asterisk
```

The renderer maps these values as follows:

```text
none: bank
xml_t: <t>bank</t>
xml_wsd: <WSD>bank</WSD>
xml_target: <target>bank</target>
square_brackets: [bank]
double_square_brackets: [[bank]]
double_asterisk: **bank**
```

The marker must be applied to the dataset target span using offsets or token identity, not by string
replacement. If the same surface form appears multiple times in the context, only the target
instance is marked.

`sense_order`

Order of candidate senses. Allowed values:

```text
frequency
random_fixed
lexicographic
dataset
```

`frequency` means WordNet frequency order when available.

`random_fixed` means a deterministic benchmark-defined shuffle. The shuffle must be stable for a
given dataset version, prompt ID, and item ID.

`lexicographic` means sorted by canonical sense ID.

`dataset` means the order provided by the dataset loader.

`candidate_format`

How each candidate sense entry is rendered inside `{{candidate_senses}}`. Allowed values:

```text
sensebench_multiline
compact_labeled_inline
```

`sensebench_multiline` is the default readable benchmark format:

```text
1. sense_key=bank%1:14:00::
   Definition: ...
   Synonyms: ...
   Examples:
   * ...
```

`compact_labeled_inline` renders each candidate as one line with labeled fields:

```text
1. sense_key=bank%1:14:00:: | definition=... | synonyms=... | examples=...
```

`include_wordnet_id`

Whether candidate sense entries include WordNet sense IDs.

`wordnet_id_kind`

Which WordNet identifier to render when `include_wordnet_id` is `true`. Allowed values:

```text
none
sense_key
synset_id
```

Use `sense_key` when prompts expose WordNet lemma sense keys, and `synset_id` only for prompts that
explicitly ask the model to answer or reason in synset IDs.

`include_definition`

Whether candidate sense entries include definitions.

`include_examples`

Whether candidate sense entries include usage examples.

`examples_max_per_sense`

Maximum number of examples to include for each candidate sense. Must be `0` when `include_examples`
is `false`.

`include_pos`

Whether candidate sense entries include part of speech.

`include_synonyms`

Whether candidate sense entries include synonyms or lemma names. The target lemma itself is always
excluded from rendered synonyms.

`synonyms_max_per_sense`

Maximum number of synonyms or lemma names to include for each candidate sense. Must be `0` when
`include_synonyms` is `false`.

## Candidate Sense Indexing

Candidate senses shown to the model are always numbered starting from `1`.

Example rendered candidate list:

```text
1. wn:01234567-n
   POS: noun
   Definition: ...
   Examples:
   * ...

2. wn:08912345-n
   POS: noun
   Definition: ...
   Examples:
   * ...
```

The model-facing answer is the candidate sense index. The runner converts that index into a
canonical sense ID when creating run results.

## Output Modes

### `json_sense_index`

The model must return valid JSON with exactly one field:

```json
{"sense_index": 1}
```

Validation rules:

* output must be valid JSON after trimming whitespace
* root value must be an object
* object must contain exactly one key, `sense_index`
* `sense_index` must be an integer
* `sense_index` must be between `1` and the number of candidate senses for that item

The prompt text should explicitly say:

```text
Return only valid JSON in this exact format:
{"sense_index": 1}
```

### `plain_sense_index`

The model must return only the chosen candidate number:

```text
1
```

Validation rules:

* output is trimmed before parsing
* trimmed output must contain only an integer
* no prose, markdown, punctuation, JSON, or labels are allowed
* integer must be between `1` and the number of candidate senses for that item

The prompt text should explicitly say:

```text
Return only the number of the correct sense.
```

## Validation

The prompt Pydantic model lives at:

```text
src/sensebench/prompts/models.py
```

Validate all registered prompts with:

```bash
uv run python tools/verify_prompt.py --all
```

The prompt validator must check:

* file is valid JSON
* file is under `src/sensebench/prompts/registered/`
* filename matches `id`, for example `src/sensebench/prompts/registered/p003.json`
* `schema_version` is supported by `PromptDefinition`
* `id` matches the required pattern
* no duplicate prompt ID exists
* required fields are present
* no unknown top-level fields exist, except explicitly allowed optional fields
* `template_type` is supported
* all template messages have valid roles and non-empty content
* all template variables are known
* required `params` fields are present and valid
* `params.target_marker` is one of the supported marker values
* `output.mode` is one of the two supported modes
* prompt text includes output instructions consistent with `output.mode`

## Run File Reference

A completed run references a registered prompt by ID:

```json
{
  "prompt": {
    "id": "p003"
  }
}
```

The run validator must confirm that:

* the prompt ID exists in the registry
* the run was produced with the output mode declared by that prompt

## Recommended First Prompts

The initial registry should be small. A useful starting set:

`p001`

5+1 context prompt with `<t>...</t>` target marking, WordNet sense keys, definitions, synonyms,
examples, frequency-ordered candidates, and JSON `sense_index` output.

`p002`

Minimal single-sentence context, definitions and at most one usage example per sense, frequency
order, plain integer output. Registered.

`p003`

Previous and next sentence, definitions, examples, POS, WordNet IDs, frequency order, JSON output.

`p004`

Same fields as `p003`, but deterministic shuffled sense order.

This gives enough variation to study output format and sense-order effects without making the first
benchmark release hard to audit.
