from __future__ import annotations

from sensebench.datasets.context import build_dataset_index
from sensebench.datasets.models import (
    DatasetBundle,
    DatasetID,
    Document,
    DocumentID,
    ItemID,
    SenseKey,
    Sentence,
    SentenceID,
    Token,
    WsdItem,
)
from sensebench.paths import P001_PROMPT_PATH, PROMPT_JSON_SUFFIX, PROMPT_REGISTRY_DIR
from sensebench.prompts.models import TEMPLATE_VARIABLE_CONTEXT
from sensebench.prompts.registry import load_prompt_definition
from sensebench.prompts.render import _render_template, render_task
from sensebench.wordnet import SenseCandidate, SynsetID, WordNetPos

P003_PROMPT_PATH = PROMPT_REGISTRY_DIR / f"p003{PROMPT_JSON_SUFFIX}"
DETOKENIZED_ITEM_ID: ItemID = "i2"
DETOKENIZED_SENTENCE_ID: SentenceID = "s2"

DATASET_ID: DatasetID = "fixture"
DATASET_VERSION: str = "1"
DOCUMENT_ID: DocumentID = "d1"
SENTENCE_ID: SentenceID = "s1"
ITEM_ID: ItemID = "i1"
TARGET_TEXT: str = "bank"
TARGET_LEMMA: str = "bank"
TARGET_POS: str = "NOUN"
SENSE_KEY: SenseKey = "bank%1:17:00::"
SYNSET_ID: SynsetID = "bank.n.01"
DEFINITION: str = "sloping land"
SYNONYM: str = "slope"
EXAMPLE: str = "he sat on the bank"
CONTEXT_VALUE: str = "HELLO"
UNKNOWN_VARIABLE: str = "unknown"
DOLLAR_TEMPLATE: str = "It costs $5 or $context or ${context}, said {{context}}."
DOLLAR_TEMPLATE_RENDERED: str = "It costs $5 or $context or ${context}, said HELLO."


def _bundle() -> DatasetBundle:
    return DatasetBundle(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        dataset_revision=None,
        content_hash=None,
        documents=[
            Document(
                document_id=DOCUMENT_ID,
                sentences=[
                    Sentence(
                        sentence_id=SENTENCE_ID,
                        tokens=[
                            Token(text="The"),
                            Token(text=TARGET_TEXT, item_id=ITEM_ID),
                            Token(text="was"),
                            Token(text="steep"),
                        ],
                    )
                ],
            )
        ],
        items=[
            WsdItem(
                item_id=ITEM_ID,
                document_id=DOCUMENT_ID,
                sentence_id=SENTENCE_ID,
                target_token_index=1,
                target_text=TARGET_TEXT,
                lemma=TARGET_LEMMA,
                pos=TARGET_POS,
                gold_sense_keys=[SENSE_KEY],
            )
        ],
    )


def test_render_task_marks_target_and_formats_candidates() -> None:
    prompt = load_prompt_definition(
        path=P001_PROMPT_PATH,
    )
    bundle = _bundle()
    rendered = render_task(
        prompt=prompt,
        item=bundle.items[0],
        dataset_index=build_dataset_index(bundle=bundle),
        candidates=[
            SenseCandidate(
                sense_key=SENSE_KEY,
                synset_id=SYNSET_ID,
                pos=WordNetPos.NOUN,
                definition=DEFINITION,
                synonyms=[SYNONYM],
                examples=[EXAMPLE],
            )
        ],
    )

    user_message = rendered.messages[1].content
    assert f"<t>{TARGET_TEXT}</t>" in user_message
    assert f"sense_key={SENSE_KEY}" in user_message
    assert rendered.render_hash.startswith("sha256:")
    assert all("{{" not in message.content for message in rendered.messages)


def _detokenized_bundle() -> DatasetBundle:
    return DatasetBundle(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        dataset_revision=None,
        content_hash=None,
        documents=[
            Document(
                document_id=DOCUMENT_ID,
                sentences=[
                    Sentence(
                        sentence_id=DETOKENIZED_SENTENCE_ID,
                        tokens=[
                            Token(text="He"),
                            Token(text="reached"),
                            Token(text="the"),
                            Token(text=TARGET_TEXT, item_id=DETOKENIZED_ITEM_ID),
                            Token(text=","),
                            Token(text="he"),
                            Token(text="said"),
                            Token(text="."),
                        ],
                    )
                ],
            )
        ],
        items=[
            WsdItem(
                item_id=DETOKENIZED_ITEM_ID,
                document_id=DOCUMENT_ID,
                sentence_id=DETOKENIZED_SENTENCE_ID,
                target_token_index=3,
                target_text=TARGET_TEXT,
                lemma=TARGET_LEMMA,
                pos=TARGET_POS,
                gold_sense_keys=[SENSE_KEY],
            )
        ],
    )


def test_render_task_detokenizes_context_for_p003() -> None:
    prompt = load_prompt_definition(path=P003_PROMPT_PATH)
    bundle = _detokenized_bundle()
    rendered = render_task(
        prompt=prompt,
        item=bundle.items[0],
        dataset_index=build_dataset_index(bundle=bundle),
        candidates=[
            SenseCandidate(
                sense_key=SENSE_KEY,
                synset_id=SYNSET_ID,
                pos=WordNetPos.NOUN,
                definition=DEFINITION,
                synonyms=[SYNONYM],
                examples=[EXAMPLE],
            )
        ],
    )

    user_message = rendered.messages[1].content
    assert f"the <t>{TARGET_TEXT}</t>, he said." in user_message
    assert " ," not in user_message
    assert " ." not in user_message


def test_render_template_substitutes_every_validator_accepted_spelling() -> None:
    variables: dict[str, str] = {TEMPLATE_VARIABLE_CONTEXT: CONTEXT_VALUE}

    assert _render_template(content="{{context}}", variables=variables) == CONTEXT_VALUE
    assert _render_template(content="{{ context }}", variables=variables) == CONTEXT_VALUE
    assert _render_template(content="{{context }}", variables=variables) == CONTEXT_VALUE
    assert _render_template(content="{{ context}}", variables=variables) == CONTEXT_VALUE
    assert _render_template(content="{{  context  }}", variables=variables) == CONTEXT_VALUE


def test_render_template_leaves_dollar_text_and_unknown_variables_alone() -> None:
    variables: dict[str, str] = {TEMPLATE_VARIABLE_CONTEXT: CONTEXT_VALUE}

    rendered = _render_template(
        content=DOLLAR_TEMPLATE,
        variables=variables,
    )

    assert rendered == DOLLAR_TEMPLATE_RENDERED
    assert (
        _render_template(content=f"{{{{{UNKNOWN_VARIABLE}}}}}", variables=variables)
        == f"{{{{{UNKNOWN_VARIABLE}}}}}"
    )
