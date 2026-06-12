from __future__ import annotations

from sensebench.datasets.context import build_dataset_index
from sensebench.datasets.models import DatasetBundle, Document, Sentence, Token, WsdItem
from sensebench.paths import PROMPT_REGISTRY_DIR
from sensebench.prompts.registry import load_prompt_definition
from sensebench.prompts.render import _render_template, render_task
from sensebench.wordnet import SenseCandidate


def _bundle() -> DatasetBundle:
    return DatasetBundle(
        dataset_id="fixture",
        dataset_version="1",
        dataset_revision=None,
        content_hash=None,
        documents=[
            Document(
                document_id="d1",
                sentences=[
                    Sentence(
                        sentence_id="s1",
                        tokens=[
                            Token(text="The"),
                            Token(text="bank", item_id="i1"),
                            Token(text="was"),
                            Token(text="steep"),
                        ],
                    )
                ],
            )
        ],
        items=[
            WsdItem(
                item_id="i1",
                document_id="d1",
                sentence_id="s1",
                target_token_index=1,
                target_text="bank",
                lemma="bank",
                pos="NOUN",
                gold_sense_keys=["bank%1:17:00::"],
            )
        ],
    )


def test_render_task_marks_target_and_formats_candidates() -> None:
    prompt = load_prompt_definition(
        path=PROMPT_REGISTRY_DIR / "p001.json",
    )
    bundle = _bundle()
    rendered = render_task(
        prompt=prompt,
        item=bundle.items[0],
        dataset_index=build_dataset_index(bundle=bundle),
        candidates=[
            SenseCandidate(
                sense_key="bank%1:17:00::",
                synset_id="bank.n.01",
                pos="n",
                definition="sloping land",
                synonyms=["slope"],
                examples=["he sat on the bank"],
            )
        ],
    )

    user_message = rendered.messages[1].content
    assert "<t>bank</t>" in user_message
    assert "sense_key=bank%1:17:00::" in user_message
    assert rendered.render_hash.startswith("sha256:")
    assert all("{{" not in message.content for message in rendered.messages)


def test_render_template_substitutes_every_validator_accepted_spelling() -> None:
    variables = {"context": "HELLO"}

    assert _render_template(content="{{context}}", variables=variables) == "HELLO"
    assert _render_template(content="{{ context }}", variables=variables) == "HELLO"
    assert _render_template(content="{{context }}", variables=variables) == "HELLO"
    assert _render_template(content="{{ context}}", variables=variables) == "HELLO"
    assert _render_template(content="{{  context  }}", variables=variables) == "HELLO"


def test_render_template_leaves_dollar_text_and_unknown_variables_alone() -> None:
    variables = {"context": "HELLO"}

    rendered = _render_template(
        content="It costs $5 or $context or ${context}, said {{context}}.",
        variables=variables,
    )

    assert rendered == "It costs $5 or $context or ${context}, said HELLO."
    assert _render_template(content="{{unknown}}", variables=variables) == "{{unknown}}"
