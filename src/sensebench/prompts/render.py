"""Deterministic prompt rendering."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from string import Template

from sensebench.datasets.context import ContextWindow, build_context_window
from sensebench.datasets.models import DatasetIndex, WsdItem
from sensebench.prompts.models import (
    CandidateFormat,
    MessageRole,
    OutputMode,
    PromptDefinition,
    SenseOrder,
    TargetMarker,
    WordNetIdKind,
)
from sensebench.wordnet import SenseCandidate

RANDOM_SEED_BYTES: int = 8


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class CandidateChoice:
    index: int
    sense_key: str
    synset_id: str


@dataclass(frozen=True, slots=True)
class RenderedTask:
    item_id: str
    prompt_id: str
    messages: list[ChatMessage]
    candidates: list[CandidateChoice]
    output_mode: OutputMode
    render_hash: str
    shuffle_seed: int | None
    context: ContextWindow


@dataclass(frozen=True, slots=True)
class OrderedCandidates:
    candidates: list[SenseCandidate]
    shuffle_seed: int | None


def _marker_text(*, marker: TargetMarker, target_text: str) -> str:
    if marker == TargetMarker.NONE:
        return target_text
    if marker == TargetMarker.XML_T:
        return f"<t>{target_text}</t>"
    if marker == TargetMarker.XML_WSD:
        return f"<WSD>{target_text}</WSD>"
    if marker == TargetMarker.XML_TARGET:
        return f"<target>{target_text}</target>"
    if marker == TargetMarker.SQUARE_BRACKETS:
        return f"[{target_text}]"
    if marker == TargetMarker.DOUBLE_SQUARE_BRACKETS:
        return f"[[{target_text}]]"
    if marker == TargetMarker.DOUBLE_ASTERISK:
        return f"**{target_text}**"
    raise ValueError(f"Unsupported target marker: {marker}")


def _marked_context(*, context: ContextWindow, marker: TargetMarker) -> str:
    target_text = context.text[context.target_start_char : context.target_end_char]
    return (
        context.text[: context.target_start_char]
        + _marker_text(marker=marker, target_text=target_text)
        + context.text[context.target_end_char :]
    )


def _shuffle_seed(*, prompt_id: str, item_id: str) -> int:
    digest = hashlib.sha256(f"{prompt_id}|{item_id}|sense_order".encode()).digest()
    return int.from_bytes(digest[:RANDOM_SEED_BYTES], byteorder="big", signed=False)


def _ordered_candidates(
    *,
    prompt: PromptDefinition,
    item: WsdItem,
    candidates: list[SenseCandidate],
) -> OrderedCandidates:
    order = prompt.params.sense_order
    if order in {SenseOrder.FREQUENCY, SenseOrder.DATASET}:
        return OrderedCandidates(candidates=list(candidates), shuffle_seed=None)
    if order == SenseOrder.LEXICOGRAPHIC:
        return OrderedCandidates(
            candidates=sorted(candidates, key=lambda candidate: candidate.sense_key),
            shuffle_seed=None,
        )
    if order == SenseOrder.RANDOM_FIXED:
        seed = _shuffle_seed(prompt_id=prompt.id, item_id=item.item_id)
        items: list[SenseCandidate] = list(candidates)
        random.Random(seed).shuffle(items)
        return OrderedCandidates(candidates=items, shuffle_seed=seed)
    raise ValueError(f"Unsupported sense order: {order}")


def _candidate_id_text(*, candidate: SenseCandidate, kind: WordNetIdKind) -> str | None:
    if kind == WordNetIdKind.NONE:
        return None
    if kind == WordNetIdKind.SENSE_KEY:
        return f"sense_key={candidate.sense_key}"
    if kind == WordNetIdKind.SYNSET_ID:
        return f"synset_id={candidate.synset_id}"
    raise ValueError(f"Unsupported WordNet id kind: {kind}")


def _candidate_parts(
    *,
    prompt: PromptDefinition,
    candidate: SenseCandidate,
) -> list[str]:
    parts: list[str] = []
    if prompt.params.include_wordnet_id:
        id_text = _candidate_id_text(candidate=candidate, kind=prompt.params.wordnet_id_kind)
        if id_text is not None:
            parts.append(id_text)
    if prompt.params.include_pos:
        parts.append(f"pos={candidate.pos}")
    if prompt.params.include_definition:
        parts.append(f"definition={candidate.definition}")
    if prompt.params.include_synonyms and len(candidate.synonyms) > 0:
        synonyms: list[str] = candidate.synonyms[: prompt.params.synonyms_max_per_sense]
        parts.append(f"synonyms={', '.join(synonyms)}")
    if prompt.params.include_examples and len(candidate.examples) > 0:
        examples: list[str] = candidate.examples[: prompt.params.examples_max_per_sense]
        parts.append(f"examples={'; '.join(examples)}")
    return parts


def _candidate_line(
    *,
    prompt: PromptDefinition,
    index: int,
    candidate: SenseCandidate,
) -> str:
    parts = _candidate_parts(prompt=prompt, candidate=candidate)
    if prompt.params.candidate_format == CandidateFormat.COMPACT_LABELED_INLINE:
        return f"{index}. " + " | ".join(parts)
    if prompt.params.candidate_format == CandidateFormat.SENSEBENCH_MULTILINE:
        lines: list[str] = [f"{index}."]
        lines.extend(f"   {part}" for part in parts)
        return "\n".join(lines)
    raise ValueError(f"Unsupported candidate format: {prompt.params.candidate_format}")


def _candidate_block(
    *,
    prompt: PromptDefinition,
    ordered_candidates: list[SenseCandidate],
) -> str:
    return "\n".join(
        _candidate_line(prompt=prompt, index=index, candidate=candidate)
        for index, candidate in enumerate(ordered_candidates, start=1)
    )


def _render_template(*, content: str, variables: dict[str, str]) -> str:
    template_text = content
    for variable_name in variables:
        template_text = template_text.replace(
            "{{" + variable_name + "}}", "${" + variable_name + "}"
        )
        template_text = template_text.replace(
            "{{ " + variable_name + " }}",
            "${" + variable_name + "}",
        )
    return Template(template_text).safe_substitute(variables)


def _render_hash(*, messages: list[ChatMessage]) -> str:
    payload: list[dict[str, str]] = [
        {
            "role": message.role.value,
            "content": message.content,
        }
        for message in messages
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def render_task(
    *,
    prompt: PromptDefinition,
    item: WsdItem,
    dataset_index: DatasetIndex,
    candidates: list[SenseCandidate],
) -> RenderedTask:
    context = build_context_window(
        index=dataset_index,
        item=item,
        previous_sentences=prompt.params.previous_sentences,
        next_sentences=prompt.params.next_sentences,
    )
    ordered = _ordered_candidates(
        prompt=prompt,
        item=item,
        candidates=candidates,
    )
    candidate_block = _candidate_block(prompt=prompt, ordered_candidates=ordered.candidates)
    variables: dict[str, str] = {
        "candidate_senses": candidate_block,
        "context": _marked_context(context=context, marker=prompt.params.target_marker),
        "item_id": item.item_id,
        "target_lemma": item.lemma,
        "target_pos": item.pos,
        "target_text": item.target_text,
    }
    messages: list[ChatMessage] = [
        ChatMessage(
            role=message.role,
            content=_render_template(content=message.content, variables=variables),
        )
        for message in prompt.template.messages
    ]
    choices: list[CandidateChoice] = [
        CandidateChoice(
            index=index,
            sense_key=candidate.sense_key,
            synset_id=candidate.synset_id,
        )
        for index, candidate in enumerate(ordered.candidates, start=1)
    ]
    return RenderedTask(
        item_id=item.item_id,
        prompt_id=prompt.id,
        messages=messages,
        candidates=choices,
        output_mode=prompt.output.mode,
        render_hash=_render_hash(messages=messages),
        shuffle_seed=ordered.shuffle_seed,
        context=context,
    )
