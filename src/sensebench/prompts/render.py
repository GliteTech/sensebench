"""Deterministic prompt rendering."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from random import Random
from re import Match
from typing import assert_never

from sensebench.datasets.context import ContextWindow, build_context_window
from sensebench.datasets.models import DatasetIndex, ItemID, SenseKey, WsdItem
from sensebench.prompts.models import (
    TEMPLATE_VARIABLE_CANDIDATE_SENSES,
    TEMPLATE_VARIABLE_CONTEXT,
    TEMPLATE_VARIABLE_ITEM_ID,
    TEMPLATE_VARIABLE_PATTERN,
    TEMPLATE_VARIABLE_TARGET_LEMMA,
    TEMPLATE_VARIABLE_TARGET_POS,
    TEMPLATE_VARIABLE_TARGET_TEXT,
    CandidateFormat,
    MessageRole,
    OutputMode,
    PromptDefinition,
    PromptID,
    SenseOrder,
    TargetMarker,
    WordNetIdKind,
)
from sensebench.wordnet import SenseCandidate, SynsetID

RANDOM_SEED_BYTES: int = 8
SHUFFLE_SEED_CONTEXT: str = "sense_order"
RENDER_HASH_PREFIX: str = "sha256:"
RENDER_HASH_ROLE_FIELD: str = "role"
RENDER_HASH_CONTENT_FIELD: str = "content"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class CandidateChoice:
    index: int
    sense_key: SenseKey
    synset_id: SynsetID


@dataclass(frozen=True, slots=True)
class RenderedTask:
    item_id: ItemID
    prompt_id: PromptID
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
    match marker:
        case TargetMarker.NONE:
            return target_text
        case TargetMarker.XML_T:
            return f"<t>{target_text}</t>"
        case TargetMarker.XML_WSD:
            return f"<WSD>{target_text}</WSD>"
        case TargetMarker.XML_TARGET:
            return f"<target>{target_text}</target>"
        case TargetMarker.SQUARE_BRACKETS:
            return f"[{target_text}]"
        case TargetMarker.DOUBLE_SQUARE_BRACKETS:
            return f"[[{target_text}]]"
        case TargetMarker.DOUBLE_ASTERISK:
            return f"**{target_text}**"
        case _:
            assert_never(marker)


def _marked_context(*, context: ContextWindow, marker: TargetMarker) -> str:
    target_text = context.text[context.target_start_char : context.target_end_char]
    return (
        context.text[: context.target_start_char]
        + _marker_text(marker=marker, target_text=target_text)
        + context.text[context.target_end_char :]
    )


def _shuffle_seed(*, prompt_id: PromptID, item_id: ItemID) -> int:
    digest = sha256(f"{prompt_id}|{item_id}|{SHUFFLE_SEED_CONTEXT}".encode()).digest()
    return int.from_bytes(bytes=digest[:RANDOM_SEED_BYTES], byteorder="big", signed=False)


def vote_shuffle_seed(*, prompt_id: PromptID, item_id: ItemID, vote_index: int) -> int:
    """A reproducible shuffle seed that varies per vote (permutation self-consistency)."""
    digest = sha256(
        f"{prompt_id}|{item_id}|{SHUFFLE_SEED_CONTEXT}|vote{vote_index}".encode()
    ).digest()
    return int.from_bytes(bytes=digest[:RANDOM_SEED_BYTES], byteorder="big", signed=False)


def _ordered_candidates(
    *,
    prompt: PromptDefinition,
    item: WsdItem,
    candidates: list[SenseCandidate],
    shuffle_seed_override: int | None = None,
) -> OrderedCandidates:
    if shuffle_seed_override is not None:
        shuffled: list[SenseCandidate] = list(candidates)
        Random(shuffle_seed_override).shuffle(shuffled)
        return OrderedCandidates(candidates=shuffled, shuffle_seed=shuffle_seed_override)
    order = prompt.params.sense_order
    match order:
        case SenseOrder.FREQUENCY | SenseOrder.DATASET:
            return OrderedCandidates(candidates=list(candidates), shuffle_seed=None)
        case SenseOrder.LEXICOGRAPHIC:
            return OrderedCandidates(
                candidates=sorted(candidates, key=lambda candidate: candidate.sense_key),
                shuffle_seed=None,
            )
        case SenseOrder.RANDOM_FIXED:
            seed = _shuffle_seed(prompt_id=prompt.id, item_id=item.item_id)
            items: list[SenseCandidate] = list(candidates)
            Random(seed).shuffle(items)
            return OrderedCandidates(candidates=items, shuffle_seed=seed)
        case _:
            assert_never(order)


def _candidate_id_text(*, candidate: SenseCandidate, kind: WordNetIdKind) -> str | None:
    match kind:
        case WordNetIdKind.NONE:
            return None
        case WordNetIdKind.SENSE_KEY:
            return f"sense_key={candidate.sense_key}"
        case WordNetIdKind.SYNSET_ID:
            return f"synset_id={candidate.synset_id}"
        case _:
            assert_never(kind)


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
        parts.append(f"pos={candidate.pos.value}")
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
    candidate_format = prompt.params.candidate_format
    match candidate_format:
        case CandidateFormat.COMPACT_LABELED_INLINE:
            return f"{index}. " + " | ".join(parts)
        case CandidateFormat.SENSEBENCH_MULTILINE:
            lines: list[str] = [f"{index}."]
            lines.extend(f"   {part}" for part in parts)
            return "\n".join(lines)
        case _:
            assert_never(candidate_format)


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
    def _substitute(match: Match[str]) -> str:
        variable_name = match.group(1)
        replacement = variables.get(variable_name)
        if replacement is None:
            return match.group(0)
        return replacement

    return TEMPLATE_VARIABLE_PATTERN.sub(repl=_substitute, string=content)


def _render_hash(*, messages: list[ChatMessage]) -> str:
    payload: list[dict[str, str]] = [
        {
            RENDER_HASH_ROLE_FIELD: message.role.value,
            RENDER_HASH_CONTENT_FIELD: message.content,
        }
        for message in messages
    ]
    raw = dumps(obj=payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return RENDER_HASH_PREFIX + sha256(raw.encode("utf-8")).hexdigest()


def render_task(
    *,
    prompt: PromptDefinition,
    item: WsdItem,
    dataset_index: DatasetIndex,
    candidates: list[SenseCandidate],
    shuffle_seed: int | None = None,
) -> RenderedTask:
    context = build_context_window(
        index=dataset_index,
        item=item,
        previous_sentences=prompt.params.previous_sentences,
        next_sentences=prompt.params.next_sentences,
        detokenize=prompt.params.detokenize,
    )
    ordered = _ordered_candidates(
        prompt=prompt,
        item=item,
        candidates=candidates,
        shuffle_seed_override=shuffle_seed,
    )
    candidate_block = _candidate_block(prompt=prompt, ordered_candidates=ordered.candidates)
    variables: dict[str, str] = {
        TEMPLATE_VARIABLE_CANDIDATE_SENSES: candidate_block,
        TEMPLATE_VARIABLE_CONTEXT: _marked_context(
            context=context,
            marker=prompt.params.target_marker,
        ),
        TEMPLATE_VARIABLE_ITEM_ID: item.item_id,
        TEMPLATE_VARIABLE_TARGET_LEMMA: item.lemma,
        TEMPLATE_VARIABLE_TARGET_POS: item.pos,
        TEMPLATE_VARIABLE_TARGET_TEXT: item.target_text,
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
