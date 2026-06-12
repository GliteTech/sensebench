"""Internal runner dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field

from sensebench.prompts.render import ChatMessage, RenderedTask
from sensebench.runs.models import AttemptKind, CallRecord, PredictionRecord


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    call_id: str
    item_id: str
    vote_index: int
    attempt_index: int
    attempt_kind: AttemptKind
    model: str
    messages: list[ChatMessage]
    parameters: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompletionResult:
    call: CallRecord


@dataclass(frozen=True, slots=True)
class ItemEvaluation:
    prediction: PredictionRecord
    calls: list[CallRecord]
    rendered: RenderedTask
