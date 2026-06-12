from __future__ import annotations

import asyncio

from sensebench.datasets.context import ContextWindow
from sensebench.prompts.models import MessageRole, OutputMode
from sensebench.prompts.render import CandidateChoice, ChatMessage, RenderedTask
from sensebench.runner.client import CompletionClient
from sensebench.runner.evaluate import EvaluationConfig, evaluate_item
from sensebench.runner.models import CompletionRequest, CompletionResult
from sensebench.runs.models import AttemptKind, CallRecord, CallStatus, MessageRecord, TokenUsage


class FakeClient(CompletionClient):
    def __init__(self, *, outputs: list[str]) -> None:
        self.outputs: list[str] = list(outputs)

    async def complete(self, *, request: CompletionRequest) -> CompletionResult:
        output = self.outputs.pop(0)
        return CompletionResult(
            call=CallRecord(
                call_id=request.call_id,
                item_id=request.item_id,
                vote_index=request.vote_index,
                attempt_index=request.attempt_index,
                attempt_kind=request.attempt_kind,
                transport_retry_count=0,
                status=CallStatus.SUCCESS,
                model=request.model,
                messages=[
                    MessageRecord(role=message.role, content=message.content)
                    for message in request.messages
                ],
                raw_output=output,
                usage=TokenUsage(input_tokens=1, cached_input_tokens=None, output_tokens=1),
                cost_usd=0.01,
                latency_seconds=0.1,
            )
        )


def _rendered(*, candidate_count: int) -> RenderedTask:
    candidates: list[CandidateChoice] = [
        CandidateChoice(index=index, sense_key=f"sense-{index}", synset_id=f"synset-{index}")
        for index in range(1, candidate_count + 1)
    ]
    return RenderedTask(
        item_id="i1",
        prompt_id="p001",
        messages=[ChatMessage(role=MessageRole.USER, content="choose")],
        candidates=candidates,
        output_mode=OutputMode.JSON_SENSE_INDEX,
        render_hash="sha256:test",
        shuffle_seed=None,
        context=ContextWindow(
            text="x",
            target_start_char=0,
            target_end_char=1,
            sentences_before=0,
            sentences_after=0,
        ),
    )


def test_invalid_output_gets_one_semantic_reask() -> None:
    evaluation = asyncio.run(
        evaluate_item(
            rendered=_rendered(candidate_count=2),
            gold_sense_keys=["sense-2"],
            client=FakeClient(outputs=["not json", '{"sense_index": 2}']),
            config=EvaluationConfig(model="fake"),
        )
    )

    assert evaluation.prediction.predicted_sense_key == "sense-2"
    assert len(evaluation.calls) == 2
    assert evaluation.calls[1].attempt_kind == AttemptKind.SEMANTIC_REASK


def test_majority_tie_uses_earliest_vote() -> None:
    evaluation = asyncio.run(
        evaluate_item(
            rendered=_rendered(candidate_count=2),
            gold_sense_keys=["sense-2"],
            client=FakeClient(outputs=['{"sense_index": 2}', '{"sense_index": 1}']),
            config=EvaluationConfig(model="fake", votes_per_item=2),
        )
    )

    assert evaluation.prediction.predicted_sense_key == "sense-2"


def test_monosemous_item_short_circuits_without_calls() -> None:
    evaluation = asyncio.run(
        evaluate_item(
            rendered=_rendered(candidate_count=1),
            gold_sense_keys=["sense-1"],
            client=FakeClient(outputs=[]),
            config=EvaluationConfig(model="fake"),
        )
    )

    assert evaluation.prediction.was_monosemous is True
    assert len(evaluation.calls) == 0
