from __future__ import annotations

from asyncio import run as run_async
from json import dumps
from pathlib import Path

from sensebench.datasets.models import ItemID
from sensebench.prompts.models import SENSE_INDEX_FIELD, MessageRole
from sensebench.runner.client import CompletionClient
from sensebench.runner.models import CompletionRequest, CompletionResult
from sensebench.runner.run import RunConfig, _model_with_resolved_snapshots, run_benchmark
from sensebench.runs.models import (
    CLOUD_LLM_KIND,
    RUN_SCHEMA_VERSION,
    AttemptKind,
    CallID,
    CallRecord,
    CallStatus,
    CloudLlmReference,
    CostBreakdown,
    CostSourceKind,
    MachineInfo,
    MessageRecord,
    ModelReference,
    ModelSourceKind,
    RunnerIdentity,
    SamplingParameters,
    TokenUsage,
)
from tests.run_fixtures import (
    FIRST_SENSE_KEY,
    FIXTURE_HOURLY_RATE_USD,
    RUNNER_GITHUB_HANDLE,
    fixture_machine,
    registered_prompt,
    renderable_dataset,
    self_hosted_model,
)

CALL_ID_1: CallID = "call-1"
CALL_ID_2: CallID = "call-2"
CALL_ID_3: CallID = "call-3"
ITEM_ID: ItemID = "item-1"
REQUESTED_MODEL: str = "gpt-5.5"
MODEL_DISPLAY_NAME: str = "GPT-5.5"
MODEL_SNAPSHOT_A: str = "gpt-5.5-2026-04-23"
MODEL_SNAPSHOT_B: str = "gpt-5.5-2026-05-01"
MESSAGE_CONTENT: str = "choose"


def _cloud_reference() -> CloudLlmReference:
    return CloudLlmReference(
        kind=CLOUD_LLM_KIND,
        display_name=MODEL_DISPLAY_NAME,
        requested_model=REQUESTED_MODEL,
        source_kind=ModelSourceKind.PROPRIETARY,
    )


def _call(
    *,
    call_id: CallID,
    model: str,
    status: CallStatus = CallStatus.SUCCESS,
) -> CallRecord:
    return CallRecord(
        call_id=call_id,
        item_id=ITEM_ID,
        vote_index=1,
        attempt_index=1,
        attempt_kind=AttemptKind.INITIAL,
        transport_retry_count=0,
        status=status,
        model=model,
        messages=[MessageRecord(role=MessageRole.USER, content=MESSAGE_CONTENT)],
        usage=TokenUsage(),
        cost=CostBreakdown(total_usd=None, source=CostSourceKind.UNAVAILABLE),
    )


def test_model_with_resolved_snapshots_sets_single_resolved_model() -> None:
    updated_model = _model_with_resolved_snapshots(
        model=_cloud_reference(),
        calls=[
            _call(call_id=CALL_ID_1, model=MODEL_SNAPSHOT_A),
            _call(call_id=CALL_ID_2, model=MODEL_SNAPSHOT_A),
            _call(
                call_id=CALL_ID_3,
                model=REQUESTED_MODEL,
                status=CallStatus.TRANSPORT_ERROR,
            ),
        ],
    )

    assert isinstance(updated_model, CloudLlmReference)
    assert updated_model.resolved_model == MODEL_SNAPSHOT_A
    assert updated_model.resolved_model_counts == {MODEL_SNAPSHOT_A: 2}


def test_model_with_resolved_snapshots_keeps_distribution_for_mixed_models() -> None:
    updated_model = _model_with_resolved_snapshots(
        model=_cloud_reference(),
        calls=[
            _call(call_id=CALL_ID_1, model=MODEL_SNAPSHOT_A),
            _call(call_id=CALL_ID_2, model=MODEL_SNAPSHOT_B),
        ],
    )

    assert isinstance(updated_model, CloudLlmReference)
    assert updated_model.resolved_model is None
    assert updated_model.resolved_model_counts == {
        MODEL_SNAPSHOT_A: 1,
        MODEL_SNAPSHOT_B: 1,
    }


class _CountingFakeClient(CompletionClient):
    def __init__(self) -> None:
        self.request_call_ids: list[CallID] = []

    async def complete(self, *, request: CompletionRequest) -> CompletionResult:
        self.request_call_ids.append(request.call_id)
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
                raw_output=dumps({SENSE_INDEX_FIELD: 1}),
                usage=TokenUsage(input_tokens=1, output_tokens=1),
                cost=CostBreakdown(source=CostSourceKind.UNAVAILABLE),
                latency_seconds=0.0,
            )
        )


def _run_config(
    *,
    tmp_path: Path,
    model: ModelReference,
    machine: MachineInfo | None,
    warmup_calls: int,
) -> RunConfig:
    return RunConfig(
        run_id="run-1",
        output_root=tmp_path,
        dataset=renderable_dataset(gold_sense_keys=[FIRST_SENSE_KEY]),
        prompt=registered_prompt(),
        model=model,
        runner=RunnerIdentity(github_handle=RUNNER_GITHUB_HANDLE),
        sampling=SamplingParameters(),
        votes_per_item=1,
        semantic_reasks_per_invalid_vote=1,
        concurrency=2,
        machine=machine,
        warmup_calls=warmup_calls,
        show_progress=False,
    )


def test_run_benchmark_records_execution_and_machine_time_cost(tmp_path: Path) -> None:
    client = _CountingFakeClient()
    config = _run_config(
        tmp_path=tmp_path,
        model=self_hosted_model(),
        machine=fixture_machine(),
        warmup_calls=2,
    )

    completed = run_async(run_benchmark(config=config, client=client))

    metadata = completed.metadata
    assert metadata.schema_version == RUN_SCHEMA_VERSION
    assert metadata.execution is not None
    assert metadata.execution.concurrency == 2
    assert metadata.execution.warmup_call_count == 2
    timing = metadata.execution.timing
    assert timing.benchmark_seconds > 0
    assert timing.setup_seconds is not None
    assert timing.benchmark_ended_at >= timing.benchmark_started_at
    assert metadata.totals.elapsed_seconds == timing.benchmark_seconds
    assert metadata.machine == fixture_machine()
    warmup_ids = [call_id for call_id in client.request_call_ids if call_id.startswith("warmup")]
    assert len(warmup_ids) == 2
    recorded_call_ids = {call.call_id for call in completed.calls}
    assert all(call_id not in recorded_call_ids for call_id in warmup_ids)
    cost = metadata.totals.cost
    assert cost.source == CostSourceKind.MACHINE_TIME_ESTIMATE
    assert cost.total_usd == timing.benchmark_seconds * FIXTURE_HOURLY_RATE_USD / 3600.0


def test_run_benchmark_without_hourly_rate_keeps_call_costs(tmp_path: Path) -> None:
    client = _CountingFakeClient()
    machine = fixture_machine().model_copy(update={"hourly_rate_usd": None})
    config = _run_config(
        tmp_path=tmp_path,
        model=self_hosted_model(),
        machine=machine,
        warmup_calls=0,
    )

    completed = run_async(run_benchmark(config=config, client=client))

    assert completed.metadata.totals.cost.source == CostSourceKind.UNAVAILABLE
    assert completed.metadata.execution is not None
    assert completed.metadata.execution.warmup_call_count == 0


def test_run_benchmark_cloud_run_has_no_machine_section(tmp_path: Path) -> None:
    client = _CountingFakeClient()
    config = _run_config(
        tmp_path=tmp_path,
        model=_cloud_reference(),
        machine=None,
        warmup_calls=0,
    )

    completed = run_async(run_benchmark(config=config, client=client))

    assert completed.metadata.machine is None
    assert completed.metadata.execution is not None
