"""Run SenseBench evaluations."""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from subprocess import run

from tqdm import tqdm

from sensebench import __version__
from sensebench.datasets.context import build_dataset_index
from sensebench.datasets.models import DatasetBundle, DatasetIndex, WsdItem
from sensebench.prompts.models import MessageRole, PromptDefinition
from sensebench.prompts.render import ChatMessage, render_task
from sensebench.runner.client import CompletionClient
from sensebench.runner.costs import sum_costs
from sensebench.runner.evaluate import EvaluationConfig, evaluate_item
from sensebench.runner.models import CompletionRequest, ItemEvaluation
from sensebench.runner.writer import write_run_artifacts
from sensebench.runs.models import (
    CLOUD_LLM_KIND,
    RUN_SCHEMA_VERSION,
    AttemptKind,
    CallRecord,
    CallStatus,
    DatasetReference,
    ModelID,
    ModelReference,
    MonosemousPolicyKind,
    PredictionRecord,
    PromptReference,
    RunID,
    RunMetadata,
    RunnerIdentity,
    RunPolicy,
    RunTotals,
    SamplingParameters,
    TieBreakKind,
    TokenUsage,
)
from sensebench.wordnet import SenseCandidate, get_candidate_senses, wordnet_version

UNKNOWN_GIT_COMMIT: str | None = None
PROGRESS_DESCRIPTION: str = "Evaluating items"
PROGRESS_UNIT: str = "item"
PREFLIGHT_CALL_ID: str = "preflight"
PREFLIGHT_PROMPT: str = "Reply with the number 1."
LLM_TEMPERATURE_PARAMETER: str = "temperature"
LLM_TOP_P_PARAMETER: str = "top_p"
LLM_MAX_TOKENS_PARAMETER: str = "max_tokens"
LLM_SEED_PARAMETER: str = "seed"
LLM_API_BASE_PARAMETER: str = "api_base"
LLM_REASONING_EFFORT_PARAMETER: str = "reasoning_effort"


@dataclass(frozen=True, slots=True)
class RunConfig:
    run_id: RunID
    output_root: Path
    dataset: DatasetBundle
    prompt: PromptDefinition
    model: ModelReference
    runner: RunnerIdentity
    sampling: SamplingParameters
    votes_per_item: int
    semantic_reasks_per_invalid_vote: int
    concurrency: int
    show_progress: bool = True


@dataclass(frozen=True, slots=True)
class CompletedRun:
    run_dir: Path
    metadata: RunMetadata
    predictions: list[PredictionRecord]
    calls: list[CallRecord]


@dataclass(frozen=True, slots=True)
class IndexedItemEvaluation:
    item_index: int
    evaluation: ItemEvaluation


def git_commit() -> str | None:
    try:
        result = run(
            args=["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return UNKNOWN_GIT_COMMIT
    return result.stdout.strip()


def _progress_postfix(
    *,
    correct_count: int,
    completed_count: int,
    known_cost_usd: float,
    has_known_cost: bool,
) -> str:
    accuracy_text = (
        f"acc {correct_count / completed_count:.3f}" if completed_count > 0 else "acc n/a"
    )
    cost_text = f"cost ${known_cost_usd:.2f}" if has_known_cost else "cost n/a"
    return f"{accuracy_text}, {cost_text}"


def _llm_parameters(*, sampling: SamplingParameters) -> dict[str, object]:
    parameters: dict[str, object] = {}
    if sampling.temperature is not None:
        parameters[LLM_TEMPERATURE_PARAMETER] = sampling.temperature
    if sampling.top_p is not None:
        parameters[LLM_TOP_P_PARAMETER] = sampling.top_p
    if sampling.max_tokens is not None:
        parameters[LLM_MAX_TOKENS_PARAMETER] = sampling.max_tokens
    if sampling.seed is not None:
        parameters[LLM_SEED_PARAMETER] = sampling.seed
    for key, value in sampling.extra.items():
        parameters[key] = value
    return parameters


def _completion_parameters(*, config: RunConfig) -> dict[str, object]:
    parameters: dict[str, object] = _llm_parameters(sampling=config.sampling)
    if config.model.endpoint_base_url is not None:
        parameters[LLM_API_BASE_PARAMETER] = config.model.endpoint_base_url
    if config.model.kind == CLOUD_LLM_KIND and config.model.reasoning_effort is not None:
        parameters[LLM_REASONING_EFFORT_PARAMETER] = config.model.reasoning_effort
    return parameters


def _requested_model_id(*, model: ModelReference) -> ModelID:
    if len(model.requested_model) > 0:
        return model.requested_model
    return model.display_name


def _sum_optional_ints(*, values: list[int | None]) -> int | None:
    observed_values: list[int] = [value for value in values if value is not None]
    if len(observed_values) == 0:
        return None
    return sum(observed_values)


def _sum_usage(*, predictions: list[PredictionRecord]) -> TokenUsage:
    return TokenUsage(
        input_tokens=_sum_optional_ints(
            values=[prediction.usage.input_tokens for prediction in predictions],
        ),
        cached_input_tokens=_sum_optional_ints(
            values=[prediction.usage.cached_input_tokens for prediction in predictions],
        ),
        output_tokens=_sum_optional_ints(
            values=[prediction.usage.output_tokens for prediction in predictions],
        ),
        reasoning_output_tokens=_sum_optional_ints(
            values=[prediction.usage.reasoning_output_tokens for prediction in predictions],
        ),
    )


def _totals(
    *,
    predictions: list[PredictionRecord],
    calls: list[CallRecord],
    elapsed_seconds: float,
) -> RunTotals:
    item_count = len(predictions)
    correct_count = sum(1 for prediction in predictions if prediction.is_correct is True)
    accuracy = correct_count / item_count if item_count > 0 else None
    return RunTotals(
        item_count=item_count,
        correct_count=correct_count,
        accuracy=accuracy,
        call_count=len(calls),
        usage=_sum_usage(predictions=predictions),
        cost=sum_costs(costs=[prediction.cost for prediction in predictions]),
        elapsed_seconds=elapsed_seconds,
    )


def _resolved_model_counts(*, calls: list[CallRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for call in calls:
        if call.status == CallStatus.SUCCESS and len(call.model) > 0:
            counts[call.model] += 1
    return dict(sorted(counts.items()))


def _single_resolved_model(*, resolved_model_counts: dict[str, int]) -> str | None:
    if len(resolved_model_counts) != 1:
        return None
    return next(iter(resolved_model_counts))


def _model_with_resolved_snapshots(
    *,
    model: ModelReference,
    calls: list[CallRecord],
) -> ModelReference:
    resolved_model_counts: dict[str, int] = _resolved_model_counts(calls=calls)
    return model.model_copy(
        update={
            "resolved_model": _single_resolved_model(
                resolved_model_counts=resolved_model_counts,
            ),
            "resolved_model_counts": resolved_model_counts,
        },
    )


async def _evaluate_one(
    *,
    item: WsdItem,
    dataset_index: DatasetIndex,
    config: RunConfig,
    client: CompletionClient,
    semaphore: asyncio.Semaphore,
) -> ItemEvaluation:
    candidates: list[SenseCandidate] = get_candidate_senses(lemma=item.lemma, pos=item.pos)
    rendered = render_task(
        prompt=config.prompt,
        item=item,
        dataset_index=dataset_index,
        candidates=candidates,
    )
    evaluation_config = EvaluationConfig(
        model=_requested_model_id(model=config.model),
        votes_per_item=config.votes_per_item,
        semantic_reasks_per_invalid_vote=config.semantic_reasks_per_invalid_vote,
        llm_parameters=_completion_parameters(config=config),
    )
    async with semaphore:
        return await evaluate_item(
            rendered=rendered,
            gold_sense_keys=item.gold_sense_keys,
            client=client,
            config=evaluation_config,
        )


async def _evaluate_one_indexed(
    *,
    item_index: int,
    item: WsdItem,
    dataset_index: DatasetIndex,
    config: RunConfig,
    client: CompletionClient,
    semaphore: asyncio.Semaphore,
) -> IndexedItemEvaluation:
    evaluation = await _evaluate_one(
        item=item,
        dataset_index=dataset_index,
        config=config,
        client=client,
        semaphore=semaphore,
    )
    return IndexedItemEvaluation(item_index=item_index, evaluation=evaluation)


async def _evaluate_items(
    *,
    dataset_index: DatasetIndex,
    config: RunConfig,
    client: CompletionClient,
    semaphore: asyncio.Semaphore,
) -> list[ItemEvaluation]:
    tasks: list[asyncio.Task[IndexedItemEvaluation]] = [
        asyncio.create_task(
            _evaluate_one_indexed(
                item_index=item_index,
                item=item,
                dataset_index=dataset_index,
                config=config,
                client=client,
                semaphore=semaphore,
            )
        )
        for item_index, item in enumerate(config.dataset.items)
    ]
    evaluations_by_index: list[ItemEvaluation | None] = [None] * len(tasks)
    progress = tqdm(
        total=len(tasks),
        desc=PROGRESS_DESCRIPTION,
        unit=PROGRESS_UNIT,
        disable=not config.show_progress,
    )
    correct_count = 0
    completed_count = 0
    known_cost_usd = 0.0
    has_known_cost = False
    try:
        for completed_task in asyncio.as_completed(tasks):
            indexed_evaluation = await completed_task
            evaluations_by_index[indexed_evaluation.item_index] = indexed_evaluation.evaluation
            prediction = indexed_evaluation.evaluation.prediction
            completed_count += 1
            if prediction.is_correct is True:
                correct_count += 1
            if prediction.cost.total_usd is not None:
                known_cost_usd += prediction.cost.total_usd
                has_known_cost = True
            progress.set_postfix_str(
                _progress_postfix(
                    correct_count=correct_count,
                    completed_count=completed_count,
                    known_cost_usd=known_cost_usd,
                    has_known_cost=has_known_cost,
                ),
                refresh=False,
            )
            progress.update(1)
    except Exception:
        for task in tasks:
            if task.done() is False:
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        progress.close()
    evaluations: list[ItemEvaluation] = []
    for evaluation in evaluations_by_index:
        assert evaluation is not None, "all item evaluations completed"
        evaluations.append(evaluation)
    return evaluations


async def preflight_model(*, config: RunConfig, client: CompletionClient) -> None:
    request = CompletionRequest(
        call_id=PREFLIGHT_CALL_ID,
        item_id=PREFLIGHT_CALL_ID,
        vote_index=1,
        attempt_index=1,
        attempt_kind=AttemptKind.INITIAL,
        model=_requested_model_id(model=config.model),
        messages=[ChatMessage(role=MessageRole.USER, content=PREFLIGHT_PROMPT)],
        parameters=_completion_parameters(config=config),
    )
    completion = await client.complete(request=request)
    if completion.call.status != CallStatus.SUCCESS:
        raise RuntimeError(
            "model preflight failed: "
            f"{completion.call.error_kind}: {completion.call.error_message}"
        )


async def run_benchmark(*, config: RunConfig, client: CompletionClient) -> CompletedRun:
    run_dir = config.output_root / config.run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    wordnet_version()
    started = time.monotonic()
    semaphore = asyncio.Semaphore(config.concurrency)
    dataset_index = build_dataset_index(bundle=config.dataset)
    evaluations: list[ItemEvaluation] = await _evaluate_items(
        dataset_index=dataset_index,
        config=config,
        client=client,
        semaphore=semaphore,
    )
    elapsed_seconds = time.monotonic() - started
    predictions: list[PredictionRecord] = [evaluation.prediction for evaluation in evaluations]
    calls: list[CallRecord] = [call for evaluation in evaluations for call in evaluation.calls]
    metadata = RunMetadata(
        schema_version=RUN_SCHEMA_VERSION,
        run_id=config.run_id,
        created_at=datetime.now(tz=UTC),
        git_commit=git_commit(),
        runner=config.runner,
        dataset=DatasetReference(
            dataset_id=config.dataset.dataset_id,
            dataset_version=config.dataset.dataset_version,
            dataset_revision=config.dataset.dataset_revision,
            content_hash=config.dataset.content_hash,
            item_count=len(config.dataset.items),
        ),
        prompt=PromptReference(id=config.prompt.id, sensebench_version=__version__),
        model=_model_with_resolved_snapshots(model=config.model, calls=calls),
        sampling=config.sampling,
        policy=RunPolicy(
            votes_per_item=config.votes_per_item,
            semantic_reasks_per_invalid_vote=config.semantic_reasks_per_invalid_vote,
            tie_break=TieBreakKind.EARLIEST_VOTE,
            monosemous_policy=MonosemousPolicyKind.SHORT_CIRCUIT,
        ),
        totals=_totals(predictions=predictions, calls=calls, elapsed_seconds=elapsed_seconds),
    )
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=predictions,
        calls=calls,
    )
    return CompletedRun(
        run_dir=run_dir,
        metadata=metadata,
        predictions=predictions,
        calls=calls,
    )
