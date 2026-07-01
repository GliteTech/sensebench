"""Repair a completed run by re-evaluating its failed items with a fallback model."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path

from sensebench.datasets.context import build_dataset_index
from sensebench.datasets.models import DatasetIndex, ItemID, WsdItem
from sensebench.prompts.render import render_task
from sensebench.runner.client import CompletionClient
from sensebench.runner.costs import sum_costs
from sensebench.runner.evaluate import EvaluationConfig, evaluate_item
from sensebench.runner.models import ItemEvaluation
from sensebench.runner.run import RunConfig, completion_parameters, git_commit
from sensebench.runner.writer import write_run_artifacts
from sensebench.runs.loaders import LoadedRun
from sensebench.runs.models import (
    CallRecord,
    ExecutionInfo,
    PredictionRecord,
    PredictionStatus,
    RunMetadata,
    RunTiming,
    RunTotals,
    TokenUsage,
)
from sensebench.wordnet import get_candidate_senses

DISPLAY_NAME_FIELD: str = "display_name"


def _sum_optional_ints(*, values: list[int | None]) -> int | None:
    observed_values: list[int] = [value for value in values if value is not None]
    if len(observed_values) == 0:
        return None
    return sum(observed_values)


def _sum_prediction_usage(*, predictions: list[PredictionRecord]) -> TokenUsage:
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


async def _repair_one_item(
    *,
    item: WsdItem,
    dataset_index: DatasetIndex,
    fallback_config: RunConfig,
    client: CompletionClient,
    semaphore: asyncio.Semaphore,
) -> ItemEvaluation:
    candidates = get_candidate_senses(lemma=item.lemma, pos=item.pos)
    rendered = render_task(
        prompt=fallback_config.prompt,
        item=item,
        dataset_index=dataset_index,
        candidates=candidates,
    )
    evaluation_config = EvaluationConfig(
        model=fallback_config.model.requested_model,
        votes_per_item=fallback_config.votes_per_item,
        semantic_reasks_per_invalid_vote=fallback_config.semantic_reasks_per_invalid_vote,
        llm_parameters=completion_parameters(config=fallback_config),
    )
    async with semaphore:
        return await evaluate_item(
            rendered=rendered,
            gold_sense_keys=item.gold_sense_keys,
            client=client,
            config=evaluation_config,
        )


async def repair_run(
    *,
    loaded: LoadedRun,
    fallback_config: RunConfig,
    client: CompletionClient,
    new_run_id: str,
    output_root: Path,
) -> Path:
    """Re-evaluate a run's ``no_valid_vote`` items with a fallback model.

    Items that already succeeded (or were monosemous / had no candidates) are
    copied through unchanged, including their original calls. Items that
    never got a valid vote have their prediction and calls fully replaced by
    the fallback model's fresh attempt — the original run directory is left
    untouched, so the discarded failed attempts remain inspectable there.
    """
    items_by_id: dict[ItemID, WsdItem] = {
        item.item_id: item for item in fallback_config.dataset.items
    }
    to_repair: list[PredictionRecord] = [
        prediction
        for prediction in loaded.predictions
        if prediction.status == PredictionStatus.NO_VALID_VOTE
    ]

    dataset_index = build_dataset_index(bundle=fallback_config.dataset)
    semaphore = asyncio.Semaphore(fallback_config.concurrency)
    repair_started = time.monotonic()
    repair_started_at = datetime.now(tz=UTC)
    evaluations: list[ItemEvaluation] = await asyncio.gather(
        *[
            _repair_one_item(
                item=items_by_id[prediction.item_id],
                dataset_index=dataset_index,
                fallback_config=fallback_config,
                client=client,
                semaphore=semaphore,
            )
            for prediction in to_repair
        ]
    )
    repair_seconds = time.monotonic() - repair_started
    repair_ended_at = datetime.now(tz=UTC)

    repaired_by_item_id: dict[ItemID, ItemEvaluation] = {
        evaluation.prediction.item_id: evaluation for evaluation in evaluations
    }
    calls_by_id: dict[str, CallRecord] = {call.call_id: call for call in loaded.calls}

    merged_predictions: list[PredictionRecord] = []
    merged_calls: list[CallRecord] = []
    for prediction in loaded.predictions:
        evaluation = repaired_by_item_id.get(prediction.item_id)
        if evaluation is None:
            merged_predictions.append(prediction)
            for vote in prediction.votes:
                merged_calls.extend(calls_by_id[call_id] for call_id in vote.call_ids)
        else:
            merged_predictions.append(evaluation.prediction)
            merged_calls.extend(evaluation.calls)

    item_count = len(merged_predictions)
    correct_count = sum(1 for prediction in merged_predictions if prediction.is_correct is True)
    accuracy = correct_count / item_count if item_count > 0 else None

    original_execution = loaded.metadata.execution
    execution: ExecutionInfo | None = None
    original_elapsed = loaded.metadata.totals.elapsed_seconds or 0.0
    elapsed_seconds = original_elapsed + repair_seconds
    if original_execution is not None:
        execution = ExecutionInfo(
            concurrency=fallback_config.concurrency,
            warmup_call_count=original_execution.warmup_call_count,
            timing=RunTiming(
                benchmark_started_at=min(
                    original_execution.timing.benchmark_started_at, repair_started_at
                ),
                benchmark_ended_at=max(
                    original_execution.timing.benchmark_ended_at, repair_ended_at
                ),
                benchmark_seconds=original_execution.timing.benchmark_seconds + repair_seconds,
                setup_seconds=original_execution.timing.setup_seconds,
            ),
        )

    primary_model = loaded.metadata.model
    if len(to_repair) > 0:
        # Leaderboard-distinct from an unassisted run of the primary model — a
        # fallback-assisted score must not be conflated with a raw single-model one.
        primary_model = primary_model.model_copy(
            update={
                DISPLAY_NAME_FIELD: (
                    f"{primary_model.display_name}+fallback:{fallback_config.model.requested_model}"
                ),
            },
        )

    metadata = RunMetadata(
        schema_version=loaded.metadata.schema_version,
        run_id=new_run_id,
        created_at=datetime.now(tz=UTC),
        git_commit=git_commit(),
        runner=loaded.metadata.runner,
        dataset=loaded.metadata.dataset,
        prompt=loaded.metadata.prompt,
        model=primary_model,
        fallback_model=fallback_config.model,
        sampling=loaded.metadata.sampling,
        policy=loaded.metadata.policy,
        machine=loaded.metadata.machine,
        execution=execution,
        totals=RunTotals(
            item_count=item_count,
            correct_count=correct_count,
            accuracy=accuracy,
            call_count=len(merged_calls),
            usage=_sum_prediction_usage(predictions=merged_predictions),
            cost=sum_costs(costs=[prediction.cost for prediction in merged_predictions]),
            elapsed_seconds=elapsed_seconds,
            fallback_used_count=len(to_repair),
        ),
    )
    new_run_dir = output_root / new_run_id
    write_run_artifacts(
        run_dir=new_run_dir,
        metadata=metadata,
        predictions=merged_predictions,
        calls=merged_calls,
    )
    return new_run_dir
