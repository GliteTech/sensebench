"""Run SenseBench evaluations."""

from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sensebench import __version__
from sensebench.datasets.context import build_dataset_index
from sensebench.datasets.models import DatasetBundle, DatasetIndex, WsdItem
from sensebench.prompts.models import PromptDefinition
from sensebench.prompts.render import render_task
from sensebench.runner.client import CompletionClient
from sensebench.runner.evaluate import EvaluationConfig, evaluate_item
from sensebench.runner.models import ItemEvaluation
from sensebench.runner.writer import write_run_artifacts
from sensebench.runs.models import (
    RUN_SCHEMA_VERSION,
    CallRecord,
    DatasetReference,
    ModelReference,
    MonosemousPolicyKind,
    PredictionRecord,
    PromptReference,
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


@dataclass(frozen=True, slots=True)
class RunConfig:
    run_id: str
    output_root: Path
    dataset: DatasetBundle
    prompt: PromptDefinition
    model: ModelReference
    runner: RunnerIdentity
    sampling: SamplingParameters
    votes_per_item: int
    semantic_reasks_per_invalid_vote: int
    concurrency: int


@dataclass(frozen=True, slots=True)
class CompletedRun:
    run_dir: Path
    metadata: RunMetadata
    predictions: list[PredictionRecord]
    calls: list[CallRecord]


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return UNKNOWN_GIT_COMMIT
    return result.stdout.strip()


def _llm_parameters(*, sampling: SamplingParameters) -> dict[str, object]:
    parameters: dict[str, object] = {}
    if sampling.temperature is not None:
        parameters["temperature"] = sampling.temperature
    if sampling.top_p is not None:
        parameters["top_p"] = sampling.top_p
    if sampling.max_tokens is not None:
        parameters["max_tokens"] = sampling.max_tokens
    if sampling.seed is not None:
        parameters["seed"] = sampling.seed
    for key, value in sampling.extra.items():
        parameters[key] = value
    return parameters


def _completion_parameters(*, config: RunConfig) -> dict[str, object]:
    parameters: dict[str, object] = _llm_parameters(sampling=config.sampling)
    if config.model.endpoint_base_url is not None:
        parameters["api_base"] = config.model.endpoint_base_url
    return parameters


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
    )


def _sum_cost(*, predictions: list[PredictionRecord]) -> float | None:
    values: list[float] = [
        prediction.cost_usd for prediction in predictions if prediction.cost_usd is not None
    ]
    if len(values) == 0:
        return None
    return sum(values)


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
        cost_usd=_sum_cost(predictions=predictions),
        elapsed_seconds=elapsed_seconds,
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
        model=config.model.requested_model or config.model.display_name,
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


async def run_benchmark(*, config: RunConfig, client: CompletionClient) -> CompletedRun:
    wordnet_version()
    started = time.monotonic()
    semaphore = asyncio.Semaphore(config.concurrency)
    dataset_index = build_dataset_index(bundle=config.dataset)
    evaluations: list[ItemEvaluation] = await asyncio.gather(
        *[
            _evaluate_one(
                item=item,
                dataset_index=dataset_index,
                config=config,
                client=client,
                semaphore=semaphore,
            )
            for item in config.dataset.items
        ]
    )
    elapsed_seconds = time.monotonic() - started
    predictions: list[PredictionRecord] = [evaluation.prediction for evaluation in evaluations]
    calls: list[CallRecord] = [call for evaluation in evaluations for call in evaluation.calls]
    metadata = RunMetadata(
        schema_version=RUN_SCHEMA_VERSION,
        run_id=config.run_id,
        created_at=datetime.now(tz=UTC).isoformat(),
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
        model=config.model,
        sampling=config.sampling,
        policy=RunPolicy(
            votes_per_item=config.votes_per_item,
            semantic_reasks_per_invalid_vote=config.semantic_reasks_per_invalid_vote,
            tie_break=TieBreakKind.EARLIEST_VOTE,
            monosemous_policy=MonosemousPolicyKind.SHORT_CIRCUIT,
        ),
        totals=_totals(predictions=predictions, calls=calls, elapsed_seconds=elapsed_seconds),
    )
    run_dir = config.output_root / config.run_id
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
