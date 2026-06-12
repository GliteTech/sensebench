from __future__ import annotations

import json
from pathlib import Path

import pytest

import sensebench.leaderboard.aggregate as aggregate_module
import sensebench.site.build as site_build_module
from sensebench.datasets.context import build_dataset_index
from sensebench.datasets.loaders import load_jsonl_dataset
from sensebench.datasets.models import DatasetBundle
from sensebench.datasets.releases import DatasetRelease
from sensebench.leaderboard.aggregate import LeaderboardBuildError
from sensebench.paths import PROMPT_JSON_SUFFIX, PROMPT_REGISTRY_DIR
from sensebench.prompts.registry import load_prompt_definition
from sensebench.prompts.render import render_task
from sensebench.runner.evaluate import prediction_is_correct
from sensebench.runner.writer import write_run_artifacts
from sensebench.runs.models import (
    CLOUD_LLM_KIND,
    RUN_SCHEMA_VERSION,
    AttemptKind,
    CallRecord,
    CallStatus,
    CandidateRecord,
    CloudLlmReference,
    CostBreakdown,
    CostSourceKind,
    DatasetReference,
    MessageRecord,
    ModelSourceKind,
    MonosemousPolicyKind,
    PredictionRecord,
    PredictionStatus,
    PromptReference,
    RunMetadata,
    RunnerIdentity,
    RunPolicy,
    RunTotals,
    SamplingParameters,
    TieBreakKind,
    TokenUsage,
    VoteRecord,
    VoteStatus,
)
from sensebench.wordnet import get_candidate_senses


def _dataset() -> DatasetBundle:
    return load_jsonl_dataset(
        path=Path("tests/data/smoke_items.jsonl"),
        dataset_id="lexen",
        dataset_version="lexen-v0.1.0",
    )


def _patch_registered_dataset(
    *,
    monkeypatch: pytest.MonkeyPatch,
    dataset: DatasetBundle,
) -> DatasetRelease:
    assert dataset.content_hash is not None
    release = DatasetRelease(
        release_id="lexen-v0.1.0",
        dataset_id="lexen",
        url="https://example.com/items.jsonl",
        content_hash=dataset.content_hash,
        item_count=len(dataset.items),
    )

    def fake_get_dataset_release(*, release_id: str) -> DatasetRelease:
        assert release_id == release.release_id
        return release

    def fake_load_registered_dataset(*, release: DatasetRelease) -> DatasetBundle:
        return dataset

    monkeypatch.setattr(aggregate_module, "get_dataset_release", fake_get_dataset_release)
    monkeypatch.setattr(
        aggregate_module,
        "load_registered_dataset",
        fake_load_registered_dataset,
    )
    monkeypatch.setattr(site_build_module, "get_dataset_release", fake_get_dataset_release)
    monkeypatch.setattr(
        site_build_module,
        "load_registered_dataset",
        fake_load_registered_dataset,
    )
    monkeypatch.setattr(site_build_module, "DATASET_RELEASES", {release.release_id: release})
    return release


def _write_verified_run(
    *,
    results_dir: Path,
    dataset: DatasetBundle,
    run_id: str,
    model_name: str = "fake-model",
    content_hash: str | None = None,
    choose_gold: bool = True,
) -> None:
    prompt = load_prompt_definition(path=PROMPT_REGISTRY_DIR / f"p001{PROMPT_JSON_SUFFIX}")
    item = dataset.items[0]
    index = build_dataset_index(bundle=dataset)
    candidates = get_candidate_senses(lemma=item.lemma, pos=item.pos)
    rendered = render_task(prompt=prompt, item=item, dataset_index=index, candidates=candidates)
    if choose_gold:
        chosen = next(
            candidate
            for candidate in rendered.candidates
            if candidate.sense_key in item.gold_sense_keys
        )
    else:
        chosen = next(
            candidate
            for candidate in rendered.candidates
            if candidate.sense_key not in item.gold_sense_keys
        )
    call_id = f"{item.item_id}__v1__a1"
    usage = TokenUsage(input_tokens=100, cached_input_tokens=0, output_tokens=10)
    cost = CostBreakdown(total_usd=0.02, source=CostSourceKind.LITELLM_ESTIMATE)
    is_correct = prediction_is_correct(
        predicted_sense_key=chosen.sense_key,
        gold_sense_keys=item.gold_sense_keys,
    )
    prediction = PredictionRecord(
        item_id=item.item_id,
        gold_sense_keys=item.gold_sense_keys,
        candidates=[
            CandidateRecord(
                index=candidate.index,
                sense_key=candidate.sense_key,
                synset_id=candidate.synset_id,
            )
            for candidate in rendered.candidates
        ],
        votes=[
            VoteRecord(
                vote_index=1,
                status=VoteStatus.SUCCESS,
                chosen_sense_index=chosen.index,
                chosen_sense_key=chosen.sense_key,
                call_ids=[call_id],
            )
        ],
        predicted_sense_index=chosen.index,
        predicted_sense_key=chosen.sense_key,
        is_correct=is_correct,
        status=PredictionStatus.SUCCESS,
        was_monosemous=False,
        usage=usage,
        cost=cost,
        latency_seconds=0.5,
    )
    metadata = RunMetadata(
        schema_version=RUN_SCHEMA_VERSION,
        run_id=run_id,
        created_at="2026-06-12T00:00:00+00:00",
        git_commit="abc123",
        runner=RunnerIdentity(github_handle="tester"),
        dataset=DatasetReference(
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            content_hash=content_hash if content_hash is not None else dataset.content_hash,
            item_count=len(dataset.items),
        ),
        prompt=PromptReference(id=prompt.id, sensebench_version="0.1.0"),
        model=CloudLlmReference(
            kind=CLOUD_LLM_KIND,
            display_name=model_name,
            requested_model=model_name,
            llm_vendor="TestVendor",
            api_provider="TestProvider",
            source_kind=ModelSourceKind.UNKNOWN,
        ),
        sampling=SamplingParameters(),
        policy=RunPolicy(
            votes_per_item=1,
            semantic_reasks_per_invalid_vote=1,
            tie_break=TieBreakKind.EARLIEST_VOTE,
            monosemous_policy=MonosemousPolicyKind.SHORT_CIRCUIT,
        ),
        totals=RunTotals(
            item_count=1,
            correct_count=1 if is_correct else 0,
            accuracy=1.0 if is_correct else 0.0,
            call_count=1,
            usage=usage,
            cost=cost,
            elapsed_seconds=0.5,
        ),
    )
    call = CallRecord(
        call_id=call_id,
        item_id=item.item_id,
        vote_index=1,
        attempt_index=1,
        attempt_kind=AttemptKind.INITIAL,
        transport_retry_count=0,
        status=CallStatus.SUCCESS,
        model=model_name,
        messages=[
            MessageRecord(role=message.role, content=message.content)
            for message in rendered.messages
        ],
        raw_output=f'{{"sense_index": {chosen.index}}}',
        usage=usage,
        cost=cost,
        latency_seconds=0.5,
    )
    write_run_artifacts(
        run_dir=results_dir / run_id,
        metadata=metadata,
        predictions=[prediction],
        calls=[call],
    )


def test_build_site_emits_static_pages_and_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset()
    _patch_registered_dataset(monkeypatch=monkeypatch, dataset=dataset)
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "_site"
    run_id = "fake-model-p001-lexen-v0.1.0-20260612"
    _write_verified_run(
        results_dir=results_dir,
        dataset=dataset,
        run_id=run_id,
        choose_gold=False,
    )

    site_build_module.build_site(
        results_dir=results_dir,
        output_dir=output_dir,
        base_url="https://example.com/sensebench/",
        strict=True,
    )

    assert (output_dir / "index.html").exists()
    assert (output_dir / "runs" / run_id / "index.html").exists()
    assert (output_dir / "data" / "runs" / f"{run_id}.json").exists()
    assert (output_dir / "assets" / "vendor" / "echarts.min.js").exists()
    assert run_id in (output_dir / "sitemap.xml").read_text(encoding="utf-8")

    site_data = json.loads((output_dir / "data" / "leaderboard.json").read_text())
    assert site_data["schema_version"] == "sensebench-site-data-v2"
    assert site_data["summary"]["verified_run_count"] == 1
    entry = site_data["entries"][0]
    assert entry["accuracy"] == 0.0
    assert entry["cost_per_million_items"] == 20_000.0
    assert entry["tokens_per_item"] == 110.0
    assert "latency_per_item" not in entry

    run_detail = json.loads((output_dir / "data" / "runs" / f"{run_id}.json").read_text())
    example = run_detail["worst_examples"][0]
    assert example["context_sentences"]
    assert example["candidates"]
    assert any(candidate["is_gold"] for candidate in example["candidates"])
    assert any(candidate["is_selected"] for candidate in example["candidates"])


def test_build_site_strict_rejects_wrong_dataset_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset()
    _patch_registered_dataset(monkeypatch=monkeypatch, dataset=dataset)
    results_dir = tmp_path / "results"
    run_id = "fake-model-p001-lexen-v0.1.0-20260612"
    _write_verified_run(
        results_dir=results_dir,
        dataset=dataset,
        run_id=run_id,
        content_hash="sha256:bad",
    )

    with pytest.raises(LeaderboardBuildError):
        site_build_module.build_site(
            results_dir=results_dir,
            output_dir=tmp_path / "_site",
            base_url="https://example.com/sensebench/",
            strict=True,
        )
