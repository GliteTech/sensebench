from __future__ import annotations

from datetime import UTC, datetime
from json import dumps
from pathlib import Path

from pytest import MonkeyPatch, raises

import sensebench.leaderboard.aggregate as aggregate_module
import sensebench.site.build as site_build_module
from sensebench.datasets.context import build_dataset_index
from sensebench.datasets.loaders import load_jsonl_dataset
from sensebench.datasets.models import DatasetBundle, DatasetID
from sensebench.datasets.releases import DatasetRelease
from sensebench.leaderboard.aggregate import LeaderboardBuildError
from sensebench.leaderboard.baselines import MFS_BASELINE_LABEL, BaselineKind
from sensebench.paths import (
    CALLS_FILENAME,
    DEFAULT_LEXEN_RELEASE_ID,
    INDEX_HTML_FILENAME,
    LEADERBOARD_JSON_PATH,
    LEXEN_DATASET_ID,
    P001_PROMPT_PATH,
    PREDICTIONS_FILENAME,
    RUN_METADATA_FILENAME,
    SITE_ASSETS_DIRNAME,
    SITE_DATA_DIRNAME,
    SITE_OUTPUT_DIR,
    SITE_RUNS_DIRNAME,
    SUBMITTED_RESULTS_DIR,
)
from sensebench.prompts.models import SENSE_INDEX_FIELD, MessageRole
from sensebench.prompts.registry import load_prompt_definition
from sensebench.prompts.render import render_task
from sensebench.runner.evaluate import prediction_is_correct
from sensebench.runner.writer import write_run_artifacts
from sensebench.runs.models import (
    CLOUD_LLM_KIND,
    RUN_SCHEMA_VERSION,
    AttemptKind,
    CallID,
    CallRecord,
    CallStatus,
    CandidateRecord,
    CloudLlmReference,
    CostBreakdown,
    CostSourceKind,
    DatasetReference,
    ExecutionInfo,
    MessageRecord,
    ModelID,
    ModelSourceKind,
    MonosemousPolicyKind,
    PredictionRecord,
    PredictionStatus,
    PromptReference,
    RunID,
    RunMetadata,
    RunnerIdentity,
    RunPolicy,
    RunTiming,
    RunTotals,
    SamplingParameters,
    TieBreakKind,
    TokenUsage,
    VoteRecord,
    VoteStatus,
)
from sensebench.wordnet import get_candidate_senses

SMOKE_ITEMS_PATH: Path = Path("tests/data/smoke_items.jsonl")
TEST_RELEASE_ID: str = DEFAULT_LEXEN_RELEASE_ID
TEST_DATASET_ID: DatasetID = LEXEN_DATASET_ID
TEST_RELEASE_URL: str = "https://example.com/items.jsonl"
TEST_RUN_ID: RunID = "fake-model-p001-lexen-v0.1.0-20260612"
TEST_MODEL_NAME: ModelID = "fake-model"
TEST_BASE_URL: str = "https://example.com/sensebench/"
TEST_CREATED_AT: datetime = datetime(2026, 6, 12, tzinfo=UTC)
TEST_GIT_COMMIT: str = "abc123"
TEST_GITHUB_HANDLE: str = "tester"
TEST_VENDOR: str = "TestVendor"
TEST_PROVIDER: str = "TestProvider"
SENSEBENCH_VERSION: str = "0.1.0"
CALL_VOTE_INDEX: int = 1
CALL_ATTEMPT_INDEX: int = 1
INPUT_TOKENS: int = 100
CACHED_INPUT_TOKENS: int = 0
OUTPUT_TOKENS: int = 10
TOTAL_COST_USD: float = 0.02
INPUT_UNCACHED_USD: float = 0.01
INPUT_CACHED_USD: float = 0.0
OUTPUT_USD: float = 0.01
INPUT_UNCACHED_PRICE_USD: float = 0.0001
INPUT_CACHED_PRICE_USD: float = 0.00001
OUTPUT_PRICE_USD: float = 0.001
LATENCY_SECONDS: float = 0.5
TEST_CONCURRENCY: int = 8
EXPECTED_COST_PER_MILLION_ITEMS: float = 20_000.0
EXPECTED_TOKENS_PER_ITEM: float = 110.0
SITE_DATA_SCHEMA_VERSION: str = "sensebench-site-data-v3"
RUN_DETAIL_SCHEMA_VERSION: str = "sensebench-run-detail-v4"
TARGET_ART_TEXT: str = "art"
TARGET_LEMMA_TEXT: str = "Target lemma: art"
SHOW_RAW_PROMPT_TEXT: str = "Show raw prompt"
ACTUAL_RUN_COST_TEXT: str = "Actual run cost"
DOWNLOAD_RAW_FILES_TEXT: str = "Download Raw Run Files"
PRICE_PER_MILLION_TEXT: str = "Price / 1M tokens"
PRICE_PER_TOKEN_TEXT: str = "Price / token"
EXPECTED_PRICE_TEXT: str = "$100"
LEGACY_PRICE_TEXT: str = "$100.00"
EXPECTED_CORRECTNESS_BITS: str = "0"
REFERENCE_BASELINES_TEXT: str = "Reference Baselines"
MAX_COST_FILTER_ID: str = "max-cost-filter"
SOURCE_FILTER_ID: str = "source-filter"
BUILT_BY_GLITE_TEXT: str = "Built by Glite"
EXPECTED_BASELINE_COUNT: int = 4
ECHARTS_VENDOR_PATH: Path = Path("vendor") / "echarts.min.js"
BAD_CONTENT_HASH: str = "sha256:bad"
GET_DATASET_RELEASE_ATTR: str = "get_dataset_release"
LOAD_REGISTERED_DATASET_ATTR: str = "load_registered_dataset"
DATASET_RELEASES_ATTR: str = "DATASET_RELEASES"


def raw_output_for_sense_index(*, sense_index: int) -> str:
    return dumps({SENSE_INDEX_FIELD: sense_index})


def _dataset() -> DatasetBundle:
    return load_jsonl_dataset(
        path=SMOKE_ITEMS_PATH,
        dataset_id=TEST_DATASET_ID,
        dataset_version=TEST_RELEASE_ID,
    )


def _patch_registered_dataset(
    *,
    monkeypatch: MonkeyPatch,
    dataset: DatasetBundle,
) -> DatasetRelease:
    assert dataset.content_hash is not None
    release = DatasetRelease(
        release_id=TEST_RELEASE_ID,
        dataset_id=TEST_DATASET_ID,
        url=TEST_RELEASE_URL,
        content_hash=dataset.content_hash,
        item_count=len(dataset.items),
    )

    def fake_get_dataset_release(*, release_id: str) -> DatasetRelease:
        assert release_id == release.release_id
        return release

    def fake_load_registered_dataset(*, release: DatasetRelease) -> DatasetBundle:
        return dataset

    monkeypatch.setattr(
        target=aggregate_module,
        name=GET_DATASET_RELEASE_ATTR,
        value=fake_get_dataset_release,
    )
    monkeypatch.setattr(
        target=aggregate_module,
        name=LOAD_REGISTERED_DATASET_ATTR,
        value=fake_load_registered_dataset,
    )
    monkeypatch.setattr(
        target=site_build_module,
        name=GET_DATASET_RELEASE_ATTR,
        value=fake_get_dataset_release,
    )
    monkeypatch.setattr(
        target=site_build_module,
        name=LOAD_REGISTERED_DATASET_ATTR,
        value=fake_load_registered_dataset,
    )
    monkeypatch.setattr(
        target=site_build_module,
        name=DATASET_RELEASES_ATTR,
        value={release.release_id: release},
    )
    return release


def _write_verified_run(
    *,
    results_dir: Path,
    dataset: DatasetBundle,
    run_id: RunID,
    model_name: ModelID = TEST_MODEL_NAME,
    content_hash: str | None = None,
    choose_gold: bool = True,
) -> None:
    prompt = load_prompt_definition(path=P001_PROMPT_PATH)
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
    call_id: CallID = f"{item.item_id}__v{CALL_VOTE_INDEX}__a{CALL_ATTEMPT_INDEX}"
    usage = TokenUsage(
        input_tokens=INPUT_TOKENS,
        cached_input_tokens=CACHED_INPUT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
    )
    cost = CostBreakdown(
        total_usd=TOTAL_COST_USD,
        input_uncached_usd=INPUT_UNCACHED_USD,
        input_cached_usd=INPUT_CACHED_USD,
        output_usd=OUTPUT_USD,
        input_uncached_unit_price_usd=INPUT_UNCACHED_PRICE_USD,
        input_cached_unit_price_usd=INPUT_CACHED_PRICE_USD,
        output_unit_price_usd=OUTPUT_PRICE_USD,
        source=CostSourceKind.LITELLM_ESTIMATE,
    )
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
                vote_index=CALL_VOTE_INDEX,
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
        latency_seconds=LATENCY_SECONDS,
    )
    metadata = RunMetadata(
        schema_version=RUN_SCHEMA_VERSION,
        run_id=run_id,
        created_at=TEST_CREATED_AT,
        git_commit=TEST_GIT_COMMIT,
        runner=RunnerIdentity(github_handle=TEST_GITHUB_HANDLE),
        dataset=DatasetReference(
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            content_hash=content_hash if content_hash is not None else dataset.content_hash,
            item_count=len(dataset.items),
        ),
        prompt=PromptReference(id=prompt.id, sensebench_version=SENSEBENCH_VERSION),
        model=CloudLlmReference(
            kind=CLOUD_LLM_KIND,
            display_name=model_name,
            requested_model=model_name,
            llm_vendor=TEST_VENDOR,
            api_provider=TEST_PROVIDER,
            source_kind=ModelSourceKind.UNKNOWN,
        ),
        sampling=SamplingParameters(),
        policy=RunPolicy(
            votes_per_item=CALL_VOTE_INDEX,
            semantic_reasks_per_invalid_vote=CALL_VOTE_INDEX,
            tie_break=TieBreakKind.EARLIEST_VOTE,
            monosemous_policy=MonosemousPolicyKind.SHORT_CIRCUIT,
        ),
        execution=ExecutionInfo(
            concurrency=TEST_CONCURRENCY,
            timing=RunTiming(
                benchmark_started_at=TEST_CREATED_AT,
                benchmark_ended_at=TEST_CREATED_AT,
                benchmark_seconds=LATENCY_SECONDS,
            ),
        ),
        totals=RunTotals(
            item_count=CALL_VOTE_INDEX,
            correct_count=1 if is_correct else 0,
            accuracy=1.0 if is_correct else 0.0,
            call_count=CALL_VOTE_INDEX,
            usage=usage,
            cost=cost,
            elapsed_seconds=LATENCY_SECONDS,
        ),
    )
    call = CallRecord(
        call_id=call_id,
        item_id=item.item_id,
        vote_index=CALL_VOTE_INDEX,
        attempt_index=CALL_ATTEMPT_INDEX,
        attempt_kind=AttemptKind.INITIAL,
        transport_retry_count=0,
        status=CallStatus.SUCCESS,
        model=model_name,
        messages=[
            MessageRecord(role=message.role, content=message.content)
            for message in rendered.messages
        ],
        raw_output=raw_output_for_sense_index(sense_index=chosen.index),
        usage=usage,
        cost=cost,
        latency_seconds=LATENCY_SECONDS,
    )
    write_run_artifacts(
        run_dir=results_dir / run_id,
        metadata=metadata,
        predictions=[prediction],
        calls=[call],
    )


def test_build_site_emits_static_pages_and_data(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset = _dataset()
    _patch_registered_dataset(monkeypatch=monkeypatch, dataset=dataset)
    results_dir = tmp_path / SUBMITTED_RESULTS_DIR
    output_dir = tmp_path / SITE_OUTPUT_DIR
    run_id = TEST_RUN_ID
    _write_verified_run(
        results_dir=results_dir,
        dataset=dataset,
        run_id=run_id,
        choose_gold=False,
    )

    site_build_module.build_site(
        results_dir=results_dir,
        output_dir=output_dir,
        base_url=TEST_BASE_URL,
        strict=True,
    )

    assert (output_dir / INDEX_HTML_FILENAME).exists()
    assert (output_dir / SITE_RUNS_DIRNAME / run_id / INDEX_HTML_FILENAME).exists()
    assert (output_dir / SITE_DATA_DIRNAME / SITE_RUNS_DIRNAME / f"{run_id}.json").exists()
    assert (output_dir / "artifacts" / SITE_RUNS_DIRNAME / run_id / RUN_METADATA_FILENAME).exists()
    assert (output_dir / "artifacts" / SITE_RUNS_DIRNAME / run_id / PREDICTIONS_FILENAME).exists()
    assert (output_dir / "artifacts" / SITE_RUNS_DIRNAME / run_id / CALLS_FILENAME).exists()
    assert (output_dir / SITE_ASSETS_DIRNAME / ECHARTS_VENDOR_PATH).exists()
    assert run_id in (output_dir / "sitemap.xml").read_text(encoding="utf-8")

    site_data = site_build_module.SiteData.model_validate_json(
        (output_dir / SITE_DATA_DIRNAME / LEADERBOARD_JSON_PATH).read_text(encoding="utf-8")
    )
    assert site_data.schema_version == SITE_DATA_SCHEMA_VERSION
    assert site_data.summary.verified_run_count == 1
    entry = site_data.entries[0]
    assert entry.accuracy == 0.0
    assert entry.cost_per_million_items == EXPECTED_COST_PER_MILLION_ITEMS
    assert entry.tokens_per_item == EXPECTED_TOKENS_PER_ITEM
    assert entry.input_uncached_tokens == INPUT_TOKENS
    assert entry.cached_input_tokens == CACHED_INPUT_TOKENS
    assert entry.output_tokens == OUTPUT_TOKENS
    assert entry.input_uncached_usd == INPUT_UNCACHED_USD
    assert entry.input_cached_usd == INPUT_CACHED_USD
    assert entry.output_usd == OUTPUT_USD
    assert entry.input_uncached_unit_price_usd == INPUT_UNCACHED_PRICE_USD
    assert entry.input_cached_unit_price_usd == INPUT_CACHED_PRICE_USD
    assert entry.output_unit_price_usd == OUTPUT_PRICE_USD
    assert entry.cost_source == CostSourceKind.LITELLM_ESTIMATE.value
    assert "latency_per_item" not in entry.model_dump()

    assert len(site_data.baselines) == EXPECTED_BASELINE_COUNT
    mfs = next(
        baseline
        for baseline in site_data.baselines
        if baseline.label == MFS_BASELINE_LABEL
    )
    assert mfs.kind == BaselineKind.COMPUTED_WORDNET_MFS
    assert mfs.accuracy == 0.0
    assert all(
        baseline.dataset_version == TEST_RELEASE_ID for baseline in site_data.baselines
    )

    run_detail = site_build_module.RunDetail.model_validate_json(
        (output_dir / SITE_DATA_DIRNAME / SITE_RUNS_DIRNAME / f"{run_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_detail.schema_version == RUN_DETAIL_SCHEMA_VERSION
    assert run_detail.metadata.run_id == run_id
    assert run_detail.metadata.totals.cost.total_usd == TOTAL_COST_USD
    assert run_detail.correctness == EXPECTED_CORRECTNESS_BITS
    assert {artifact.filename for artifact in run_detail.artifacts} == {
        RUN_METADATA_FILENAME,
        PREDICTIONS_FILENAME,
        CALLS_FILENAME,
    }
    example = run_detail.worst_examples[0]
    assert len(example.context_sentences) == 2
    assert any(
        f"<mark>{TARGET_ART_TEXT}</mark>" in sentence.html
        for sentence in example.context_sentences
    )
    assert len(example.candidates) > 0
    assert any(candidate.is_gold for candidate in example.candidates)
    assert any(candidate.is_selected for candidate in example.candidates)
    assert example.prompt_messages[0].role == MessageRole.SYSTEM
    assert TARGET_LEMMA_TEXT in example.prompt_messages[1].content

    run_html = (output_dir / SITE_RUNS_DIRNAME / run_id / INDEX_HTML_FILENAME).read_text(
        encoding="utf-8"
    )
    assert SHOW_RAW_PROMPT_TEXT in run_html
    assert TARGET_LEMMA_TEXT in run_html
    assert "&lt;t&gt;art&lt;/t&gt;" in run_html
    assert "<t>art</t>" not in run_html
    assert "<mark>art</mark>" in run_html
    assert ACTUAL_RUN_COST_TEXT in run_html
    assert DOWNLOAD_RAW_FILES_TEXT in run_html
    assert PRICE_PER_MILLION_TEXT in run_html
    assert PRICE_PER_TOKEN_TEXT not in run_html
    assert EXPECTED_PRICE_TEXT in run_html
    assert LEGACY_PRICE_TEXT not in run_html
    assert f"artifacts/runs/{TEST_RUN_ID}/{RUN_METADATA_FILENAME}" in run_html

    index_html = (output_dir / INDEX_HTML_FILENAME).read_text(encoding="utf-8")
    assert REFERENCE_BASELINES_TEXT in index_html
    assert MAX_COST_FILTER_ID in index_html
    assert SOURCE_FILTER_ID in index_html
    assert BUILT_BY_GLITE_TEXT in index_html


def test_format_money_rounds_by_magnitude() -> None:
    assert site_build_module._format_money(None) == "n/a"
    assert site_build_module._format_money(20_000.0) == "$20,000"
    assert site_build_module._format_money(791.0716) == "$791"
    assert site_build_module._format_money(100.0) == "$100"
    assert site_build_module._format_money(7.929) == "$7.93"
    assert site_build_module._format_money(0.5) == "$0.500"
    assert site_build_module._format_money(0.0042) == "$0.00420"
    assert site_build_module._format_money(0.0) == "$0.00"


def test_build_site_strict_rejects_wrong_dataset_hash(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset = _dataset()
    _patch_registered_dataset(monkeypatch=monkeypatch, dataset=dataset)
    results_dir = tmp_path / SUBMITTED_RESULTS_DIR
    run_id = TEST_RUN_ID
    _write_verified_run(
        results_dir=results_dir,
        dataset=dataset,
        run_id=run_id,
        content_hash=BAD_CONTENT_HASH,
    )

    with raises(LeaderboardBuildError):
        site_build_module.build_site(
            results_dir=results_dir,
            output_dir=tmp_path / SITE_OUTPUT_DIR,
            base_url=TEST_BASE_URL,
            strict=True,
        )
