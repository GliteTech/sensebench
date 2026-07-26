from __future__ import annotations

from datetime import UTC, datetime
from json import dumps
from pathlib import Path

from pytest import MonkeyPatch, raises

from sensebench.datasets.context import build_dataset_index
from sensebench.datasets.loaders import load_jsonl_dataset
from sensebench.datasets.models import DatasetBundle, DatasetID
from sensebench.datasets.releases import DatasetRelease
from sensebench.leaderboard.aggregate import LeaderboardBuildError
from sensebench.leaderboard.baselines import (
    BASELINE_PREDICTION_SPECS,
    MFS_BASELINE_LABEL,
    BaselineKind,
)
from sensebench.leaderboard.gpu import (
    GPU_REFERENCE_HOURLY_RATE_USD,
    GPU_REFERENCE_RATES_AS_OF,
)
from sensebench.paths import (
    CALLS_FILENAME,
    CNAME_FILENAME,
    DEFAULT_LEXEN_RELEASE_ID,
    INDEX_HTML_FILENAME,
    LEADERBOARD_JSON_PATH,
    LEXEN_DATASET_ID,
    P001_PROMPT_PATH,
    PREDICTIONS_FILENAME,
    RUN_ARTIFACT_ROOT,
    RUN_METADATA_FILENAME,
    SITE_ASSETS_DIRNAME,
    SITE_DATA_DIRNAME,
    SITE_OUTPUT_DIR,
    SITE_RUNS_DIRNAME,
    SMOKE_ITEMS_PATH,
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
    ModelHostingKind,
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
from sensebench.site.build import (
    DEFAULT_CUSTOM_DOMAIN,
    PROMPTS_ROUTE_PREFIX,
    UNKNOWN_DATASET_VERSION_LABEL,
    RunDetail,
    SiteData,
    _dataset_version_label,
    _format_money,
    _reasoning_effort_label,
    _reference_rate_sentence,
    _run_page_description,
    _run_page_title,
    _static_pages,
    build_site,
)
from sensebench.wordnet import get_candidate_senses
from tests.run_fixtures import (
    FIXTURE_GPU_NAME,
    FIXTURE_INFERENCE_ENGINE,
    FIXTURE_PROVIDER,
    FIXTURE_QUANTIZATION,
    SELF_HOSTED_MODEL_NAME,
    fixture_machine,
    self_hosted_model,
)

TEST_RELEASE_ID: str = DEFAULT_LEXEN_RELEASE_ID
TEST_DATASET_ID: DatasetID = LEXEN_DATASET_ID
TEST_RELEASE_URL: str = "https://example.com/items.jsonl"
TEST_RUN_ID: RunID = "fake-model-p001-lexen-v1-20260612"
SELF_HOSTED_RUN_ID: RunID = "fake-local-p001-lexen-v1-20260612"
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
SITE_DATA_SCHEMA_VERSION: str = "sensebench-site-data-v8"
METHODOLOGY_SLUG: str = "methodology"
RUN_DETAIL_SCHEMA_VERSION: str = "sensebench-run-detail-v7"
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
HOSTING_FILTER_ID: str = "hosting-filter"
GPU_FILTER_ID: str = "gpu-filter"
QUANT_FILTER_ID: str = "quant-filter"
X_METRIC_SELECT_ID: str = "x-metric-select"
X_SCALE_SELECT_ID: str = "x-scale-select"
MACHINE_TIMING_HEADING_TEXT: str = "Machine &amp; Timing"
TITLE_OPEN_TAG: str = "<title>"
TITLE_CLOSE_TAG: str = "</title>"
WSD_TASK_PHRASE_TEXT: str = "word sense disambiguation"
ITEMS_CORRECT_TEXT: str = "items correct"
OMITTED_REASONING_EFFORT: str = "none"
SELF_DESCRIBING_REASONING_EFFORT: str = "reasoning"
XHIGH_REASONING_EFFORT: str = "xhigh"
XHIGH_REASONING_EFFORT_LABEL: str = "xhigh reasoning"
LEXEN_DISPLAY_LABEL: str = "lexEN v1"
UNREGISTERED_DATASET_VERSION: str = "custom-v9"
ALTERNATE_QUANTIZATION: str = "awq-int4"
RERUN_CREATED_AT: str = "2026-07-20T00:00:00+00:00"
RERUN_DATE_LABEL: str = "2026-07-20"
MACHINE_HOURS_TEXT: str = "Machine-hours / 1M items"
SPEED_COLUMN_HEADER_TEXT: str = "Speed (s / item)"
SPEED_SORT_BUTTON_TEXT: str = 'data-sort="seconds_per_item"'
SORT_ARROW_TEXT: str = '<span class="sort-arrow"'
RANK_SORT_BUTTON_TEXT: str = 'data-sort="rank"'
EXPECTED_GPU_LABEL: str = "H100 80GB"
DOWNLOAD_CSV_BUTTON_ID: str = "download-csv"
DOWNLOAD_JSON_BUTTON_ID: str = "download-json"
TABLE_DOWNLOADS_CLASS: str = "table-downloads"
SITE_JS_FILENAME: str = "site.js"
SITE_JS_EXPORT_COLUMNS_MARKER: str = "EXPORT_COLUMNS"
SITE_JS_SPOT_PRICED_MARKER: str = "function isSpotPriced"
SITE_JS_SPOT_PRICED_CELL_MARKER: str = "spotPricedNoteHtml(entry)"
SITE_JS_FRONTIER_BADGE_MARKER: str = "function frontierBadgeHtml"
COST_FLAG_CLASS: str = "cost-flag"
SITE_JS_DOWNLOAD_CSV_MARKER: str = "function downloadCsv"
DESCRIPTIVE_GLITE_CSV_PREFIX: str = "lexen_glite_coarse"
DESCRIPTIVE_CSI_CSV_PREFIX: str = "lexen_csi_coarse"
LABEL_SCHEMES_ROUTE_PREFIX: str = "label-schemes"
LABEL_SCHEME_TABLE_CLASS: str = "label-scheme-table"
LABEL_SCHEME_TABLE_WRAP_CLASS: str = "label-scheme-table-wrap"
LABEL_SCHEME_CSI_DATA_LABEL: str = 'data-label="CSI coarse-grained (Lacerra 2020)"'
BUILT_BY_GLITE_TEXT: str = "Built by Glite"
EXPECTED_BASELINE_COUNT: int = 1 + len(BASELINE_PREDICTION_SPECS)
ECHARTS_VENDOR_PATH: Path = Path("vendor") / "echarts.min.js"
BAD_CONTENT_HASH: str = "sha256:bad"
GET_DATASET_RELEASE_ATTR: str = "get_dataset_release"
LOAD_REGISTERED_DATASET_ATTR: str = "load_registered_dataset"
DATASET_RELEASES_ATTR: str = "DATASET_RELEASES"
AGGREGATE_GET_DATASET_RELEASE_TARGET: str = "sensebench.leaderboard.aggregate.get_dataset_release"
AGGREGATE_LOAD_REGISTERED_DATASET_TARGET: str = (
    "sensebench.leaderboard.aggregate.load_registered_dataset"
)
SITE_GET_DATASET_RELEASE_TARGET: str = "sensebench.site.build.get_dataset_release"
SITE_LOAD_REGISTERED_DATASET_TARGET: str = "sensebench.site.build.load_registered_dataset"
SITE_DATASET_RELEASES_TARGET: str = "sensebench.site.build.DATASET_RELEASES"


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

    monkeypatch.setattr(AGGREGATE_GET_DATASET_RELEASE_TARGET, fake_get_dataset_release)
    monkeypatch.setattr(AGGREGATE_LOAD_REGISTERED_DATASET_TARGET, fake_load_registered_dataset)
    monkeypatch.setattr(SITE_GET_DATASET_RELEASE_TARGET, fake_get_dataset_release)
    monkeypatch.setattr(SITE_LOAD_REGISTERED_DATASET_TARGET, fake_load_registered_dataset)
    monkeypatch.setattr(SITE_DATASET_RELEASES_TARGET, {release.release_id: release})
    return release


def _write_verified_run(
    *,
    results_dir: Path,
    dataset: DatasetBundle,
    run_id: RunID,
    model_name: ModelID = TEST_MODEL_NAME,
    content_hash: str | None = None,
    choose_gold: bool = True,
    self_hosted: bool = False,
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
        model=self_hosted_model(model_name=model_name)
        if self_hosted
        else CloudLlmReference(
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
        machine=fixture_machine() if self_hosted else None,
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

    build_site(
        results_dir=results_dir,
        output_dir=output_dir,
        base_url=TEST_BASE_URL,
        strict=True,
    )

    assert (output_dir / INDEX_HTML_FILENAME).exists()
    assert (output_dir / SITE_RUNS_DIRNAME / run_id / INDEX_HTML_FILENAME).exists()
    assert (output_dir / SITE_DATA_DIRNAME / SITE_RUNS_DIRNAME / f"{run_id}.json").exists()
    assert (output_dir / RUN_ARTIFACT_ROOT / run_id / RUN_METADATA_FILENAME).exists()
    assert (output_dir / RUN_ARTIFACT_ROOT / run_id / PREDICTIONS_FILENAME).exists()
    assert (output_dir / RUN_ARTIFACT_ROOT / run_id / CALLS_FILENAME).exists()
    assert (output_dir / SITE_ASSETS_DIRNAME / ECHARTS_VENDOR_PATH).exists()
    assert (output_dir / CNAME_FILENAME).read_text(encoding="utf-8") == (
        f"{DEFAULT_CUSTOM_DOMAIN}\n"
    )
    assert run_id in (output_dir / "sitemap.xml").read_text(encoding="utf-8")

    # Prompts index + per-prompt pages (details, rendered examples, JSON download).
    prompts_index = output_dir / PROMPTS_ROUTE_PREFIX / INDEX_HTML_FILENAME
    assert prompts_index.exists()
    prompts_index_html = prompts_index.read_text(encoding="utf-8")
    assert "prompts/p001/" in prompts_index_html
    assert "prompts/p002/" in prompts_index_html
    assert "prompt-table" in prompts_index_html
    p001_page = output_dir / PROMPTS_ROUTE_PREFIX / "p001" / INDEX_HTML_FILENAME
    assert p001_page.exists()
    assert (output_dir / PROMPTS_ROUTE_PREFIX / "p001" / "p001.json").exists()
    p001_html = p001_page.read_text(encoding="utf-8")
    assert "Rendered Item Examples" in p001_html
    assert "prompts/p001/p001.json" in p001_html
    assert "prompt-code" in p001_html
    # Nav exposes Prompts, and prompt mentions link to the prompt page.
    index_html = (output_dir / INDEX_HTML_FILENAME).read_text(encoding="utf-8")
    assert ">Prompts</a>" in index_html
    assert _reference_rate_sentence() in index_html, (
        "the leaderboard's own method notes must state the rates its cost column is priced at"
    )
    run_html = (output_dir / SITE_RUNS_DIRNAME / run_id / INDEX_HTML_FILENAME).read_text(
        encoding="utf-8"
    )
    assert "prompts/p001/" in run_html
    label_schemes_page = output_dir / LABEL_SCHEMES_ROUTE_PREFIX / INDEX_HTML_FILENAME
    assert label_schemes_page.exists()
    label_schemes_html = label_schemes_page.read_text(encoding="utf-8")
    assert LABEL_SCHEME_TABLE_CLASS in label_schemes_html
    assert LABEL_SCHEME_TABLE_WRAP_CLASS in label_schemes_html
    assert LABEL_SCHEME_CSI_DATA_LABEL in label_schemes_html

    site_data = SiteData.model_validate_json(
        (output_dir / SITE_DATA_DIRNAME / LEADERBOARD_JSON_PATH).read_text(encoding="utf-8")
    )
    assert site_data.schema_version == SITE_DATA_SCHEMA_VERSION
    assert site_data.summary.verified_run_count == 1
    assert site_data.summary.gpus == []
    assert site_data.summary.quantizations == []
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
    mfs = next(baseline for baseline in site_data.baselines if baseline.label == MFS_BASELINE_LABEL)
    assert mfs.kind == BaselineKind.COMPUTED_WORDNET_MFS
    assert mfs.accuracy == 0.0
    assert all(baseline.dataset_version == TEST_RELEASE_ID for baseline in site_data.baselines)

    run_detail = RunDetail.model_validate_json(
        (output_dir / SITE_DATA_DIRNAME / SITE_RUNS_DIRNAME / f"{run_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_detail.schema_version == RUN_DETAIL_SCHEMA_VERSION
    assert run_detail.metadata.run_id == run_id
    assert run_detail.metadata.totals.cost.total_usd == TOTAL_COST_USD
    assert run_detail.correctness_by_scheme["lexen_fine"] == EXPECTED_CORRECTNESS_BITS
    assert {artifact.filename for artifact in run_detail.artifacts} == {
        RUN_METADATA_FILENAME,
        PREDICTIONS_FILENAME,
        CALLS_FILENAME,
    }
    example = run_detail.worst_examples[0]
    assert len(example.context_sentences) == 2
    assert any(
        f"<mark>{TARGET_ART_TEXT}</mark>" in sentence.html for sentence in example.context_sentences
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
    assert (RUN_ARTIFACT_ROOT / TEST_RUN_ID / RUN_METADATA_FILENAME).as_posix() in run_html
    assert MACHINE_TIMING_HEADING_TEXT not in run_html

    index_html = (output_dir / INDEX_HTML_FILENAME).read_text(encoding="utf-8")
    assert REFERENCE_BASELINES_TEXT in index_html
    assert MAX_COST_FILTER_ID in index_html
    assert SOURCE_FILTER_ID in index_html
    assert HOSTING_FILTER_ID in index_html
    assert X_METRIC_SELECT_ID in index_html
    assert X_SCALE_SELECT_ID in index_html
    assert GPU_FILTER_ID not in index_html
    assert QUANT_FILTER_ID not in index_html
    assert BUILT_BY_GLITE_TEXT in index_html
    assert SPEED_COLUMN_HEADER_TEXT not in index_html
    assert SPEED_SORT_BUTTON_TEXT not in index_html
    assert SORT_ARROW_TEXT in index_html
    assert RANK_SORT_BUTTON_TEXT in index_html
    # The model name itself is now the run link (the separate "view" column is gone).
    assert f'title="{run_id}">Fake Model</a>' in index_html
    assert "model-link" in index_html
    assert ">view</a>" not in index_html
    assert f">{run_id}</a>" not in index_html
    # New controls / card affordances.
    assert 'id="sort-select"' in index_html
    assert 'id="compare-bar"' in index_html
    # CSV / JSON download buttons and the client-side export logic that backs them.
    assert f'id="{DOWNLOAD_CSV_BUTTON_ID}"' in index_html
    assert f'id="{DOWNLOAD_JSON_BUTTON_ID}"' in index_html
    assert TABLE_DOWNLOADS_CLASS in index_html
    site_js = (output_dir / SITE_ASSETS_DIRNAME / SITE_JS_FILENAME).read_text(encoding="utf-8")
    assert SITE_JS_EXPORT_COLUMNS_MARKER in site_js
    assert SITE_JS_DOWNLOAD_CSV_MARKER in site_js
    assert DESCRIPTIVE_GLITE_CSV_PREFIX in site_js
    assert DESCRIPTIVE_CSI_CSV_PREFIX in site_js
    assert SITE_JS_SPOT_PRICED_MARKER in site_js
    assert SITE_JS_SPOT_PRICED_CELL_MARKER in site_js, (
        "the cost cell must flag rows that are not priced at a reference rate"
    )
    assert SITE_JS_FRONTIER_BADGE_MARKER in site_js, (
        "the frontier star must carry the caveat when the row is not reference-priced"
    )
    assert COST_FLAG_CLASS in (output_dir / SITE_ASSETS_DIRNAME / "site.css").read_text(
        encoding="utf-8"
    )
    assert COST_FLAG_CLASS in index_html, "method notes must explain the cost flag they reference"
    # TestVendor has no bundled logo, so the colored-initial fallback renders.
    assert 'class="vendor-initial' in index_html


def test_build_site_self_hosted_run(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset = _dataset()
    _patch_registered_dataset(monkeypatch=monkeypatch, dataset=dataset)
    results_dir = tmp_path / SUBMITTED_RESULTS_DIR
    output_dir = tmp_path / SITE_OUTPUT_DIR
    _write_verified_run(
        results_dir=results_dir,
        dataset=dataset,
        run_id=TEST_RUN_ID,
    )
    _write_verified_run(
        results_dir=results_dir,
        dataset=dataset,
        run_id=SELF_HOSTED_RUN_ID,
        model_name=SELF_HOSTED_MODEL_NAME,
        self_hosted=True,
    )

    build_site(
        results_dir=results_dir,
        output_dir=output_dir,
        base_url=TEST_BASE_URL,
        strict=True,
    )

    site_data = SiteData.model_validate_json(
        (output_dir / SITE_DATA_DIRNAME / LEADERBOARD_JSON_PATH).read_text(encoding="utf-8")
    )
    assert site_data.summary.verified_run_count == 2
    assert site_data.summary.gpus == [EXPECTED_GPU_LABEL]
    assert site_data.summary.quantizations == [FIXTURE_QUANTIZATION]
    self_hosted_entry = next(
        entry for entry in site_data.entries if entry.run_id == SELF_HOSTED_RUN_ID
    )
    assert self_hosted_entry.hosting_kind == ModelHostingKind.SELF_HOSTED
    assert self_hosted_entry.gpu == EXPECTED_GPU_LABEL
    assert self_hosted_entry.quantization == FIXTURE_QUANTIZATION
    assert self_hosted_entry.seconds_per_item == LATENCY_SECONDS
    assert self_hosted_entry.machine_hours_per_million_items == LATENCY_SECONDS * 1_000_000 / 3600
    cloud_entry = next(entry for entry in site_data.entries if entry.run_id == TEST_RUN_ID)
    assert cloud_entry.seconds_per_item is None
    assert cloud_entry.machine_hours_per_million_items is None

    index_html = (output_dir / INDEX_HTML_FILENAME).read_text(encoding="utf-8")
    assert GPU_FILTER_ID in index_html
    assert EXPECTED_GPU_LABEL in index_html
    assert QUANT_FILTER_ID in index_html
    assert FIXTURE_QUANTIZATION in index_html
    assert MACHINE_HOURS_TEXT in index_html

    self_hosted_run_html = (
        output_dir / SITE_RUNS_DIRNAME / SELF_HOSTED_RUN_ID / INDEX_HTML_FILENAME
    ).read_text(encoding="utf-8")
    assert MACHINE_TIMING_HEADING_TEXT in self_hosted_run_html
    assert FIXTURE_GPU_NAME in self_hosted_run_html
    assert FIXTURE_PROVIDER in self_hosted_run_html
    assert FIXTURE_INFERENCE_ENGINE in self_hosted_run_html

    cloud_run_html = (output_dir / SITE_RUNS_DIRNAME / TEST_RUN_ID / INDEX_HTML_FILENAME).read_text(
        encoding="utf-8"
    )
    assert MACHINE_TIMING_HEADING_TEXT not in cloud_run_html

    run_detail = RunDetail.model_validate_json(
        (
            output_dir / SITE_DATA_DIRNAME / SITE_RUNS_DIRNAME / f"{SELF_HOSTED_RUN_ID}.json"
        ).read_text(encoding="utf-8")
    )
    assert run_detail.schema_version == RUN_DETAIL_SCHEMA_VERSION
    assert run_detail.metadata.machine is not None
    assert run_detail.metadata.machine.gpu is not None
    assert run_detail.metadata.machine.gpu.name == FIXTURE_GPU_NAME


def test_methodology_page_names_every_reference_rate() -> None:
    """The published methodology must state the rates the leaderboard actually prices at."""
    methodology = next(page for page in _static_pages() if page.slug == METHODOLOGY_SLUG)
    prose = " ".join(
        paragraph for section in methodology.sections for paragraph in section.paragraphs
    )

    for label, rate in GPU_REFERENCE_HOURLY_RATE_USD.items():
        assert f"${rate:.2f}/h for {label}" in prose, (
            f"methodology does not state the {label} reference rate readers are being "
            "ranked by"
        )
    assert GPU_REFERENCE_RATES_AS_OF in prose, "methodology does not date the reference rates"


def test_reference_rate_sentence_is_derived_from_the_registry() -> None:
    sentence = _reference_rate_sentence()

    assert len(GPU_REFERENCE_HOURLY_RATE_USD) > 0
    for label, rate in GPU_REFERENCE_HOURLY_RATE_USD.items():
        assert f"${rate:.2f}/h for {label}" in sentence
    assert GPU_REFERENCE_RATES_AS_OF in sentence, (
        "readers cannot tell how stale the rates are without an as-of date"
    )


def test_reference_rate_sentence_lists_rates_cheapest_first() -> None:
    sentence = _reference_rate_sentence()
    positions = [
        sentence.index(f"${rate:.2f}/h for {label}")
        for label, rate in sorted(GPU_REFERENCE_HOURLY_RATE_USD.items(), key=lambda i: i[1])
    ]

    assert positions == sorted(positions), "rates should read as a price ladder"


def test_format_money_rounds_by_magnitude() -> None:
    assert _format_money(None) == "n/a"
    assert _format_money(20_000.0) == "$20,000"
    assert _format_money(791.0716) == "$791"
    assert _format_money(100.0) == "$100"
    assert _format_money(7.929) == "$7.93"
    assert _format_money(0.5) == "$0.500"
    assert _format_money(0.0042) == "$0.00420"
    assert _format_money(0.0) == "$0.00"


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
        build_site(
            results_dir=results_dir,
            output_dir=tmp_path / SITE_OUTPUT_DIR,
            base_url=TEST_BASE_URL,
            strict=True,
        )


def _page_title(html_text: str) -> str:
    start = html_text.index(TITLE_OPEN_TAG) + len(TITLE_OPEN_TAG)
    return html_text[start : html_text.index(TITLE_CLOSE_TAG, start)]


def test_reasoning_effort_label_omits_absent_and_never_repeats_the_word() -> None:
    assert _reasoning_effort_label(None) is None
    assert _reasoning_effort_label(OMITTED_REASONING_EFFORT) is None
    assert (
        _reasoning_effort_label(SELF_DESCRIBING_REASONING_EFFORT)
        == SELF_DESCRIBING_REASONING_EFFORT
    )
    assert _reasoning_effort_label(XHIGH_REASONING_EFFORT) == XHIGH_REASONING_EFFORT_LABEL


def test_dataset_version_label_prefers_the_registered_display_name() -> None:
    assert _dataset_version_label(DEFAULT_LEXEN_RELEASE_ID) == LEXEN_DISPLAY_LABEL
    assert _dataset_version_label(UNREGISTERED_DATASET_VERSION) == UNREGISTERED_DATASET_VERSION
    assert _dataset_version_label(None) == UNKNOWN_DATASET_VERSION_LABEL


def test_run_page_titles_name_the_task_and_separate_hardware_variants(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset = _dataset()
    _patch_registered_dataset(monkeypatch=monkeypatch, dataset=dataset)
    results_dir = tmp_path / SUBMITTED_RESULTS_DIR
    output_dir = tmp_path / SITE_OUTPUT_DIR
    _write_verified_run(results_dir=results_dir, dataset=dataset, run_id=TEST_RUN_ID)
    _write_verified_run(
        results_dir=results_dir,
        dataset=dataset,
        run_id=SELF_HOSTED_RUN_ID,
        model_name=SELF_HOSTED_MODEL_NAME,
        self_hosted=True,
    )

    build_site(
        results_dir=results_dir,
        output_dir=output_dir,
        base_url=TEST_BASE_URL,
        strict=True,
    )

    cloud_title = _page_title(
        (output_dir / SITE_RUNS_DIRNAME / TEST_RUN_ID / INDEX_HTML_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    self_hosted_title = _page_title(
        (output_dir / SITE_RUNS_DIRNAME / SELF_HOSTED_RUN_ID / INDEX_HTML_FILENAME).read_text(
            encoding="utf-8"
        )
    )

    assert WSD_TASK_PHRASE_TEXT in cloud_title
    assert WSD_TASK_PHRASE_TEXT in self_hosted_title
    assert cloud_title != self_hosted_title
    assert FIXTURE_QUANTIZATION in self_hosted_title
    assert EXPECTED_GPU_LABEL in self_hosted_title

    site_data = SiteData.model_validate_json(
        (output_dir / SITE_DATA_DIRNAME / LEADERBOARD_JSON_PATH).read_text(encoding="utf-8")
    )
    self_hosted_entry = next(
        entry for entry in site_data.entries if entry.run_id == SELF_HOSTED_RUN_ID
    )
    requantized_entry = self_hosted_entry.model_copy(
        update={"quantization": ALTERNATE_QUANTIZATION}
    )
    assert _run_page_title(requantized_entry) != _run_page_title(self_hosted_entry)

    rerun_entry = self_hosted_entry.model_copy(update={"created_at": RERUN_CREATED_AT})
    assert _run_page_title(rerun_entry) != _run_page_title(self_hosted_entry)
    assert RERUN_DATE_LABEL in _run_page_title(rerun_entry)

    description = _run_page_description(self_hosted_entry)
    assert WSD_TASK_PHRASE_TEXT in description
    assert ITEMS_CORRECT_TEXT in description
    assert FIXTURE_QUANTIZATION in description
