"""Build the static SenseBench leaderboard website."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from html import escape
from importlib.resources import files
from importlib.resources.abc import Traversable
from math import floor, log10
from pathlib import Path
from shutil import copy2, rmtree
from typing import NamedTuple, assert_never
from urllib.parse import urljoin, urlparse

from jinja2 import Environment, PackageLoader
from pydantic import BaseModel, ConfigDict

from sensebench.datasets.context import build_context_window, build_dataset_index
from sensebench.datasets.detokenize import DetokenizedPiece, detokenize_pieces
from sensebench.datasets.models import (
    DatasetBundle,
    DatasetIndex,
    ItemID,
    SenseKey,
    Sentence,
    WsdItem,
)
from sensebench.datasets.releases import (
    DATASET_RELEASES,
    get_dataset_release,
    load_registered_dataset,
)
from sensebench.leaderboard.aggregate import (
    LeaderboardBuildError,
    LeaderboardCollection,
    LeaderboardEntry,
    collect_leaderboard_entries,
)
from sensebench.leaderboard.baselines import Baseline, BaselineKind, score_baselines
from sensebench.leaderboard.schemes import (
    DEFAULT_SCHEME_ID,
    GOLD_SOURCE_LABELS,
    GRANULARITY_LABELS,
    SCHEMES,
    GoldSource,
    Granularity,
    gold_fine_keys,
    is_scoreable,
    load_concept_map,
    scheme_correct,
)
from sensebench.paths import (
    CALLS_FILENAME,
    CNAME_FILENAME,
    DEFAULT_LEXEN_RELEASE_ID,
    INDEX_HTML_FILENAME,
    LEADERBOARD_JSON_PATH,
    NOT_FOUND_FILENAME,
    P001_PROMPT_PATH,
    PREDICTIONS_FILENAME,
    PROMPT_JSON_SUFFIX,
    PROMPT_REGISTRY_DIR,
    ROBOTS_FILENAME,
    RUN_ARTIFACT_ROOT,
    RUN_METADATA_FILENAME,
    SITE_ASSETS_DIRNAME,
    SITE_DATA_DIRNAME,
    SITE_RUNS_DIRNAME,
    SITEMAP_FILENAME,
    SUBMITTED_RESULTS_DIR,
)
from sensebench.prompts.models import MessageRole, PromptDefinition
from sensebench.prompts.registry import load_prompt_definition, registered_prompt_paths
from sensebench.prompts.render import render_task
from sensebench.runner.evaluate import prediction_is_correct
from sensebench.runs.loaders import LoadedRun, load_run_directory
from sensebench.runs.models import (
    CallID,
    CallRecord,
    CallStatus,
    ModelSourceKind,
    PredictionRecord,
    RunID,
    RunMetadata,
    VoteStatus,
)
from sensebench.wordnet import SenseCandidate, SynsetID, get_candidate_senses

DEFAULT_CUSTOM_DOMAIN: str = "sense-bench.com"
DEFAULT_SITE_BASE_URL: str = f"https://{DEFAULT_CUSTOM_DOMAIN}/"
DEFAULT_REPOSITORY_TREE_URL: str = "https://github.com/GliteTech/sensebench/tree/main"
SITE_DATA_SCHEMA_VERSION: str = "sensebench-site-data-v7"
RUN_DETAIL_SCHEMA_VERSION: str = "sensebench-run-detail-v7"
MAX_ERROR_EXAMPLES: int = 12
PACKAGE_NAME: str = "sensebench.site"
TEMPLATE_PACKAGE_PATH: str = "templates"
STATIC_PACKAGE_PATH: str = "static"
PCT_FILTER_NAME: str = "pct"
NUM_FILTER_NAME: str = "num"
MONEY_FILTER_NAME: str = "money"
MILLION_TOKEN_PRICE_FILTER_NAME: str = "million_token_price"
SECONDS_FILTER_NAME: str = "seconds"
MACHINE_HOURS_FILTER_NAME: str = "machine_hours"
BYTES_FILTER_NAME: str = "bytes"
SOURCE_LABEL_FILTER_NAME: str = "source_label"
SOURCE_KIND_LABELS: dict[ModelSourceKind, str] = {
    ModelSourceKind.OPEN_SOURCE: "Open weights",
    ModelSourceKind.PROPRIETARY: "Proprietary",
}
UNKNOWN_SOURCE_LABEL: str = "Unknown source"
BASELINE_KIND_LABEL_FILTER_NAME: str = "baseline_kind_label"
BASELINE_KIND_LABELS: dict[BaselineKind, str] = {
    BaselineKind.COMPUTED_WORDNET_MFS: "Computed at build time",
    BaselineKind.PUBLISHED_PREDICTIONS: "Published predictions",
    BaselineKind.REPRODUCED_PREDICTIONS: "Reproduced predictions",
}
CORRECT_BIT: str = "1"
INCORRECT_BIT: str = "0"
MONEY_SMALL_VALUE_SIGNIFICANT_FIGURES: int = 3
PAGE_CONTEXT_KEY: str = "page"
FRONTIER_RUN_IDS_CONTEXT_KEY: str = "frontier_run_ids"
PROMPT_CONTEXT_KEY: str = "prompt"
PARAMS_JSON_CONTEXT_KEY: str = "params_json"
PROMPT_EXAMPLES_CONTEXT_KEY: str = "prompt_examples"
PROMPT_DOWNLOAD_CONTEXT_KEY: str = "download_name"
PROMPTS_CONTEXT_KEY: str = "prompts"
PROMPT_EXAMPLE_COUNT: int = 3
DETAIL_CONTEXT_KEY: str = "detail"
REPOSITORY_ARTIFACT_URL_CONTEXT_KEY: str = "repository_artifact_url"
ENTRIES_CONTEXT_KEY: str = "entries"
SITE_DATA_CONTEXT_KEY: str = "site_data"
DATASETS_CONTEXT_KEY: str = "datasets"
ASSET_VERSION_GLOBAL_KEY: str = "asset_version"
SOURCE_DATASET_METADATA_KEY: str = "source_dataset"
ERROR_BUCKET_GROUP: str = "Error"
ANY_BUCKET_VALUE: str = "Any"
RUNS_ROUTE: str = "runs/"
PROMPTS_ROUTE_PREFIX: str = "prompts"
RUNS_ROUTE_PREFIX: str = "runs"
STATIC_PAGE_SLUGS: tuple[str, ...] = (
    "about",
    "methodology",
    "submit",
    "changelog",
    "citation",
)


class SliceGroup(StrEnum):
    POS = "POS"
    CANDIDATE_COUNT = "Candidate Count"
    SOURCE_DATASET = "Source Dataset"


class SliceKey(NamedTuple):
    group: SliceGroup
    value: str | None


class SiteModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SiteSummary(SiteModel):
    verified_run_count: int
    model_count: int
    dataset_versions: list[str]
    prompt_ids: list[str]
    gpus: list[str]
    quantizations: list[str]
    top_accuracy: float | None
    generated_at: str


class SiteData(SiteModel):
    schema_version: str
    summary: SiteSummary
    entries: list[LeaderboardEntry]
    baselines: list[Baseline]


class SliceSummary(SiteModel):
    group: SliceGroup
    value: str | None
    correct_count: int
    item_count: int
    accuracy: float | None


class ExampleContextSentence(SiteModel):
    html: str
    is_target_sentence: bool


class ExampleCandidate(SiteModel):
    index: int
    sense_key: str
    synset_id: SynsetID
    definition: str | None
    synonyms: list[str]
    examples: list[str]
    is_gold: bool
    is_gold_maru2022: bool
    is_gold_raganato: bool
    is_selected: bool


class ExamplePromptMessage(SiteModel):
    role: MessageRole
    content: str


class RunExample(SiteModel):
    bucket_group: SliceGroup | str
    bucket_value: str | None
    item_id: ItemID
    lemma: str
    target_text: str
    pos: str
    source_dataset: str | None
    candidate_count: int
    is_correct: bool | None
    predicted_sense_index: int | None
    predicted_sense_key: str | None
    gold_sense_keys: list[str]
    gold_sense_keys_maru2022: list[str]
    gold_sense_keys_raganato: list[str]
    predicted_matches_fine: dict[str, bool | None]
    context_sentences: list[ExampleContextSentence]
    candidates: list[ExampleCandidate]
    prompt_messages: list[ExamplePromptMessage]


class RunArtifact(SiteModel):
    label: str
    filename: str
    path: str
    size_bytes: int
    description: str


class RunDetail(SiteModel):
    schema_version: str
    entry: LeaderboardEntry
    metadata: RunMetadata
    artifacts: list[RunArtifact]
    slices: list[SliceSummary]
    worst_examples: list[RunExample]
    correctness_by_scheme: dict[str, str]


@dataclass(frozen=True, slots=True)
class PageSection:
    title: str
    paragraphs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StaticPage:
    slug: str
    title: str
    description: str
    sections: tuple[PageSection, ...]


@dataclass(frozen=True, slots=True)
class RunArtifactSpec:
    filename: str
    label: str
    description: str


RUN_ARTIFACT_SPECS: tuple[RunArtifactSpec, ...] = (
    RunArtifactSpec(
        filename=RUN_METADATA_FILENAME,
        label="Run Metadata",
        description="Submitted run.json with model, dataset, prompt, policy, totals, and costs.",
    ),
    RunArtifactSpec(
        filename=PREDICTIONS_FILENAME,
        label="Predictions",
        description="One JSON record per benchmark item with candidates, votes, and correctness.",
    ),
    RunArtifactSpec(
        filename=CALLS_FILENAME,
        label="Raw Calls",
        description="Gzipped JSONL call log with prompts, raw outputs, token usage, and costs.",
    ),
)


def _base_url(*, base_url: str) -> str:
    if base_url.endswith("/"):
        return base_url
    return f"{base_url}/"


def _base_path(*, base_url: str) -> str:
    parsed = urlparse(_base_url(base_url=base_url))
    path = parsed.path
    if len(path) == 0:
        return "/"
    if not path.startswith("/"):
        path = f"/{path}"
    if not path.endswith("/"):
        path = f"{path}/"
    return path


def _absolute_url(*, base_url: str, path: str) -> str:
    return urljoin(base=_base_url(base_url=base_url), url=path)


def _template_env() -> Environment:
    env = Environment(
        loader=PackageLoader(
            package_name=PACKAGE_NAME,
            package_path=TEMPLATE_PACKAGE_PATH,
        ),
        autoescape=True,
    )
    env.filters[PCT_FILTER_NAME] = _format_percent
    env.filters[NUM_FILTER_NAME] = _format_number
    env.filters[MONEY_FILTER_NAME] = _format_money
    env.filters[MILLION_TOKEN_PRICE_FILTER_NAME] = _format_million_token_price
    env.filters[SECONDS_FILTER_NAME] = _format_seconds
    env.filters[MACHINE_HOURS_FILTER_NAME] = _format_machine_hours
    env.filters[BYTES_FILTER_NAME] = _format_bytes
    env.filters[SOURCE_LABEL_FILTER_NAME] = _format_source_label
    env.filters[BASELINE_KIND_LABEL_FILTER_NAME] = _format_baseline_kind
    env.globals["scheme_gold_sources"] = [
        {"value": source.value, "label": GOLD_SOURCE_LABELS[source]}
        for source in (GoldSource.LEXEN, GoldSource.MARU2022, GoldSource.RAGANATO)
    ]
    env.globals["scheme_granularities"] = [
        {"value": granularity.value, "label": GRANULARITY_LABELS[granularity]}
        for granularity in (Granularity.FINE, Granularity.COARSE, Granularity.CSI)
    ]
    env.globals["default_scheme_id"] = DEFAULT_SCHEME_ID
    return env


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    return f"{value:,.4f}"


def _format_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 100:
        return f"${value:,.0f}"
    if value >= 1:
        return f"${value:,.2f}"
    if value <= 0:
        return "$0.00"
    decimals = (MONEY_SMALL_VALUE_SIGNIFICANT_FIGURES - 1) - floor(log10(value))
    return f"${value:.{decimals}f}"


def _format_source_label(value: str | None) -> str:
    if value is None:
        return UNKNOWN_SOURCE_LABEL
    try:
        source_kind = ModelSourceKind(value)
    except ValueError:
        return UNKNOWN_SOURCE_LABEL
    return SOURCE_KIND_LABELS.get(source_kind, UNKNOWN_SOURCE_LABEL)


def _format_baseline_kind(value: str) -> str:
    try:
        baseline_kind = BaselineKind(value)
    except ValueError:
        return value
    return BASELINE_KIND_LABELS.get(baseline_kind, value)


def _format_million_token_price(value: float | None) -> str:
    if value is None:
        return "n/a"
    return _format_money(value * 1_000_000)


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 60:
        return f"{value:.2f}s"
    minutes = value / 60
    return f"{minutes:.2f}m"


def _format_machine_hours(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 10:
        return f"{value:.2f} h/M"
    return f"{value:,.1f} h/M"


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    if value < 1024:
        return f"{value} B"
    if value < 1024 * 1024:
        return f"{value / 1024:.1f} KB"
    return f"{value / (1024 * 1024):.1f} MB"


def _write_text(*, path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data=text, encoding="utf-8")


def _write_json(*, path: Path, value: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = value.model_dump_json(indent=2)
    path.write_text(data=f"{serialized}\n", encoding="utf-8")


def _page_file_path(*, output_dir: Path, route: str) -> Path:
    return output_dir / Path(route) / INDEX_HTML_FILENAME


def _run_artifact_route(*, run_id: RunID, filename: str) -> str:
    return (RUN_ARTIFACT_ROOT / run_id / filename).as_posix()


def _copy_tree(*, source: Traversable, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        child_target = target / child.name
        if child.is_dir():
            _copy_tree(source=child, target=child_target)
        else:
            child_target.write_bytes(child.read_bytes())


def _copy_static_assets(*, output_dir: Path) -> None:
    static_root = files(PACKAGE_NAME).joinpath(STATIC_PACKAGE_PATH)
    _copy_tree(source=static_root, target=output_dir / SITE_ASSETS_DIRNAME)


def _copy_run_artifacts(
    *,
    output_dir: Path,
    run_dir: Path,
    run_id: RunID,
) -> list[RunArtifact]:
    copied: list[RunArtifact] = []
    target_dir = output_dir / RUN_ARTIFACT_ROOT / run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    for spec in RUN_ARTIFACT_SPECS:
        source = run_dir / spec.filename
        if not source.exists():
            continue
        target = target_dir / spec.filename
        copy2(src=source, dst=target)
        copied.append(
            RunArtifact(
                label=spec.label,
                filename=spec.filename,
                path=_run_artifact_route(run_id=run_id, filename=spec.filename),
                size_bytes=source.stat().st_size,
                description=spec.description,
            )
        )
    return copied


def _site_summary(*, collection: LeaderboardCollection) -> SiteSummary:
    entries: list[LeaderboardEntry] = collection.entries
    top_accuracy = entries[0].accuracy if len(entries) > 0 else None
    return SiteSummary(
        verified_run_count=len(entries),
        model_count=len({entry.model for entry in entries}),
        dataset_versions=sorted(
            {entry.dataset_version for entry in entries if entry.dataset_version is not None}
        ),
        prompt_ids=sorted({entry.prompt_id for entry in entries}),
        gpus=sorted({entry.gpu for entry in entries if entry.gpu is not None}),
        quantizations=sorted(
            {entry.quantization for entry in entries if entry.quantization is not None}
        ),
        top_accuracy=top_accuracy,
        generated_at=datetime.now(tz=UTC).isoformat(),
    )


def _site_data(*, collection: LeaderboardCollection, baselines: list[Baseline]) -> SiteData:
    return SiteData(
        schema_version=SITE_DATA_SCHEMA_VERSION,
        summary=_site_summary(collection=collection),
        entries=collection.entries,
        baselines=baselines,
    )


def _site_baselines(
    *,
    collection: LeaderboardCollection,
    dataset_cache: dict[str, DatasetBundle],
) -> list[Baseline]:
    versions: list[str] = sorted(
        {entry.dataset_version for entry in collection.entries if entry.dataset_version is not None}
    )
    baselines: list[Baseline] = []
    for version in versions:
        baselines.extend(
            score_baselines(dataset=_dataset_for_version(version=version, cache=dataset_cache))
        )
    return baselines


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    run_id: RunID
    accuracy: float
    cost_per_million_items: float


def _pareto_frontier_run_ids(*, entries: list[LeaderboardEntry]) -> set[RunID]:
    points: list[FrontierPoint] = [
        FrontierPoint(
            run_id=entry.run_id,
            accuracy=entry.accuracy,
            cost_per_million_items=entry.cost_per_million_items,
        )
        for entry in entries
        if entry.accuracy is not None and entry.cost_per_million_items is not None
    ]
    frontier: set[RunID] = set()
    for point in points:
        dominated = any(
            other.accuracy >= point.accuracy
            and other.cost_per_million_items <= point.cost_per_million_items
            and (
                other.accuracy > point.accuracy
                or other.cost_per_million_items < point.cost_per_million_items
            )
            for other in points
        )
        if not dominated:
            frontier.add(point.run_id)
    return frontier


def _candidate_bucket(*, count: int) -> str:
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 5:
        return "3-5"
    if count <= 10:
        return "6-10"
    return "11+"


def _accuracy(*, predictions: list[PredictionRecord]) -> float | None:
    if len(predictions) == 0:
        return None
    correct_count = sum(1 for prediction in predictions if prediction.is_correct is True)
    return correct_count / len(predictions)


def _slice_summary(
    *,
    group: SliceGroup,
    value: str | None,
    predictions: list[PredictionRecord],
) -> SliceSummary:
    correct_count = sum(1 for prediction in predictions if prediction.is_correct is True)
    return SliceSummary(
        group=group,
        value=value,
        correct_count=correct_count,
        item_count=len(predictions),
        accuracy=_accuracy(predictions=predictions),
    )


def _source_dataset(*, item: WsdItem) -> str | None:
    value = item.metadata.get(SOURCE_DATASET_METADATA_KEY)
    if value is None or len(value) == 0:
        return None
    return value


def _slice_value(
    *,
    group: SliceGroup,
    prediction: PredictionRecord,
    item: WsdItem,
) -> str | None:
    match group:
        case SliceGroup.POS:
            return item.pos
        case SliceGroup.CANDIDATE_COUNT:
            return _candidate_bucket(count=len(prediction.candidates))
        case SliceGroup.SOURCE_DATASET:
            return _source_dataset(item=item)
        case _:
            assert_never(group)


def _slice_summaries(*, loaded: LoadedRun, dataset: DatasetBundle) -> list[SliceSummary]:
    items_by_id: dict[ItemID, WsdItem] = {item.item_id: item for item in dataset.items}
    groups: dict[SliceKey, list[PredictionRecord]] = defaultdict(list)
    for prediction in loaded.predictions:
        item = items_by_id.get(prediction.item_id)
        if item is None:
            continue
        for group in SliceGroup:
            groups[
                SliceKey(
                    group=group,
                    value=_slice_value(group=group, prediction=prediction, item=item),
                )
            ].append(
                prediction,
            )
    summaries: list[SliceSummary] = [
        _slice_summary(group=key.group, value=key.value, predictions=predictions)
        for key, predictions in groups.items()
    ]
    return sorted(
        summaries,
        key=lambda summary: (
            summary.group.value,
            summary.value if summary.value is not None else "",
        ),
    )


def _context_sentences(
    *,
    index: DatasetIndex,
    item: WsdItem,
    prompt: PromptDefinition,
) -> list[ExampleContextSentence] | None:
    try:
        build_context_window(
            index=index,
            item=item,
            previous_sentences=prompt.params.previous_sentences,
            next_sentences=prompt.params.next_sentences,
        )
    except (KeyError, ValueError):
        return None

    document = index.documents_by_id[item.document_id]
    sentence_indexes = index.document_sentence_indexes[item.document_id]
    target_sentence_index = sentence_indexes[item.sentence_id]
    first_sentence_index = max(0, target_sentence_index - prompt.params.previous_sentences)
    last_sentence_exclusive = min(
        len(document.sentences),
        target_sentence_index + prompt.params.next_sentences + 1,
    )
    context: list[ExampleContextSentence] = []
    has_marked_target = False
    for sentence_index in range(first_sentence_index, last_sentence_exclusive):
        sentence = document.sentences[sentence_index]
        is_target_sentence = sentence_index == target_sentence_index
        target_token_index = item.target_token_index if is_target_sentence else None
        sentence_html = _context_sentence_html(
            sentence=sentence,
            target_token_index=target_token_index,
            detokenize=prompt.params.detokenize,
        )
        if target_token_index is not None:
            has_marked_target = True
        context.append(
            ExampleContextSentence(
                html=sentence_html,
                is_target_sentence=is_target_sentence,
            )
        )
    if not has_marked_target:
        return None
    return context


def _context_sentence_html(
    *,
    sentence: Sentence,
    target_token_index: int | None,
    detokenize: bool,
) -> str:
    surfaces: list[str] = [token.text for token in sentence.tokens]
    pieces: list[DetokenizedPiece] | None = (
        detokenize_pieces(surfaces=surfaces) if detokenize else None
    )
    html = ""
    for token_index, token in enumerate(sentence.tokens):
        rendered = pieces[token_index].text if pieces is not None else token.text
        if token_index == target_token_index:
            token_html = f"<mark>{escape(token.text)}</mark>"
        else:
            token_html = escape(rendered)
        leading_space = pieces[token_index].leading_space if pieces is not None else token_index > 0
        if leading_space and len(html) > 0:
            html += " "
        html += token_html
    return html


def _calls_by_id(*, calls: list[CallRecord]) -> dict[CallID, CallRecord]:
    return {call.call_id: call for call in calls}


def _calls_by_item(*, calls: list[CallRecord]) -> dict[ItemID, list[CallRecord]]:
    grouped: dict[ItemID, list[CallRecord]] = defaultdict(list)
    for call in calls:
        grouped[call.item_id].append(call)
    return grouped


def _call_for_prediction(
    *,
    prediction: PredictionRecord,
    calls_by_id: dict[CallID, CallRecord],
    calls_by_item: dict[ItemID, list[CallRecord]],
) -> CallRecord | None:
    for vote in prediction.votes:
        if vote.status != VoteStatus.SUCCESS:
            continue
        for call_id in vote.call_ids:
            call = calls_by_id.get(call_id)
            if call is not None and call.status == CallStatus.SUCCESS:
                return call

    item_calls = calls_by_item.get(prediction.item_id, [])
    for call in item_calls:
        if call.status == CallStatus.SUCCESS:
            return call
    if len(item_calls) > 0:
        return item_calls[0]
    return None


def _prompt_messages_for_prediction(
    *,
    prediction: PredictionRecord,
    calls_by_id: dict[CallID, CallRecord],
    calls_by_item: dict[ItemID, list[CallRecord]],
) -> list[ExamplePromptMessage]:
    call = _call_for_prediction(
        prediction=prediction,
        calls_by_id=calls_by_id,
        calls_by_item=calls_by_item,
    )
    if call is None:
        return []
    return [
        ExamplePromptMessage(
            role=message.role,
            content=message.content,
        )
        for message in call.messages
    ]


def _example_candidates(
    *,
    prediction: PredictionRecord,
    item: WsdItem,
) -> list[ExampleCandidate]:
    wordnet_by_key: dict[SenseKey, SenseCandidate] = {
        candidate.sense_key: candidate
        for candidate in get_candidate_senses(lemma=item.lemma, pos=item.pos)
    }
    maru2022_gold: set[SenseKey] = set(gold_fine_keys(item=item, gold_source=GoldSource.MARU2022))
    raganato_gold: set[SenseKey] = set(gold_fine_keys(item=item, gold_source=GoldSource.RAGANATO))
    candidates: list[ExampleCandidate] = []
    for candidate in prediction.candidates:
        wordnet_candidate = wordnet_by_key.get(candidate.sense_key)
        candidates.append(
            ExampleCandidate(
                index=candidate.index,
                sense_key=candidate.sense_key,
                synset_id=candidate.synset_id,
                definition=wordnet_candidate.definition if wordnet_candidate is not None else None,
                synonyms=wordnet_candidate.synonyms if wordnet_candidate is not None else [],
                examples=wordnet_candidate.examples if wordnet_candidate is not None else [],
                is_gold=candidate.sense_key in prediction.gold_sense_keys,
                is_gold_maru2022=candidate.sense_key in maru2022_gold,
                is_gold_raganato=candidate.sense_key in raganato_gold,
                is_selected=candidate.sense_key == prediction.predicted_sense_key,
            )
        )
    return candidates


def _predicted_matches_fine(
    *,
    predicted_sense_key: SenseKey | None,
    item: WsdItem,
) -> dict[str, bool | None]:
    matches: dict[str, bool | None] = {}
    for source in (GoldSource.LEXEN, GoldSource.MARU2022, GoldSource.RAGANATO):
        gold_keys = gold_fine_keys(item=item, gold_source=source)
        if len(gold_keys) == 0:
            matches[source.value] = None
        else:
            matches[source.value] = prediction_is_correct(
                predicted_sense_key=predicted_sense_key,
                gold_sense_keys=gold_keys,
            )
    return matches


def _example(
    *,
    bucket_group: SliceGroup | str,
    bucket_value: str | None,
    prediction: PredictionRecord,
    item: WsdItem,
    index: DatasetIndex,
    prompt: PromptDefinition,
    calls_by_id: dict[CallID, CallRecord],
    calls_by_item: dict[ItemID, list[CallRecord]],
) -> RunExample | None:
    context_sentences = _context_sentences(
        index=index,
        item=item,
        prompt=prompt,
    )
    if context_sentences is None:
        return None
    return RunExample(
        bucket_group=bucket_group,
        bucket_value=bucket_value,
        item_id=prediction.item_id,
        lemma=item.lemma,
        target_text=item.target_text,
        pos=item.pos,
        source_dataset=_source_dataset(item=item),
        candidate_count=len(prediction.candidates),
        is_correct=prediction.is_correct,
        predicted_sense_index=prediction.predicted_sense_index,
        predicted_sense_key=prediction.predicted_sense_key,
        gold_sense_keys=list(prediction.gold_sense_keys),
        gold_sense_keys_maru2022=gold_fine_keys(item=item, gold_source=GoldSource.MARU2022),
        gold_sense_keys_raganato=gold_fine_keys(item=item, gold_source=GoldSource.RAGANATO),
        predicted_matches_fine=_predicted_matches_fine(
            predicted_sense_key=prediction.predicted_sense_key, item=item
        ),
        context_sentences=context_sentences,
        candidates=_example_candidates(prediction=prediction, item=item),
        prompt_messages=_prompt_messages_for_prediction(
            prediction=prediction,
            calls_by_id=calls_by_id,
            calls_by_item=calls_by_item,
        ),
    )


def _worst_examples(
    *,
    loaded: LoadedRun,
    dataset: DatasetBundle,
    prompt: PromptDefinition,
    slices: list[SliceSummary],
) -> list[RunExample]:
    index = build_dataset_index(bundle=dataset)
    items_by_id: dict[ItemID, WsdItem] = {item.item_id: item for item in dataset.items}
    calls_by_id: dict[CallID, CallRecord] = _calls_by_id(calls=loaded.calls)
    calls_by_item: dict[ItemID, list[CallRecord]] = _calls_by_item(calls=loaded.calls)
    errors: list[PredictionRecord] = [
        prediction
        for prediction in sorted(loaded.predictions, key=lambda candidate: candidate.item_id)
        if prediction.is_correct is False and prediction.item_id in items_by_id
    ]
    selected: list[RunExample] = []
    selected_items: set[ItemID] = set()
    ranked_slices: list[SliceSummary] = sorted(
        [
            summary
            for summary in slices
            if summary.group in (SliceGroup.POS, SliceGroup.CANDIDATE_COUNT)
            and summary.accuracy is not None
        ],
        key=lambda summary: (
            summary.accuracy,
            -summary.item_count,
            summary.group.value,
            summary.value if summary.value is not None else "",
        ),
    )
    for summary in ranked_slices:
        for prediction in errors:
            if prediction.item_id in selected_items:
                continue
            item = items_by_id[prediction.item_id]
            if _slice_value(group=summary.group, prediction=prediction, item=item) != summary.value:
                continue
            example = _example(
                bucket_group=summary.group,
                bucket_value=summary.value,
                prediction=prediction,
                item=item,
                index=index,
                prompt=prompt,
                calls_by_id=calls_by_id,
                calls_by_item=calls_by_item,
            )
            if example is None:
                continue
            selected.append(example)
            selected_items.add(prediction.item_id)
            if len(selected) >= MAX_ERROR_EXAMPLES:
                return selected
    for prediction in errors:
        if prediction.item_id in selected_items:
            continue
        item = items_by_id[prediction.item_id]
        example = _example(
            bucket_group=ERROR_BUCKET_GROUP,
            bucket_value=ANY_BUCKET_VALUE,
            prediction=prediction,
            item=item,
            index=index,
            prompt=prompt,
            calls_by_id=calls_by_id,
            calls_by_item=calls_by_item,
        )
        if example is None:
            continue
        selected.append(example)
        if len(selected) >= MAX_ERROR_EXAMPLES:
            break
    return selected


def _dataset_for_version(*, version: str, cache: dict[str, DatasetBundle]) -> DatasetBundle:
    dataset = cache.get(version)
    if dataset is None:
        release = get_dataset_release(release_id=version)
        dataset = load_registered_dataset(release=release)
        cache[version] = dataset
    return dataset


def _dataset_for_entry(
    *,
    entry: LeaderboardEntry,
    cache: dict[str, DatasetBundle],
) -> DatasetBundle:
    if entry.dataset_version is None:
        raise ValueError(f"run {entry.run_id} has no dataset_version")
    return _dataset_for_version(version=entry.dataset_version, cache=cache)


def _correctness_by_scheme(*, loaded: LoadedRun, dataset: DatasetBundle) -> dict[str, str]:
    concept_map = load_concept_map()
    predicted_by_item: dict[ItemID, SenseKey | None] = {
        prediction.item_id: prediction.predicted_sense_key for prediction in loaded.predictions
    }
    result: dict[str, str] = {}
    for scheme in SCHEMES:
        bits: list[str] = []
        for item in dataset.items:
            predicted = predicted_by_item.get(item.item_id)
            correct = is_scoreable(item=item, gold_source=scheme.gold_source) and scheme_correct(
                scheme=scheme,
                predicted_sense_key=predicted,
                item=item,
                concept_map=concept_map,
            )
            bits.append(CORRECT_BIT if correct else INCORRECT_BIT)
        result[scheme.scheme_id] = "".join(bits)
    return result


def _run_detail(
    *,
    entry: LeaderboardEntry,
    results_dir: Path,
    dataset_cache: dict[str, DatasetBundle],
    artifacts: list[RunArtifact],
) -> RunDetail:
    loaded = load_run_directory(run_dir=results_dir / entry.run_id)
    dataset = _dataset_for_entry(entry=entry, cache=dataset_cache)
    prompt = load_prompt_definition(
        path=P001_PROMPT_PATH
        if entry.prompt_id == P001_PROMPT_PATH.stem
        else PROMPT_REGISTRY_DIR / f"{entry.prompt_id}{PROMPT_JSON_SUFFIX}",
    )
    slices = _slice_summaries(loaded=loaded, dataset=dataset)
    return RunDetail(
        schema_version=RUN_DETAIL_SCHEMA_VERSION,
        entry=entry,
        metadata=loaded.metadata,
        artifacts=artifacts,
        slices=slices,
        worst_examples=_worst_examples(
            loaded=loaded,
            dataset=dataset,
            prompt=prompt,
            slices=slices,
        ),
        correctness_by_scheme=_correctness_by_scheme(loaded=loaded, dataset=dataset),
    )


def _render(
    *,
    env: Environment,
    template_name: str,
    base_url: str,
    title: str,
    description: str,
    path: str,
    context: dict[str, object],
) -> str:
    template = env.get_template(template_name)
    return template.render(
        base_path=_base_path(base_url=base_url),
        canonical_url=_absolute_url(base_url=base_url, path=path),
        title=title,
        description=description,
        default_dataset_version=DEFAULT_LEXEN_RELEASE_ID,
        **context,
    )


def _static_pages() -> list[StaticPage]:
    return [
        StaticPage(
            slug="about",
            title="About SenseBench",
            description="What SenseBench measures and why the leaderboard exists.",
            sections=(
                PageSection(
                    title="Purpose",
                    paragraphs=(
                        "SenseBench evaluates English word sense disambiguation with auditable "
                        "LLM run artifacts.",
                        "Each model receives a target word in context and candidate WordNet "
                        "senses, then returns the chosen sense index.",
                    ),
                ),
                PageSection(
                    title="Design",
                    paragraphs=(
                        "The leaderboard is static, reproducible, and rebuilt from verified "
                        "submissions.",
                        "Scores are recomputed from predictions and checked against the "
                        "registered dataset and prompt.",
                    ),
                ),
            ),
        ),
        StaticPage(
            slug="methodology",
            title="Methodology",
            description="SenseBench scoring, verification, and leaderboard ranking rules.",
            sections=(
                PageSection(
                    title="Scoring",
                    paragraphs=(
                        "Accuracy is the fraction of dataset items whose predicted WordNet "
                        "sense key matches the gold sense key set.",
                        "Confidence intervals are bootstrap intervals over item correctness "
                        "with a fixed seed, shown as a ± half-width next to accuracy.",
                        "Rank ranges list the positions a run could plausibly occupy among "
                        "the visible rows given overlapping 95% confidence intervals.",
                        "The compare view tests paired per-item differences between runs on "
                        "the same dataset version with McNemar's test, which is far more "
                        "sensitive than comparing overlapping intervals.",
                        "Reference baselines (MFS, BEM, ESCHER, ConSeC, Glite LENS) are "
                        "scored from per-item system predictions on exactly the same dataset "
                        "items as the model runs.",
                    ),
                ),
                PageSection(
                    title="Verification",
                    paragraphs=(
                        "Every public run is reloaded, replayed, and checked before it "
                        "appears on the site.",
                        "Verification checks run metadata, prompt references, dataset "
                        "hashes, candidate sets, raw output extraction, vote decisions, "
                        "and correctness.",
                    ),
                ),
                PageSection(
                    title="Ranking",
                    paragraphs=(
                        "Runs sort by higher accuracy, then lower cost per million items "
                        "when available, then newer creation time.",
                        "The default leaderboard view lists every verified run; the "
                        "collapsed view keeps only the best verified run per model and "
                        "dataset version, across prompts and reasoning efforts.",
                    ),
                ),
                PageSection(
                    title="Self-Hosted Runs",
                    paragraphs=(
                        "Self-hosted runs record the GPU machine they ran on and a "
                        "benchmark time that covers only the per-item evaluation loop, "
                        "excluding model download, weight loading, and inference engine "
                        "startup.",
                        "Machine-hours per 1M items is the benchmark time divided by the "
                        "item count, scaled to one million items and expressed in machine "
                        "hours; it is comparable only across runs on the same GPU "
                        "configuration.",
                        "When the machine's hourly rate is known, run cost is estimated "
                        "as machine time multiplied by that rate (cost source "
                        "machine_time_estimate); otherwise cost is unavailable.",
                    ),
                ),
                PageSection(
                    title="Comparing Across GPUs and Quantization",
                    paragraphs=(
                        "Self-hosted rows record the quantization used (for example fp8 "
                        "or bf16) alongside the GPU. The same model may appear at "
                        "different quantization on different GPUs because each GPU is run "
                        "at its best practical configuration: native fp8 on H100 and "
                        "H200, and bf16 on A100, which has no native fp8 hardware. A "
                        "cross-GPU accuracy difference for one model therefore reflects "
                        "both the hardware and the quantization, and the two should not "
                        "be attributed to the GPU alone.",
                        "Quantized inference is not bit-identical across GPU "
                        "architectures, so the same model under greedy decoding can "
                        "produce slightly different accuracy on different GPUs. Small "
                        "cross-GPU accuracy differences for an identical model and "
                        "quantization are expected and are a property of the kernels, "
                        "not a measurement error.",
                        "Throughput is measured at a fixed per-GPU concurrency, reported "
                        "as machine-hours per 1M items, so figures are comparable at a "
                        "standard load rather than at each model's individually tuned "
                        "optimum. Accuracy is computed under greedy decoding "
                        "(temperature 0) and is deterministic given the weights; "
                        "reported confidence intervals are fixed-seed bootstrap "
                        "intervals and pairwise comparisons use McNemar's test.",
                        "The per-item generation cap is a runaway guard, not a scoring "
                        "knob: it is set high enough that compliant models reach their "
                        "answer and stop at the end-of-sequence token well before the "
                        "cap. Submissions whose outputs are truncated by the cap on a "
                        "material fraction of items are rejected by verification.",
                    ),
                ),
            ),
        ),
        StaticPage(
            slug="submit",
            title="Submit A Run",
            description="How to add a verified SenseBench run to the public leaderboard.",
            sections=(
                PageSection(
                    title="Workflow",
                    paragraphs=(
                        "Run the benchmark locally with the registered dataset and prompt, "
                        "verify the result, then add the complete run directory under "
                        "results/<run-id>/ in a pull request.",
                        "Submissions must identify the runner: pass --github-handle to "
                        "sensebench run, or stamp an existing run with sensebench "
                        "set-runner.",
                        "Pull request CI rebuilds the site and fails if any submitted "
                        "result is invalid; a maintainer reviews every submission, and "
                        "runs appear on the leaderboard only after the pull request is "
                        "accepted and merged.",
                    ),
                ),
                PageSection(
                    title="Commands",
                    paragraphs=(
                        "Generate a run with: sensebench run --model <model> --prompt p001 "
                        "--github-handle <your-handle>",
                        "Verify it with: sensebench verify runs/<run-id> --dataset "
                        "lexen-v1 --prompt p001",
                    ),
                ),
            ),
        ),
        StaticPage(
            slug="changelog",
            title="Changelog",
            description="Leaderboard and benchmark release notes.",
            sections=(
                PageSection(
                    title="Initial Site",
                    paragraphs=(
                        "GitHub Pages leaderboard with static run pages, generated JSON, "
                        "ECharts charts, and strict PR validation.",
                    ),
                ),
            ),
        ),
        StaticPage(
            slug="citation",
            title="Citation",
            description="Citation information for SenseBench and lexEN.",
            sections=(
                PageSection(
                    title="SenseBench",
                    paragraphs=(
                        "Cite the SenseBench repository and the exact dataset and prompt "
                        "versions used in comparisons.",
                        "Run pages expose the dataset hash, prompt ID, SenseBench version, "
                        "and run commit for reproducibility.",
                    ),
                ),
            ),
        ),
    ]


def _render_static_pages(
    *,
    env: Environment,
    output_dir: Path,
    base_url: str,
) -> list[str]:
    paths: list[str] = []
    for page in _static_pages():
        path = f"{page.slug}/"
        html_text = _render(
            env=env,
            template_name="simple_page.html.j2",
            base_url=base_url,
            title=page.title,
            description=page.description,
            path=path,
            context={PAGE_CONTEXT_KEY: page},
        )
        _write_text(path=_page_file_path(output_dir=output_dir, route=path), text=html_text)
        paths.append(path)
    return paths


def _prompt_examples(
    *,
    prompt: PromptDefinition,
    dataset: DatasetBundle,
    limit: int,
) -> list[dict[str, object]]:
    """Render `prompt` against the first few polysemous dataset items for display."""
    index = build_dataset_index(bundle=dataset)
    examples: list[dict[str, object]] = []
    for item in dataset.items:
        if len(examples) >= limit:
            break
        candidates = get_candidate_senses(lemma=item.lemma, pos=item.pos)
        if len(candidates) < 2:
            continue
        rendered = render_task(
            prompt=prompt,
            item=item,
            dataset_index=index,
            candidates=candidates,
        )
        examples.append(
            {
                "item_id": item.item_id,
                "lemma": item.lemma,
                "pos": item.pos,
                "target_text": item.target_text,
                "candidate_count": len(candidates),
                "messages": [
                    {"role": message.role.value, "content": message.content}
                    for message in rendered.messages
                ],
            }
        )
    return examples


def _render_prompts_index(
    *,
    env: Environment,
    output_dir: Path,
    base_url: str,
) -> str:
    prompts = [load_prompt_definition(path=path) for path in registered_prompt_paths()]
    route = f"{PROMPTS_ROUTE_PREFIX}/"
    html_text = _render(
        env=env,
        template_name="prompts_index.html.j2",
        base_url=base_url,
        title="Prompts",
        description="Prompt formats SenseBench runs use to query models.",
        path=route,
        context={PROMPTS_CONTEXT_KEY: prompts},
    )
    _write_text(path=_page_file_path(output_dir=output_dir, route=route), text=html_text)
    return route


def _render_prompt_pages(
    *,
    env: Environment,
    output_dir: Path,
    base_url: str,
    dataset_cache: dict[str, DatasetBundle],
) -> list[str]:
    paths: list[str] = []
    dataset = _dataset_for_version(version=DEFAULT_LEXEN_RELEASE_ID, cache=dataset_cache)
    for prompt_path in registered_prompt_paths():
        prompt = load_prompt_definition(path=prompt_path)
        route = f"{PROMPTS_ROUTE_PREFIX}/{prompt.id}/"
        page_dir = output_dir / PROMPTS_ROUTE_PREFIX / prompt.id
        page_dir.mkdir(parents=True, exist_ok=True)
        download_name = f"{prompt.id}{PROMPT_JSON_SUFFIX}"
        copy2(src=prompt_path, dst=page_dir / download_name)
        html_text = _render(
            env=env,
            template_name="prompt.html.j2",
            base_url=base_url,
            title=f"{prompt.id} Prompt",
            description=prompt.description,
            path=route,
            context={
                PROMPT_CONTEXT_KEY: prompt,
                PARAMS_JSON_CONTEXT_KEY: prompt.params.model_dump_json(indent=2),
                PROMPT_EXAMPLES_CONTEXT_KEY: _prompt_examples(
                    prompt=prompt, dataset=dataset, limit=PROMPT_EXAMPLE_COUNT
                ),
                PROMPT_DOWNLOAD_CONTEXT_KEY: download_name,
            },
        )
        _write_text(path=_page_file_path(output_dir=output_dir, route=route), text=html_text)
        paths.append(route)
    return paths


def _sitemap(*, base_url: str, paths: list[str]) -> str:
    urls = "\n".join(
        f"  <url><loc>{escape(_absolute_url(base_url=base_url, path=path))}</loc></url>"
        for path in paths
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}\n"
        "</urlset>\n"
    )


def _render_404(
    *,
    env: Environment,
    output_dir: Path,
    base_url: str,
) -> None:
    html_text = _render(
        env=env,
        template_name="simple_page.html.j2",
        base_url=base_url,
        title="Not Found",
        description="The requested SenseBench page was not found.",
        path=NOT_FOUND_FILENAME,
        context={
            PAGE_CONTEXT_KEY: StaticPage(
                slug="404",
                title="Not Found",
                description="The requested page was not found.",
                sections=(
                    PageSection(
                        title="Missing Page",
                        paragraphs=("Return to the leaderboard or choose another site section.",),
                    ),
                ),
            )
        },
    )
    _write_text(path=output_dir / NOT_FOUND_FILENAME, text=html_text)


def _render_run_pages(
    *,
    env: Environment,
    output_dir: Path,
    base_url: str,
    results_dir: Path,
    entries: list[LeaderboardEntry],
    dataset_cache: dict[str, DatasetBundle],
) -> list[str]:
    paths: list[str] = []
    for entry in entries:
        run_dir = results_dir / entry.run_id
        artifacts = _copy_run_artifacts(
            output_dir=output_dir,
            run_dir=run_dir,
            run_id=entry.run_id,
        )
        detail = _run_detail(
            entry=entry,
            results_dir=results_dir,
            dataset_cache=dataset_cache,
            artifacts=artifacts,
        )
        data_path = output_dir / SITE_DATA_DIRNAME / SITE_RUNS_DIRNAME / f"{entry.run_id}.json"
        _write_json(path=data_path, value=detail)
        path = f"{RUNS_ROUTE_PREFIX}/{entry.run_id}/"
        html_text = _render(
            env=env,
            template_name="run.html.j2",
            base_url=base_url,
            title=f"{entry.model} SenseBench Run",
            description=f"Verified SenseBench run {entry.run_id}.",
            path=path,
            context={
                DETAIL_CONTEXT_KEY: detail,
                REPOSITORY_ARTIFACT_URL_CONTEXT_KEY: (
                    f"{DEFAULT_REPOSITORY_TREE_URL}/"
                    f"{SUBMITTED_RESULTS_DIR.as_posix()}/{entry.run_id}/"
                ),
            },
        )
        _write_text(path=_page_file_path(output_dir=output_dir, route=path), text=html_text)
        paths.append(path)
    return paths


def _render_runs_index(
    *,
    env: Environment,
    output_dir: Path,
    base_url: str,
    entries: list[LeaderboardEntry],
) -> str:
    path = RUNS_ROUTE
    html_text = _render(
        env=env,
        template_name="runs_index.html.j2",
        base_url=base_url,
        title="Run Archive",
        description="All verified SenseBench leaderboard submissions.",
        path=path,
        context={ENTRIES_CONTEXT_KEY: entries},
    )
    _write_text(path=_page_file_path(output_dir=output_dir, route=path), text=html_text)
    return path


def _render_label_schemes(*, env: Environment, output_dir: Path, base_url: str) -> str:
    path = "label-schemes/"
    html_text = _render(
        env=env,
        template_name="label_schemes.html.j2",
        base_url=base_url,
        title="Label schemes — SenseBench",
        description=(
            "The nine gold-label and sense-granularity scoring schemes "
            "on the SenseBench leaderboard."
        ),
        path=path,
        context={},
    )
    _write_text(path=_page_file_path(output_dir=output_dir, route=path), text=html_text)
    return path


def _render_coarsening(*, env: Environment, output_dir: Path, base_url: str) -> str:
    path = "coarsening/"
    html_text = _render(
        env=env,
        template_name="coarsening.html.j2",
        base_url=base_url,
        title="Sense coarsening — SenseBench",
        description=(
            "What sense coarsening is, how it changes WSD results, and how the Glite and CSI "
            "coarse-grained inventories work."
        ),
        path=path,
        context={},
    )
    _write_text(path=_page_file_path(output_dir=output_dir, route=path), text=html_text)
    return path


def _render_index(
    *,
    env: Environment,
    output_dir: Path,
    base_url: str,
    site_data: SiteData,
    frontier_run_ids: set[RunID],
) -> str:
    path = ""
    html_text = _render(
        env=env,
        template_name="index.html.j2",
        base_url=base_url,
        title="SenseBench Leaderboard",
        description="Verified leaderboard for English word sense disambiguation with LLMs.",
        path=path,
        context={
            SITE_DATA_CONTEXT_KEY: site_data,
            DATASETS_CONTEXT_KEY: sorted(DATASET_RELEASES),
            FRONTIER_RUN_IDS_CONTEXT_KEY: frontier_run_ids,
        },
    )
    _write_text(path=output_dir / INDEX_HTML_FILENAME, text=html_text)
    return path


def _clean_output_dir(*, output_dir: Path) -> None:
    if output_dir.exists():
        rmtree(output_dir)
    output_dir.mkdir(parents=True)


def build_site(
    *,
    results_dir: Path,
    output_dir: Path,
    base_url: str = DEFAULT_SITE_BASE_URL,
    strict: bool = True,
    on_invalid: Callable[[LeaderboardBuildError], None] | None = None,
) -> Path:
    try:
        collection = collect_leaderboard_entries(
            results_dir=results_dir,
            official=True,
            fail_on_invalid=strict,
        )
    except LeaderboardBuildError as exc:
        if on_invalid is not None:
            on_invalid(exc)
        raise

    env = _template_env()
    _clean_output_dir(output_dir=output_dir)
    _copy_static_assets(output_dir=output_dir)
    dataset_cache: dict[str, DatasetBundle] = {}
    site_data = _site_data(
        collection=collection,
        baselines=_site_baselines(collection=collection, dataset_cache=dataset_cache),
    )
    env.globals[ASSET_VERSION_GLOBAL_KEY] = site_data.summary.generated_at.replace(":", "").replace(
        "+", ""
    )
    _write_json(
        path=output_dir / SITE_DATA_DIRNAME / LEADERBOARD_JSON_PATH,
        value=site_data,
    )

    paths: list[str] = []
    paths.append(
        _render_index(
            env=env,
            output_dir=output_dir,
            base_url=base_url,
            site_data=site_data,
            frontier_run_ids=_pareto_frontier_run_ids(entries=collection.entries),
        )
    )
    paths.append(
        _render_runs_index(
            env=env,
            output_dir=output_dir,
            base_url=base_url,
            entries=collection.entries,
        )
    )
    paths.extend(
        _render_run_pages(
            env=env,
            output_dir=output_dir,
            base_url=base_url,
            results_dir=results_dir,
            entries=collection.entries,
            dataset_cache=dataset_cache,
        )
    )
    paths.append(_render_prompts_index(env=env, output_dir=output_dir, base_url=base_url))
    paths.extend(
        _render_prompt_pages(
            env=env, output_dir=output_dir, base_url=base_url, dataset_cache=dataset_cache
        )
    )
    paths.extend(_render_static_pages(env=env, output_dir=output_dir, base_url=base_url))
    paths.append(_render_label_schemes(env=env, output_dir=output_dir, base_url=base_url))
    paths.append(_render_coarsening(env=env, output_dir=output_dir, base_url=base_url))
    _render_404(env=env, output_dir=output_dir, base_url=base_url)
    _write_text(
        path=output_dir / SITEMAP_FILENAME,
        text=_sitemap(base_url=base_url, paths=paths),
    )
    _write_text(
        path=output_dir / ROBOTS_FILENAME,
        text=(
            "User-agent: *\n"
            "Allow: /\n"
            f"Sitemap: {_absolute_url(base_url=base_url, path=SITEMAP_FILENAME)}\n"
        ),
    )
    _write_text(path=output_dir / CNAME_FILENAME, text=f"{DEFAULT_CUSTOM_DOMAIN}\n")
    return output_dir
