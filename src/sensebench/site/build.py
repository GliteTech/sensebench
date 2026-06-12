"""Build the static SenseBench leaderboard website."""

from __future__ import annotations

import html
import json
import shutil
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from urllib.parse import urljoin, urlparse

from jinja2 import Environment, PackageLoader, select_autoescape
from pydantic import BaseModel, ConfigDict

from sensebench.datasets.context import build_context_window, build_dataset_index
from sensebench.datasets.models import DatasetBundle, DatasetIndex, ItemID, WsdItem
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
from sensebench.paths import (
    CALLS_FILENAME,
    DEFAULT_LEXEN_RELEASE_ID,
    PREDICTIONS_FILENAME,
    PROMPT_JSON_SUFFIX,
    PROMPT_REGISTRY_DIR,
    RUN_METADATA_FILENAME,
)
from sensebench.prompts.models import MessageRole, PromptDefinition
from sensebench.prompts.registry import load_prompt_definition, registered_prompt_paths
from sensebench.runs.loaders import LoadedRun, load_run_directory
from sensebench.runs.models import (
    CallID,
    CallRecord,
    CallStatus,
    PredictionRecord,
    RunMetadata,
    VoteStatus,
)
from sensebench.wordnet import get_candidate_senses

DEFAULT_SITE_BASE_URL: str = "https://glitetech.github.io/sensebench/"
DEFAULT_REPOSITORY_TREE_URL: str = "https://github.com/GliteTech/sensebench/tree/main"
SITE_DATA_SCHEMA_VERSION: str = "sensebench-site-data-v2"
RUN_DETAIL_SCHEMA_VERSION: str = "sensebench-run-detail-v3"
RUN_ARTIFACT_ROOT: str = "artifacts/runs"
SLICE_POS: str = "POS"
SLICE_CANDIDATE_COUNT: str = "Candidate Count"
SLICE_SOURCE_DATASET: str = "Source Dataset"
MAX_ERROR_EXAMPLES: int = 12
STATIC_PAGE_SLUGS: tuple[str, ...] = (
    "about",
    "methodology",
    "submit",
    "changelog",
    "citation",
)


class SiteModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SiteSummary(SiteModel):
    verified_run_count: int
    model_count: int
    dataset_versions: list[str]
    prompt_ids: list[str]
    top_accuracy: float | None
    generated_at: str


class SiteData(SiteModel):
    schema_version: str
    summary: SiteSummary
    entries: list[LeaderboardEntry]


class SliceSummary(SiteModel):
    group: str
    value: str
    correct_count: int
    item_count: int
    accuracy: float | None


class ExampleContextSentence(SiteModel):
    html: str
    is_target_sentence: bool


class ExampleCandidate(SiteModel):
    index: int
    sense_key: str
    synset_id: str
    definition: str | None
    synonyms: list[str]
    examples: list[str]
    is_gold: bool
    is_selected: bool


class ExamplePromptMessage(SiteModel):
    role: MessageRole
    content: str


class RunExample(SiteModel):
    bucket_group: str
    bucket_value: str
    item_id: ItemID
    lemma: str
    target_text: str
    pos: str
    source_dataset: str
    candidate_count: int
    is_correct: bool | None
    predicted_sense_index: int | None
    predicted_sense_key: str | None
    gold_sense_keys: list[str]
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
    return urljoin(_base_url(base_url=base_url), path)


def _template_env() -> Environment:
    env = Environment(
        loader=PackageLoader("sensebench.site", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["pct"] = _format_percent
    env.filters["num"] = _format_number
    env.filters["money"] = _format_money
    env.filters["million_token_price"] = _format_million_token_price
    env.filters["seconds"] = _format_seconds
    env.filters["bytes"] = _format_bytes
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
    if value < 1:
        return f"${value:.4f}"
    return f"${value:,.2f}"


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
    path.write_text(text, encoding="utf-8")


def _write_json(*, path: Path, value: BaseModel | dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, BaseModel):
        serialized = value.model_dump_json(indent=2)
    else:
        serialized = json.dumps(value, indent=2, ensure_ascii=False)
    path.write_text(f"{serialized}\n", encoding="utf-8")


def _copy_tree(*, source: Traversable, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        child_target = target / child.name
        if child.is_dir():
            _copy_tree(source=child, target=child_target)
        else:
            child_target.write_bytes(child.read_bytes())


def _copy_static_assets(*, output_dir: Path) -> None:
    static_root = files("sensebench.site").joinpath("static")
    _copy_tree(source=static_root, target=output_dir / "assets")


def _copy_run_artifacts(
    *,
    output_dir: Path,
    run_dir: Path,
    run_id: str,
) -> list[RunArtifact]:
    copied: list[RunArtifact] = []
    target_dir = output_dir / RUN_ARTIFACT_ROOT / run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    for spec in RUN_ARTIFACT_SPECS:
        source = run_dir / spec.filename
        if not source.exists():
            continue
        target = target_dir / spec.filename
        shutil.copy2(src=source, dst=target)
        copied.append(
            RunArtifact(
                label=spec.label,
                filename=spec.filename,
                path=f"{RUN_ARTIFACT_ROOT}/{run_id}/{spec.filename}",
                size_bytes=source.stat().st_size,
                description=spec.description,
            )
        )
    return copied


def _site_summary(*, collection: LeaderboardCollection) -> SiteSummary:
    entries = collection.entries
    top_accuracy = entries[0].accuracy if len(entries) > 0 else None
    return SiteSummary(
        verified_run_count=len(entries),
        model_count=len({entry.model for entry in entries}),
        dataset_versions=sorted(
            {entry.dataset_version for entry in entries if entry.dataset_version is not None}
        ),
        prompt_ids=sorted({entry.prompt_id for entry in entries}),
        top_accuracy=top_accuracy,
        generated_at=datetime.now(tz=UTC).isoformat(),
    )


def _site_data(*, collection: LeaderboardCollection) -> SiteData:
    return SiteData(
        schema_version=SITE_DATA_SCHEMA_VERSION,
        summary=_site_summary(collection=collection),
        entries=collection.entries,
    )


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
    group: str,
    value: str,
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


def _source_dataset(*, item: WsdItem) -> str:
    value = item.metadata.get("source_dataset")
    if value is None or len(value) == 0:
        return "unknown"
    return value


def _slice_value(*, group: str, prediction: PredictionRecord, item: WsdItem) -> str:
    if group == SLICE_POS:
        return item.pos
    if group == SLICE_CANDIDATE_COUNT:
        return _candidate_bucket(count=len(prediction.candidates))
    if group == SLICE_SOURCE_DATASET:
        return _source_dataset(item=item)
    raise ValueError(f"unknown slice group: {group}")


def _slice_summaries(*, loaded: LoadedRun, dataset: DatasetBundle) -> list[SliceSummary]:
    items_by_id = {item.item_id: item for item in dataset.items}
    groups: dict[tuple[str, str], list[PredictionRecord]] = defaultdict(list)
    for prediction in loaded.predictions:
        item = items_by_id.get(prediction.item_id)
        if item is None:
            continue
        for group in (SLICE_POS, SLICE_CANDIDATE_COUNT, SLICE_SOURCE_DATASET):
            groups[(group, _slice_value(group=group, prediction=prediction, item=item))].append(
                prediction
            )
    summaries = [
        _slice_summary(group=group, value=value, predictions=predictions)
        for (group, value), predictions in groups.items()
    ]
    return sorted(summaries, key=lambda summary: (summary.group, summary.value))


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
        pieces: list[str] = []
        for token_index, token in enumerate(sentence.tokens):
            token_text = html.escape(token.text)
            if is_target_sentence and token_index == item.target_token_index:
                pieces.append(f"<mark>{token_text}</mark>")
                has_marked_target = True
            else:
                pieces.append(token_text)
        context.append(
            ExampleContextSentence(
                html=" ".join(pieces),
                is_target_sentence=is_target_sentence,
            )
        )
    if not has_marked_target:
        return None
    return context


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
    wordnet_by_key = {
        candidate.sense_key: candidate
        for candidate in get_candidate_senses(lemma=item.lemma, pos=item.pos)
    }
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
                is_selected=candidate.sense_key == prediction.predicted_sense_key,
            )
        )
    return candidates


def _example(
    *,
    bucket_group: str,
    bucket_value: str,
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
    items_by_id = {item.item_id: item for item in dataset.items}
    calls_by_id = _calls_by_id(calls=loaded.calls)
    calls_by_item = _calls_by_item(calls=loaded.calls)
    errors = [
        prediction
        for prediction in sorted(loaded.predictions, key=lambda candidate: candidate.item_id)
        if prediction.is_correct is False and prediction.item_id in items_by_id
    ]
    selected: list[RunExample] = []
    selected_items: set[ItemID] = set()
    ranked_slices = sorted(
        [
            summary
            for summary in slices
            if summary.group in (SLICE_POS, SLICE_CANDIDATE_COUNT)
            and summary.accuracy is not None
        ],
        key=lambda summary: (summary.accuracy, -summary.item_count, summary.group, summary.value),
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
            bucket_group="Error",
            bucket_value="Any",
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


def _dataset_for_entry(
    *,
    entry: LeaderboardEntry,
    cache: dict[str, DatasetBundle],
) -> DatasetBundle:
    if entry.dataset_version is None:
        raise ValueError(f"run {entry.run_id} has no dataset_version")
    dataset = cache.get(entry.dataset_version)
    if dataset is None:
        release = get_dataset_release(release_id=entry.dataset_version)
        dataset = load_registered_dataset(release=release)
        cache[entry.dataset_version] = dataset
    return dataset


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
        path=PROMPT_REGISTRY_DIR / f"{entry.prompt_id}{PROMPT_JSON_SUFFIX}",
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
                        "with a fixed seed.",
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
                        "The default leaderboard view shows the best verified run per "
                        "model, dataset version, and prompt ID.",
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
                        "Pull request CI rebuilds the site and fails if any submitted "
                        "result is invalid.",
                    ),
                ),
                PageSection(
                    title="Commands",
                    paragraphs=(
                        "Generate a run with: sensebench run --model <model> --prompt p001",
                        "Verify it with: sensebench verify runs/<run-id> --dataset "
                        "lexen-v0.1.0 --prompt p001",
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
            context={"page": page},
        )
        _write_text(path=output_dir / page.slug / "index.html", text=html_text)
        paths.append(path)
    return paths


def _render_dataset_page(
    *,
    env: Environment,
    output_dir: Path,
    base_url: str,
) -> str:
    release = get_dataset_release(release_id=DEFAULT_LEXEN_RELEASE_ID)
    path = f"datasets/{release.release_id}/"
    html_text = _render(
        env=env,
        template_name="dataset.html.j2",
        base_url=base_url,
        title=f"{release.release_id} Dataset",
        description="Registered lexEN dataset release used by the SenseBench leaderboard.",
        path=path,
        context={"release": release},
    )
    _write_text(path=output_dir / path / "index.html", text=html_text)
    return path


def _render_prompt_pages(
    *,
    env: Environment,
    output_dir: Path,
    base_url: str,
) -> list[str]:
    paths: list[str] = []
    for prompt_path in registered_prompt_paths():
        prompt = load_prompt_definition(path=prompt_path)
        path = f"prompts/{prompt.id}/"
        html_text = _render(
            env=env,
            template_name="prompt.html.j2",
            base_url=base_url,
            title=f"{prompt.id} Prompt",
            description=prompt.description,
            path=path,
            context={
                "prompt": prompt,
                "params_json": json.dumps(prompt.params.model_dump(mode="json"), indent=2),
            },
        )
        _write_text(path=output_dir / path / "index.html", text=html_text)
        paths.append(path)
    return paths


def _sitemap(*, base_url: str, paths: list[str]) -> str:
    urls = "\n".join(
        "  <url><loc>"
        f"{html.escape(_absolute_url(base_url=base_url, path=path))}"
        "</loc></url>"
        for path in paths
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}\n"
        "</urlset>\n"
    )


def _render_404(*, env: Environment, output_dir: Path, base_url: str) -> None:
    html_text = _render(
        env=env,
        template_name="simple_page.html.j2",
        base_url=base_url,
        title="Not Found",
        description="The requested SenseBench page was not found.",
        path="404.html",
        context={
            "page": StaticPage(
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
    _write_text(path=output_dir / "404.html", text=html_text)


def _render_run_pages(
    *,
    env: Environment,
    output_dir: Path,
    base_url: str,
    results_dir: Path,
    entries: list[LeaderboardEntry],
) -> list[str]:
    paths: list[str] = []
    dataset_cache: dict[str, DatasetBundle] = {}
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
        data_path = output_dir / "data" / "runs" / f"{entry.run_id}.json"
        _write_json(path=data_path, value=detail)
        path = f"runs/{entry.run_id}/"
        html_text = _render(
            env=env,
            template_name="run.html.j2",
            base_url=base_url,
            title=f"{entry.model} SenseBench Run",
            description=f"Verified SenseBench run {entry.run_id}.",
            path=path,
            context={
                "detail": detail,
                "repository_artifact_url": (
                    f"{DEFAULT_REPOSITORY_TREE_URL}/results/{entry.run_id}/"
                ),
            },
        )
        _write_text(path=output_dir / path / "index.html", text=html_text)
        paths.append(path)
    return paths


def _render_runs_index(
    *,
    env: Environment,
    output_dir: Path,
    base_url: str,
    entries: list[LeaderboardEntry],
) -> str:
    path = "runs/"
    html_text = _render(
        env=env,
        template_name="runs_index.html.j2",
        base_url=base_url,
        title="Run Archive",
        description="All verified SenseBench leaderboard submissions.",
        path=path,
        context={"entries": entries},
    )
    _write_text(path=output_dir / path / "index.html", text=html_text)
    return path


def _render_index(
    *,
    env: Environment,
    output_dir: Path,
    base_url: str,
    site_data: SiteData,
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
            "site_data": site_data,
            "datasets": sorted(DATASET_RELEASES),
        },
    )
    _write_text(path=output_dir / "index.html", text=html_text)
    return path


def _clean_output_dir(*, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
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
    site_data = _site_data(collection=collection)
    env.globals["asset_version"] = (
        site_data.summary.generated_at.replace(":", "").replace("+", "")
    )
    _write_json(path=output_dir / "data" / "leaderboard.json", value=site_data)

    paths: list[str] = []
    paths.append(
        _render_index(
            env=env,
            output_dir=output_dir,
            base_url=base_url,
            site_data=site_data,
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
        )
    )
    paths.append(_render_dataset_page(env=env, output_dir=output_dir, base_url=base_url))
    paths.extend(_render_prompt_pages(env=env, output_dir=output_dir, base_url=base_url))
    paths.extend(_render_static_pages(env=env, output_dir=output_dir, base_url=base_url))
    _render_404(env=env, output_dir=output_dir, base_url=base_url)
    _write_text(path=output_dir / "sitemap.xml", text=_sitemap(base_url=base_url, paths=paths))
    _write_text(
        path=output_dir / "robots.txt",
        text=(
            "User-agent: *\n"
            "Allow: /\n"
            f"Sitemap: {_absolute_url(base_url=base_url, path='sitemap.xml')}\n"
        ),
    )
    return output_dir
