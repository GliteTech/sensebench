"""SenseBench command line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import assert_never

from dotenv import find_dotenv, load_dotenv

from sensebench.datasets.context import build_dataset_index
from sensebench.datasets.loaders import load_jsonl_dataset
from sensebench.datasets.models import DatasetBundle
from sensebench.datasets.releases import (
    fetch_dataset_release,
    get_dataset_release,
    load_registered_dataset,
)
from sensebench.leaderboard.aggregate import LeaderboardBuildError, emit_leaderboard
from sensebench.paths import (
    DEFAULT_LEXEN_RELEASE_ID,
    LEADERBOARD_JSON_PATH,
    LOCAL_RUNS_DIR,
    PROMPT_JSON_SUFFIX,
    PROMPT_REGISTRY_DIR,
    RUN_METADATA_FILENAME,
    SITE_OUTPUT_DIR,
    SUBMITTED_RESULTS_DIR,
)
from sensebench.prompts.models import PromptDefinition
from sensebench.prompts.registry import load_prompt_definition
from sensebench.prompts.render import render_task
from sensebench.runner.client import LiteLlmClient
from sensebench.runner.endpoint import (
    VLLM_ENGINE_NAME,
    is_local_endpoint,
    litellm_model_id,
    probe_openai_endpoint,
    served_model_id,
)
from sensebench.runner.machine import collect_machine_info
from sensebench.runner.run import CompletedRun, RunConfig, preflight_model, run_benchmark
from sensebench.runs.models import (
    CLOUD_LLM_KIND,
    SELF_HOSTED_LLM_KIND,
    CloudLlmReference,
    MachineInfo,
    ModelHostingKind,
    ModelSourceKind,
    PredictionStatus,
    RunMetadata,
    RunnerIdentity,
    SamplingParameters,
    SelfHostedLlmReference,
    VoteStatus,
)
from sensebench.site.build import DEFAULT_SITE_BASE_URL, build_site
from sensebench.verify.runs import RunValidationReport, verify_run_directory
from sensebench.wordnet import SenseCandidate, get_candidate_senses, wordnet_version

CommandHandler = Callable[..., int]
DEFAULT_RUN_CONCURRENCY: int = 512
DEFAULT_SELF_HOSTED_CONCURRENCY: int = 256
DEFAULT_MAX_TOKENS: int = 512
MISSING_MACHINE_GPU_WARNING: str = (
    "warning: no GPU details collected for this self-hosted run; run "
    "`sensebench machine-info` on the GPU host and pass --machine-info-json "
    "(self-hosted submissions without GPU details fail verification)"
)
RUN_ID_DATE_FORMAT: str = "%Y%m%d"
RUN_ID_COLLISION_TIME_FORMAT: str = "%H%M%S"
RUN_ID_SLUG_KEEP_CHARACTERS: str = "._-"
SENSEBENCH_REPO_URL: str = "https://github.com/GliteTech/sensebench"
MISSING_HANDLE_WARNING: str = (
    "warning: --github-handle not set; this run will not be leaderboard-eligible "
    "(fix later with: sensebench set-runner <run-dir> --github-handle <handle>)"
)


def _load_local_env() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    if len(dotenv_path) > 0:
        load_dotenv(dotenv_path=dotenv_path, override=False)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative number")
    return parsed


def _prompt_path(*, prompt: str) -> Path:
    raw_path = Path(prompt)
    if raw_path.exists():
        return raw_path
    return PROMPT_REGISTRY_DIR / f"{prompt}{PROMPT_JSON_SUFFIX}"


def _optional_arg(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _load_prompt(*, prompt: str) -> PromptDefinition:
    return load_prompt_definition(path=_prompt_path(prompt=prompt))


def _print_run_report(*, report: RunValidationReport) -> None:
    if not report.has_errors():
        print(f"{report.run_dir}: OK")
        return
    print(f"{report.run_dir}: FAILED")
    for issue in report.issues:
        print(f"  [{issue.rule.value}] {issue.location}: {issue.message}")


def _add_dataset_args(*, parser: argparse.ArgumentParser, default_release: str | None) -> None:
    parser.add_argument(
        "--dataset",
        default=default_release,
        help="Registered dataset release ID (downloaded and cached automatically).",
    )
    parser.add_argument(
        "--dataset-jsonl",
        default=None,
        help="Local SenseBench JSONL dataset file (overrides --dataset).",
    )
    parser.add_argument(
        "--dataset-id",
        default="local",
        help="Dataset identifier recorded in metadata for --dataset-jsonl runs.",
    )
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="Dataset version recorded in metadata for --dataset-jsonl runs.",
    )


def _resolve_dataset(*, args: argparse.Namespace) -> DatasetBundle | None:
    if args.dataset_jsonl is not None:
        return load_jsonl_dataset(
            path=Path(str(args.dataset_jsonl)),
            dataset_id=str(args.dataset_id),
            dataset_version=_optional_arg(value=args.dataset_version),
        )
    if args.dataset is None:
        return None
    release = get_dataset_release(release_id=str(args.dataset))
    return load_registered_dataset(release=release)


def _slugify(*, value: str) -> str:
    slug_characters = [
        character if character.isalnum() or character in RUN_ID_SLUG_KEEP_CHARACTERS else "-"
        for character in value.lower()
    ]
    slug = "".join(slug_characters).strip("-")
    if len(slug) == 0:
        return "model"
    return slug


def _default_run_id(
    *,
    model: str,
    prompt_id: str,
    dataset_label: str,
    output_root: Path,
) -> str:
    now = datetime.now(tz=UTC)
    base = (
        f"{_slugify(value=model)}-{prompt_id}-{dataset_label}-{now.strftime(RUN_ID_DATE_FORMAT)}"
    )
    if not (output_root / base).exists():
        return base
    return f"{base}-{now.strftime(RUN_ID_COLLISION_TIME_FORMAT)}"


def _warn_if_unpriced(*, model: str) -> None:
    if model.startswith("openrouter/"):
        return
    import litellm

    try:
        litellm.get_model_info(model)
    except Exception:
        print(
            f"warning: no litellm pricing for {model}; run cost will be recorded as unavailable",
            file=sys.stderr,
        )


def _print_run_preamble(*, config: RunConfig) -> None:
    model = config.model
    effort = model.reasoning_effort if isinstance(model, CloudLlmReference) else None
    model_text = (
        model.requested_model
        if effort is None
        else f"{model.requested_model} (reasoning effort: {effort})"
    )
    dataset = config.dataset
    dataset_label = (
        dataset.dataset_version if dataset.dataset_version is not None else dataset.dataset_id
    )
    print(f"run:     {config.run_id}", file=sys.stderr)
    print(f"model:   {model_text}", file=sys.stderr)
    print(f"prompt:  {config.prompt.id} — {config.prompt.name}", file=sys.stderr)
    print(f"dataset: {dataset_label} ({len(dataset.items):,} items)", file=sys.stderr)
    print(
        "policy:  "
        f"votes_per_item={config.votes_per_item} "
        f"concurrency={config.concurrency} "
        f"max_tokens={config.sampling.max_tokens}",
        file=sys.stderr,
    )
    machine = config.machine
    if machine is not None and machine.gpu is not None:
        rate_text = (
            f" @ ${machine.hourly_rate_usd}/h" if machine.hourly_rate_usd is not None else ""
        )
        print(
            f"machine: {machine.gpu.count}x {machine.gpu.name}{rate_text}",
            file=sys.stderr,
        )
    handle = config.runner.github_handle
    if handle is not None and len(handle.strip()) > 0:
        print(f"runner:  github:{handle}", file=sys.stderr)
    else:
        print(MISSING_HANDLE_WARNING, file=sys.stderr)


def _submission_blockers(*, metadata: RunMetadata) -> list[str]:
    blockers: list[str] = []
    release = None
    version = metadata.dataset.dataset_version
    if version is None:
        blockers.append("the dataset is not a registered release")
    else:
        try:
            release = get_dataset_release(release_id=version)
        except Exception:
            blockers.append(f"dataset {version} is not a registered release")
    if release is not None and metadata.dataset.item_count != release.item_count:
        blockers.append(
            f"partial run: {metadata.dataset.item_count} of {release.item_count} items evaluated"
        )
    prompt_path = PROMPT_REGISTRY_DIR / f"{metadata.prompt.id}{PROMPT_JSON_SUFFIX}"
    if not prompt_path.exists():
        blockers.append(f"prompt {metadata.prompt.id} is not a registered prompt")
    handle = metadata.runner.github_handle
    if handle is None or len(handle.strip()) == 0:
        blockers.append(
            "runner identity is missing "
            "(run: sensebench set-runner <run-dir> --github-handle <handle>)"
        )
    return blockers


def _print_submission_guidance(*, completed: CompletedRun) -> None:
    metadata = completed.metadata
    blockers = _submission_blockers(metadata=metadata)
    print()
    if len(blockers) > 0:
        print("This run is not eligible for leaderboard submission:")
        for blocker in blockers:
            print(f"  - {blocker}")
        return
    print("Submit this run to the public leaderboard:")
    print(
        f"  1. Verify it: sensebench verify {completed.run_dir} "
        f"--dataset {metadata.dataset.dataset_version} --prompt {metadata.prompt.id}"
    )
    print(
        f"  2. Fork {SENSEBENCH_REPO_URL} and copy the run directory "
        f"to results/{metadata.run_id}/"
    )
    print(f"  3. Open a pull request titled submit-{metadata.run_id}")
    print(
        "CI re-verifies every submitted run from the raw artifacts; a maintainer reviews "
        "each submission, and the leaderboard updates when the pull request is merged."
    )


def _print_run_summary(*, completed: CompletedRun) -> None:
    totals = completed.metadata.totals
    status_counts: Counter[PredictionStatus] = Counter(
        prediction.status for prediction in completed.predictions
    )
    invalid_output_votes = sum(
        1
        for prediction in completed.predictions
        for vote in prediction.votes
        if vote.status == VoteStatus.INVALID_OUTPUT
    )
    transport_error_votes = sum(
        1
        for prediction in completed.predictions
        for vote in prediction.votes
        if vote.status == VoteStatus.TRANSPORT_ERROR
    )
    accuracy_text = "n/a" if totals.accuracy is None else f"{totals.accuracy:.4f}"
    cost_text = "unavailable" if totals.cost.total_usd is None else f"${totals.cost.total_usd:.2f}"
    elapsed_text = "n/a" if totals.elapsed_seconds is None else f"{totals.elapsed_seconds:.1f}s"
    print(f"run_id: {completed.metadata.run_id}")
    print(f"accuracy: {accuracy_text} ({totals.correct_count}/{totals.item_count} correct)")
    print(
        "items: "
        f"success={status_counts.get(PredictionStatus.SUCCESS, 0)} "
        f"monosemous={status_counts.get(PredictionStatus.MONOSEMOUS, 0)} "
        f"no_valid_vote={status_counts.get(PredictionStatus.NO_VALID_VOTE, 0)} "
        f"no_candidates={status_counts.get(PredictionStatus.NO_CANDIDATES, 0)}"
    )
    print(f"votes: invalid_output={invalid_output_votes} transport_error={transport_error_votes}")
    print(f"calls: {totals.call_count}  cost: {cost_text}  elapsed: {elapsed_text}")
    print(f"artifacts: {completed.run_dir}")


def _cmd_render(*, args: argparse.Namespace) -> int:
    wordnet_version()
    dataset = _resolve_dataset(args=args)
    assert dataset is not None, "the render command defines a default dataset release"
    prompt = _load_prompt(prompt=str(args.prompt))
    index = build_dataset_index(bundle=dataset)
    rendered_count = 0
    for item in dataset.items:
        if args.item_id is not None and item.item_id != args.item_id:
            continue
        if args.limit is not None and rendered_count >= args.limit:
            break
        candidates: list[SenseCandidate] = get_candidate_senses(lemma=item.lemma, pos=item.pos)
        rendered = render_task(prompt=prompt, item=item, dataset_index=index, candidates=candidates)
        print(
            json.dumps(
                {
                    "item_id": item.item_id,
                    "messages": [
                        {
                            "role": message.role.value,
                            "content": message.content,
                        }
                        for message in rendered.messages
                    ],
                    "candidates": [
                        {
                            "index": candidate.index,
                            "sense_key": candidate.sense_key,
                            "synset_id": candidate.synset_id,
                        }
                        for candidate in rendered.candidates
                    ],
                },
                ensure_ascii=False,
            )
        )
        rendered_count += 1
    return 0


def _cloud_model_reference(*, args: argparse.Namespace) -> CloudLlmReference:
    return CloudLlmReference(
        kind=CLOUD_LLM_KIND,
        display_name=str(args.model),
        requested_model=str(args.model),
        resolved_model=None,
        llm_vendor=args.vendor,
        api_provider=args.api_provider,
        source_kind=ModelSourceKind(str(args.source_kind)),
        license=args.license,
        model_url=args.model_url,
        reasoning_effort=args.reasoning_effort,
        endpoint_base_url=args.endpoint_base_url,
    )


def _self_hosted_model_reference(*, args: argparse.Namespace) -> SelfHostedLlmReference:
    endpoint_base_url = _optional_arg(value=args.endpoint_base_url)
    if endpoint_base_url is None:
        raise ValueError("--endpoint-base-url is required for --hosting-kind self_hosted")
    return SelfHostedLlmReference(
        kind=SELF_HOSTED_LLM_KIND,
        display_name=str(args.model),
        requested_model=litellm_model_id(model=str(args.model)),
        resolved_model=None,
        llm_vendor=args.vendor,
        source_kind=ModelSourceKind(str(args.source_kind)),
        license=args.license,
        model_url=args.model_url,
        hf_revision=args.hf_revision,
        quantization=args.quantization,
        inference_engine=args.inference_engine,
        inference_engine_version=args.inference_engine_version,
        container_image=args.container_image,
        serve_command=args.serve_command,
        endpoint_base_url=endpoint_base_url,
    )


def _model_reference(*, args: argparse.Namespace) -> CloudLlmReference | SelfHostedLlmReference:
    hosting_kind = ModelHostingKind(str(args.hosting_kind))
    if hosting_kind == ModelHostingKind.CLOUD_API:
        return _cloud_model_reference(args=args)
    if hosting_kind == ModelHostingKind.SELF_HOSTED:
        return _self_hosted_model_reference(args=args)
    assert_never(hosting_kind)


def _sampling(*, args: argparse.Namespace) -> SamplingParameters:
    return SamplingParameters(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )


def _machine_overrides(*, args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.provider is not None:
        overrides["provider"] = str(args.provider)
    if args.instance_id is not None:
        overrides["instance_id"] = str(args.instance_id)
    if args.hourly_rate_usd is not None:
        overrides["hourly_rate_usd"] = float(args.hourly_rate_usd)
    return overrides


def _machine_info(*, args: argparse.Namespace, endpoint_base_url: str) -> MachineInfo | None:
    machine: MachineInfo | None = None
    if args.machine_info_json is not None:
        machine_path = Path(str(args.machine_info_json))
        machine = MachineInfo.model_validate_json(machine_path.read_text(encoding="utf-8"))
    elif is_local_endpoint(base_url=endpoint_base_url):
        machine = collect_machine_info()
    overrides = _machine_overrides(args=args)
    if machine is None:
        if len(overrides) == 0:
            return None
        machine = MachineInfo()
    if len(overrides) > 0:
        machine = machine.model_copy(update=overrides)
    return machine


def _resolved_concurrency(*, args: argparse.Namespace, hosting_kind: ModelHostingKind) -> int:
    if args.concurrency is not None:
        return int(args.concurrency)
    if hosting_kind == ModelHostingKind.SELF_HOSTED:
        return DEFAULT_SELF_HOSTED_CONCURRENCY
    return DEFAULT_RUN_CONCURRENCY


def _probe_self_hosted_endpoint(*, model: SelfHostedLlmReference) -> SelfHostedLlmReference:
    probe = probe_openai_endpoint(base_url=model.endpoint_base_url)
    served = served_model_id(requested_model=model.requested_model)
    if served not in probe.served_model_ids:
        raise RuntimeError(
            f"model {served} is not served by {model.endpoint_base_url}; "
            f"served models: {', '.join(probe.served_model_ids) or '(none)'}"
        )
    if model.inference_engine is None and probe.engine_version is not None:
        return model.model_copy(
            update={
                "inference_engine": VLLM_ENGINE_NAME,
                "inference_engine_version": probe.engine_version,
            }
        )
    return model


async def _run_async(*, args: argparse.Namespace) -> int:
    _load_local_env()
    dataset = _resolve_dataset(args=args)
    assert dataset is not None, "the run command defines a default dataset release"
    if args.limit is not None:
        dataset = replace(dataset, items=dataset.items[: int(args.limit)])
        print(
            f"PARTIAL RUN: --limit {args.limit}; this run is not eligible for the leaderboard",
            file=sys.stderr,
        )
    prompt = _load_prompt(prompt=str(args.prompt))
    model = _model_reference(args=args)
    machine: MachineInfo | None = None
    if isinstance(model, SelfHostedLlmReference):
        if not bool(args.skip_preflight):
            model = _probe_self_hosted_endpoint(model=model)
        machine = _machine_info(args=args, endpoint_base_url=model.endpoint_base_url)
        if machine is None or machine.gpu is None:
            print(MISSING_MACHINE_GPU_WARNING, file=sys.stderr)
    output_root = Path(args.output_root)
    run_id = _optional_arg(value=args.run_id)
    if run_id is None:
        dataset_label = (
            dataset.dataset_version if dataset.dataset_version is not None else dataset.dataset_id
        )
        run_id = _default_run_id(
            model=model.display_name,
            prompt_id=prompt.id,
            dataset_label=dataset_label,
            output_root=output_root,
        )
    config = RunConfig(
        run_id=run_id,
        output_root=output_root,
        dataset=dataset,
        prompt=prompt,
        model=model,
        runner=RunnerIdentity(
            github_handle=args.github_handle,
            name=args.runner_name,
            contact=args.runner_contact,
        ),
        sampling=_sampling(args=args),
        votes_per_item=int(args.votes),
        semantic_reasks_per_invalid_vote=1,
        concurrency=_resolved_concurrency(
            args=args,
            hosting_kind=ModelHostingKind(str(args.hosting_kind)),
        ),
        machine=machine,
        warmup_calls=int(args.warmup_calls),
        show_progress=not bool(args.no_progress),
    )
    _print_run_preamble(config=config)
    client = LiteLlmClient()
    if not bool(args.skip_preflight):
        await preflight_model(config=config, client=client)
        print(f"preflight OK: {model.requested_model}", file=sys.stderr)
    if isinstance(model, CloudLlmReference):
        _warn_if_unpriced(model=model.requested_model)
    completed = await run_benchmark(config=config, client=client)
    _print_run_summary(completed=completed)
    _print_submission_guidance(completed=completed)
    return 0


def _cmd_run(*, args: argparse.Namespace) -> int:
    return asyncio.run(_run_async(args=args))


def _cmd_set_runner(*, args: argparse.Namespace) -> int:
    metadata_path = Path(args.run_dir) / RUN_METADATA_FILENAME
    metadata = RunMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    updated_runner = RunnerIdentity(
        github_handle=str(args.github_handle),
        name=args.runner_name if args.runner_name is not None else metadata.runner.name,
        contact=args.runner_contact if args.runner_contact is not None else metadata.runner.contact,
    )
    updated = metadata.model_copy(update={"runner": updated_runner})
    metadata_path.write_text(updated.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"{metadata_path}: runner github_handle={updated_runner.github_handle}")
    return 0


def _cmd_verify(*, args: argparse.Namespace) -> int:
    wordnet_version()
    dataset = _resolve_dataset(args=args)
    prompt = None
    if args.prompt is not None:
        prompt = _load_prompt(prompt=str(args.prompt))
    report = verify_run_directory(run_dir=Path(args.run_dir), dataset=dataset, prompt=prompt)
    _print_run_report(report=report)
    return 1 if report.has_errors() else 0


def _cmd_leaderboard(*, args: argparse.Namespace) -> int:
    try:
        emit_leaderboard(
            results_dir=Path(args.results_dir),
            output_path=Path(args.output),
            official=bool(args.official),
            strict=bool(args.strict),
        )
    except LeaderboardBuildError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(args.output)
    return 0


def _cmd_site_build(*, args: argparse.Namespace) -> int:
    try:
        build_site(
            results_dir=Path(args.results_dir),
            output_dir=Path(args.output_dir),
            base_url=str(args.base_url),
            strict=bool(args.strict),
        )
    except LeaderboardBuildError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(args.output_dir)
    return 0


def _cmd_fetch_dataset(*, args: argparse.Namespace) -> int:
    release = get_dataset_release(release_id=str(args.release))
    path = fetch_dataset_release(release=release)
    print(path)
    return 0


def _cmd_machine_info(*, args: argparse.Namespace) -> int:
    machine = collect_machine_info(
        provider=_optional_arg(value=args.provider),
        instance_id=_optional_arg(value=args.instance_id),
        hourly_rate_usd=args.hourly_rate_usd,
    )
    print(machine.model_dump_json(indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SenseBench runner and verifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser(
        "render", help="Render prompt messages for dataset items."
    )
    _add_dataset_args(parser=render_parser, default_release=DEFAULT_LEXEN_RELEASE_ID)
    render_parser.add_argument("--prompt", required=True)
    render_parser.add_argument("--item-id")
    render_parser.add_argument("--limit", type=_positive_int)
    render_parser.set_defaults(func=_cmd_render)

    run_parser = subparsers.add_parser("run", help="Run a model and write local run artifacts.")
    _add_dataset_args(parser=run_parser, default_release=DEFAULT_LEXEN_RELEASE_ID)
    run_parser.add_argument("--prompt", required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier; generated from model, prompt, and dataset when omitted.",
    )
    run_parser.add_argument("--output-root", default=str(LOCAL_RUNS_DIR))
    run_parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Evaluate only the first N items (smoke runs; not leaderboard-eligible).",
    )
    run_parser.add_argument("--skip-preflight", action="store_true")
    run_parser.add_argument("--votes", type=_positive_int, default=1)
    run_parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=None,
        help=(
            f"Concurrent items in flight (default {DEFAULT_RUN_CONCURRENCY} for cloud APIs, "
            f"{DEFAULT_SELF_HOSTED_CONCURRENCY} for self-hosted endpoints)."
        ),
    )
    run_parser.add_argument("--no-progress", action="store_true")
    run_parser.add_argument("--temperature", type=float)
    run_parser.add_argument("--top-p", type=float)
    run_parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--vendor")
    run_parser.add_argument("--api-provider")
    run_parser.add_argument(
        "--hosting-kind",
        choices=[kind.value for kind in ModelHostingKind],
        default=ModelHostingKind.CLOUD_API.value,
    )
    run_parser.add_argument(
        "--source-kind",
        choices=[kind.value for kind in ModelSourceKind],
        default=ModelSourceKind.UNKNOWN.value,
    )
    run_parser.add_argument("--license")
    run_parser.add_argument("--model-url")
    run_parser.add_argument("--reasoning-effort")
    run_parser.add_argument("--hf-revision")
    run_parser.add_argument("--quantization")
    run_parser.add_argument("--inference-engine")
    run_parser.add_argument("--inference-engine-version")
    run_parser.add_argument(
        "--container-image",
        help="Serving container image (with digest if known), e.g. vllm/vllm-openai:v0.22.1.",
    )
    run_parser.add_argument(
        "--serve-command",
        help="Exact inference server launch command, recorded for reproducibility.",
    )
    run_parser.add_argument("--endpoint-base-url")
    run_parser.add_argument(
        "--machine-info-json",
        default=None,
        help="MachineInfo JSON file (from `sensebench machine-info` on the GPU host).",
    )
    run_parser.add_argument("--provider", help="Machine provider, e.g. vast.ai.")
    run_parser.add_argument("--instance-id", help="Provider instance identifier.")
    run_parser.add_argument(
        "--hourly-rate-usd",
        type=_non_negative_float,
        default=None,
        help="Machine hourly rate; enables machine-time cost estimation.",
    )
    run_parser.add_argument(
        "--warmup-calls",
        type=_non_negative_int,
        default=0,
        help="Unrecorded warmup completions before the timed benchmark loop.",
    )
    run_parser.add_argument("--github-handle")
    run_parser.add_argument("--runner-name")
    run_parser.add_argument("--runner-contact")
    run_parser.set_defaults(func=_cmd_run)

    verify_parser = subparsers.add_parser("verify", help="Verify a run directory.")
    verify_parser.add_argument("run_dir")
    _add_dataset_args(parser=verify_parser, default_release=None)
    verify_parser.add_argument("--prompt")
    verify_parser.set_defaults(func=_cmd_verify)

    set_runner_parser = subparsers.add_parser(
        "set-runner",
        help="Stamp runner identity into an existing run.json (required for submissions).",
    )
    set_runner_parser.add_argument("run_dir")
    set_runner_parser.add_argument("--github-handle", required=True)
    set_runner_parser.add_argument("--runner-name")
    set_runner_parser.add_argument("--runner-contact")
    set_runner_parser.set_defaults(func=_cmd_set_runner)

    leaderboard_parser = subparsers.add_parser("leaderboard", help="Emit leaderboard.json.")
    leaderboard_parser.add_argument("--results-dir", default=str(SUBMITTED_RESULTS_DIR))
    leaderboard_parser.add_argument("--output", default=str(LEADERBOARD_JSON_PATH))
    leaderboard_parser.add_argument(
        "--official",
        action="store_true",
        help="Verify against registered official dataset releases.",
    )
    leaderboard_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of skipping invalid submitted results.",
    )
    leaderboard_parser.set_defaults(func=_cmd_leaderboard)

    site_parser = subparsers.add_parser("site", help="Build static website assets.")
    site_subparsers = site_parser.add_subparsers(dest="site_command", required=True)
    site_build_parser = site_subparsers.add_parser(
        "build", help="Build the static GitHub Pages site."
    )
    site_build_parser.add_argument("--results-dir", default=str(SUBMITTED_RESULTS_DIR))
    site_build_parser.add_argument("--output-dir", default=str(SITE_OUTPUT_DIR))
    site_build_parser.add_argument("--base-url", default=DEFAULT_SITE_BASE_URL)
    site_build_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any submitted result is invalid or ineligible.",
    )
    site_build_parser.set_defaults(func=_cmd_site_build)

    fetch_parser = subparsers.add_parser(
        "fetch-dataset", help="Download and cache a registered dataset release."
    )
    fetch_parser.add_argument("release", nargs="?", default=DEFAULT_LEXEN_RELEASE_ID)
    fetch_parser.set_defaults(func=_cmd_fetch_dataset)

    machine_info_parser = subparsers.add_parser(
        "machine-info",
        help="Print this machine's hardware details as MachineInfo JSON.",
    )
    machine_info_parser.add_argument("--provider", help="Machine provider, e.g. vast.ai.")
    machine_info_parser.add_argument("--instance-id", help="Provider instance identifier.")
    machine_info_parser.add_argument(
        "--hourly-rate-usd",
        type=_non_negative_float,
        default=None,
        help="Machine hourly rate; enables machine-time cost estimation.",
    )
    machine_info_parser.set_defaults(func=_cmd_machine_info)
    return parser


def _run_command_handler(*, args: argparse.Namespace) -> int:
    handler: object = args.func
    assert callable(handler), "args.func is callable"
    result = handler(args=args)
    assert isinstance(result, int), "command handler returns int"
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _run_command_handler(args=args)


if __name__ == "__main__":
    sys.exit(main())
