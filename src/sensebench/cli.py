"""SenseBench command line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from sensebench.datasets.context import build_dataset_index
from sensebench.datasets.loaders import load_jsonl_dataset
from sensebench.leaderboard.aggregate import emit_leaderboard
from sensebench.paths import (
    LEADERBOARD_JSON_PATH,
    LOCAL_RUNS_DIR,
    PROMPT_JSON_SUFFIX,
    PROMPT_REGISTRY_DIR,
    SUBMITTED_RESULTS_DIR,
)
from sensebench.prompts.models import PromptDefinition
from sensebench.prompts.registry import load_prompt_definition
from sensebench.prompts.render import render_task
from sensebench.runner.client import LiteLlmClient
from sensebench.runner.run import RunConfig, run_benchmark
from sensebench.runs.models import (
    ModelExecutionKind,
    ModelHostingKind,
    ModelReference,
    ModelSourceKind,
    RunnerIdentity,
    SamplingParameters,
)
from sensebench.verify.runs import RunValidationReport, verify_run_directory
from sensebench.wordnet import SenseCandidate, get_candidate_senses

CommandHandler = Callable[..., int]


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


def _add_dataset_args(*, parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-jsonl", required=True, help="Local SenseBench JSONL dataset file."
    )
    parser.add_argument(
        "--dataset-id", default="local", help="Dataset identifier recorded in metadata."
    )
    parser.add_argument(
        "--dataset-version", default=None, help="Dataset version recorded in metadata."
    )


def _cmd_render(*, args: argparse.Namespace) -> int:
    dataset = load_jsonl_dataset(
        path=Path(args.dataset_jsonl),
        dataset_id=str(args.dataset_id),
        dataset_version=_optional_arg(value=args.dataset_version),
    )
    prompt = _load_prompt(prompt=str(args.prompt))
    index = build_dataset_index(bundle=dataset)
    rendered_count = 0
    for item in dataset.items:
        if args.item_id is not None and item.item_id != args.item_id:
            continue
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
        if args.limit is not None and rendered_count >= args.limit:
            break
    return 0


def _model_reference(*, args: argparse.Namespace) -> ModelReference:
    return ModelReference(
        execution_kind=ModelExecutionKind.LLM,
        display_name=str(args.model),
        requested_model=str(args.model),
        resolved_model=None,
        vendor=args.vendor,
        api_provider=args.api_provider,
        hosting_kind=ModelHostingKind(str(args.hosting_kind)),
        source_kind=ModelSourceKind(str(args.source_kind)),
        license=args.license,
        model_url=args.model_url,
        reasoning_effort=args.reasoning_effort,
        quantization=args.quantization,
        inference_engine=args.inference_engine,
        inference_engine_version=args.inference_engine_version,
        endpoint_base_url=args.endpoint_base_url,
        gpu=args.gpu,
        cpu=args.cpu,
    )


def _sampling(*, args: argparse.Namespace) -> SamplingParameters:
    return SamplingParameters(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )


async def _run_async(*, args: argparse.Namespace) -> int:
    dataset = load_jsonl_dataset(
        path=Path(args.dataset_jsonl),
        dataset_id=str(args.dataset_id),
        dataset_version=_optional_arg(value=args.dataset_version),
    )
    prompt = _load_prompt(prompt=str(args.prompt))
    config = RunConfig(
        run_id=str(args.run_id),
        output_root=Path(args.output_root),
        dataset=dataset,
        prompt=prompt,
        model=_model_reference(args=args),
        runner=RunnerIdentity(
            github_handle=args.github_handle,
            name=args.runner_name,
            contact=args.runner_contact,
        ),
        sampling=_sampling(args=args),
        votes_per_item=int(args.votes),
        semantic_reasks_per_invalid_vote=1,
        concurrency=int(args.concurrency),
    )
    client = LiteLlmClient()
    completed = await run_benchmark(config=config, client=client)
    print(completed.run_dir)
    return 0


def _cmd_run(*, args: argparse.Namespace) -> int:
    return asyncio.run(_run_async(args=args))


def _cmd_verify(*, args: argparse.Namespace) -> int:
    dataset = None
    prompt = None
    if args.dataset_jsonl is not None:
        dataset = load_jsonl_dataset(
            path=Path(args.dataset_jsonl),
            dataset_id=str(args.dataset_id),
            dataset_version=_optional_arg(value=args.dataset_version),
        )
    if args.prompt is not None:
        prompt = _load_prompt(prompt=str(args.prompt))
    report = verify_run_directory(run_dir=Path(args.run_dir), dataset=dataset, prompt=prompt)
    _print_run_report(report=report)
    return 1 if report.has_errors() else 0


def _cmd_leaderboard(*, args: argparse.Namespace) -> int:
    emit_leaderboard(results_dir=Path(args.results_dir), output_path=Path(args.output))
    print(args.output)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SenseBench runner and verifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser(
        "render", help="Render prompt messages for dataset items."
    )
    _add_dataset_args(parser=render_parser)
    render_parser.add_argument("--prompt", required=True)
    render_parser.add_argument("--item-id")
    render_parser.add_argument("--limit", type=int)
    render_parser.set_defaults(func=_cmd_render)

    run_parser = subparsers.add_parser("run", help="Run a model and write local run artifacts.")
    _add_dataset_args(parser=run_parser)
    run_parser.add_argument("--prompt", required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--output-root", default=str(LOCAL_RUNS_DIR))
    run_parser.add_argument("--votes", type=int, default=1)
    run_parser.add_argument("--concurrency", type=int, default=4)
    run_parser.add_argument("--temperature", type=float)
    run_parser.add_argument("--top-p", type=float)
    run_parser.add_argument("--max-tokens", type=int, default=32)
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--vendor")
    run_parser.add_argument("--api-provider")
    run_parser.add_argument("--hosting-kind", default=ModelHostingKind.CLOUD_API.value)
    run_parser.add_argument("--source-kind", default=ModelSourceKind.UNKNOWN.value)
    run_parser.add_argument("--license")
    run_parser.add_argument("--model-url")
    run_parser.add_argument("--reasoning-effort")
    run_parser.add_argument("--quantization")
    run_parser.add_argument("--inference-engine")
    run_parser.add_argument("--inference-engine-version")
    run_parser.add_argument("--endpoint-base-url")
    run_parser.add_argument("--gpu")
    run_parser.add_argument("--cpu")
    run_parser.add_argument("--github-handle")
    run_parser.add_argument("--runner-name")
    run_parser.add_argument("--runner-contact")
    run_parser.set_defaults(func=_cmd_run)

    verify_parser = subparsers.add_parser("verify", help="Verify a run directory.")
    verify_parser.add_argument("run_dir")
    verify_parser.add_argument("--dataset-jsonl")
    verify_parser.add_argument("--dataset-id", default="local")
    verify_parser.add_argument("--dataset-version", default=None)
    verify_parser.add_argument("--prompt")
    verify_parser.set_defaults(func=_cmd_verify)

    leaderboard_parser = subparsers.add_parser("leaderboard", help="Emit leaderboard.json.")
    leaderboard_parser.add_argument("--results-dir", default=str(SUBMITTED_RESULTS_DIR))
    leaderboard_parser.add_argument("--output", default=str(LEADERBOARD_JSON_PATH))
    leaderboard_parser.set_defaults(func=_cmd_leaderboard)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = cast(CommandHandler, args.func)
    return handler(args=args)


if __name__ == "__main__":
    sys.exit(main())
