"""Validate registered SenseBench prompt files."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import ValidationError

from sensebench.paths import PROMPT_JSON_SUFFIX, PROMPT_REGISTRY_DIR
from sensebench.prompts.models import PromptDefinition, PromptID
from sensebench.prompts.registry import load_prompt_definition, registered_prompt_paths


class ValidationRule(StrEnum):
    DUPLICATE_PROMPT_ID = "duplicate_prompt_id"
    FILE = "file"
    FILENAME_MATCHES_ID = "filename_matches_id"
    PYDANTIC_MODEL = "pydantic_model"
    REGISTRY_PATH = "registry_path"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    rule: ValidationRule
    location: str
    message: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    path: Path
    issues: list[ValidationIssue]

    def has_errors(self) -> bool:
        return len(self.issues) > 0


@dataclass(frozen=True, slots=True)
class CliArgs:
    all_prompts: bool
    prompt_paths: list[Path]


def _is_under_registry(*, path: Path) -> bool:
    resolved_path: Path = path.resolve()
    resolved_registry: Path = PROMPT_REGISTRY_DIR.resolve()
    return resolved_path == resolved_registry or resolved_registry in resolved_path.parents


def _validation_location(*, raw_location: object) -> str:
    if isinstance(raw_location, tuple):
        parts: list[str] = [str(part) for part in raw_location]
        if len(parts) == 0:
            return "$"
        return "$." + ".".join(parts)
    return "$"


def _issues_from_validation_error(*, exc: ValidationError) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for error in exc.errors():
        issues.append(
            ValidationIssue(
                rule=ValidationRule.PYDANTIC_MODEL,
                location=_validation_location(raw_location=error.get("loc", ())),
                message=str(error.get("msg", "validation error")),
            )
        )
    return issues


def _path_issues(*, path: Path, prompt: PromptDefinition) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not _is_under_registry(path=path):
        issues.append(
            ValidationIssue(
                rule=ValidationRule.REGISTRY_PATH,
                location=str(path),
                message=f"prompt file must be under {PROMPT_REGISTRY_DIR}",
            )
        )

    expected_filename = f"{prompt.id}{PROMPT_JSON_SUFFIX}"
    if path.name != expected_filename:
        issues.append(
            ValidationIssue(
                rule=ValidationRule.FILENAME_MATCHES_ID,
                location=str(path),
                message=f"filename must be {expected_filename}",
            )
        )
    return issues


def validate_prompt_file(*, path: Path) -> ValidationReport:
    try:
        prompt: PromptDefinition = load_prompt_definition(path=path)
    except OSError as exc:
        return ValidationReport(
            path=path,
            issues=[
                ValidationIssue(
                    rule=ValidationRule.FILE,
                    location=str(path),
                    message=str(exc),
                ),
            ],
        )
    except ValidationError as exc:
        return ValidationReport(
            path=path,
            issues=_issues_from_validation_error(exc=exc),
        )

    return ValidationReport(path=path, issues=_path_issues(path=path, prompt=prompt))


def validate_prompt_registry() -> list[ValidationReport]:
    prompt_paths: list[Path] = registered_prompt_paths()
    reports: list[ValidationReport] = [
        validate_prompt_file(path=prompt_path) for prompt_path in prompt_paths
    ]

    prompt_ids: dict[PromptID, Path] = {}
    for prompt_path in prompt_paths:
        try:
            prompt: PromptDefinition = load_prompt_definition(path=prompt_path)
        except (OSError, ValidationError):
            continue

        existing_path: Path | None = prompt_ids.get(prompt.id)
        if existing_path is not None:
            reports.append(
                ValidationReport(
                    path=prompt_path,
                    issues=[
                        ValidationIssue(
                            rule=ValidationRule.DUPLICATE_PROMPT_ID,
                            location="id",
                            message=f"duplicates {existing_path}",
                        ),
                    ],
                )
            )
        else:
            prompt_ids[prompt.id] = prompt_path

    return reports


def _print_report(*, report: ValidationReport) -> None:
    if report.has_errors():
        print(f"{report.path}: FAILED")
        for issue in report.issues:
            print(f"  [{issue.rule.value}] {issue.location}: {issue.message}")
        return
    print(f"{report.path}: OK")


def _namespace_prompt_paths(*, namespace: argparse.Namespace) -> list[str]:
    raw_prompt_paths: object = namespace.prompt_paths
    assert isinstance(raw_prompt_paths, list), "namespace prompt_paths is a list"
    return [str(raw_path) for raw_path in raw_prompt_paths]


def _parse_args(*, argv: Sequence[str] | None) -> CliArgs:
    parser = argparse.ArgumentParser(description="Validate SenseBench prompt JSON files.")
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_prompts",
        help="Validate every prompt in the registered prompt directory.",
    )
    parser.add_argument(
        "prompt_paths",
        nargs="*",
        help="Prompt JSON files to validate.",
    )
    namespace = parser.parse_args(argv)
    all_prompts = bool(namespace.all_prompts)
    prompt_paths: list[Path] = [
        Path(raw_path) for raw_path in _namespace_prompt_paths(namespace=namespace)
    ]
    if not all_prompts and len(prompt_paths) == 0:
        parser.error("pass one or more prompt files, or use --all")
    if all_prompts and len(prompt_paths) > 0:
        parser.error("--all cannot be combined with explicit prompt paths")
    return CliArgs(
        all_prompts=all_prompts,
        prompt_paths=prompt_paths,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args: CliArgs = _parse_args(argv=argv)
    reports: list[ValidationReport]
    if args.all_prompts:
        reports = validate_prompt_registry()
    else:
        reports = [validate_prompt_file(path=path) for path in args.prompt_paths]

    failed_count = 0
    for report in reports:
        _print_report(report=report)
        if report.has_errors():
            failed_count += 1

    if failed_count > 0:
        print(f"{failed_count} prompt file(s) failed validation.")
        return 1
    print(f"{len(reports)} prompt file(s) passed validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
