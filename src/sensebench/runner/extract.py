"""Model-output extraction for SenseBench prompt modes."""

from __future__ import annotations

from dataclasses import dataclass
from json import JSONDecodeError, loads
from typing import assert_never

from sensebench.prompts.models import SENSE_INDEX_FIELD, OutputMode
from sensebench.runs.models import InvalidOutputReason


@dataclass(frozen=True, slots=True)
class ValidSenseIndexExtraction:
    sense_index: int


@dataclass(frozen=True, slots=True)
class InvalidSenseIndexExtraction:
    invalid_reason: InvalidOutputReason


type SenseIndexExtraction = ValidSenseIndexExtraction | InvalidSenseIndexExtraction


def _validate_range(*, value: int, candidate_count: int) -> SenseIndexExtraction:
    if value < 1 or value > candidate_count:
        return InvalidSenseIndexExtraction(
            invalid_reason=InvalidOutputReason.INDEX_OUT_OF_RANGE,
        )
    return ValidSenseIndexExtraction(sense_index=value)


def _parse_json_output(*, text: str, candidate_count: int) -> SenseIndexExtraction:
    try:
        parsed: object = loads(text)
    except JSONDecodeError:
        return InvalidSenseIndexExtraction(invalid_reason=InvalidOutputReason.INVALID_JSON)
    if not isinstance(parsed, dict):
        return InvalidSenseIndexExtraction(
            invalid_reason=InvalidOutputReason.JSON_NOT_OBJECT,
        )
    if set(parsed.keys()) != {SENSE_INDEX_FIELD}:
        return InvalidSenseIndexExtraction(
            invalid_reason=InvalidOutputReason.JSON_WRONG_KEYS,
        )
    raw_value = parsed[SENSE_INDEX_FIELD]
    if not isinstance(raw_value, int):
        return InvalidSenseIndexExtraction(
            invalid_reason=InvalidOutputReason.SENSE_INDEX_NOT_INT,
        )
    return _validate_range(value=raw_value, candidate_count=candidate_count)


def _parse_plain_output(*, text: str, candidate_count: int) -> SenseIndexExtraction:
    if not text.isdigit():
        return InvalidSenseIndexExtraction(
            invalid_reason=InvalidOutputReason.PLAIN_NOT_INTEGER,
        )
    return _validate_range(value=int(text), candidate_count=candidate_count)


def extract_sense_index(
    *,
    text: str | None,
    output_mode: OutputMode,
    candidate_count: int,
) -> SenseIndexExtraction:
    if text is None or len(text.strip()) == 0:
        return InvalidSenseIndexExtraction(
            invalid_reason=InvalidOutputReason.EMPTY_OUTPUT,
        )
    stripped = text.strip()
    if output_mode == OutputMode.JSON_SENSE_INDEX:
        return _parse_json_output(text=stripped, candidate_count=candidate_count)
    if output_mode == OutputMode.PLAIN_SENSE_INDEX:
        return _parse_plain_output(text=stripped, candidate_count=candidate_count)
    assert_never(output_mode)
