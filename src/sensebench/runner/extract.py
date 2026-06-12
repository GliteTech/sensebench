"""Model-output extraction for SenseBench prompt modes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sensebench.prompts.models import SENSE_INDEX_FIELD, OutputMode


class InvalidOutputReason(StrEnum):
    EMPTY_OUTPUT = "empty_output"
    INVALID_JSON = "invalid_json"
    JSON_NOT_OBJECT = "json_not_object"
    JSON_WRONG_KEYS = "json_wrong_keys"
    SENSE_INDEX_NOT_INT = "sense_index_not_int"
    PLAIN_NOT_INTEGER = "plain_not_integer"
    INDEX_OUT_OF_RANGE = "index_out_of_range"


@dataclass(frozen=True, slots=True)
class ExtractedSenseIndex:
    sense_index: int | None
    invalid_reason: InvalidOutputReason | None


def _validate_range(*, value: int, candidate_count: int) -> ExtractedSenseIndex:
    if value < 1 or value > candidate_count:
        return ExtractedSenseIndex(
            sense_index=None,
            invalid_reason=InvalidOutputReason.INDEX_OUT_OF_RANGE,
        )
    return ExtractedSenseIndex(sense_index=value, invalid_reason=None)


def _parse_json_output(*, text: str, candidate_count: int) -> ExtractedSenseIndex:
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        return ExtractedSenseIndex(
            sense_index=None, invalid_reason=InvalidOutputReason.INVALID_JSON
        )
    if not isinstance(parsed, dict):
        return ExtractedSenseIndex(
            sense_index=None,
            invalid_reason=InvalidOutputReason.JSON_NOT_OBJECT,
        )
    if set(parsed.keys()) != {SENSE_INDEX_FIELD}:
        return ExtractedSenseIndex(
            sense_index=None,
            invalid_reason=InvalidOutputReason.JSON_WRONG_KEYS,
        )
    raw_value = parsed[SENSE_INDEX_FIELD]
    if not isinstance(raw_value, int):
        return ExtractedSenseIndex(
            sense_index=None,
            invalid_reason=InvalidOutputReason.SENSE_INDEX_NOT_INT,
        )
    return _validate_range(value=raw_value, candidate_count=candidate_count)


def _parse_plain_output(*, text: str, candidate_count: int) -> ExtractedSenseIndex:
    if not text.isdigit():
        return ExtractedSenseIndex(
            sense_index=None,
            invalid_reason=InvalidOutputReason.PLAIN_NOT_INTEGER,
        )
    return _validate_range(value=int(text), candidate_count=candidate_count)


def extract_sense_index(
    *,
    text: str | None,
    output_mode: OutputMode,
    candidate_count: int,
) -> ExtractedSenseIndex:
    if text is None or len(text.strip()) == 0:
        return ExtractedSenseIndex(
            sense_index=None,
            invalid_reason=InvalidOutputReason.EMPTY_OUTPUT,
        )
    stripped = text.strip()
    if output_mode == OutputMode.JSON_SENSE_INDEX:
        return _parse_json_output(text=stripped, candidate_count=candidate_count)
    if output_mode == OutputMode.PLAIN_SENSE_INDEX:
        return _parse_plain_output(text=stripped, candidate_count=candidate_count)
    raise ValueError(f"Unsupported output mode: {output_mode}")
