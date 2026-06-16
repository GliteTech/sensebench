"""Model-output extraction for SenseBench prompt modes."""

from __future__ import annotations

from ast import literal_eval
from dataclasses import dataclass
from json import JSONDecodeError, JSONDecoder, loads
from re import DOTALL, IGNORECASE, Match, Pattern, compile
from typing import assert_never

from sensebench.prompts.models import SENSE_INDEX_FIELD, OutputMode
from sensebench.runs.models import InvalidOutputReason

FENCED_BLOCK_PATTERN: Pattern[str] = compile(
    r"```[A-Za-z0-9_-]*\s*\n?(?P<body>.*?)\n?```",
    DOTALL,
)
FULL_FENCED_BLOCK_PATTERN: Pattern[str] = compile(
    r"\A```[A-Za-z0-9_-]*\s*\n?(?P<body>.*?)\n?```\Z",
    DOTALL,
)
SENSE_INDEX_LABEL_PATTERN: Pattern[str] = compile(
    rf'\A"?{SENSE_INDEX_FIELD}"?\s*[:=]\s*(?P<value>\d+)\.?\Z',
    IGNORECASE,
)
PLAIN_ANSWER_LABEL_PATTERN: Pattern[str] = compile(
    r"\A(?:answer|index)\s*[:=]\s*(?P<value>\d+)\.?\Z",
    IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ValidSenseIndexExtraction:
    sense_index: int
    repeated_json_objects: bool = False
    repeated_plain_integer: bool = False


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


def _validate_repeated_plain_integer(
    *,
    text: str,
    candidate_count: int,
) -> SenseIndexExtraction | None:
    if not text.isdigit() or len(text) % 2 != 0:
        return None
    midpoint = len(text) // 2
    first_half = text[:midpoint]
    second_half = text[midpoint:]
    if first_half != second_half:
        return None
    repeated_value = int(first_half)
    if repeated_value < 1 or repeated_value > candidate_count:
        return None
    return ValidSenseIndexExtraction(
        sense_index=repeated_value,
        repeated_plain_integer=True,
    )


def _parse_numeric_text(*, text: str, candidate_count: int) -> SenseIndexExtraction | None:
    direct_value = _coerce_sense_index_value(value=text)
    if direct_value is None:
        return None
    direct_extraction = _validate_range(value=direct_value, candidate_count=candidate_count)
    if isinstance(direct_extraction, ValidSenseIndexExtraction):
        return direct_extraction
    repeated_extraction = _validate_repeated_plain_integer(
        text=text,
        candidate_count=candidate_count,
    )
    if repeated_extraction is not None:
        return repeated_extraction
    return direct_extraction


def _coerce_sense_index_value(*, value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _extraction_from_parsed_value(
    *,
    parsed: object,
    candidate_count: int,
) -> SenseIndexExtraction:
    if isinstance(parsed, str):
        numeric_text_extraction = _parse_numeric_text(
            text=parsed.strip(),
            candidate_count=candidate_count,
        )
        if numeric_text_extraction is not None:
            return numeric_text_extraction
    else:
        direct_value = _coerce_sense_index_value(value=parsed)
        if direct_value is not None:
            return _validate_range(value=direct_value, candidate_count=candidate_count)
    if not isinstance(parsed, dict):
        return InvalidSenseIndexExtraction(
            invalid_reason=InvalidOutputReason.JSON_NOT_OBJECT,
        )
    if SENSE_INDEX_FIELD not in parsed:
        return InvalidSenseIndexExtraction(
            invalid_reason=InvalidOutputReason.JSON_WRONG_KEYS,
        )
    sense_index = _coerce_sense_index_value(value=parsed[SENSE_INDEX_FIELD])
    if sense_index is None:
        return InvalidSenseIndexExtraction(
            invalid_reason=InvalidOutputReason.SENSE_INDEX_NOT_INT,
        )
    return _validate_range(value=sense_index, candidate_count=candidate_count)


def _load_jsonish_value(*, text: str) -> object | None:
    try:
        parsed_json: object = loads(text)
        return parsed_json
    except JSONDecodeError:
        pass
    try:
        parsed_literal: object = literal_eval(text)
        return parsed_literal
    except (SyntaxError, ValueError):
        return None


def _append_unique_text(*, values: list[str], value: str) -> None:
    stripped = value.strip()
    if len(stripped) == 0:
        return
    if stripped not in values:
        values.append(stripped)


def _json_object_substrings(*, text: str) -> list[str]:
    substrings: list[str] = []
    start_index: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start_index = index
            depth += 1
        elif character == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_index is not None:
                substrings.append(text[start_index : index + 1])
                start_index = None
    return substrings


def _candidate_json_texts(*, text: str) -> list[str]:
    candidates: list[str] = []
    _append_unique_text(values=candidates, value=text)
    full_fence_match = FULL_FENCED_BLOCK_PATTERN.fullmatch(text)
    if full_fence_match is not None:
        _append_unique_text(values=candidates, value=full_fence_match.group("body"))
    fence_matches: list[Match[str]] = list(FENCED_BLOCK_PATTERN.finditer(text))
    if len(fence_matches) == 1:
        _append_unique_text(values=candidates, value=fence_matches[0].group("body"))
    json_substrings = _json_object_substrings(text=text)
    if len(json_substrings) == 1:
        _append_unique_text(values=candidates, value=json_substrings[0])
    return candidates


def _labeled_sense_index(*, text: str) -> int | None:
    for pattern in [SENSE_INDEX_LABEL_PATTERN, PLAIN_ANSWER_LABEL_PATTERN]:
        match = pattern.fullmatch(text)
        if match is not None:
            return int(match.group("value"))
    return None


def _parse_json_output(*, text: str, candidate_count: int) -> SenseIndexExtraction:
    repeated_json_extraction = _parse_repeated_json_objects(
        text=text,
        candidate_count=candidate_count,
    )
    if repeated_json_extraction is not None:
        return repeated_json_extraction
    last_invalid: InvalidSenseIndexExtraction | None = None
    for candidate_text in _candidate_json_texts(text=text):
        parsed = _load_jsonish_value(text=candidate_text)
        if parsed is not None:
            extraction = _extraction_from_parsed_value(
                parsed=parsed,
                candidate_count=candidate_count,
            )
            if isinstance(extraction, ValidSenseIndexExtraction):
                return extraction
            last_invalid = extraction
        labeled_index = _labeled_sense_index(text=candidate_text)
        if labeled_index is not None:
            return _validate_range(value=labeled_index, candidate_count=candidate_count)
    if last_invalid is not None:
        return last_invalid
    return InvalidSenseIndexExtraction(invalid_reason=InvalidOutputReason.INVALID_JSON)


def _parse_repeated_json_objects(
    *,
    text: str,
    candidate_count: int,
) -> SenseIndexExtraction | None:
    json_decoder = JSONDecoder()
    values: list[object] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            value, end_index = json_decoder.raw_decode(text, index)
        except JSONDecodeError:
            return None
        values.append(value)
        index = end_index
    if len(values) < 2:
        return None
    sense_indexes: list[int] = []
    for value in values:
        extraction = _extraction_from_parsed_value(
            parsed=value,
            candidate_count=candidate_count,
        )
        if not isinstance(extraction, ValidSenseIndexExtraction):
            return extraction
        sense_indexes.append(extraction.sense_index)
    if len(set(sense_indexes)) != 1:
        return InvalidSenseIndexExtraction(invalid_reason=InvalidOutputReason.INVALID_JSON)
    return ValidSenseIndexExtraction(
        sense_index=sense_indexes[0],
        repeated_json_objects=True,
    )


def _parse_plain_output(*, text: str, candidate_count: int) -> SenseIndexExtraction:
    for candidate_text in _candidate_json_texts(text=text):
        numeric_text_extraction = _parse_numeric_text(
            text=candidate_text,
            candidate_count=candidate_count,
        )
        if numeric_text_extraction is not None:
            return numeric_text_extraction
        labeled_index = _labeled_sense_index(text=candidate_text)
        if labeled_index is not None:
            return _validate_range(value=labeled_index, candidate_count=candidate_count)
        parsed = _load_jsonish_value(text=candidate_text)
        if parsed is None:
            continue
        extraction = _extraction_from_parsed_value(
            parsed=parsed,
            candidate_count=candidate_count,
        )
        if isinstance(extraction, ValidSenseIndexExtraction):
            return extraction
    return InvalidSenseIndexExtraction(
        invalid_reason=InvalidOutputReason.PLAIN_NOT_INTEGER,
    )


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
