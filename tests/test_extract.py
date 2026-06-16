from __future__ import annotations

from json import dumps

from sensebench.prompts.models import SENSE_INDEX_FIELD, OutputMode
from sensebench.runner.extract import (
    InvalidSenseIndexExtraction,
    ValidSenseIndexExtraction,
    extract_sense_index,
)
from sensebench.runs.models import InvalidOutputReason

VALID_SENSE_INDEX: int = 2
INVALID_SENSE_INDEX_TEXT: str = "2"
OUT_OF_RANGE_SENSE_INDEX: int = 4
CANDIDATE_COUNT: int = 3
JSON_SENSE_OUTPUT: str = dumps({SENSE_INDEX_FIELD: VALID_SENSE_INDEX})
FENCED_JSON_OUTPUT: str = f"```json\n{JSON_SENSE_OUTPUT}\n```"
PROSE_WITH_JSON_OUTPUT: str = f"The best answer is {JSON_SENSE_OUTPUT}."
JSON_WITH_EXTRA_KEY_OUTPUT: str = dumps({SENSE_INDEX_FIELD: VALID_SENSE_INDEX, "reason": "context"})
SINGLE_QUOTED_JSON_OUTPUT: str = f"{{'{SENSE_INDEX_FIELD}': {VALID_SENSE_INDEX}}}"
LABELED_INDEX_OUTPUT: str = f"{SENSE_INDEX_FIELD}: {VALID_SENSE_INDEX}"
FENCED_LABELED_INDEX_OUTPUT: str = f"```\n{LABELED_INDEX_OUTPUT}\n```"
FENCED_PLAIN_OUTPUT: str = "```\n2\n```"
MULTIPLE_JSON_OBJECTS_OUTPUT: str = f"{JSON_SENSE_OUTPUT} then {dumps({SENSE_INDEX_FIELD: 1})}"
REPEATED_JSON_OBJECTS_OUTPUT: str = f"{JSON_SENSE_OUTPUT}{JSON_SENSE_OUTPUT}"
DISAGREEING_JSON_OBJECTS_OUTPUT: str = f"{JSON_SENSE_OUTPUT}{dumps({SENSE_INDEX_FIELD: 1})}"
BOOLEAN_SENSE_INDEX_OUTPUT: str = dumps({SENSE_INDEX_FIELD: True})


def test_extract_json_sense_index() -> None:
    extracted = extract_sense_index(
        text=dumps({SENSE_INDEX_FIELD: VALID_SENSE_INDEX}),
        output_mode=OutputMode.JSON_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, ValidSenseIndexExtraction), "extraction succeeds"
    assert extracted.sense_index == VALID_SENSE_INDEX, "sense index is parsed"


def test_extract_accepts_string_sense_index() -> None:
    extracted = extract_sense_index(
        text=dumps({SENSE_INDEX_FIELD: INVALID_SENSE_INDEX_TEXT}),
        output_mode=OutputMode.JSON_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, ValidSenseIndexExtraction), "string digit extraction succeeds"
    assert extracted.sense_index == VALID_SENSE_INDEX, "string digit sense index is parsed"


def test_extract_accepts_fenced_json_sense_index() -> None:
    extracted = extract_sense_index(
        text=FENCED_JSON_OUTPUT,
        output_mode=OutputMode.JSON_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, ValidSenseIndexExtraction), "fenced JSON extraction succeeds"
    assert extracted.sense_index == VALID_SENSE_INDEX, "fenced JSON sense index is parsed"


def test_extract_accepts_embedded_json_sense_index() -> None:
    extracted = extract_sense_index(
        text=PROSE_WITH_JSON_OUTPUT,
        output_mode=OutputMode.JSON_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, ValidSenseIndexExtraction), "embedded JSON extraction succeeds"
    assert extracted.sense_index == VALID_SENSE_INDEX, "embedded JSON sense index is parsed"


def test_extract_accepts_json_with_extra_keys() -> None:
    extracted = extract_sense_index(
        text=JSON_WITH_EXTRA_KEY_OUTPUT,
        output_mode=OutputMode.JSON_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, ValidSenseIndexExtraction), "extra JSON keys are tolerated"
    assert extracted.sense_index == VALID_SENSE_INDEX, "sense index is parsed from extra-key JSON"


def test_extract_accepts_single_quoted_json_sense_index() -> None:
    extracted = extract_sense_index(
        text=SINGLE_QUOTED_JSON_OUTPUT,
        output_mode=OutputMode.JSON_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, ValidSenseIndexExtraction), "Python-like dict extraction succeeds"
    assert extracted.sense_index == VALID_SENSE_INDEX, "single-quoted dict sense index is parsed"


def test_extract_accepts_labeled_json_sense_index() -> None:
    extracted = extract_sense_index(
        text=LABELED_INDEX_OUTPUT,
        output_mode=OutputMode.JSON_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, ValidSenseIndexExtraction), "labeled index extraction succeeds"
    assert extracted.sense_index == VALID_SENSE_INDEX, "labeled sense index is parsed"


def test_extract_accepts_fenced_labeled_json_sense_index() -> None:
    extracted = extract_sense_index(
        text=FENCED_LABELED_INDEX_OUTPUT,
        output_mode=OutputMode.JSON_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, ValidSenseIndexExtraction), "fenced label extraction succeeds"
    assert extracted.sense_index == VALID_SENSE_INDEX, "fenced label sense index is parsed"


def test_extract_plain_accepts_fenced_integer() -> None:
    extracted = extract_sense_index(
        text=FENCED_PLAIN_OUTPUT,
        output_mode=OutputMode.PLAIN_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, ValidSenseIndexExtraction), "fenced integer extraction succeeds"
    assert extracted.sense_index == VALID_SENSE_INDEX, "fenced integer sense index is parsed"


def test_extract_plain_repairs_duplicated_integer_when_full_index_is_out_of_range() -> None:
    extracted = extract_sense_index(
        text="22",
        output_mode=OutputMode.PLAIN_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, ValidSenseIndexExtraction), "duplicated integer is repaired"
    assert extracted.sense_index == VALID_SENSE_INDEX, "repeated half is used as sense index"
    assert extracted.repeated_plain_integer is True, "repeated integer repair is recorded"


def test_extract_plain_keeps_direct_integer_when_it_is_in_range() -> None:
    extracted = extract_sense_index(
        text="11",
        output_mode=OutputMode.PLAIN_SENSE_INDEX,
        candidate_count=11,
    )

    assert isinstance(extracted, ValidSenseIndexExtraction), "direct integer is parsed"
    assert extracted.sense_index == 11, "in-range direct integer wins over repeated half"
    assert extracted.repeated_plain_integer is False, "no repeated repair is recorded"


def test_extract_plain_rejects_duplicated_integer_when_half_is_out_of_range() -> None:
    extracted = extract_sense_index(
        text="44",
        output_mode=OutputMode.PLAIN_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, InvalidSenseIndexExtraction), "out-of-range duplicate fails"
    assert extracted.invalid_reason == InvalidOutputReason.INDEX_OUT_OF_RANGE, (
        "out-of-range sense index is still rejected"
    )


def test_extract_plain_repairs_repeated_multi_digit_integer() -> None:
    extracted = extract_sense_index(
        text="1010",
        output_mode=OutputMode.PLAIN_SENSE_INDEX,
        candidate_count=10,
    )

    assert isinstance(extracted, ValidSenseIndexExtraction), "multi-digit duplicate is repaired"
    assert extracted.sense_index == 10, "multi-digit repeated half is parsed"
    assert extracted.repeated_plain_integer is True, "repeated integer repair is recorded"


def test_extract_rejects_ambiguous_json_objects() -> None:
    extracted = extract_sense_index(
        text=MULTIPLE_JSON_OBJECTS_OUTPUT,
        output_mode=OutputMode.JSON_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, InvalidSenseIndexExtraction), "ambiguous extraction fails"
    assert extracted.invalid_reason == InvalidOutputReason.INVALID_JSON, (
        "ambiguous JSON is rejected"
    )


def test_extract_accepts_repeated_identical_json_objects() -> None:
    extracted = extract_sense_index(
        text=REPEATED_JSON_OBJECTS_OUTPUT,
        output_mode=OutputMode.JSON_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, ValidSenseIndexExtraction), "repeated JSON extraction succeeds"
    assert extracted.sense_index == VALID_SENSE_INDEX, "sense index is parsed"
    assert extracted.repeated_json_objects is True, "repeated JSON repair is recorded"


def test_extract_rejects_repeated_disagreeing_json_objects() -> None:
    extracted = extract_sense_index(
        text=DISAGREEING_JSON_OBJECTS_OUTPUT,
        output_mode=OutputMode.JSON_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, InvalidSenseIndexExtraction), "disagreeing JSON fails"
    assert extracted.invalid_reason == InvalidOutputReason.INVALID_JSON, (
        "disagreeing JSON is rejected"
    )


def test_extract_rejects_boolean_sense_index() -> None:
    extracted = extract_sense_index(
        text=BOOLEAN_SENSE_INDEX_OUTPUT,
        output_mode=OutputMode.JSON_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, InvalidSenseIndexExtraction), "boolean extraction fails"
    assert extracted.invalid_reason == InvalidOutputReason.SENSE_INDEX_NOT_INT, (
        "boolean sense index is rejected"
    )


def test_extract_rejects_out_of_range_index() -> None:
    extracted = extract_sense_index(
        text=str(OUT_OF_RANGE_SENSE_INDEX),
        output_mode=OutputMode.PLAIN_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, InvalidSenseIndexExtraction), "extraction fails"
    assert extracted.invalid_reason == InvalidOutputReason.INDEX_OUT_OF_RANGE, (
        "out-of-range sense index is rejected"
    )
