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


def test_extract_json_sense_index() -> None:
    extracted = extract_sense_index(
        text=dumps({SENSE_INDEX_FIELD: VALID_SENSE_INDEX}),
        output_mode=OutputMode.JSON_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, ValidSenseIndexExtraction), "extraction succeeds"
    assert extracted.sense_index == VALID_SENSE_INDEX, "sense index is parsed"


def test_extract_rejects_string_sense_index() -> None:
    extracted = extract_sense_index(
        text=dumps({SENSE_INDEX_FIELD: INVALID_SENSE_INDEX_TEXT}),
        output_mode=OutputMode.JSON_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, InvalidSenseIndexExtraction), "extraction fails"
    assert (
        extracted.invalid_reason == InvalidOutputReason.SENSE_INDEX_NOT_INT
    ), "string sense index is rejected"


def test_extract_rejects_out_of_range_index() -> None:
    extracted = extract_sense_index(
        text=str(OUT_OF_RANGE_SENSE_INDEX),
        output_mode=OutputMode.PLAIN_SENSE_INDEX,
        candidate_count=CANDIDATE_COUNT,
    )

    assert isinstance(extracted, InvalidSenseIndexExtraction), "extraction fails"
    assert (
        extracted.invalid_reason == InvalidOutputReason.INDEX_OUT_OF_RANGE
    ), "out-of-range sense index is rejected"
