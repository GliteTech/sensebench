from __future__ import annotations

from sensebench.prompts.models import OutputMode
from sensebench.runner.extract import InvalidOutputReason, extract_sense_index


def test_extract_json_sense_index() -> None:
    extracted = extract_sense_index(
        text='{"sense_index": 2}',
        output_mode=OutputMode.JSON_SENSE_INDEX,
        candidate_count=3,
    )

    assert extracted.sense_index == 2
    assert extracted.invalid_reason is None


def test_extract_rejects_string_sense_index() -> None:
    extracted = extract_sense_index(
        text='{"sense_index": "2"}',
        output_mode=OutputMode.JSON_SENSE_INDEX,
        candidate_count=3,
    )

    assert extracted.sense_index is None
    assert extracted.invalid_reason == InvalidOutputReason.SENSE_INDEX_NOT_INT


def test_extract_rejects_out_of_range_index() -> None:
    extracted = extract_sense_index(
        text="4",
        output_mode=OutputMode.PLAIN_SENSE_INDEX,
        candidate_count=3,
    )

    assert extracted.sense_index is None
    assert extracted.invalid_reason == InvalidOutputReason.INDEX_OUT_OF_RANGE
