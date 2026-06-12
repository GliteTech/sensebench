from __future__ import annotations

from sensebench.datasets.models import ItemID
from sensebench.prompts.models import MessageRole
from sensebench.runner.run import _llm_parameters, _model_with_resolved_snapshots
from sensebench.runs.models import (
    CLOUD_LLM_KIND,
    AttemptKind,
    CallID,
    CallRecord,
    CallStatus,
    CloudLlmReference,
    CostBreakdown,
    CostSourceKind,
    MessageRecord,
    ModelSourceKind,
    SamplingParameters,
    TokenUsage,
)

CALL_ID_1: CallID = "call-1"
CALL_ID_2: CallID = "call-2"
CALL_ID_3: CallID = "call-3"
ITEM_ID: ItemID = "item-1"
REQUESTED_MODEL: str = "gpt-5.5"
MODEL_DISPLAY_NAME: str = "GPT-5.5"
MODEL_SNAPSHOT_A: str = "gpt-5.5-2026-04-23"
MODEL_SNAPSHOT_B: str = "gpt-5.5-2026-05-01"
MESSAGE_CONTENT: str = "choose"


def _cloud_reference() -> CloudLlmReference:
    return CloudLlmReference(
        kind=CLOUD_LLM_KIND,
        display_name=MODEL_DISPLAY_NAME,
        requested_model=REQUESTED_MODEL,
        source_kind=ModelSourceKind.PROPRIETARY,
    )


def _call(
    *,
    call_id: CallID,
    model: str,
    status: CallStatus = CallStatus.SUCCESS,
) -> CallRecord:
    return CallRecord(
        call_id=call_id,
        item_id=ITEM_ID,
        vote_index=1,
        attempt_index=1,
        attempt_kind=AttemptKind.INITIAL,
        transport_retry_count=0,
        status=status,
        model=model,
        messages=[MessageRecord(role=MessageRole.USER, content=MESSAGE_CONTENT)],
        usage=TokenUsage(),
        cost=CostBreakdown(total_usd=None, source=CostSourceKind.UNAVAILABLE),
    )


def test_model_with_resolved_snapshots_sets_single_resolved_model() -> None:
    updated_model = _model_with_resolved_snapshots(
        model=_cloud_reference(),
        calls=[
            _call(call_id=CALL_ID_1, model=MODEL_SNAPSHOT_A),
            _call(call_id=CALL_ID_2, model=MODEL_SNAPSHOT_A),
            _call(
                call_id=CALL_ID_3,
                model=REQUESTED_MODEL,
                status=CallStatus.TRANSPORT_ERROR,
            ),
        ],
    )

    assert isinstance(updated_model, CloudLlmReference)
    assert updated_model.resolved_model == MODEL_SNAPSHOT_A
    assert updated_model.resolved_model_counts == {MODEL_SNAPSHOT_A: 2}


def test_model_with_resolved_snapshots_keeps_distribution_for_mixed_models() -> None:
    updated_model = _model_with_resolved_snapshots(
        model=_cloud_reference(),
        calls=[
            _call(call_id=CALL_ID_1, model=MODEL_SNAPSHOT_A),
            _call(call_id=CALL_ID_2, model=MODEL_SNAPSHOT_B),
        ],
    )

    assert isinstance(updated_model, CloudLlmReference)
    assert updated_model.resolved_model is None
    assert updated_model.resolved_model_counts == {
        MODEL_SNAPSHOT_A: 1,
        MODEL_SNAPSHOT_B: 1,
    }


def test_llm_parameters_maps_disabled_thinking() -> None:
    parameters = _llm_parameters(
        sampling=SamplingParameters(extra={"thinking": "disabled"}),
    )

    assert parameters["thinking"] == {"type": "disabled"}
    assert parameters["allowed_openai_params"] == ["thinking"]
