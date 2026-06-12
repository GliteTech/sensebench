"""Pydantic models for submitted run artifacts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from sensebench.datasets.models import DatasetID, ItemID, SenseKey
from sensebench.prompts.models import MessageRole, PromptID
from sensebench.wordnet import SynsetID

type RunSchemaVersion = Literal["sensebench-run-v1"]
type CloudLlmKind = Literal["cloud_llm"]
type SelfHostedLlmKind = Literal["self_hosted_llm"]

RUN_SCHEMA_VERSION: RunSchemaVersion = "sensebench-run-v1"
CLOUD_LLM_KIND: CloudLlmKind = "cloud_llm"
SELF_HOSTED_LLM_KIND: SelfHostedLlmKind = "self_hosted_llm"
MODEL_REFERENCE_KIND_FIELD: str = "kind"
USD_CURRENCY_CODE: Literal["USD"] = "USD"
MIN_HTTP_STATUS_CODE: int = 100
MAX_HTTP_STATUS_CODE: int = 599

type RunID = str
type CallID = str
type ModelID = str

type NonNegativeInt = Annotated[int, Field(ge=0)]
type PositiveInt = Annotated[int, Field(ge=1)]
type NonNegativeFloat = Annotated[float, Field(ge=0.0)]
type Probability = Annotated[float, Field(ge=0.0, le=1.0)]
type HttpStatusCode = Annotated[
    int,
    Field(ge=MIN_HTTP_STATUS_CODE, le=MAX_HTTP_STATUS_CODE),
]


class StrictRunModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelHostingKind(StrEnum):
    CLOUD_API = "cloud_api"
    SELF_HOSTED = "self_hosted"


class ModelSourceKind(StrEnum):
    OPEN_SOURCE = "open_source"
    PROPRIETARY = "proprietary"
    UNKNOWN = "unknown"


class AttemptKind(StrEnum):
    INITIAL = "initial"
    SEMANTIC_REASK = "semantic_reask"


class CallStatus(StrEnum):
    SUCCESS = "success"
    TRANSPORT_ERROR = "transport_error"


class VoteStatus(StrEnum):
    SUCCESS = "success"
    INVALID_OUTPUT = "invalid_output"
    TRANSPORT_ERROR = "transport_error"


class PredictionStatus(StrEnum):
    SUCCESS = "success"
    MONOSEMOUS = "monosemous"
    NO_CANDIDATES = "no_candidates"
    NO_VALID_VOTE = "no_valid_vote"


class TieBreakKind(StrEnum):
    EARLIEST_VOTE = "earliest_vote"


class MonosemousPolicyKind(StrEnum):
    SHORT_CIRCUIT = "short_circuit"


class CostSourceKind(StrEnum):
    LITELLM_ESTIMATE = "litellm_estimate"
    NO_CALLS = "no_calls"
    UNAVAILABLE = "unavailable"


class InvalidOutputReason(StrEnum):
    EMPTY_OUTPUT = "empty_output"
    INVALID_JSON = "invalid_json"
    JSON_NOT_OBJECT = "json_not_object"
    JSON_WRONG_KEYS = "json_wrong_keys"
    SENSE_INDEX_NOT_INT = "sense_index_not_int"
    PLAIN_NOT_INTEGER = "plain_not_integer"
    INDEX_OUT_OF_RANGE = "index_out_of_range"


class DatasetReference(StrictRunModel):
    dataset_id: DatasetID
    dataset_version: str | None = None
    dataset_revision: str | None = None
    content_hash: str | None = None
    item_count: NonNegativeInt


class PromptReference(StrictRunModel):
    id: PromptID
    sensebench_version: str | None = None


class CloudLlmReference(StrictRunModel):
    kind: CloudLlmKind
    display_name: str
    requested_model: ModelID
    resolved_model: ModelID | None = None
    resolved_model_counts: dict[ModelID, NonNegativeInt] = Field(default_factory=dict)
    llm_vendor: str | None = None
    api_provider: str | None = None
    source_kind: ModelSourceKind
    license: str | None = None
    model_url: str | None = None
    reasoning_effort: str | None = None
    endpoint_base_url: str | None = None


class SelfHostedLlmReference(StrictRunModel):
    kind: SelfHostedLlmKind
    display_name: str
    requested_model: ModelID
    resolved_model: ModelID | None = None
    resolved_model_counts: dict[ModelID, NonNegativeInt] = Field(default_factory=dict)
    llm_vendor: str | None = None
    source_kind: ModelSourceKind
    license: str | None = None
    model_url: str | None = None
    hf_revision: str | None = None
    quantization: str | None = None
    inference_engine: str | None = None
    inference_engine_version: str | None = None
    endpoint_base_url: str
    gpu: str | None = None
    cpu: str | None = None


type ModelReference = Annotated[
    CloudLlmReference | SelfHostedLlmReference,
    Field(discriminator=MODEL_REFERENCE_KIND_FIELD),
]


class SamplingParameters(StrictRunModel):
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: PositiveInt | None = None
    seed: NonNegativeInt | None = None
    extra: dict[str, str] = Field(default_factory=dict)


class RunPolicy(StrictRunModel):
    votes_per_item: PositiveInt
    semantic_reasks_per_invalid_vote: NonNegativeInt
    tie_break: TieBreakKind
    monosemous_policy: MonosemousPolicyKind


class TokenUsage(StrictRunModel):
    input_tokens: NonNegativeInt | None = None
    cached_input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    reasoning_output_tokens: NonNegativeInt | None = None


class CostBreakdown(StrictRunModel):
    currency: Literal["USD"] = USD_CURRENCY_CODE
    total_usd: NonNegativeFloat | None = None
    input_uncached_usd: NonNegativeFloat | None = None
    input_cached_usd: NonNegativeFloat | None = None
    output_usd: NonNegativeFloat | None = None
    input_uncached_unit_price_usd: NonNegativeFloat | None = None
    input_cached_unit_price_usd: NonNegativeFloat | None = None
    output_unit_price_usd: NonNegativeFloat | None = None
    source: CostSourceKind


class RunTotals(StrictRunModel):
    item_count: NonNegativeInt
    correct_count: NonNegativeInt
    accuracy: Probability | None = None
    call_count: NonNegativeInt
    usage: TokenUsage
    cost: CostBreakdown
    elapsed_seconds: NonNegativeFloat | None = None


class RunnerIdentity(StrictRunModel):
    github_handle: str | None = None
    name: str | None = None
    contact: str | None = None


class RunMetadata(StrictRunModel):
    schema_version: RunSchemaVersion
    run_id: RunID
    created_at: datetime
    git_commit: str | None
    runner: RunnerIdentity
    dataset: DatasetReference
    prompt: PromptReference
    model: ModelReference
    sampling: SamplingParameters
    policy: RunPolicy
    totals: RunTotals

    @field_serializer("created_at")
    def serialize_created_at(self, value: datetime) -> str:
        return value.isoformat()


class CandidateRecord(StrictRunModel):
    index: PositiveInt
    sense_key: SenseKey
    synset_id: SynsetID


class VoteRecord(StrictRunModel):
    vote_index: PositiveInt
    status: VoteStatus
    chosen_sense_index: PositiveInt | None = None
    chosen_sense_key: SenseKey | None = None
    call_ids: list[CallID] = Field(default_factory=list)
    invalid_reason: InvalidOutputReason | str | None = None


class PredictionRecord(StrictRunModel):
    item_id: ItemID
    gold_sense_keys: list[SenseKey]
    candidates: list[CandidateRecord]
    votes: list[VoteRecord]
    predicted_sense_index: PositiveInt | None = None
    predicted_sense_key: SenseKey | None = None
    is_correct: bool | None = None
    status: PredictionStatus
    was_monosemous: bool
    usage: TokenUsage
    cost: CostBreakdown
    latency_seconds: NonNegativeFloat | None = None


class MessageRecord(StrictRunModel):
    role: MessageRole
    content: str


class CallRecord(StrictRunModel):
    call_id: CallID
    item_id: ItemID
    vote_index: PositiveInt
    attempt_index: PositiveInt
    attempt_kind: AttemptKind
    transport_retry_count: NonNegativeInt
    status: CallStatus
    model: ModelID
    messages: list[MessageRecord]
    raw_output: str | None = None
    raw_response: dict[str, object] | None = None
    usage: TokenUsage
    cost: CostBreakdown
    latency_seconds: NonNegativeFloat | None = None
    http_status: HttpStatusCode | None = None
    provider_request_id: str | None = None
    error_kind: str | None = None
    error_message: str | None = None
