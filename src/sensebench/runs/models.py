"""Pydantic models for submitted run artifacts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from sensebench.datasets.models import DatasetID, ItemID, SenseKey
from sensebench.prompts.models import MessageRole, PromptID
from sensebench.wordnet import SynsetID

RUN_SCHEMA_VERSION: Literal["sensebench-run-v1"] = "sensebench-run-v1"
CLOUD_LLM_KIND: Literal["cloud_llm"] = "cloud_llm"
SELF_HOSTED_LLM_KIND: Literal["self_hosted_llm"] = "self_hosted_llm"

type RunID = str
type CallID = str


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


class DatasetReference(StrictRunModel):
    dataset_id: DatasetID
    dataset_version: str | None = None
    dataset_revision: str | None = None
    content_hash: str | None = None
    item_count: int = Field(ge=0)


class PromptReference(StrictRunModel):
    id: PromptID
    sensebench_version: str | None = None


class CloudLlmReference(StrictRunModel):
    kind: Literal["cloud_llm"]
    display_name: str
    requested_model: str
    resolved_model: str | None = None
    resolved_model_counts: dict[str, int] = Field(default_factory=dict)
    llm_vendor: str | None = None
    api_provider: str | None = None
    source_kind: ModelSourceKind
    license: str | None = None
    model_url: str | None = None
    reasoning_effort: str | None = None
    endpoint_base_url: str | None = None


class SelfHostedLlmReference(StrictRunModel):
    kind: Literal["self_hosted_llm"]
    display_name: str
    requested_model: str
    resolved_model: str | None = None
    resolved_model_counts: dict[str, int] = Field(default_factory=dict)
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
    Field(discriminator="kind"),
]


class SamplingParameters(StrictRunModel):
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    extra: dict[str, str] = Field(default_factory=dict)


class RunPolicy(StrictRunModel):
    votes_per_item: int = Field(ge=1)
    semantic_reasks_per_invalid_vote: int = Field(ge=0)
    tie_break: TieBreakKind
    monosemous_policy: MonosemousPolicyKind


class TokenUsage(StrictRunModel):
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_output_tokens: int | None = None


class CostBreakdown(StrictRunModel):
    currency: Literal["USD"] = "USD"
    total_usd: float | None = None
    input_uncached_usd: float | None = None
    input_cached_usd: float | None = None
    output_usd: float | None = None
    input_uncached_unit_price_usd: float | None = None
    input_cached_unit_price_usd: float | None = None
    output_unit_price_usd: float | None = None
    source: CostSourceKind


class RunTotals(StrictRunModel):
    item_count: int = Field(ge=0)
    correct_count: int = Field(ge=0)
    accuracy: float | None = None
    call_count: int = Field(ge=0)
    usage: TokenUsage
    cost: CostBreakdown
    elapsed_seconds: float | None = None


class RunnerIdentity(StrictRunModel):
    github_handle: str | None = None
    name: str | None = None
    contact: str | None = None


class RunMetadata(StrictRunModel):
    schema_version: Literal["sensebench-run-v1"]
    run_id: RunID
    created_at: str
    git_commit: str | None
    runner: RunnerIdentity
    dataset: DatasetReference
    prompt: PromptReference
    model: ModelReference
    sampling: SamplingParameters
    policy: RunPolicy
    totals: RunTotals


class CandidateRecord(StrictRunModel):
    index: int = Field(ge=1)
    sense_key: SenseKey
    synset_id: SynsetID


class VoteRecord(StrictRunModel):
    vote_index: int = Field(ge=1)
    status: VoteStatus
    chosen_sense_index: int | None = None
    chosen_sense_key: SenseKey | None = None
    call_ids: list[CallID] = Field(default_factory=list)
    invalid_reason: str | None = None


class PredictionRecord(StrictRunModel):
    item_id: ItemID
    gold_sense_keys: list[SenseKey]
    candidates: list[CandidateRecord]
    votes: list[VoteRecord]
    predicted_sense_index: int | None = None
    predicted_sense_key: SenseKey | None = None
    is_correct: bool | None = None
    status: PredictionStatus
    was_monosemous: bool
    usage: TokenUsage
    cost: CostBreakdown
    latency_seconds: float | None = None


class MessageRecord(StrictRunModel):
    role: MessageRole
    content: str


class CallRecord(StrictRunModel):
    call_id: CallID
    item_id: ItemID
    vote_index: int = Field(ge=1)
    attempt_index: int = Field(ge=1)
    attempt_kind: AttemptKind
    transport_retry_count: int = Field(ge=0)
    status: CallStatus
    model: str
    messages: list[MessageRecord]
    raw_output: str | None = None
    raw_response: dict[str, object] | None = None
    usage: TokenUsage
    cost: CostBreakdown
    latency_seconds: float | None = None
    http_status: int | None = None
    provider_request_id: str | None = None
    error_kind: str | None = None
    error_message: str | None = None
