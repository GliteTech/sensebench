"""Typed JSON models for self-hosted benchmark helper scripts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_serializer

type GpuPresetKey = str
type JobID = str

CREATED_AT_FIELD: str = "created_at"
DESTROYED_AT_FIELD: str = "destroyed_at"


class StrictToolModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GpuPresetConfig(StrictToolModel):
    gpu_label: str = Field(min_length=1)
    search: str = Field(min_length=1)
    disk_gb: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    max_hourly_usd: float = Field(ge=0.0)


class ManifestSampling(StrictToolModel):
    temperature: float | None = None
    top_p: float | None = None


class ManifestJob(StrictToolModel):
    job_id: JobID = Field(min_length=1)
    gpus: list[GpuPresetKey] = Field(default_factory=list)
    model: str = Field(min_length=1)
    served_checkpoint: str | None = None
    hf_revision: str = Field(min_length=1)
    quantization: str | None = None
    serve_args: list[str] = Field(default_factory=list)
    vendor: str | None = None
    license: str | None = None
    model_url: str | None = None
    image_override: str | None = None
    notes: str | None = None


class SelfHostedManifest(StrictToolModel):
    dataset: str | None = None
    prompts: list[str] = Field(default_factory=list)
    prompt_max_tokens: dict[str, int] = Field(default_factory=dict)
    sampling: ManifestSampling = Field(default_factory=ManifestSampling)
    warmup_calls: int = Field(default=0, ge=0)
    default_image: str = Field(min_length=1)
    gpu_presets: dict[GpuPresetKey, GpuPresetConfig] = Field(default_factory=dict)
    jobs: list[ManifestJob] = Field(default_factory=list)


class InstanceRecord(StrictToolModel):
    provider: str = Field(min_length=1)
    instance_id: int
    gpu_preset: GpuPresetKey
    gpu_label: str
    offer_id: int
    image: str
    disk_gb: int
    ssh_host: str
    ssh_port: int
    ssh_user: str
    hourly_rate_usd: float
    label: str
    created_at: datetime
    destroyed_at: datetime | None = None

    @field_serializer(CREATED_AT_FIELD, DESTROYED_AT_FIELD)
    def serialize_timestamps(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return value.isoformat()
