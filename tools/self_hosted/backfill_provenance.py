"""Backfill provenance metadata into self-hosted run artifacts.

Enriches already-produced ``run.json`` files with the resolved checkpoint
revision, container image, and exact serve command, reconstructed from the
job manifest. This lets runs that were executed before provenance capture
satisfy verification without re-running them. The model outputs are never
touched, so accuracy/replay verification is unaffected.

Usage:
    uv run python tools/self_hosted/backfill_provenance.py \
        --runs-dir runs --manifest tools/self_hosted/manifest.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sensebench.paths import RUN_METADATA_FILENAME, SELF_HOSTED_MANIFEST_PATH
from sensebench.runs.models import RunMetadata, SelfHostedLlmReference
from tools.self_hosted.models import ManifestJob, SelfHostedManifest

RUN_ID_PREFIX: str = "vllm-"
HF_REVISION_FIELD: str = "hf_revision"
SERVE_COMMAND_FIELD: str = "serve_command"
CONTAINER_IMAGE_FIELD: str = "container_image"
MODEL_FIELD: str = "model"
GIT_COMMIT_FIELD: str = "git_commit"
FIXED_SERVE_FLAGS: tuple[str, ...] = (
    "--port",
    "8000",
    "--max-model-len",
    "8192",
    "--gpu-memory-utilization",
    "0.90",
)


def _manifest_model(*, manifest: SelfHostedManifest | dict[str, object]) -> SelfHostedManifest:
    if isinstance(manifest, SelfHostedManifest):
        return manifest
    return SelfHostedManifest.model_validate(manifest)


def _match_job(*, run_id: str, jobs: list[ManifestJob]) -> ManifestJob | None:
    matches: list[ManifestJob] = [
        job for job in jobs if run_id.startswith(f"{RUN_ID_PREFIX}{job.job_id}-")
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _serve_command(*, job: ManifestJob) -> str:
    checkpoint = (
        job.served_checkpoint
        if job.served_checkpoint is not None and len(job.served_checkpoint) > 0
        else job.model
    )
    serve_args: list[str] = list(job.serve_args)
    parts = ["vllm", "serve", checkpoint, *FIXED_SERVE_FLAGS, *serve_args]
    return " ".join(str(part) for part in parts)


def _container_image(*, job: ManifestJob, manifest: SelfHostedManifest) -> str:
    if job.image_override is not None and len(job.image_override) > 0:
        return job.image_override
    return manifest.default_image


def _missing_text(*, value: str | None) -> bool:
    return value is None or len(value.strip()) == 0


def backfill_run(
    *,
    run_dir: Path,
    manifest: SelfHostedManifest | dict[str, object],
    git_commit: str | None = None,
) -> str:
    manifest_model = _manifest_model(manifest=manifest)
    metadata_path = run_dir / RUN_METADATA_FILENAME
    metadata = RunMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    model = metadata.model
    if not isinstance(model, SelfHostedLlmReference):
        return "skip (cloud run)"
    job = _match_job(run_id=metadata.run_id, jobs=manifest_model.jobs)
    if job is None:
        return "skip (no manifest job matched run_id)"
    model_updates: dict[str, object] = {}
    if _missing_text(value=model.hf_revision):
        model_updates[HF_REVISION_FIELD] = job.hf_revision
    if _missing_text(value=model.serve_command):
        model_updates[SERVE_COMMAND_FIELD] = _serve_command(job=job)
    if _missing_text(value=model.container_image):
        model_updates[CONTAINER_IMAGE_FIELD] = _container_image(
            job=job,
            manifest=manifest_model,
        )

    updated_model = model.model_copy(update=model_updates) if len(model_updates) > 0 else model
    metadata_updates: dict[str, object] = {}
    if len(model_updates) > 0:
        metadata_updates[MODEL_FIELD] = updated_model
    # The on-box runner invokes sensebench outside the repo dir, so git_commit is
    # recorded as null; backfill it with the released sensebench commit.
    if git_commit is not None and _missing_text(value=metadata.git_commit):
        metadata_updates[GIT_COMMIT_FIELD] = git_commit
    if len(metadata_updates) == 0:
        return "ok (already complete)"
    updated_metadata = metadata.model_copy(update=metadata_updates)
    metadata_path.write_text(
        updated_metadata.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    assert updated_model.hf_revision is not None, "backfilled model has a revision"
    return f"backfilled (rev {updated_model.hf_revision[:12]})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--manifest", default=str(SELF_HOSTED_MANIFEST_PATH))
    parser.add_argument(
        "--git-commit",
        default=None,
        help="Released sensebench commit to record when the run captured none.",
    )
    args = parser.parse_args(argv)
    manifest = SelfHostedManifest.model_validate_json(
        Path(args.manifest).read_text(encoding="utf-8"),
    )
    runs_dir = Path(args.runs_dir)
    run_dirs: list[Path] = sorted(
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and (path / RUN_METADATA_FILENAME).exists()
    )
    for run_dir in run_dirs:
        result = backfill_run(run_dir=run_dir, manifest=manifest, git_commit=args.git_commit)
        print(f"{run_dir.name}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
