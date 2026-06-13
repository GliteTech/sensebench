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
import json
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST_PATH: Path = Path("tools/self_hosted/manifest.json")
RUN_METADATA_FILENAME: str = "run.json"
RUN_ID_PREFIX: str = "vllm-"
SELF_HOSTED_KIND: str = "self_hosted_llm"
FIXED_SERVE_FLAGS: tuple[str, ...] = (
    "--port",
    "8000",
    "--max-model-len",
    "8192",
    "--gpu-memory-utilization",
    "0.90",
)


def _load_json(*, path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        loaded: Any = json.load(handle)
    assert isinstance(loaded, dict), f"{path} is a JSON object"
    return loaded


def _match_job(*, run_id: str, jobs: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [job for job in jobs if run_id.startswith(f"{RUN_ID_PREFIX}{job['job_id']}-")]
    if len(matches) != 1:
        return None
    return matches[0]


def _serve_command(*, job: dict[str, Any]) -> str:
    checkpoint = job.get("served_checkpoint") or job["model"]
    serve_args = job.get("serve_args") or []
    parts = ["vllm", "serve", checkpoint, *FIXED_SERVE_FLAGS, *serve_args]
    return " ".join(str(part) for part in parts)


def _container_image(*, job: dict[str, Any], manifest: dict[str, Any]) -> str:
    override = job.get("image_override")
    if isinstance(override, str) and len(override) > 0:
        return override
    return str(manifest["default_image"])


def backfill_run(*, run_dir: Path, manifest: dict[str, Any]) -> str:
    metadata_path = run_dir / RUN_METADATA_FILENAME
    metadata = _load_json(path=metadata_path)
    model = metadata.get("model", {})
    if model.get("kind") != SELF_HOSTED_KIND:
        return "skip (cloud run)"
    job = _match_job(run_id=metadata["run_id"], jobs=manifest["jobs"])
    if job is None:
        return "skip (no manifest job matched run_id)"
    updated = False
    if not model.get("hf_revision"):
        model["hf_revision"] = job["hf_revision"]
        updated = True
    if not model.get("serve_command"):
        model["serve_command"] = _serve_command(job=job)
        updated = True
    if not model.get("container_image"):
        model["container_image"] = _container_image(job=job, manifest=manifest)
        updated = True
    if not updated:
        return "ok (already complete)"
    metadata["model"] = model
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return f"backfilled (rev {model['hf_revision'][:12]})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    args = parser.parse_args(argv)
    manifest = _load_json(path=Path(args.manifest))
    runs_dir = Path(args.runs_dir)
    run_dirs = sorted(
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and (path / RUN_METADATA_FILENAME).exists()
    )
    for run_dir in run_dirs:
        result = backfill_run(run_dir=run_dir, manifest=manifest)
        print(f"{run_dir.name}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
