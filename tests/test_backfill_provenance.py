from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from sensebench.prompts.models import SENSE_INDEX_FIELD
from sensebench.runner.writer import write_run_artifacts
from sensebench.runs.models import RunMetadata, SelfHostedLlmReference
from sensebench.verify.runs import RunValidationRule, verify_run_directory
from tests.run_fixtures import (
    SECOND_SENSE_KEY,
    fixture_machine,
    issue_rules,
    make_metadata,
    self_hosted_model,
    success_call,
    voted_prediction,
)

BACKFILL_SCRIPT_PATH: Path = Path("tools/self_hosted/backfill_provenance.py")
JOB_ID: str = "granite-4.1-8b-fp8"
GPU_PRESET: str = "h100"
RUN_ID: str = f"vllm-{JOB_ID}-{GPU_PRESET}-p001-lexen-v1-20260614"
CHECKPOINT: str = "ibm-granite/granite-4.1-8b-fp8"
PINNED_REVISION: str = "070021b3608433b6107a00733d561c9779b9937e"
DEFAULT_IMAGE: str = "vllm/vllm-openai:v0.22.1"
RELEASE_COMMIT: str = "947a5470000000000000000000000000deadbeef"
HF_REVISION_FIELD: str = "hf_revision"
GIT_COMMIT_FIELD: str = "git_commit"


def _load_backfill_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("backfill_provenance", BACKFILL_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None, "backfill script is importable"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest() -> dict[str, object]:
    return {
        "default_image": DEFAULT_IMAGE,
        "jobs": [
            {
                "job_id": JOB_ID,
                "model": CHECKPOINT,
                "served_checkpoint": CHECKPOINT,
                "hf_revision": PINNED_REVISION,
                "quantization": "fp8",
                "serve_args": ["--max-num-batched-tokens", "8192"],
                "image_override": None,
            }
        ],
    }


def _write_self_hosted_run(*, run_dir: Path) -> None:
    model = self_hosted_model(model_name=CHECKPOINT).model_copy(
        update={HF_REVISION_FIELD: None},
    )
    metadata = make_metadata(
        item_count=1,
        correct_count=1,
        accuracy=1.0,
        call_count=1,
        run_id=RUN_ID,
        model=model,
        machine=fixture_machine(),
    ).model_copy(update={GIT_COMMIT_FIELD: None})
    prediction = voted_prediction(
        chosen_index=2,
        gold_sense_keys=[SECOND_SENSE_KEY],
        is_correct=True,
    )
    write_run_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        predictions=[prediction],
        calls=[success_call(raw_output=json.dumps({SENSE_INDEX_FIELD: 2}))],
    )


def test_backfill_enriches_self_hosted_run(tmp_path: Path) -> None:
    module = _load_backfill_module()
    run_dir = tmp_path / RUN_ID
    _write_self_hosted_run(run_dir=run_dir)

    # Provenance rule fails before backfill.
    before = verify_run_directory(run_dir=run_dir)
    assert RunValidationRule.MODEL_PROVENANCE in issue_rules(report=before)

    result = module.backfill_run(
        run_dir=run_dir,
        manifest=_manifest(),
        git_commit=RELEASE_COMMIT,
    )
    assert "backfilled" in result

    metadata = RunMetadata.model_validate_json(
        (run_dir / "run.json").read_text(encoding="utf-8"),
    )
    assert isinstance(metadata.model, SelfHostedLlmReference)
    assert metadata.model.hf_revision == PINNED_REVISION
    assert metadata.model.container_image == DEFAULT_IMAGE
    assert metadata.git_commit == RELEASE_COMMIT
    assert metadata.model.serve_command is not None
    assert CHECKPOINT in metadata.model.serve_command
    assert "--max-num-batched-tokens 8192" in metadata.model.serve_command

    # Provenance rule now passes; the backfill is idempotent.
    after = verify_run_directory(run_dir=run_dir)
    assert RunValidationRule.MODEL_PROVENANCE not in issue_rules(report=after)
    assert "already complete" in module.backfill_run(run_dir=run_dir, manifest=_manifest())
