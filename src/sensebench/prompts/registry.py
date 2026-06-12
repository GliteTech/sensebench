"""Load registered SenseBench prompt definitions."""

from pathlib import Path

from sensebench.paths import PROMPT_JSON_GLOB, PROMPT_REGISTRY_DIR
from sensebench.prompts.models import PromptDefinition


def registered_prompt_paths() -> list[Path]:
    return sorted(PROMPT_REGISTRY_DIR.glob(PROMPT_JSON_GLOB))


def load_prompt_definition(*, path: Path) -> PromptDefinition:
    raw_json: str = path.read_text(encoding="utf-8")
    return PromptDefinition.model_validate_json(json_data=raw_json)
