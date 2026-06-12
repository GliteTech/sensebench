"""Repository paths used by SenseBench tooling."""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
PROMPT_JSON_SUFFIX: str = ".json"
PROMPT_JSON_GLOB: str = f"*{PROMPT_JSON_SUFFIX}"
PROMPT_REGISTRY_DIR: Path = REPO_ROOT / "src" / "sensebench" / "prompts" / "registered"
LOCAL_RUNS_DIR: Path = REPO_ROOT / "runs"
SUBMITTED_RESULTS_DIR: Path = REPO_ROOT / "results"
LEADERBOARD_JSON_PATH: Path = REPO_ROOT / "leaderboard.json"
