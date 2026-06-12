"""Paths and filename constants used by SenseBench tooling."""

from pathlib import Path

PACKAGE_DIR: Path = Path(__file__).resolve().parent
PROMPT_JSON_SUFFIX: str = ".json"
PROMPT_JSON_GLOB: str = f"*{PROMPT_JSON_SUFFIX}"
PROMPT_REGISTRY_DIR: Path = PACKAGE_DIR / "prompts" / "registered"
# Workflow paths resolve against the current working directory.
LOCAL_RUNS_DIR: Path = Path("runs")
SUBMITTED_RESULTS_DIR: Path = Path("results")
LEADERBOARD_JSON_PATH: Path = Path("leaderboard.json")
RUN_METADATA_FILENAME: str = "run.json"
PREDICTIONS_FILENAME: str = "predictions.jsonl"
CALLS_FILENAME: str = "calls.jsonl.gz"
DEFAULT_LEXEN_RELEASE_ID: str = "lexen-v0.1.0"
