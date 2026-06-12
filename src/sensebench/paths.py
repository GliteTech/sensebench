"""Paths and filename constants used by SenseBench tooling."""

from pathlib import Path

PACKAGE_DIR: Path = Path(__file__).resolve().parent
PROMPT_JSON_SUFFIX: str = ".json"
PROMPT_JSON_GLOB: str = f"*{PROMPT_JSON_SUFFIX}"
PROMPT_REGISTRY_DIR: Path = PACKAGE_DIR / "prompts" / "registered"
BASELINE_PREDICTIONS_DIR: Path = PACKAGE_DIR / "leaderboard" / "baselines"
P001_PROMPT_FILENAME: str = "p001.json"
P001_PROMPT_PATH: Path = PROMPT_REGISTRY_DIR / P001_PROMPT_FILENAME
# Workflow paths resolve against the current working directory.
LOCAL_RUNS_DIR: Path = Path("runs")
SUBMITTED_RESULTS_DIR: Path = Path("results")
LEADERBOARD_JSON_PATH: Path = Path("leaderboard.json")
SITE_OUTPUT_DIR: Path = Path("_site")
SITE_DATA_DIRNAME: str = "data"
SITE_RUNS_DIRNAME: str = "runs"
SITE_ASSETS_DIRNAME: str = "assets"
INDEX_HTML_FILENAME: str = "index.html"
RUN_METADATA_FILENAME: str = "run.json"
PREDICTIONS_FILENAME: str = "predictions.jsonl"
CALLS_FILENAME: str = "calls.jsonl.gz"
DEFAULT_LEXEN_RELEASE_ID: str = "lexen-v0.1.0"
LEXEN_DATASET_ID: str = "lexen"
LEXEN_ITEMS_FILENAME: str = "items.jsonl"
