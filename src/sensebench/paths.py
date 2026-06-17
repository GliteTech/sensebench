"""Paths and filename constants used by SenseBench tooling."""

from pathlib import Path

PACKAGE_DIR: Path = Path(__file__).resolve().parent
PROMPT_JSON_SUFFIX: str = ".json"
PROMPT_JSON_GLOB: str = f"*{PROMPT_JSON_SUFFIX}"
PROMPT_REGISTRY_DIR: Path = PACKAGE_DIR / "prompts" / "registered"
BASELINE_PREDICTIONS_DIR: Path = PACKAGE_DIR / "leaderboard" / "baselines"
P001_PROMPT_FILENAME: str = "p001.json"
P001_PROMPT_PATH: Path = PROMPT_REGISTRY_DIR / P001_PROMPT_FILENAME
P002_PROMPT_FILENAME: str = "p002.json"
P002_PROMPT_PATH: Path = PROMPT_REGISTRY_DIR / P002_PROMPT_FILENAME
BEM_BASELINE_FILENAME: str = "bem.key.txt"
ESCHER_BASELINE_FILENAME: str = "escher.key.txt"
CONSEC_BASELINE_FILENAME: str = "consec.key.txt"
BEM_BASELINE_PATH: Path = BASELINE_PREDICTIONS_DIR / BEM_BASELINE_FILENAME
ESCHER_BASELINE_PATH: Path = BASELINE_PREDICTIONS_DIR / ESCHER_BASELINE_FILENAME
CONSEC_BASELINE_PATH: Path = BASELINE_PREDICTIONS_DIR / CONSEC_BASELINE_FILENAME
# Glite coarse-sense mapping vendored from the lexEN release (sources/glite-coarsening/).
GLITE_COARSENING_DIR: Path = PACKAGE_DIR / "datasets" / "glite"
GLITE_CONCEPT_MAP_FILENAME: str = "wordnet_sense_key_to_glite_concept.jsonl"
GLITE_ALIASES_FILENAME: str = "lexen_report_aliases.json"
GLITE_CONCEPT_MAP_PATH: Path = GLITE_COARSENING_DIR / GLITE_CONCEPT_MAP_FILENAME
GLITE_ALIASES_PATH: Path = GLITE_COARSENING_DIR / GLITE_ALIASES_FILENAME
# CSI coarse-sense mapping (Lacerra 2020) vendored from the lexEN CSI add-on (coarsenings/csi/).
CSI_COARSENING_DIR: Path = PACKAGE_DIR / "datasets" / "csi"
CSI_CONCEPT_MAP_FILENAME: str = "wordnet_sense_key_to_csi_concept.jsonl"
CSI_ALIASES_FILENAME: str = "csi_aliases.json"
CSI_CONCEPT_MAP_PATH: Path = CSI_COARSENING_DIR / CSI_CONCEPT_MAP_FILENAME
CSI_ALIASES_PATH: Path = CSI_COARSENING_DIR / CSI_ALIASES_FILENAME
# Workflow paths resolve against the current working directory.
LOCAL_RUNS_DIR: Path = Path("runs")
SUBMITTED_RESULTS_DIR: Path = Path("results")
LEADERBOARD_JSON_PATH: Path = Path("leaderboard.json")
SITE_OUTPUT_DIR: Path = Path("_site")
RUN_ARTIFACT_ROOT: Path = Path("artifacts") / "runs"
SITE_DATA_DIRNAME: str = "data"
SITE_RUNS_DIRNAME: str = "runs"
SITE_ASSETS_DIRNAME: str = "assets"
INDEX_HTML_FILENAME: str = "index.html"
SITEMAP_FILENAME: str = "sitemap.xml"
ROBOTS_FILENAME: str = "robots.txt"
NOT_FOUND_FILENAME: str = "404.html"
RUN_METADATA_FILENAME: str = "run.json"
PREDICTIONS_FILENAME: str = "predictions.jsonl"
CALLS_FILENAME: str = "calls.jsonl.gz"
DEFAULT_LEXEN_RELEASE_ID: str = "lexen-v1"
LEXEN_DATASET_ID: str = "lexen"
LEXEN_ITEMS_FILENAME: str = "items.jsonl"
DEFAULT_CACHE_DIRNAME: str = ".cache"
SENSEBENCH_CACHE_DIRNAME: str = "sensebench"
DATASETS_CACHE_DIRNAME: str = "datasets"
DATASET_FILENAME: str = LEXEN_ITEMS_FILENAME
DOWNLOAD_SUFFIX: str = ".download"
PROC_CPUINFO_PATH: Path = Path("/proc/cpuinfo")
PROC_MEMINFO_PATH: Path = Path("/proc/meminfo")
SELF_HOSTED_MANIFEST_PATH: Path = Path("tools/self_hosted/manifest.json")
SELF_HOSTED_BACKFILL_SCRIPT_PATH: Path = Path("tools/self_hosted/backfill_provenance.py")
WORK_ROOT: Path = Path("work")
INSTANCE_FILENAME: str = "instance.json"
DEFAULT_SSH_KEY_PATH: Path = Path.home() / ".ssh" / "id_ed25519"
TEST_DATA_DIR: Path = Path("tests") / "data"
SMOKE_ITEMS_PATH: Path = TEST_DATA_DIR / "smoke_items.jsonl"
