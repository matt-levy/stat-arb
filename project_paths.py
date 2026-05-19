import os
import shutil
from pathlib import Path

CODE_ROOT_DIR = Path(__file__).resolve().parent


def _default_runtime_root() -> Path:
    """Use the repo locally and Lambda's writable scratch space in AWS."""
    if os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        return Path("/tmp/stat-arb")
    return CODE_ROOT_DIR


ROOT_DIR = Path(os.getenv("STAT_ARB_RUNTIME_DIR", _default_runtime_root())).resolve()
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = ROOT_DIR / "reports"
LOGS_DIR = ROOT_DIR / "logs"
PLOTS_DIR = REPORTS_DIR / "pair_plots"


def ensure_project_directories() -> None:
    """Create standard output directories for generated artifacts."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _seed_runtime_data()


def _seed_runtime_data() -> None:
    """Copy bundled data files into the writable runtime directory when needed."""
    source_data_dir = CODE_ROOT_DIR / "data"
    if ROOT_DIR == CODE_ROOT_DIR or not source_data_dir.exists():
        return

    for source_path in source_data_dir.iterdir():
        if not source_path.is_file():
            continue
        target_path = DATA_DIR / source_path.name
        if not target_path.exists():
            shutil.copy2(source_path, target_path)
