"""Load the bundled OpenPI trainer without legacy PiPER paths."""
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = Path(os.environ.get("OPENPI_ROOT", str(ROOT / "openpi")))

def load_trainer():
    path = OPENPI_ROOT / "custom_scripts/train.py"
    spec = importlib.util.spec_from_file_location("shared_openpi_trainer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

