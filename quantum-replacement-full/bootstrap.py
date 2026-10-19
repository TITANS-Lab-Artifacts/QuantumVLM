"""Locate the existing unified parent project; do not modify its files."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
required = ['model.py', 'experiment_config.py', 'experiment_train.py', 'resource_monitor.py']
if not all((PARENT / name).is_file() for name in required):
    raise RuntimeError('Place quantum-replacement-full directly inside the delivered vlm_robustness_unified parent project.')
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))
