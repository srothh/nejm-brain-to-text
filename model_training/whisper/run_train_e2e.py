import sys
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]  # .../nejm-brain-to-text

sys.path[:0] = [
    str(ROOT),
    str(ROOT / "model_training"),
    str(ROOT / "model_training" / "whisper"),
]

runpy.run_path(str(ROOT / "model_training" / "whisper" / "train_e2e_model.py"), run_name="__main__")
