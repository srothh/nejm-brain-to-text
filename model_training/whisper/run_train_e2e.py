import os, sys, runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MT   = ROOT / "model_training"
WSP  = MT / "whisper"

sys.path[:0] = [str(ROOT), str(MT), str(WSP)]

os.chdir(ROOT)

runpy.run_path(str(WSP / "train_e2e_model.py"), run_name="__main__")
