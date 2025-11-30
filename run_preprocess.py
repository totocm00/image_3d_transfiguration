import sys
from pathlib import Path

# src/를 PYTHONPATH에 추가
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image_3d_transfiguration.preprocess.yolo_sam_pipeline import main

if __name__ == "__main__":
    main()