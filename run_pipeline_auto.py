import sys
from pathlib import Path

'''
python run_pipeline_auto.py \
  --input assets/images/robot.jpg

# Replace robot with your filename (and extension)
# robot을 본인의 파일 이름과 확장자로 바꿔서 사용하세요
'''

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from image_3d_transfiguration.pipeline.full_pipeline import run_full_pipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_pipeline(
        input_path=args.input,
        mode="auto",
        use_sam=True,
    )