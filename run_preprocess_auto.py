import sys
from pathlib import Path

'''
python run_preprocess_auto.py \
  --input assets/images/cat.png \
  --output assets/images/cat_auto.png \
  --output_masked assets/images/cat_auto_masked.png \
  --output_box_vis assets/images/cat_auto_boxes.png \
  --mode single

# Replace cat with your filename (and extension)
# cat을 본인의 파일 이름과 확장자로 바꿔서 사용하세요
'''


# src/를 PYTHONPATH에 추가
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image_3d_transfiguration.preprocess.yolo_sam_pipeline import main

if __name__ == "__main__":
    main()