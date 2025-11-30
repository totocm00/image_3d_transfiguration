import cv2
import os
import argparse
'''
python run_preprocess_manual.py \
  --input assets/images/cat.png \
  --output assets/images/cat_manual.png \
  --output_masked assets/images/cat_manual_masked.png \
  --output_box_vis assets/images/cat_manual_boxes.png

# Replace cat with your filename (and extension)
# cat을 본인의 파일 이름과 확장자로 바꿔서 사용하세요
'''



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--output_masked", type=str, default=None)
    p.add_argument("--output_box_vis", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print("[ERROR] Failed to load image:", args.input)
        return

    r = cv2.selectROI("manual ROI", img, False, False)
    cv2.destroyWindow("manual ROI")

    x, y, w, h = r
    roi = img[y:y+h, x:x+w]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, roi)

    if args.output_masked:
        masked = img.copy()
        mask = cv2.rectangle(masked, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.imwrite(args.output_masked, masked)

    if args.output_box_vis:
        vis = img.copy()
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 3)
        cv2.imwrite(args.output_box_vis, vis)

    print("[SUCCESS] Manual ROI saved to:", args.output)

if __name__ == "__main__":
    main()