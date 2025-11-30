import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse

def yolo_crop(img_path, out_path, conf=0.25):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    model = YOLO("yolov8n.pt")  # COCO 모델
    results = model.predict(img, conf=conf, verbose=False)

    if not results or len(results[0].boxes) == 0:
        print("[WARN] YOLO detect 실패. 원본 이미지를 그대로 저장합니다.")
        cv2.imwrite(out_path, img)
        return out_path

    # 가장 큰 bbox 선택
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    best_idx = int(np.argmax(areas))
    x1, y1, x2, y2 = boxes[best_idx]

    # 약간 margin
    h, w = img.shape[:2]
    margin = int(max(w, h) * 0.03)
    x1 = max(x1 - margin, 0)
    y1 = max(y1 - margin, 0)
    x2 = min(x2 + margin, w)
    y2 = min(y2 + margin, h)

    cropped = img[y1:y2, x1:x2]
    cv2.imwrite(out_path, cropped)
    print(f"[INFO] YOLO Crop saved → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="assets/images/robot.jpg")
    parser.add_argument("--output", type=str, default="assets/images/robot_yolo_crop.png")
    args = parser.parse_args()

    yolo_crop(args.input, args.output)


if __name__ == "__main__":
    main()
