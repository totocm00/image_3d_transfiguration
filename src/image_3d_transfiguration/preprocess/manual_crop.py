import cv2
import numpy as np
from pathlib import Path
import argparse


def interactive_grabcut_crop(
    img_path: str,
    out_crop_path: str,
    out_masked_path: str,
    iter_count: int = 5,
):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {img_path}")

    clone = img.copy()
    h, w = img.shape[:2]

    print("[INFO] 마우스로 로봇을 포함하는 영역을 드래그한 뒤, Enter 또는 Space를 누르세요.")
    print("[INFO] ROI 선택을 취소하려면 ESC를 누르면 됩니다.")

    # ROI 선택 (x, y, w, h)
    r = cv2.selectROI("Select Robot ROI", clone, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Robot ROI")

    x, y, rw, rh = r
    if rw == 0 or rh == 0:
        print("[WARN] ROI가 선택되지 않았습니다. 원본 이미지를 그대로 저장합니다.")
        Path(out_crop_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_masked_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_crop_path, img)
        cv2.imwrite(out_masked_path, img)
        return

    # GrabCut 초기화용 마스크
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (x, y, rw, rh)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)

    # 0,2 → 배경 / 1,3 → 전경
    grabcut_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype("uint8")

    # 전체 이미지에서 배경을 날린 버전
    masked_img = cv2.bitwise_and(img, img, mask=grabcut_mask)

    # ROI 영역만 크롭한 버전 (조금 margin 추가 가능)
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + rw, w)
    y1 = min(y + rh, h)
    cropped = masked_img[y0:y1, x0:x1].copy()

    Path(out_crop_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_masked_path).parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(out_crop_path, cropped)
    cv2.imwrite(out_masked_path, masked_img)

    print(f"[INFO] Cropped (ROI+GrabCut) → {out_crop_path}")
    print(f"[INFO] Full masked image     → {out_masked_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="assets/images/robot.jpg",
        help="원본 로봇 이미지 경로",
    )
    parser.add_argument(
        "--output_crop",
        type=str,
        default="assets/images/robot_manual_crop.png",
        help="ROI+GrabCut로 크롭된 이미지 저장 경로",
    )
    parser.add_argument(
        "--output_masked",
        type=str,
        default="assets/images/robot_manual_masked.png",
        help="배경이 제거된 전체 이미지 저장 경로",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=5,
        help="GrabCut 반복 횟수 (기본 5)",
    )
    args = parser.parse_args()

    interactive_grabcut_crop(
        img_path=args.input,
        out_crop_path=args.output_crop,
        out_masked_path=args.output_masked,
        iter_count=args.iter,
    )


if __name__ == "__main__":
    main()
