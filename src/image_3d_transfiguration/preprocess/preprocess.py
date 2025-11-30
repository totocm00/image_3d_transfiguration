import cv2
import numpy as np
from pathlib import Path
import argparse


def auto_crop_main_object(img: np.ndarray, margin_ratio: float = 0.1) -> np.ndarray:
    """
    이미지에서 가장 큰 물체를 찾아 약간의 여유(margin)를 둔 박스로 크롭.
    """
    h, w = img.shape[:2]

    # 그레이스케일 + 블러
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 엣지/이진화 (배경 대비가 어느 정도 있다는 가정)
    # threshold가 더 잘 먹으면 아래 threshold만 써도 됨
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    # 노이즈 제거용 모폴로지
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 컨투어 검색
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # 실패 시 원본 그대로 반환
        return img

    # 가장 큰 컨투어 선택
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)

    # 여유 margin 추가
    margin = int(max(bw, bh) * margin_ratio)
    x0 = max(x - margin, 0)
    y0 = max(y - margin, 0)
    x1 = min(x + bw + margin, w)
    y1 = min(y + bh + margin, h)

    cropped = img[y0:y1, x0:x1].copy()
    return cropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="assets/images/robot.jpg",
        help="원본 이미지 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets/images/robot_cropped.png",
        help="크롭된 출력 이미지 경로",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    img = cv2.imread(str(input_path))
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {input_path}")

    cropped = auto_crop_main_object(img, margin_ratio=0.12)

    # 출력 디렉토리 없으면 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), cropped)
    print(f"[INFO] Cropped image saved to: {output_path}")


if __name__ == "__main__":
    main()
