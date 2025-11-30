import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor


def ensure_dir(path: str | Path) -> None:
    path = Path(path)
    if path.suffix:  # file path
        path.parent.mkdir(parents=True, exist_ok=True)
    else:  # dir
        path.mkdir(parents=True, exist_ok=True)


def load_image_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {path}")
    return img


def interactive_select_roi(img_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    사용자가 직접 ROI(관심 영역)를 드래그해서 선택.
    GUI를 사용할 수 없는 환경(OpenCV highgui 에러)에서는
    전체 이미지를 ROI로 사용하도록 fallback.
    반환: (x, y, w, h)
    """
    h, w = img_bgr.shape[:2]

    try:
        clone = img_bgr.copy()
        print("[INFO] 먼저 관심 영역(ROI)을 직접 지정합니다.")
        print("[INFO] 로봇이 포함된 큰 영역을 마우스로 드래그한 뒤, Enter/Space를 누르세요. (ESC: 취소)")
        r = cv2.selectROI("Select ROI (전체 로봇 영역)", clone, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI (전체 로봇 영역)")

        x, y, rw, rh = r
        if rw == 0 or rh == 0:
            print("[WARN] ROI가 선택되지 않았습니다. 전체 이미지를 ROI로 사용합니다.")
            return 0, 0, w, h

        return int(x), int(y), int(rw), int(rh)

    except cv2.error as e:
        # 바로 너가 본 에러 케이스 처리
        print("[WARN] OpenCV GUI 창을 열 수 없습니다. (selectROI 실패)")
        print("[WARN] GUI 없는 환경이므로, 전체 이미지를 ROI로 사용합니다.")
        return 0, 0, w, h


def run_yolo_on_roi(
    img_bgr: np.ndarray,
    roi_xywh: tuple[int, int, int, int],
    yolo_model_path: str = "yolov8n.pt",
    conf: float = 0.2,
    min_area: int = 500,
) -> list[tuple[int, int, int, int]]:
    """
    ROI 영역 내에서만 YOLO 디텍션 수행.
    반환: 원본 이미지 기준 박스 리스트 [(x1, y1, x2, y2), ...]
    """
    x, y, w, h = roi_xywh
    roi_img = img_bgr[y : y + h, x : x + w]

    model = YOLO(yolo_model_path)
    results = model.predict(roi_img, conf=conf, verbose=False)
    if not results or len(results) == 0:
        return []

    result = results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy().astype(int)  # (N, 4)
    confs = boxes.conf.cpu().numpy()
    areas = []

    global_boxes: list[tuple[int, int, int, int]] = []
    for (x1, y1, x2, y2), cf in zip(xyxy, confs):
        gw1 = x1 + x
        gh1 = y1 + y
        gw2 = x2 + x
        gh2 = y2 + y
        area = max(0, gw2 - gw1) * max(0, gh2 - gh1)
        if area < min_area:
            continue
        areas.append(area)
        global_boxes.append((int(gw1), int(gh1), int(gw2), int(gh2)))

    return global_boxes


def pick_single_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """
    단일 객체 모드에서 사용할 대표 박스 선택 (가장 큰 면적 박스).
    """
    if not boxes:
        raise ValueError("boxes 리스트가 비어 있습니다.")
    areas = []
    for (x1, y1, x2, y2) in boxes:
        area = max(0, x2 - x1) * max(0, y2 - y1)
        areas.append(area)
    idx = int(np.argmax(areas))
    return boxes[idx]


def draw_roi_and_boxes(
    img_bgr: np.ndarray,
    roi_xywh: tuple[int, int, int, int],
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """
    ROI(파란색)와 YOLO 박스들(초록색)을 한 이미지에 그려줌.
    """
    x, y, w, h = roi_xywh
    vis = img_bgr.copy()
    # ROI (파란색)
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # YOLO 박스들 (초록색)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return vis


def load_sam(
    checkpoint_path: str,
    model_type: str = "vit_h",
    device: str | None = None,
) -> SamPredictor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def run_sam_with_box(
    predictor: SamPredictor,
    img_bgr: np.ndarray,
    box_xyxy: tuple[int, int, int, int],
    multimask_output: bool = False,
) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    x1, y1, x2, y2 = box_xyxy
    box = np.array([x1, y1, x2, y2])[None, :]  # (1, 4)

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=multimask_output,
    )

    if multimask_output:
        mask = masks[0]
    else:
        mask = masks
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]

    mask = (mask > 0.5).astype(np.uint8)  # (H, W) 0/1
    return mask


def postprocess_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return (mask > 0).astype(np.uint8)


def apply_mask_and_crop(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    pad_ratio: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    mask: (H, W) 0/1
    반환:
      masked_full: 전체 이미지에서 배경 제거된 버전
      cropped: 마스크 영역 bbox 기준으로 크롭한 버전
    """
    h, w = img_bgr.shape[:2]
    mask_3c = np.repeat(mask[:, :, None], 3, axis=2)
    masked_full = img_bgr * mask_3c

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return masked_full, img_bgr

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    pad = int(max(w, h) * pad_ratio)
    x0 = max(x_min - pad, 0)
    y0 = max(y_min - pad, 0)
    x1 = min(x_max + pad, w - 1)
    y1 = min(y_max + pad, h - 1)

    cropped = masked_full[y0 : y1 + 1, x0 : x1 + 1].copy()
    return masked_full, cropped


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
        default="assets/images/robot_yolo_sam.png",
        help="(단일 모드) 최종 전처리 결과 이미지 경로 또는 (다중 모드) 파일명 베이스",
    )
    parser.add_argument(
        "--output_masked",
        type=str,
        default="assets/images/robot_yolo_sam_masked.png",
        help="(단일 모드) 전체 배경제거 이미지 경로",
    )
    parser.add_argument(
        "--output_box_vis",
        type=str,
        default="assets/images/robot_yolo_sam_boxes.png",
        help="ROI + YOLO 박스를 그린 디버그용 이미지 경로",
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="yolov8n.pt",
        help="YOLO 모델 가중치 경로 또는 이름",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="assets/models/sam/sam_vit_h_4b8939.pth",
        help="SAM 체크포인트(.pth) 경로",
    )
    parser.add_argument(
        "--sam_type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM 모델 타입",
    )
    parser.add_argument(
        "--yolo_conf",
        type=float,
        default=0.2,
        help="YOLO confidence threshold",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=500,
        help="YOLO bbox 최소 면적 (픽셀^2)",
    )
    parser.add_argument(
        "--pad_ratio",
        type=float,
        default=0.05,
        help="크롭 시 여유 padding 비율",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "multi"],
        help="단일 객체 모드(single) / 다중 객체 모드(multi)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_masked_path = Path(args.output_masked)
    output_box_path = Path(args.output_box_vis)

    print(f"[INFO] 입력 이미지: {input_path}")
    img_bgr = load_image_bgr(input_path)

    # 0) 수동 ROI 선택
    roi = interactive_select_roi(img_bgr)
    if roi is None:
        print("[WARN] ROI를 선택하지 않아 원본 이미지를 그대로 저장합니다.")
        ensure_dir(output_path)
        cv2.imwrite(str(output_path), img_bgr)
        ensure_dir(output_masked_path)
        cv2.imwrite(str(output_masked_path), img_bgr)
        ensure_dir(output_box_path)
        cv2.imwrite(str(output_box_path), img_bgr)
        return

    x, y, w, h = roi
    print(f"[INFO] 선택된 ROI: ({x}, {y}, {w}, {h})")

    # 1) ROI 안에서만 YOLO 실행
    print("[INFO] ROI 내부에서 YOLO 디텍션 중...")
    boxes = run_yolo_on_roi(
        img_bgr,
        roi_xywh=roi,
        yolo_model_path=args.yolo_model,
        conf=args.yolo_conf,
        min_area=args.min_area,
    )

    # YOLO 결과 박스가 없으면, ROI 자체를 하나의 박스로 사용
    if not boxes:
        print("[WARN] ROI 내부에서 YOLO 박스를 찾지 못했습니다. ROI 전체를 단일 박스로 사용합니다.")
        boxes = [(x, y, x + w, y + h)]

    # ROI + YOLO 박스 시각화 이미지 저장
    box_vis = draw_roi_and_boxes(img_bgr, roi_xywh=roi, boxes=boxes)
    ensure_dir(output_box_path)
    cv2.imwrite(str(output_box_path), box_vis)
    print(f"[INFO] ROI + YOLO 박스 디버그 이미지 저장: {output_box_path}")

    # 모드에 따라 사용할 박스 리스트 결정
    if args.mode == "single":
        # 가장 큰 박스 하나만 선택
        target_boxes = [pick_single_box(boxes)]
        print(f"[INFO] 단일 객체 모드: 박스 1개만 사용합니다. → {target_boxes[0]}")
    else:
        # 다중 객체 모드: 모든 박스를 사용
        target_boxes = boxes
        print(f"[INFO] 다중 객체 모드: 총 {len(target_boxes)}개 박스를 사용합니다.")

    # 2) SAM 로드
    if not os.path.exists(args.sam_checkpoint):
        raise FileNotFoundError(
            f"SAM 체크포인트를 찾을 수 없습니다: {args.sam_checkpoint}\n"
            f"공식 SAM 모델을 다운로드 후 해당 경로에 두세요."
        )

    print("[INFO] SAM 로드 중...")
    predictor = load_sam(
        checkpoint_path=args.sam_checkpoint,
        model_type=args.sam_type,
        device=None,
    )

    # 3) 각 박스마다 SAM → 마스크 → 배경제거+크롭 → 저장
    if args.mode == "single":
        box = target_boxes[0]
        print(f"[INFO] SAM 마스크 생성 (단일 박스): {box}")
        raw_mask = run_sam_with_box(
            predictor,
            img_bgr=img_bgr,
            box_xyxy=box,
            multimask_output=False,
        )
        mask = postprocess_mask(raw_mask, kernel_size=5)

        masked_full, cropped = apply_mask_and_crop(
            img_bgr,
            mask=mask,
            pad_ratio=args.pad_ratio,
        )

        ensure_dir(output_masked_path)
        ensure_dir(output_path)
        cv2.imwrite(str(output_masked_path), masked_full)
        cv2.imwrite(str(output_path), cropped)

        print(f"[INFO] (단일) 전체 배경제거 이미지 저장: {output_masked_path}")
        print(f"[INFO] (단일) 크롭+배경제거 이미지 저장: {output_path}")
        print("[INFO] 이제 이 이미지로 3D 파이프라인을 실행할 수 있습니다.")
        print(
            f"예: python3 scripts/run_3d.py --config config/config.yaml "
            f"--image_name {output_path.name}"
        )

    else:
        # 다중 모드: output_path를 베이스로 번호 붙여 저장
        base = Path(output_path)
        stem = base.stem
        suffix = base.suffix or ".png"  # 확장자 없으면 .png

        masked_base = Path(output_masked_path)
        masked_stem = masked_base.stem
        masked_suffix = masked_base.suffix or ".png"

        for idx, box in enumerate(target_boxes, start=1):
            print(f"[INFO] SAM 마스크 생성 (다중 #{idx}): {box}")
            raw_mask = run_sam_with_box(
                predictor,
                img_bgr=img_bgr,
                box_xyxy=box,
                multimask_output=False,
            )
            mask = postprocess_mask(raw_mask, kernel_size=5)

            masked_full, cropped = apply_mask_and_crop(
                img_bgr,
                mask=mask,
                pad_ratio=args.pad_ratio,
            )

            out_crop = base.with_name(f"{stem}_obj{idx}{suffix}")
            out_mask = masked_base.with_name(f"{masked_stem}_obj{idx}{masked_suffix}")

            ensure_dir(out_crop)
            ensure_dir(out_mask)
            cv2.imwrite(str(out_mask), masked_full)
            cv2.imwrite(str(out_crop), cropped)

            print(f"[INFO] (다중 #{idx}) 전체 배경제거 이미지 저장: {out_mask}")
            print(f"[INFO] (다중 #{idx}) 크롭+배경제거 이미지 저장: {out_crop}")
            print(
                f" → 3D 실행 예: python3 scripts/run_3d.py --config config/config.yaml "
                f"--image_name {out_crop.name}"
            )

        print("[INFO] 다중 객체 모드 전처리가 완료되었습니다.")


if __name__ == "__main__":
    main()