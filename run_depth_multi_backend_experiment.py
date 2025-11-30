import argparse
import glob
import os
from typing import List

import cv2

from depth.experiment_multi_backend import MultiBackendDepthExperiment


def _list_images(input_dir: str) -> List[str]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    files.sort()
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DepthAnything + ZoeDepth multi-backend 품질 실험 스크립트",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="입력 이미지 폴더 경로",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="결과(depth 시각화, JSON) 저장 폴더",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='사용 디바이스: "auto" | "cpu" | "cuda"',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = _list_images(args.input_dir)
    if not image_paths:
        print(f"[!] 입력 폴더에 이미지가 없습니다: {args.input_dir}")
        return

    print(f"[+] 입력 이미지 개수: {len(image_paths)}")
    print(f"[+] 결과 저장 폴더: {args.output_dir}")

    experiment = MultiBackendDepthExperiment(device=args.device)

    for idx, img_path in enumerate(image_paths):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{idx+1}/{len(image_paths)}] 처리 중: {base_name}")

        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"  [!] 이미지 로드 실패, 건너뜀: {img_path}")
            continue

        # 결과는 depth/experiment_multi_backend 안에서 JSON, PNG 네 장으로 저장됨
        experiment.run_on_image_and_save_json(
            image_bgr=image_bgr,
            out_dir=args.output_dir,
            base_name=base_name,
        )

    print("[+] 모든 이미지 처리 완료.")


if __name__ == "__main__":
    main()