import sys
import os
import argparse

# src/ 디렉터리를 Python 경로에 추가해서
# image_3d_transfiguration 패키지를 import 할 수 있게 만든다.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from image_3d_transfiguration.config_loader import load_config
from image_3d_transfiguration.pipeline import run_image_3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_name",
        type=str,
        required=True,
        help="assets/images 안에 있는 파일 이름 (예: robot.png)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="설정 파일 경로",
    )
    args = parser.parse_args()

    # 설정 파일 로드
    cfg = load_config(args.config)

    # 입력 이미지 절대 경로 만들기
    image_path = os.path.join(cfg.paths.input_image_dir, args.image_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"입력 이미지가 없습니다: {image_path}")

    # 2D → 3D 변환 실행
    result = run_image_3d(image_path=image_path, cfg=cfg)

    # 결과 경로 출력
    print("=== Image 3D Transfiguration 결과 ===")
    print(f"depth:       {result['depth_path']}")
    print(f"point cloud: {result['point_cloud_path']}")


if __name__ == "__main__":
    main()