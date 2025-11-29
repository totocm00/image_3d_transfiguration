import argparse
import os

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

    cfg = load_config(args.config)

    image_path = os.path.join(cfg.paths.input_image_dir, args.image_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"입력 이미지가 없습니다: {image_path}")

    result = run_image_3d(image_path=image_path, cfg=cfg)

    print("=== Image 3D Transfiguration 결과 ===")
    print(f"depth:       {result['depth_path']}")
    print(f"point cloud: {result['point_cloud_path']}")


if __name__ == "__main__":
    main()