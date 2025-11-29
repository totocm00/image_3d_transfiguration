#!/usr/bin/env python3
import argparse
import os
import sys

# src 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from image_3d_transfiguration.config_loader import load_config
from image_3d_transfiguration.logging_utils import setup_logging
from image_3d_transfiguration.pipeline import run_full_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="image_3d_transfiguration v2 - 이미지 → 3D 파이프라인 실행"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="설정 파일 경로",
    )
    parser.add_argument(
        "--image_name",
        type=str,
        required=True,
        help="paths.image_dir 기준의 이미지 파일 이름 (예: robot.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.system.get("log_level", "INFO"))

    results = run_full_pipeline(cfg, image_name=args.image_name)

    print("=== image_3d_transfiguration v2 결과 ===")
    for k, v in results.items():
        print(f"{k:10s}: {v}")


if __name__ == "__main__":
    main()