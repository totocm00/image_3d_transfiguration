import argparse

from image_3d_transfiguration.pipeline import run_3d_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="image_3d_transfiguration: sam3d-style 3D reconstruction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="YAML config path",
    )
    parser.add_argument(
        "--image_name",
        type=str,
        required=True,
        help="입력 이미지 파일 이름 (input_root 기준)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='"auto" | "cpu" | "cuda"',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_3d_pipeline(
        cfg_path=args.config,
        image_name=args.image_name,
        device=args.device,
    )


if __name__ == "__main__":
    main()