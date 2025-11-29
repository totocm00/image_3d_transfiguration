#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import open3d as o3d

from image_3d_transfiguration.config_loader import load_config
from image_3d_transfiguration.logging_utils import setup_logging
from image_3d_transfiguration.mesh import build_mesh_from_pointcloud, export_mesh, project_texture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="기존 pointcloud(.ply)로부터 mesh만 별도 생성"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="설정 파일 경로",
    )
    parser.add_argument(
        "--pc_path",
        type=str,
        required=True,
        help="입력 pointcloud .ply 파일 경로",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="",
        help="텍스처용 원본 이미지 경로 (옵션)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="mesh 출력 경로 (지정 안 하면 config 기반 자동 생성)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.system.get("log_level", "INFO"))

    mesh_cfg = cfg.mesh
    texture_cfg = cfg.texture
    fmt = mesh_cfg.get("export_format", "ply").lower()

    pcd = o3d.io.read_point_cloud(args.pc_path)
    mesh = build_mesh_from_pointcloud(
        pcd,
        method=mesh_cfg.get("method", "poisson"),
        smoothing_iterations=int(mesh_cfg.get("smoothing_iterations", 5)),
        target_reduction=float(mesh_cfg.get("target_reduction", 0.5)),
    )

    if texture_cfg.get("enabled", True) and args.image_path:
        import cv2

        image_bgr = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
        mesh = project_texture(
            mesh,
            image_bgr=image_bgr,
            scale=float(texture_cfg.get("scale", 1.0)),
        )

    if args.output_path:
        out_path = args.output_path
    else:
        base_name = os.path.splitext(os.path.basename(args.pc_path))[0]
        mesh_dir = os.path.join(
            cfg.paths.get("output_root", "assets/outputs"),
            cfg.paths.get("mesh_dir", "mesh"),
        )
        os.makedirs(mesh_dir, exist_ok=True)
        out_path = os.path.join(mesh_dir, f"{base_name}_mesh.{fmt}")

    out_path = export_mesh(mesh, out_path, fmt=fmt)
    print(f"Mesh 저장: {out_path}")


if __name__ == "__main__":
    main()