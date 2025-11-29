#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import open3d as o3d

from image_3d_transfiguration.config_loader import load_config
from image_3d_transfiguration.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pointcloud 또는 mesh 파일을 Open3D로 시각화"
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
        default="",
        help="pointcloud .ply 파일 경로",
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        default="",
        help="mesh .ply/.obj 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.system.get("log_level", "INFO"))

    viewer_cfg = cfg.viewer
    bg = viewer_cfg.get("background_color", [0.0, 0.0, 0.0])
    point_size = float(viewer_cfg.get("point_size", 2.0))
    line_width = float(viewer_cfg.get("line_width", 1.0))

    geometries = []
    if args.pc_path:
        pcd = o3d.io.read_point_cloud(args.pc_path)
        geometries.append(pcd)
    if args.mesh_path:
        mesh = o3d.io.read_triangle_mesh(args.mesh_path)
        mesh.compute_vertex_normals()
        geometries.append(mesh)

    if not geometries:
        mode = viewer_cfg.get("default_mode", "mesh")
        raise SystemExit(
            f"pc_path 또는 mesh_path를 지정해야 합니다. (기본 모드: {mode})"
        )

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = bg
    opt.point_size = point_size
    opt.line_width = line_width

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()