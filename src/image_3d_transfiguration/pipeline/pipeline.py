# src/image_3d_transfiguration/pipeline/pipeline.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import open3d as o3d

from ..config_loader import load_config
from ..depth.depth_anything_backend import DepthAnythingBackend
from ..depth.zoe_depth_backend import ZoeDepthBackend
from ..depth.postprocess import DepthPostprocessConfig, postprocess_depth
from ..logging_utils import get_logger
from ..mesh.texture import project_texture
from ..pointcloud.generator import (
    CameraConfig,
    depth_to_pointcloud,
    make_default_camera,
)


logger = get_logger(__name__)


@dataclass
class DepthConfig:
    backend_primary: str = "depth_anything"
    backend_secondary: str = ""
    fusion_alpha: float = 0.7  # primary 가중치


def _build_depth_backends(
    depth_cfg: DepthConfig,
    device: str = "auto",
):
    primary = None
    secondary = None

    if depth_cfg.backend_primary == "depth_anything":
        primary = DepthAnythingBackend(device=device)
    else:
        raise ValueError(f"Unknown primary depth backend: {depth_cfg.backend_primary}")

    if depth_cfg.backend_secondary == "zoe_depth":
        try:
            secondary = ZoeDepthBackend(device=device)
        except Exception as e:
            logger.warning(
                "ZoeDepth backend 생성 실패, primary 만 사용합니다: %s", e
            )
            secondary = None

    return primary, secondary


def _infer_depth(
    image_bgr: np.ndarray,
    depth_cfg: DepthConfig,
    device: str = "auto",
) -> np.ndarray:
    primary, secondary = _build_depth_backends(depth_cfg, device=device)

    depth_primary = primary.predict(image_bgr)
    primary.close()

    if secondary is None:
        depth = depth_primary
    else:
        depth_secondary = secondary.predict(image_bgr)
        secondary.close()
        a = float(depth_cfg.fusion_alpha)
        depth = a * depth_primary + (1.0 - a) * depth_secondary

    return depth


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def run_3d_pipeline(
    cfg_path: str,
    image_name: str,
    device: str = "auto",
) -> None:
    """
    sam3d 스타일의 전체 파이프라인 (단일 뷰, monocular depth 기반).

    1) config 로드
    2) 이미지 로드
    3) depth 추론 + 후처리
    4) point cloud 생성
    5) mesh 생성 + smoothing
    6) texture projection
    7) 결과 저장
    """
    cfg = load_config(cfg_path)

    paths_cfg = cfg.get("paths", {})
    input_root = paths_cfg.get("input_root", "assets/images")
    output_root = paths_cfg.get("output_root", "assets/outputs")

    img_path = os.path.join(input_root, image_name)
    if not os.path.isfile(img_path):
        raise FileNotFoundError(img_path)

    logger.info("Input image: %s", img_path)

    image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"이미지 로드 실패: {img_path}")

    h, w = image_bgr.shape[:2]

    # ---- 1) Depth 예측 + 후처리 ----
    depth_section = cfg.get("depth", {})
    depth_cfg = DepthConfig(
        backend_primary=depth_section.get("backend_primary", "depth_anything"),
        backend_secondary=depth_section.get("backend_secondary", ""),
        fusion_alpha=float(depth_section.get("fusion_alpha", 0.7)),
    )

    logger.info(
        "Depth backends: primary=%s, secondary=%s",
        depth_cfg.backend_primary,
        depth_cfg.backend_secondary or "None",
    )

    depth_raw = _infer_depth(image_bgr, depth_cfg, device=device)

    pp_cfg = DepthPostprocessConfig()
    depth_norm = postprocess_depth(depth_raw, cfg=pp_cfg)

    # 저장 (디버그용)
    depth_out_dir = os.path.join(output_root, "depth")
    os.makedirs(depth_out_dir, exist_ok=True)
    depth_vis_path = os.path.join(
        depth_out_dir, f"{os.path.splitext(image_name)[0]}_depth.png"
    )
    depth_vis = (depth_norm * 255.0).clip(0, 255).astype("uint8")
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
    cv2.imwrite(depth_vis_path, depth_vis)
    logger.info("Depth visualization saved: %s", depth_vis_path)

    # ---- 2) Point cloud 생성 ----
    cam_cfg = make_default_camera(h, w, z_scale=1.0)
    pc = depth_to_pointcloud(depth_norm, image_bgr=image_bgr, cam=cam_cfg)

    pc_out_dir = os.path.join(output_root, "cloud")
    os.makedirs(pc_out_dir, exist_ok=True)
    pc_path = os.path.join(
        pc_out_dir, f"{os.path.splitext(image_name)[0]}_cloud.ply"
    )
    o3d.io.write_point_cloud(pc_path, pc)
    logger.info("Point cloud saved: %s", pc_path)

    # ---- 3) Mesh 생성 (Poisson) ----
    mesh_out_dir = os.path.join(output_root, "mesh")
    os.makedirs(mesh_out_dir, exist_ok=True)
    mesh_path = os.path.join(
        mesh_out_dir, f"{os.path.splitext(image_name)[0]}_mesh.ply"
    )

    # 포인트 노멀 계산
    pc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05, max_nn=30
        )
    )

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pc, depth=9
    )
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    mesh.compute_vertex_normals()

    # ---- 4) Texture projection ----
    mesh = project_texture(mesh, image_bgr=image_bgr, cam=cam_cfg)

    o3d.io.write_triangle_mesh(mesh_path, mesh)
    logger.info("Mesh saved: %s", mesh_path)