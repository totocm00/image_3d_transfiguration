from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

from .depth.sam3d_backend import Sam3DConfig, run_sam3d_pointmap
from .config_loader import load_config
from .logging_utils import get_logger
from .depth.depth_anything_backend import DepthAnythingBackend
from .depth.zoe_depth_backend import ZoeDepthBackend
from .depth.postprocess import DepthPostprocessConfig, postprocess_depth
from .pointcloud.generator import (
    CameraConfig,
    depth_to_pointcloud,
    make_default_camera,
)
from .mesh.texture import project_texture

logger = get_logger(__name__)


@dataclass
class DepthConfig:
    backend_primary: str = "depth_anything"
    backend_secondary: str = ""
    fusion_alpha: float = 0.7  # primary 가중치


def _build_depth_backends(
    depth_cfg: DepthConfig,
    device: str = "auto",
) -> Tuple[DepthAnythingBackend, Optional[ZoeDepthBackend]]:
    primary = None
    secondary = None

    # ----- primary -----
    if depth_cfg.backend_primary == "depth_anything":
        primary = DepthAnythingBackend(device=device)
    else:
        raise ValueError(f"Unknown primary depth backend: {depth_cfg.backend_primary}")

    # ----- secondary (옵션) -----
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
    """
    DepthAnything (+ ZoeDepth 옵션)으로 깊이 추론하고,
    필요시 두 결과를 가중합으로 fusion.
    """
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


def _save_depth_vis(
    depth_norm: np.ndarray,
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    depth_img = (depth_norm * 255.0).clip(0, 255).astype("uint8")
    depth_color = cv2.applyColorMap(depth_img, cv2.COLORMAP_MAGMA)
    cv2.imwrite(save_path, depth_color)


def _save_pointcloud(pc: o3d.geometry.PointCloud, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    o3d.io.write_point_cloud(save_path, pc)


def _save_mesh(mesh: o3d.geometry.TriangleMesh, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    o3d.io.write_triangle_mesh(save_path, mesh)


def run_3d_pipeline(
    cfg_path: str,
    image_name: str,
    device: str = "auto",
) -> None:
    """
    image_3d_transfiguration 메인 파이프라인.

    - backend_primary 가 "sam3d" 인 경우:
        sam-3d 백엔드를 서브프로세스로 호출하여
        곧바로 mesh .ply 를 생성하고 종료.

    - 그 외("depth_anything", "zoe_depth"):
        1) config 로드
        2) 입력 이미지 로드
        3) DepthAnything(+Zoe) 깊이 추론 + 후처리
        4) point cloud 생성
        5) Poisson mesh + smoothing
        6) 카메라 기반 텍스처 projection
        7) depth / cloud / mesh 저장
    """
    cfg = load_config(cfg_path)

    # --- paths / depth 섹션 파싱 (dict / dataclass 둘 다 지원) ---
    if isinstance(cfg, dict):
        paths_cfg = cfg.get("paths", {})
        depth_section = cfg.get("depth", {})
    else:
        paths_cfg = getattr(cfg, "paths", {})
        depth_section = getattr(cfg, "depth", {})

    # paths_cfg도 dict일 수도 있고, 객체일 수도 있음
    if isinstance(paths_cfg, dict):
        image_dir = paths_cfg.get(
            "image_dir",
            paths_cfg.get("input_root", "assets/images"),
        )
        output_root = paths_cfg.get("output_root", "assets/outputs")
        depth_dir_name = paths_cfg.get("depth_dir", "depth")
        pc_dir_name = paths_cfg.get("pointcloud_dir", "pointcloud")
        mesh_dir_name = paths_cfg.get("mesh_dir", "mesh")
    else:
        image_dir = getattr(paths_cfg, "image_dir", "assets/images")
        output_root = getattr(paths_cfg, "output_root", "assets/outputs")
        depth_dir_name = getattr(paths_cfg, "depth_dir", "depth")
        pc_dir_name = getattr(paths_cfg, "pointcloud_dir", "pointcloud")
        mesh_dir_name = getattr(paths_cfg, "mesh_dir", "mesh")

    img_path = os.path.join(image_dir, image_name)
    if not os.path.isfile(img_path):
        raise FileNotFoundError(img_path)

    logger.info("Input image: %s", img_path)

    # ----- Depth 섹션 읽기 -----
    if isinstance(depth_section, dict):
        backend_primary = depth_section.get("backend_primary", "depth_anything")
        backend_secondary = depth_section.get("backend_secondary", "")
        fusion_alpha = float(depth_section.get("fusion_alpha", 0.7))

        sam3d_root = depth_section.get(
            "sam3d_root",
            "/home/toto/parent/sam-3d-objects",
        )
        sam3d_python = depth_section.get("sam3d_python", "python")
    else:
        backend_primary = getattr(depth_section, "backend_primary", "depth_anything")
        backend_secondary = getattr(depth_section, "backend_secondary", "")
        fusion_alpha = float(getattr(depth_section, "fusion_alpha", 0.7))

        sam3d_root = getattr(depth_section, "sam3d_root", "/home/toto/parent/sam-3d-objects")
        sam3d_python = getattr(depth_section, "sam3d_python", "python")

    # =========================================================
    #  A) sam-3d 백엔드 모드: sam-3d에 직접 메쉬 생성을 맡긴다
    # =========================================================
    if backend_primary == "sam3d":
        sam_cfg = Sam3DConfig(
            sam3d_root=sam3d_root,
            python_cmd=sam3d_python,
        )

        mesh_out_dir = os.path.join(output_root, mesh_dir_name)
        os.makedirs(mesh_out_dir, exist_ok=True)
        mesh_path = os.path.join(
            mesh_out_dir,
            f"{os.path.splitext(image_name)[0]}_mesh.ply",
        )

        logger.info(
            "Running sam-3d backend: root=%s, python=%s",
            sam_cfg.sam3d_root,
            sam_cfg.python_cmd,
        )
        logger.info("sam-3d input image: %s", img_path)
        logger.info("sam-3d output mesh: %s", mesh_path)

        run_sam3d_pointmap(
            cfg=sam_cfg,
            input_path=os.path.abspath(img_path),
            output_ply=os.path.abspath(mesh_path),
        )

        logger.info("sam-3d mesh saved: %s", mesh_path)
        return

    # =========================================================
    #  B) 기존 DepthAnything / ZoeDepth 파이프라인
    # =========================================================
    image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"이미지 로드 실패: {img_path}")

    h, w = image_bgr.shape[:2]

    depth_cfg = DepthConfig(
        backend_primary=backend_primary,
        backend_secondary=backend_secondary,
        fusion_alpha=fusion_alpha,
    )

    logger.info(
        "Depth backends: primary=%s, secondary=%s, fusion_alpha=%.2f",
        depth_cfg.backend_primary,
        depth_cfg.backend_secondary or "None",
        depth_cfg.fusion_alpha,
    )

    # ----- 1) Raw depth 추론 -----
    depth_raw = _infer_depth(image_bgr, depth_cfg, device=device)

    # ----- 2) Depth 후처리 (sam3d 느낌 smoothing) -----
    pp_cfg = DepthPostprocessConfig()
    depth_norm = postprocess_depth(depth_raw, cfg=pp_cfg)  # 0~1

    # depth 시각화 저장
    depth_out_dir = os.path.join(output_root, depth_dir_name)
    depth_vis_path = os.path.join(
        depth_out_dir, f"{os.path.splitext(image_name)[0]}_depth.png"
    )
    _save_depth_vis(depth_norm, depth_vis_path)
    logger.info("Depth visualization saved: %s", depth_vis_path)

    # ----- 3) Point cloud 생성 -----
    cam_cfg: CameraConfig = make_default_camera(h, w, z_scale=1.0)
    pc = depth_to_pointcloud(
        depth_norm=depth_norm,
        image_bgr=image_bgr,
        cam=cam_cfg,
    )

    pc_out_dir = os.path.join(output_root, pc_dir_name)
    pc_path = os.path.join(
        pc_out_dir, f"{os.path.splitext(image_name)[0]}_cloud.ply"
    )
    _save_pointcloud(pc, pc_path)
    logger.info("Point cloud saved: %s", pc_path)

    # ----- 4) Mesh 생성 (Poisson + smoothing) -----
    mesh_out_dir = os.path.join(output_root, mesh_dir_name)
    mesh_path = os.path.join(
        mesh_out_dir, f"{os.path.splitext(image_name)[0]}_mesh.ply"
    )

    if len(pc.points) == 0:
        raise RuntimeError("Point cloud 가 비어 있습니다. depth 결과를 확인하세요.")

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

    # ----- 5) Texture projection -----
    mesh = project_texture(mesh, image_bgr=image_bgr, cam=cam_cfg)

    _save_mesh(mesh, mesh_path)
    logger.info("Mesh saved: %s", mesh_path)