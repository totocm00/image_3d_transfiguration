import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

from .config_loader import Config
from .logging_utils import get_logger
from .mask import create_mask
from .depth import (
    DepthAnythingBackend,
    ZoeDepthBackend,
    FakeDepthBackend,
    DepthBackend,
)
from .fusion import fuse_depth_maps
from .pointcloud import depth_to_pointcloud, postprocess_pointcloud
from .mesh import build_mesh_from_pointcloud, project_texture, export_mesh

logger = get_logger(__name__)


def _build_depth_backend(name: str, device: str) -> DepthBackend:
    name = name.lower()
    if name == "depth_anything":
        return DepthAnythingBackend(device=device)
    if name == "zoe_depth":
        return ZoeDepthBackend(device=device)
    if name == "fake":
        return FakeDepthBackend(device=device)
    raise ValueError(f"알 수 없는 depth backend: {name}")


def _resolve_device(device_cfg: str) -> str:
    import torch  # torch가 없으면 여기서 ImportError 발생할 수 있음

    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_cfg in ("cuda", "cpu"):
        return device_cfg
    return "cpu"


def run_full_pipeline(
    cfg: Config,
    image_name: str,
    external_mask: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """
    전체 파이프라인:
      이미지 → 마스크 → depth → fusion → pointcloud → postprocess → mesh → export
    결과 파일 경로들을 dict로 반환합니다.
    """
    # 경로 구성
    paths_cfg = cfg.paths
    out_cfg = cfg.output
    mesh_cfg = cfg.mesh
    texture_cfg = cfg.texture
    depth_cfg = cfg.depth
    fusion_cfg = cfg.fusion
    input_cfg = cfg.input

    image_dir = paths_cfg.get("image_dir", "assets/images")
    output_root = paths_cfg.get("output_root", "assets/outputs")
    depth_dir = os.path.join(output_root, paths_cfg.get("depth_dir", "depth"))
    pc_dir = os.path.join(output_root, paths_cfg.get("pointcloud_dir", "pointcloud"))
    mesh_dir = os.path.join(output_root, paths_cfg.get("mesh_dir", "mesh"))

    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pc_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)

    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    logger.info(f"입력 이미지: {image_path}")
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"이미지를 읽을 수 없습니다: {image_path}")

    h, w = image_bgr.shape[:2]

    # 마스크 생성
    use_mask = bool(input_cfg.get("use_mask", False))
    mask_mode = input_cfg.get("mask_mode", "full")
    if use_mask:
        mask = create_mask(image_bgr, mode=mask_mode, external_mask=external_mask)
    else:
        mask = np.ones((h, w), dtype=bool)

    # depth backend 준비
    device = _resolve_device(cfg.system.get("device", "auto"))
    logger.info(f"Depth device: {device}")

    primary_name = depth_cfg.get("backend_primary", "depth_anything")
    secondary_name = depth_cfg.get("backend_secondary", "")

    primary_backend = _build_depth_backend(primary_name, device=device)
    secondary_backend = None
    if secondary_name:
        secondary_backend = _build_depth_backend(secondary_name, device=device)

    # depth 예측
    logger.info(f"Primary depth backend: {primary_name}")
    depth_primary = primary_backend.predict(image_bgr)

    depth_secondary = None
    if secondary_backend is not None:
        logger.info(f"Secondary depth backend: {secondary_name}")
        depth_secondary = secondary_backend.predict(image_bgr)

    # fusion
    if fusion_cfg.get("enabled", False):
        depth = fuse_depth_maps(
            depth_primary,
            depth_secondary,
            method=fusion_cfg.get("method", "mean"),
            primary_weight=float(fusion_cfg.get("primary_weight", 0.7)),
        )
    else:
        depth = depth_primary

    # depth 저장
    results: Dict[str, str] = {}
    if out_cfg.get("save_depth_png", True):
        depth_png_path = os.path.join(
            depth_dir, os.path.splitext(os.path.basename(image_name))[0] + "_depth.png"
        )
        _save_depth_as_png(
            depth,
            depth_png_path,
            grayscale=out_cfg.get("depth_grayscale", True),
        )
        logger.info(f"Depth PNG 저장: {depth_png_path}")
        results["depth"] = depth_png_path

    # pointcloud 생성
    pcd = depth_to_pointcloud(
        depth=depth,
        mask=mask,
        point_step=int(out_cfg.get("point_step", 2)),
        clip_min=float(out_cfg.get("clip_min", 0.05)),
        clip_max=float(out_cfg.get("clip_max", 0.95)),
    )

    # pointcloud 후처리
    pp_cfg = out_cfg.get("postprocess", {})
    pcd = postprocess_pointcloud(
        pcd,
        voxel_size=float(pp_cfg.get("voxel_size", 0.0)),
        remove_statistical_outlier=bool(
            pp_cfg.get("remove_statistical_outlier", True)
        ),
        nb_neighbors=int(pp_cfg.get("nb_neighbors", 20)),
        std_ratio=float(pp_cfg.get("std_ratio", 2.0)),
    )

    # pointcloud 저장
    if out_cfg.get("save_pointcloud", True):
        pc_path = os.path.join(
            pc_dir, os.path.splitext(os.path.basename(image_name))[0] + "_pc.ply"
        )
        o3d.io.write_point_cloud(pc_path, pcd)
        logger.info(f"PointCloud 저장: {pc_path}")
        results["pointcloud"] = pc_path

    # mesh 생성/저장
    if out_cfg.get("save_mesh", True):
        mesh = build_mesh_from_pointcloud(
            pcd,
            method=mesh_cfg.get("method", "poisson"),
            smoothing_iterations=int(mesh_cfg.get("smoothing_iterations", 5)),
            target_reduction=float(mesh_cfg.get("target_reduction", 0.5)),
        )

        if texture_cfg.get("enabled", True):
            mesh = project_texture(
                mesh,
                image_bgr=image_bgr,
                scale=float(texture_cfg.get("scale", 1.0)),
            )

        base_name = os.path.splitext(os.path.basename(image_name))[0]
        fmt = mesh_cfg.get("export_format", "ply").lower()
        mesh_path = os.path.join(mesh_dir, f"{base_name}_mesh.{fmt}")
        mesh_path = export_mesh(mesh, mesh_path, fmt=fmt)
        logger.info(f"Mesh 저장: {mesh_path}")
        results["mesh"] = mesh_path

    return results


def _save_depth_as_png(
    depth: np.ndarray,
    path: str,
    grayscale: bool = True,
) -> None:
    """
    depth map을 PNG로 저장합니다.
    grayscale=True면 0~255 단일 채널로 저장.
    """
    d = depth.astype(np.float32)
    d_min, d_max = float(d.min()), float(d.max())
    d = (d - d_min) / (d_max - d_min + 1e-8)
    d_255 = (d * 255.0).clip(0, 255).astype(np.uint8)
    if not grayscale:
        d_255 = cv2.applyColorMap(d_255, cv2.COLORMAP_JET)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, d_255)