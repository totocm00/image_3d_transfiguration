import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import open3d as o3d

from image_3d_transfiguration.config_loader import AppConfig, build_output_paths


def _select_device(request: str) -> str:
    if request == "cpu":
        return "cpu"
    if request == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_depth_model(cfg: AppConfig):
    device = _select_device(cfg.model.device)
    processor = AutoImageProcessor.from_pretrained(cfg.model.id)
    model = AutoModelForDepthEstimation.from_pretrained(cfg.model.id)
    model = model.to(device)
    return processor, model, device


def save_depth_png(depth_norm: np.ndarray, out_path: str, grayscale: bool = True):
    if grayscale:
        depth_img = (depth_norm * 255.0).astype(np.uint8)
        depth_pil = Image.fromarray(depth_img)
    else:
        # 간단 컬러맵 (magma 같은 거 직접 구현해도 되고, 일단 gray 고정해도 됨)
        depth_img = (depth_norm * 255.0).astype(np.uint8)
        depth_pil = Image.fromarray(depth_img)

    depth_pil.save(out_path)


def run_image_3d(
    image_path: str,
    cfg: AppConfig,
):
    # 파일명에서 베이스 이름 추출
    basename = os.path.splitext(os.path.basename(image_path))[0]
    depth_path, pc_path = build_output_paths(cfg, basename)

    processor, model, device = load_depth_model(cfg)

    # --- depth ---
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    post_processed = processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )
    depth_tensor = post_processed[0]["predicted_depth"].cpu()
    depth = depth_tensor.squeeze().numpy()
    d_min, d_max = float(depth.min()), float(depth.max())
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)

    # --- depth 저장 (config 기반) ---
    if cfg.output.save_depth_png:
        save_depth_png(
            depth_norm=depth_norm,
            out_path=depth_path,
            grayscale=cfg.output.depth_grayscale,
        )

    # --- point cloud (config 기반) ---
    if cfg.output.save_pointcloud:
        _depth_to_pointcloud(
            image=image,
            depth=depth,
            depth_norm=depth_norm,
            out_ply=pc_path,
            step=cfg.output.point_step,
            clip_min=cfg.output.clip_min,
            clip_max=cfg.output.clip_max,
        )

    return {
        "depth_path": depth_path if cfg.output.save_depth_png else None,
        "point_cloud_path": pc_path if cfg.output.save_pointcloud else None,
    }


def _depth_to_pointcloud(
    image: Image.Image,
    depth: np.ndarray,
    depth_norm: np.ndarray,
    out_ply: str,
    step: int = 2,
    clip_min: float = 0.05,
    clip_max: float = 0.95,
):
    H, W = depth.shape
    rgb = np.array(image)

    mask_valid = (depth_norm > clip_min) & (depth_norm < clip_max)

    fx = fy = max(H, W) * 0.7
    cx = W / 2.0
    cy = H / 2.0

    points = []
    colors = []

    for v in range(0, H, step):
        for u in range(0, W, step):
            if not mask_valid[v, u]:
                continue

            z = float(depth_norm[v, u])
            if z <= 0:
                continue

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points.append([x, -y, z])
            r, g, b = rgb[v, u] / 255.0
            colors.append([r, g, b])

    if not points:
        raise RuntimeError("No valid points for point cloud.")

    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)

    centroid = points.mean(axis=0, keepdims=True)
    points = points - centroid

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    o3d.io.write_point_cloud(out_ply, pcd)