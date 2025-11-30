# src/image_3d_transfiguration/pointcloud/generator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import open3d as o3d


@dataclass
class CameraConfig:
    fx: float
    fy: float
    cx: float
    cy: float
    z_scale: float = 1.0   # 깊이 범위 스케일링 (0~1 → 실제 Z)


def make_default_camera(h: int, w: int, z_scale: float = 1.0) -> CameraConfig:
    """
    간단한 pinhole 카메라 파라미터 생성.
    - FOV ≈ 60~70deg 정도 되도록 f 설정.
    """
    f = 1.2 * max(h, w)
    cx = w / 2.0
    cy = h / 2.0
    return CameraConfig(fx=f, fy=f, cx=cx, cy=cy, z_scale=z_scale)


def depth_to_pointcloud(
    depth_norm: np.ndarray,
    image_bgr: Optional[np.ndarray] = None,
    cam: Optional[CameraConfig] = None,
    depth_min: float = 0.05,
    depth_max: float = 1.0,
    mask: Optional[np.ndarray] = None,
) -> o3d.geometry.PointCloud:
    """
    정규화(0~1) depth 맵을 point cloud 로 변환.

    - depth_norm: (H, W) 0~1
    - image_bgr: vertex color 로 사용할 이미지 (선택)
    - cam: 카메라 파라미터 (없으면 자동)
    - depth_min / depth_max: 실제 Z 범위 (meters 비슷하게)
    - mask: True/1 인 위치만 사용 (선택)
    """
    h, w = depth_norm.shape[:2]
    if cam is None:
        cam = make_default_camera(h, w)

    z = depth_min + depth_norm * (depth_max - depth_min)
    z = z * cam.z_scale

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    if mask is not None:
        valid = (mask > 0) & (z > 0)
    else:
        valid = z > 0

    xx = xx[valid].astype("float32")
    yy = yy[valid].astype("float32")
    zz = z[valid].astype("float32")

    # pinhole 모델: X = (x - cx) * Z / fx, Y = (y - cy) * Z / fy
    x = (xx - cam.cx) * zz / cam.fx
    y = (yy - cam.cy) * zz / cam.fy

    pts = np.stack([x, -y, zz], axis=-1)  # y 축 반전(화면↓ → 월드↑)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)

    if image_bgr is not None:
        if image_bgr.ndim == 2:
            image_bgr = np.repeat(image_bgr[..., None], 3, axis=-1)
        img = image_bgr.astype("float32") / 255.0
        img_rgb = img[..., ::-1]  # BGR → RGB

        colors = img_rgb[yy.astype(int), xx.astype(int)]
        pc.colors = o3d.utility.Vector3dVector(colors)

    return pc