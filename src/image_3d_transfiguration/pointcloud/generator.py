from typing import Tuple

import numpy as np
import open3d as o3d


def depth_to_pointcloud(
    depth: np.ndarray,
    mask: np.ndarray,
    point_step: int = 2,
    clip_min: float = 0.05,
    clip_max: float = 0.95,
) -> o3d.geometry.PointCloud:
    """
    depth map + mask를 사용하여 PointCloud 생성.

    - depth: (H, W) float32
    - mask : (H, W) bool
    """
    h, w = depth.shape

    # depth 정규화 후 클리핑
    d = depth.astype(np.float32)
    d_min, d_max = np.percentile(d, [clip_min * 100, clip_max * 100])
    d = np.clip(d, d_min, d_max)
    d = (d - d_min) / (d_max - d_min + 1e-8)

    # 좌표 생성 (간단한 pinhole 가정)
    ys, xs = np.mgrid[0:h, 0:w]
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)

    # 중심 기준으로 정규화
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    fx = fy = max(h, w)  # focal length 비슷하게

    X = (xs - cx) / fx * d
    Y = (ys - cy) / fy * d
    Z = d

    # mask 및 point_step 반영
    valid = mask & (d > 0)
    valid[::point_step, ::point_step] = valid[::point_step, ::point_step] & True
    valid_indices = np.where(valid)

    points = np.stack(
        [
            X[valid_indices],
            -Y[valid_indices],  # 위/아래 반전 (보기 좋게)
            Z[valid_indices],
        ],
        axis=-1,
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd