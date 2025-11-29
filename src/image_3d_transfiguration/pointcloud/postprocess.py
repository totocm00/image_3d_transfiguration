from typing import Optional

import open3d as o3d


def postprocess_pointcloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.0,
    remove_statistical_outlier: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    """
    PointCloud에 대한 후처리:
      - voxel 다운샘플링
      - 통계 기반 outlier 제거
    """
    out = pcd

    if voxel_size > 0:
        out = out.voxel_down_sample(voxel_size=voxel_size)

    if remove_statistical_outlier:
        out, _ = out.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )

    return out