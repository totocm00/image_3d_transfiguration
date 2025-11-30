from typing import Literal, Tuple

import numpy as np
import open3d as o3d


def estimate_normals(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05,
            max_nn=30,
        )
    )
    return pcd


def build_mesh_from_pointcloud(
    pcd: o3d.geometry.PointCloud,
    method: Literal["poisson", "ball_pivoting"] = "poisson",
    smoothing_iterations: int = 5,
    target_reduction: float = 0.5,
) -> o3d.geometry.TriangleMesh:
    """
    PointCloud로부터 Mesh 생성.
    """
    if not pcd.has_normals():
        pcd = estimate_normals(pcd)

    if method == "poisson":
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        # bounding box 기반 crop (필요 시)
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)
    elif method == "ball_pivoting":
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = float(np.mean(distances))
        radius = 3.0 * avg_dist
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(
                [radius, radius * 2.0]
            ),
        )
    else:
        raise ValueError(f"지원하지 않는 mesh method: {method}")

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    if smoothing_iterations > 0:
        mesh = mesh.filter_smooth_simple(number_of_iterations=smoothing_iterations)

    if 0.0 < target_reduction < 1.0:
        mesh = mesh.simplify_quadric_decimation(
            int(len(mesh.triangles) * (1.0 - target_reduction))
        )

    mesh.compute_vertex_normals()
    return mesh