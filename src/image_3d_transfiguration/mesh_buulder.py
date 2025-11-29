import open3d as o3d
import numpy as np


def load_pointcloud(ply_path: str):
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(np.asarray(pcd.points)) == 0:
        raise ValueError("PointCloud is empty or invalid.")
    return pcd


def estimate_normals(pcd, radius=0.02, max_nn=30):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn
        )
    )
    pcd.orient_normals_consistent_tangent_plane(30)
    return pcd


def poisson_mesh_reconstruction(pcd, depth=9):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    return mesh


def remove_low_density(mesh, densities, threshold=0.01):
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, threshold)
    vertices_to_keep = densities > density_threshold
    mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])
    return mesh


def save_mesh(mesh, out_path: str):
    o3d.io.write_triangle_mesh(out_path, mesh)
    return out_path


def pointcloud_to_mesh(
    ply_path: str,
    output_path: str = "assets/outputs/mesh/mesh.obj",
    depth: int = 9,
):
    print("[1] Loading point cloud...")
    pcd = load_pointcloud(ply_path)

    print("[2] Estimating normals...")
    pcd = estimate_normals(pcd)

    print("[3] Running Poisson reconstruction (creating mesh)...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    print("[4] Cleaning mesh (removing low-density vertices)...")
    mesh = remove_low_density(mesh, densities, threshold=0.02)

    print("[5] Saving mesh...")
    save_mesh(mesh, output_path)

    print(f"Mesh saved at: {output_path}")
    return output_path