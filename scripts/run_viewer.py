import argparse

import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple Open3D viewer with optional smoothing",
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        help="Mesh file path (.ply/.obj/.glb 등)",
    )
    parser.add_argument(
        "--pc_path",
        type=str,
        help="Point cloud file path (.ply 등)",
    )
    parser.add_argument(
        "--smooth_iter",
        type=int,
        default=10,
        help="Taubin smoothing iteration count (mesh only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    geometries = []

    if args.mesh_path:
        mesh = o3d.io.read_triangle_mesh(args.mesh_path)
        if mesh.is_empty():
            print(f"[!] Failed to load mesh: {args.mesh_path}")
        else:
            mesh.compute_vertex_normals()
            if args.smooth_iter > 0:
                print(f"[+] Smoothing mesh ({args.smooth_iter} iters)...")
                mesh = mesh.filter_smooth_taubin(
                    number_of_iterations=args.smooth_iter
                )
                mesh.compute_vertex_normals()
            geometries.append(mesh)

    if args.pc_path:
        pc = o3d.io.read_point_cloud(args.pc_path)
        if pc.is_empty():
            print(f"[!] Failed to load point cloud: {args.pc_path}")
        else:
            geometries.append(pc)

    if not geometries:
        print("[!] No geometry to show. Use --mesh_path or --pc_path")
        return

    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    main()