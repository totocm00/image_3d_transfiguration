import sys
from pathlib import Path
import open3d as o3d

def main():
    if len(sys.argv) < 2:
        print("사용 방법:")
        print("  python3 view_ply.py <파일경로>")
        print("예:")
        print("  python3 view_ply.py assets/outputs/pointcloud/input_pc.ply")
        print("  python3 view_ply.py assets/outputs/mesh/input_mesh.ply")
        sys.exit(1)

    path = Path(sys.argv[1])
    
    if not path.exists():
        print(f"[ERROR] 파일을 찾을 수 없습니다: {path}")
        sys.exit(1)

    print(f"[INFO] 로드 중: {path}")

    # 먼저 mesh로 시도
    try:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh.has_triangles():
            mesh.compute_vertex_normals()
            print("[INFO] Mesh 파일로 감지 → 시각화")
            o3d.visualization.draw_geometries([mesh])
            return
    except Exception:
        pass

    # fallback: point cloud로 시도
    pc = o3d.io.read_point_cloud(str(path))
    print("[INFO] PointCloud 파일로 감지 → 시각화")
    o3d.visualization.draw_geometries([pc])


if __name__ == "__main__":
    main()