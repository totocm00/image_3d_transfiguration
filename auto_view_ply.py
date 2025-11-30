import sys
from pathlib import Path
import open3d as o3d
import glob
import os

POINTCLOUD_DIR = "assets/outputs/pointcloud"
MESH_DIR = "assets/outputs/mesh"

def find_latest_ply():
    """pointcloud 및 mesh 폴더에서 최신 .ply 파일 자동 탐색"""
    candidates = []

    for folder in [POINTCLOUD_DIR, MESH_DIR]:
        ply_files = list(Path(folder).glob("*.ply"))
        for f in ply_files:
            candidates.append((f, f.stat().st_mtime))

    if not candidates:
        return None

    # 수정 날짜 기준 최신 파일
    latest = max(candidates, key=lambda x: x[1])[0]
    return latest

def view_ply(path: Path):
    """mesh인지 pointcloud인지 자동 판별 후 시각화"""
    print(f"[INFO] 로드 중: {path}")

    # Mesh 시도
    try:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh.has_triangles():
            mesh.compute_vertex_normals()
            print("[INFO] Mesh로 판별됨 → 시각화")
            o3d.visualization.draw_geometries([mesh])
            return
    except Exception:
        pass

    # PointCloud fallback
    pc = o3d.io.read_point_cloud(str(path))
    print("[INFO] PointCloud로 판별됨 → 시각화")
    o3d.visualization.draw_geometries([pc])


def main():
    # 수동 지정 모드
    if len(sys.argv) > 1:
        manual_path = Path(sys.argv[1])
        if not manual_path.exists():
            print(f"[ERROR] 파일을 찾을 수 없습니다: {manual_path}")
            sys.exit(1)
        view_ply(manual_path)
        return

    # 자동 모드
    print("[INFO] 인자 없음 → 자동으로 최신 .ply 파일 탐색 중...")
    latest_file = find_latest_ply()

    if latest_file is None:
        print("[ERROR] 최신 .ply 파일을 찾을 수 없습니다.")
        print("pointcloud 또는 mesh 폴더 안에 결과가 있는지 확인하세요.")
        sys.exit(1)

    print(f"[INFO] 최신 파일 선택됨 → {latest_file}")
    view_ply(latest_file)


if __name__ == "__main__":
    main()