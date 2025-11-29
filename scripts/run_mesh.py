import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from image_3d_transfiguration.mesh_builder import pointcloud_to_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pc_path", required=True, help="PointCloud .ply file path")
    parser.add_argument(
        "--out",
        default="assets/outputs/mesh/mesh.obj",
        help="Output mesh file path",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pointcloud_to_mesh(args.pc_path, args.out)


if __name__ == "__main__":
    main()