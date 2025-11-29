import os
from typing import Literal, Tuple

import open3d as o3d


def export_mesh(
    mesh: o3d.geometry.TriangleMesh,
    output_path: str,
    fmt: Literal["ply", "obj", "glb"] = "ply",
) -> str:
    """
    Mesh를 지정된 포맷으로 저장합니다.
    (glb는 open3d 단독으로는 직접 지원이 제한적이므로
     필요 시 trimesh 등 외부 라이브러리를 활용하는 확장 포인트입니다.)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ext = fmt.lower()
    if ext == "ply":
        if not output_path.endswith(".ply"):
            output_path += ".ply"
        o3d.io.write_triangle_mesh(output_path, mesh)
    elif ext == "obj":
        if not output_path.endswith(".obj"):
            output_path += ".obj"
        o3d.io.write_triangle_mesh(output_path, mesh)
    elif ext == "glb":
        # Open3D만으로 glb를 직접 쓰기는 조금 까다로워서,
        # 여기서는 PLY로 저장 후, 향후 별도 변환 스텝을 두는 방식 권장.
        if not output_path.endswith(".ply"):
            output_path += ".ply"
        o3d.io.write_triangle_mesh(output_path, mesh)
        # TODO: 필요 시 trimesh/pygltflib 등으로 glb 변환 로직 추가
    else:
        raise ValueError(f"지원하지 않는 mesh export 포맷: {fmt}")

    return output_path