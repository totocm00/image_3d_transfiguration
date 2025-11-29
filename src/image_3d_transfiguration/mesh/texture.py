from typing import Tuple

import numpy as np
import open3d as o3d


def project_texture(
    mesh: o3d.geometry.TriangleMesh,
    image_bgr: np.ndarray,
    scale: float = 1.0,
) -> o3d.geometry.TriangleMesh:
    """
    아주 단순한 UV 매핑:
      - 이미지 좌표계를 XY 평면에 투영하고
      - 정규화된 (x, y)를 texture 좌표로 사용.
    고급 텍스처링이 필요하면 별도 툴(Blender, Kaolin 등)로 후처리.
    """
    # mesh 좌표를 [0,1] 범위로 정규화
    vertices = np.asarray(mesh.vertices)
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    v_range = (v_max - v_min) + 1e-8
    v_norm = (vertices - v_min) / v_range

    # x, y만 사용해 uv 생성
    u = v_norm[:, 0]  # x
    v = 1.0 - v_norm[:, 1]  # y (위/아래 반전)

    mesh.triangle_uvs = o3d.utility.Vector2dVector(
        np.stack([u, v], axis=-1)[mesh.triangles.reshape(-1)]
    )

    # 텍스처 이미지를 저장하고 external tool에서 사용할 수 있도록 두는 방식 추천.
    # 여기서는 mesh에 image를 직접 붙이지 않고 UV 정보만 셋팅.
    return mesh