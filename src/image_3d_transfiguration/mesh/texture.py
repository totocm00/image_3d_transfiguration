# src/image_3d_transfiguration/mesh/texture.py

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import open3d as o3d

from ..pointcloud.generator import CameraConfig, make_default_camera


def project_texture(
    mesh: o3d.geometry.TriangleMesh,
    image_bgr: np.ndarray,
    cam: Optional[CameraConfig] = None,
) -> o3d.geometry.TriangleMesh:
    """
    카메라 모델 기반 텍스처 프로젝션.

    1) mesh vertex 를 카메라 좌표계로 역투영했다고 가정
    2) 각 vertex 를 다시 이미지 평면으로 project → (u,v)
    3) (u,v) 에서 rgb 샘플링 → vertex_colors 세팅

    sam3d 처럼 완전한 UV / Multi-view 는 아니지만,
    최소한 카메라 중심 기준으로 찌그러짐이 줄어든 버전.
    """
    if mesh.is_empty():
        return mesh

    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_f = img_rgb.astype("float32") / 255.0
    h, w = img_rgb_f.shape[:2]

    vertices = np.asarray(mesh.vertices)
    if vertices.shape[0] == 0:
        return mesh

    # 메쉬가 이미 카메라 좌표계에 있다고 가정 (pointcloud 생성 단계 기준)
    # Z>0 인 쪽만 신뢰
    X = vertices[:, 0]
    Y = vertices[:, 1]
    Z = vertices[:, 2]

    # Z가 너무 작거나 음수면 버린다 → 나중에 색 대신 회색 사용
    valid = Z > (Z.mean() * 0.05)

    if cam is None:
        # Z 범위를 고려하지 못하지만, FOV 정도 맞추는 용도
        cam = make_default_camera(h, w)

    # project: x_img = fx * X / Z + cx
    x_img = cam.fx * X / (Z + 1e-8) + cam.cx
    y_img = cam.fy * (-Y) / (Z + 1e-8) + cam.cy

    x_img = np.clip(x_img, 0, w - 1).astype("int32")
    y_img = np.clip(y_img, 0, h - 1).astype("int32")

    colors = img_rgb_f[y_img, x_img]

    # 유효하지 않은 Z는 회색으로
    gray = np.full_like(colors, 0.5)
    colors[~valid] = gray[~valid]

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    return mesh