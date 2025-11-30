# src/image_3d_transfiguration/pipeline/full_pipeline.py
import os
from pathlib import Path
import cv2
import numpy as np

from image_3d_transfiguration.preprocess.yolo_sam_pipeline import run_yolo_sam  # TODO: 네가 만들 함수 이름에 맞춰
from image_3d_transfiguration.preprocess.roi_manual import select_manual_roi     # TODO: manual ROI 함수
from image_3d_transfiguration.depth.depth_anything import DepthAnythingBackend
from image_3d_transfiguration.mesh.build_mesh import depth_to_pointcloud_and_mesh  # TODO: 실제 함수명 맞추기


def run_full_pipeline(
    input_path: str,
    mode: str = "auto",      # "auto" or "manual"
    use_sam: bool = True,
    output_prefix: str | None = None,
    config: dict | None = None,
):
    input_path = Path(input_path)
    image = cv2.imread(str(input_path))
    if image is None:
        raise RuntimeError(f"Failed to load image: {input_path}")

    if output_prefix is None:
        output_prefix = input_path.stem  # robot_manual 등

    # 1) ROI 결정
    if mode == "manual":
        # 수동 ROI 선택 (x, y, w, h 반환하도록 설계)
        x, y, w, h = select_manual_roi(image)
        roi = image[y:y+h, x:x+w]
        roi_bbox = (x, y, w, h)
    elif mode == "auto":
        # YOLO로 박스 후보 얻고, 내부에서 규칙대로 하나 선택
        # run_yolo_sam 이 (masked_img, bbox, mask)를 리턴하게 설계해도 좋음
        masked_img, roi_bbox, mask = run_yolo_sam(image)
        roi = masked_img
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 2) SAM 마스크 사용 여부 (auto에서 이미 마스크면 그대로 사용)
    #    manual 모드에서도 나중에 SAM 붙일 수 있음
    if not use_sam and mode == "manual":
        mask = None  # ROI 전체를 대상으로 depth 추출

    # 3) Depth 추정
    depth_backend = DepthAnythingBackend()
    depth_backend.setup(device="cpu")   # TODO: 나중에 cuda 로 바꿀 수 있게
    depth = depth_backend.predict(roi)  # HxW float32

    # 4) pointcloud / mesh 변환
    pc, mesh = depth_to_pointcloud_and_mesh(
        depth=depth,
        rgb=roi,
        mask=mask,
        config=config,
    )

    # 5) 결과 저장
    out_base = Path("assets/outputs")
    depth_dir = out_base / "depth"
    pc_dir    = out_base / "pointcloud"
    mesh_dir  = out_base / "mesh"

    depth_dir.mkdir(parents=True, exist_ok=True)
    pc_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    depth_path = depth_dir / f"{output_prefix}_depth.png"
    pc_path    = pc_dir / f"{output_prefix}_pc.ply"
    mesh_path  = mesh_dir / f"{output_prefix}_mesh.ply"

    # depth 저장 (0~1 → 0~255)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth_u8 = (depth_norm * 255).astype(np.uint8)
    cv2.imwrite(str(depth_path), depth_u8)

    # pointcloud/mesh 저장
    # 실제 구현은 depth_to_pointcloud_and_mesh 안에서 Open3D로 파일 저장해도 되고,
    # 여기에서 o3d.io.write_point_cloud / write_triangle_mesh를 호출해도 됨.
    depth_to_pointcloud_and_mesh(
        depth=depth,
        rgb=roi,
        mask=mask,
        config={
            "pc_out": str(pc_path),
            "mesh_out": str(mesh_path),
        },
    )

    print("=== image_3d_transfiguration pipeline result ===")
    print("input      :", input_path)
    print("roi_bbox   :", roi_bbox)
    print("depth      :", depth_path)
    print("pointcloud :", pc_path)
    print("mesh       :", mesh_path)

    return {
        "depth": depth_path,
        "pointcloud": pc_path,
        "mesh": mesh_path,
        "roi_bbox": roi_bbox,
    }