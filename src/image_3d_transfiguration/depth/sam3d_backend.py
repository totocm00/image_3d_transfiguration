# src/image_3d_transfiguration/depth/sam3d_backend.py

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass
class Sam3DConfig:
    sam3d_root: str
    python_cmd: str = "python"


def run_sam3d_pointmap(
    cfg: Sam3DConfig,
    input_path: str,
    output_ply: str,
) -> None:
    """
    sam3d_objects.pipeline.inference_pipeline_pointmap 를 서브프로세스로 호출해서
    input_path → output_ply (PLY 메쉬 or 포인트맵) 생성.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    os.makedirs(os.path.dirname(output_ply), exist_ok=True)

    cmd = [
        cfg.python_cmd,
        "-m",
        "sam3d_objects.pipeline.inference_pipeline_pointmap",
        "--input",
        input_path,
        "--output",
        output_ply,
    ]

    # sam-3d 프로젝트 루트에서 실행
    proc = subprocess.run(
        cmd,
        cwd=cfg.sam3d_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"sam-3d inference 실패 (code={proc.returncode}):\n{proc.stdout}"
        )