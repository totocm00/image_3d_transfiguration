import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .depth_anything import DepthAnythingBackend
from .zoe_depth import ZoeDepthBackend


def _normalize_depth(
    depth: np.ndarray,
    clip_percent: float = 2.0,
) -> np.ndarray:
    """
    depth 맵을 0~1 범위로 정규화.
    - 하위/상위 clip_percent% 구간을 잘라서 outlier 영향 줄임.
    """
    d = depth.astype("float32")
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    flat = d.flatten()
    lo = np.percentile(flat, clip_percent)
    hi = np.percentile(flat, 100.0 - clip_percent)

    if hi <= lo:
        # 이상한 경우 전체 범위 사용
        lo = flat.min()
        hi = flat.max()

    if hi == lo:
        return np.zeros_like(d, dtype="float32")

    d = (d - lo) / (hi - lo)
    d = np.clip(d, 0.0, 1.0)
    return d.astype("float32")


def _colormap_depth(depth_norm: np.ndarray) -> np.ndarray:
    """
    0~1 정규화된 depth 맵을 컬러맵(BGR) 이미지로 변환.
    """
    depth_uint8 = (depth_norm * 255.0).clip(0, 255).astype("uint8")
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
    return depth_color


@dataclass
class DepthStats:
    mean: float
    std: float
    min: float
    max: float


@dataclass
class DepthExperimentResult:
    """
    한 장의 이미지에 대해:
    - depth_anything / zoe / fused 각각의 depth 맵과 정규화 버전
    - 간단한 통계
    - 두 모델의 차이(norm) 통계
    """

    stats_depth_anything: DepthStats
    stats_zoe: DepthStats
    stats_fused: DepthStats
    stats_diff: DepthStats

    # 경로 정보 (시각화 저장 위치)
    path_depth_anything_vis: str
    path_zoe_vis: str
    path_fused_vis: str
    path_diff_vis: str


class MultiBackendDepthExperiment:
    """
    DepthAnything + ZoeDepth 두 개를 동시에 돌려서:
    - 각자 depth 추정
    - 0~1 정규화
    - 단순 평균 융합 (fused)
    - 시각화 이미지 저장
    - 간단한 통계 JSON 저장
    """

    def __init__(self, device: str = "auto") -> None:
        self.depth_anything = DepthAnythingBackend(device=device)
        self.zoe_depth = ZoeDepthBackend(device=device)

    def run_on_image(
        self,
        image_bgr: np.ndarray,
        out_dir: str,
        base_name: str,
    ) -> DepthExperimentResult:
        os.makedirs(out_dir, exist_ok=True)

        # ----- 1. 두 백엔드로 depth 추정 -----
        depth_da = self.depth_anything.predict(image_bgr)
        depth_zoe = self.zoe_depth.predict(image_bgr)

        # ----- 2. 정규화 (0~1) -----
        depth_da_norm = _normalize_depth(depth_da)
        depth_zoe_norm = _normalize_depth(depth_zoe)

        # ----- 3. 단순 평균 융합 -----
        depth_fused_norm = 0.5 * depth_da_norm + 0.5 * depth_zoe_norm

        # ----- 4. 차이 맵 (절대값) -----
        depth_diff_norm = np.abs(depth_da_norm - depth_zoe_norm)
        depth_diff_norm = _normalize_depth(depth_diff_norm)

        # ----- 5. 시각화 저장 -----
        vis_da = _colormap_depth(depth_da_norm)
        vis_zoe = _colormap_depth(depth_zoe_norm)
        vis_fused = _colormap_depth(depth_fused_norm)
        vis_diff = _colormap_depth(depth_diff_norm)

        path_da = os.path.join(out_dir, f"{base_name}_depth_da.png")
        path_zoe = os.path.join(out_dir, f"{base_name}_depth_zoe.png")
        path_fused = os.path.join(out_dir, f"{base_name}_depth_fused.png")
        path_diff = os.path.join(out_dir, f"{base_name}_depth_diff.png")

        cv2.imwrite(path_da, vis_da)
        cv2.imwrite(path_zoe, vis_zoe)
        cv2.imwrite(path_fused, vis_fused)
        cv2.imwrite(path_diff, vis_diff)

        # ----- 6. 통계 계산 -----
        def _stats(d: np.ndarray) -> DepthStats:
            return DepthStats(
                mean=float(d.mean()),
                std=float(d.std()),
                min=float(d.min()),
                max=float(d.max()),
            )

        stats_da = _stats(depth_da_norm)
        stats_zoe = _stats(depth_zoe_norm)
        stats_fused = _stats(depth_fused_norm)
        stats_diff = _stats(depth_diff_norm)

        return DepthExperimentResult(
            stats_depth_anything=stats_da,
            stats_zoe=stats_zoe,
            stats_fused=stats_fused,
            stats_diff=stats_diff,
            path_depth_anything_vis=path_da,
            path_zoe_vis=path_zoe,
            path_fused_vis=path_fused,
            path_diff_vis=path_diff,
        )

    def run_on_image_and_save_json(
        self,
        image_bgr: np.ndarray,
        out_dir: str,
        base_name: str,
    ) -> DepthExperimentResult:
        """
        run_on_image + JSON 요약 저장까지 같이 수행.
        """
        result = self.run_on_image(image_bgr, out_dir, base_name)

        summary = {
            "stats_depth_anything": asdict(result.stats_depth_anything),
            "stats_zoe": asdict(result.stats_zoe),
            "stats_fused": asdict(result.stats_fused),
            "stats_diff": asdict(result.stats_diff),
            "path_depth_anything_vis": result.path_depth_anything_vis,
            "path_zoe_vis": result.path_zoe_vis,
            "path_fused_vis": result.path_fused_vis,
            "path_diff_vis": result.path_diff_vis,
        }

        json_path = os.path.join(out_dir, f"{base_name}_depth_experiment.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return result