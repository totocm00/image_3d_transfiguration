# src/image_3d_transfiguration/depth/postprocess.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class DepthPostprocessConfig:
    clip_low_percent: float = 1.0
    clip_high_percent: float = 1.0
    # 가우시안 블러 강도 ↑
    blur_kernel: int = 9          # 가우시안 블러 커널
    # 양방향 필터 강도 ↑
    bilateral_kernel: int = 11    # 양방향 필터 커널
    bilateral_sigma_color: float = 55.0
    bilateral_sigma_space: float = 55.0
    min_valid_ratio: float = 0.2  # 너무 비어 있으면 그대로 사용


def _percentile_clip(d: np.ndarray, low: float, high: float) -> np.ndarray:
    flat = d.reshape(-1)
    lo = np.percentile(flat, low)
    hi = np.percentile(flat, 100.0 - high)
    if hi <= lo:
        return d
    d = np.clip(d, lo, hi)
    return d


def postprocess_depth(
    depth_raw: np.ndarray,
    cfg: Optional[DepthPostprocessConfig] = None,
) -> np.ndarray:
    """
    DepthAnything / ZoeDepth 로부터 얻은 depth 를 sam3d 느낌으로 후처리.

    1) NaN 제거 + 음수 제거
    2) 상하위 percentile 클리핑
    3) 양방향 필터 + 가우시안 스무딩
    4) 0~1 정규화
    """
    if cfg is None:
        cfg = DepthPostprocessConfig()

    d = depth_raw.astype("float32")
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d = np.maximum(d, 0.0)

    # 유효한 값이 너무 적으면 그냥 normalize만 한다
    valid = d > 0
    if valid.mean() < cfg.min_valid_ratio:
        d = d - d.min()
        d /= (d.max() + 1e-8)
        return d

    # 1) percentile clip
    d = _percentile_clip(d, cfg.clip_low_percent, cfg.clip_high_percent)

    # 2) bilateral filter (엣지 보존)
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
    d_norm_uint8 = (d_norm * 255.0).astype("uint8")
    d_bi = cv2.bilateralFilter(
        d_norm_uint8,
        d=cfg.bilateral_kernel,
        sigmaColor=cfg.bilateral_sigma_color,
        sigmaSpace=cfg.bilateral_sigma_space,
    )

    # 3) Gaussian smoothing (커널이 1보다 클 때만)
    if cfg.blur_kernel > 1:
        k = cfg.blur_kernel if cfg.blur_kernel % 2 == 1 else cfg.blur_kernel + 1
        d_bi = cv2.GaussianBlur(d_bi, (k, k), 0)

    d_smooth = d_bi.astype("float32") / 255.0
    d_smooth = np.clip(d_smooth, 0.0, 1.0)
    return d_smooth