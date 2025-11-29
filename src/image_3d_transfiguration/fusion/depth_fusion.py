from typing import Literal, Optional

import numpy as np


def fuse_depth_maps(
    depth_primary: np.ndarray,
    depth_secondary: Optional[np.ndarray],
    method: Literal["mean", "weighted", "min", "max"] = "mean",
    primary_weight: float = 0.7,
) -> np.ndarray:
    """
    두 개의 depth map을 하나로 합성합니다.
    depth_secondary가 None이면 depth_primary를 그대로 반환합니다.
    """
    if depth_secondary is None:
        return depth_primary

    if depth_primary.shape != depth_secondary.shape:
        raise ValueError("두 depth map의 shape가 다릅니다.")

    if method == "mean":
        return (depth_primary + depth_secondary) / 2.0

    if method == "weighted":
        w = float(primary_weight)
        w = max(0.0, min(1.0, w))
        return w * depth_primary + (1.0 - w) * depth_secondary

    if method == "min":
        return np.minimum(depth_primary, depth_secondary)

    if method == "max":
        return np.maximum(depth_primary, depth_secondary)

    raise ValueError(f"지원하지 않는 fusion method: {method}")