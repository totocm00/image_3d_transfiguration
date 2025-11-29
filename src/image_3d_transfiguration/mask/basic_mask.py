import numpy as np
from typing import Optional


def create_mask(
    image: np.ndarray,
    mode: str = "full",
    external_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    이미지에 대한 마스크 생성.

    mode:
      - "full": 전체 이미지를 1로 가지는 마스크
      - "manual": external_mask 사용
      - "auto": 향후 SAM/YOLO 세그멘테이션 연동 지점 (현재는 full과 동일 동작)
    """
    h, w = image.shape[:2]

    if mode == "manual":
        if external_mask is None:
            raise ValueError("manual 모드에서는 external_mask가 필요합니다.")
        if external_mask.shape[:2] != (h, w):
            raise ValueError("external_mask의 크기가 원본 이미지와 다릅니다.")
        mask = external_mask.astype(bool)

    elif mode in ("full", "auto"):
        # 현재 auto는 아직 구현 전이므로 full과 동일한 마스크 사용
        mask = np.ones((h, w), dtype=bool)

    else:
        raise ValueError(f"지원하지 않는 mask mode: {mode}")

    return mask