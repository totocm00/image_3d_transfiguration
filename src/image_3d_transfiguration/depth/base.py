# src/image_3d_transfiguration/depth/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class DepthBackend(ABC):
    """
    심도(Depth) 추론용 공통 백엔드 인터페이스.

    - 입력: BGR uint8 이미지 (H, W, 3)
    - 출력: float32 depth (H, W)  (상대/절대는 구현체에 따름)
    """

    def __init__(self, device: str = "auto") -> None:
        self.device = device
        self._setup(device=device)

    @abstractmethod
    def _setup(self, device: str) -> None:
        ...

    @abstractmethod
    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        ...

    def close(self) -> None:
        """필요하면 리소스 정리용으로 override."""
        return