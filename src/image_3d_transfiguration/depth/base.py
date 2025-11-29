from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class DepthBackend(ABC):
    """
    Depth 예측용 백엔드 공통 인터페이스.
    """

    def __init__(self, device: str = "auto") -> None:
        self.device = device
        self._setup(device=device)

    @abstractmethod
    def _setup(self, device: str) -> None:
        """
        모델 로드/초기화 등 세팅 작업을 수행합니다.
        """
        ...

    @abstractmethod
    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        BGR 이미지(0~255 uint8)를 입력받아 depth map(float32)을 반환합니다.
        반환 shape: (H, W), 값 범위: 任意 (후처리에서 정규화).
        """
        ...