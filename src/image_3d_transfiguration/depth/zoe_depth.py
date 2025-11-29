import numpy as np

from .base import DepthBackend


class ZoeDepthBackend(DepthBackend):
    """
    ZoeDepth 기반 백엔드.
    구조만 정의하고, 실제 모델 연결은 사용자가 채울 수 있도록 합니다.
    """

    def _setup(self, device: str) -> None:
        self.device = device
        self._use_fake_fallback = True

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        if getattr(self, "_use_fake_fallback", False):
            h, w = image_bgr.shape[:2]
            y = np.linspace(0.1, 1.0, h, dtype=np.float32)
            depth = np.tile(y[:, None], (1, w))
            return depth

        raise NotImplementedError("ZoeDepth 모델 연결이 필요합니다.")