import numpy as np

from .base import DepthBackend


class FakeDepthBackend(DepthBackend):
    """
    개발/테스트용 더미 DepthBackend.
    실제 프로젝트에서는 depth_anything / zoe_depth 사용.
    """

    def _setup(self, device: str) -> None:
        self.device = device

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        x = np.linspace(0.1, 1.0, w, dtype=np.float32)
        depth = np.tile(x, (h, 1))
        return depth