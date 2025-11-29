from typing import Any

import numpy as np

from .base import DepthBackend


class DepthAnythingBackend(DepthBackend):
    """
    Depth Anything 기반 백엔드.
    실제 사용 시에는 사용자가 모델 로딩 코드를 채워 넣어야 합니다.
    (여기서는 구조만 제공하고, 핵심 부분은 TODO로 남겨둡니다.)
    """

    def _setup(self, device: str) -> None:
        self.device = device
        # TODO:
        # - torch / transformers / depth-anything 모델 로드
        # - self.model = ...
        # - device에 따라 to("cuda") 또는 to("cpu")

        # 일단은 fake와 동일하게 동작하도록 해놓고,
        # 모델을 연결하면 predict만 수정해도 전체 파이프라인이 그대로 사용 가능.
        self._use_fake_fallback = True

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        if getattr(self, "_use_fake_fallback", False):
            # 임시: fake-like fallback
            h, w = image_bgr.shape[:2]
            x = np.linspace(0.1, 1.0, w, dtype=np.float32)
            depth = np.tile(x, (h, 1))
            return depth

        # 여기에 실제 DepthAnything inference 코드를 쓰면 됨.
        # 예시 구조:
        #   image_rgb = image_bgr[..., ::-1]
        #   tensor = preprocess(image_rgb)
        #   with torch.no_grad():
        #       pred = self.model(tensor.to(self.device))
        #   depth = postprocess(pred)
        #   return depth
        raise NotImplementedError("DepthAnything 모델 연결이 필요합니다.")