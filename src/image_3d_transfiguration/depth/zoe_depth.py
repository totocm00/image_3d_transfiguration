from typing import Any

import numpy as np

from .base import DepthBackend


class ZoeDepthBackend(DepthBackend):
    """
    Hugging Face Transformers 기반 ZoeDepth 백엔드.

    - 모델: Intel/zoedepth-nyu-kitti
    - 입력: BGR uint8 이미지 (OpenCV 형식)
    - 출력: float32 depth (H, W), 원본 해상도에 맞춰 리샘플
    """

    def _setup(self, device: str) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            import cv2  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "[ZoeDepthBackend] 필요한 패키지가 없습니다.\n"
                "필수: torch, transformers, Pillow, opencv-python\n"
                "예시) pip install torch torchvision torchaudio\n"
                "     pip install transformers Pillow opencv-python"
            ) from e

        self._torch = torch
        self._AutoImageProcessor = AutoImageProcessor
        self._AutoModelForDepthEstimation = AutoModelForDepthEstimation

        # 디바이스 결정
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        # HF ZoeDepth 체크포인트
        checkpoint = "Intel/zoedepth-nyu-kitti"
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForDepthEstimation.from_pretrained(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        BGR(0~255) 이미지를 받아 ZoeDepth로 깊이 추정.
        반환: float32 np.ndarray, shape=(H, W)
        """
        import cv2
        from PIL import Image

        torch = self._torch

        # BGR → RGB PIL.Image
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # 전처리
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth  # (B, H', W')

        # 원본 사이즈로 보간
        h, w = image_bgr.shape[:2]
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        depth = prediction.squeeze().cpu().numpy().astype("float32")

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = np.maximum(depth, 0.0)
        return depth