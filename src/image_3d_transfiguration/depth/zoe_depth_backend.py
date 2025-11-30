# src/image_3d_transfiguration/depth/zoe_depth_backend.py

from __future__ import annotations

import numpy as np

from .base import DepthBackend


class ZoeDepthBackend(DepthBackend):
    """
    Hugging Face ZoeDepth 백엔드.

    checkpoint: Intel/zoedepth-nyu-kitti
    (설치 안 되어 있으면 ImportError 발생 → 상위에서 fallback)
    """

    def _setup(self, device: str) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise RuntimeError(
                "[ZoeDepthBackend] zoedepth 사용을 위해 torch / transformers 가 필요합니다."
            ) from e

        self._torch = torch
        self._AutoImageProcessor = AutoImageProcessor
        self._AutoModelForDepthEstimation = AutoModelForDepthEstimation

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        ckpt = "Intel/zoedepth-nyu-kitti"
        self.processor = AutoImageProcessor.from_pretrained(ckpt)
        self.model = AutoModelForDepthEstimation.from_pretrained(ckpt)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        import cv2
        from PIL import Image

        torch = self._torch

        h, w = image_bgr.shape[:2]
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)

        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            pred = outputs.predicted_depth

        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        depth = pred.squeeze().cpu().numpy().astype("float32")
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = np.maximum(depth, 0.0)
        return depth