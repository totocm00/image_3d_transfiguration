# src/image_3d_transfiguration/depth/depth_anything_backend.py

from __future__ import annotations

from typing import Any

import numpy as np

from .base import DepthBackend


class DepthAnythingBackend(DepthBackend):
    """
    Hugging Face Depth-Anything 백엔드.

    checkpoint: LiheYoung/depth-anything-small-hf
    """

    def _setup(self, device: str) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise RuntimeError(
                "[DepthAnythingBackend] torch / transformers 가 필요합니다.\n"
                "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124\n"
                "  pip install transformers Pillow opencv-python"
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

        ckpt = "LiheYoung/depth-anything-small-hf"
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
            pred = outputs.predicted_depth  # (B, H', W')

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