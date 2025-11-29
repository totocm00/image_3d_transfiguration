from .base import DepthBackend
from .depth_anything import DepthAnythingBackend
from .zoe_depth import ZoeDepthBackend
from .fake import FakeDepthBackend

__all__ = [
    "DepthBackend",
    "DepthAnythingBackend",
    "ZoeDepthBackend",
    "FakeDepthBackend",
]