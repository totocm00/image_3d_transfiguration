import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def paths(self) -> Dict[str, Any]:
        return self.raw.get("paths", {})

    @property
    def input(self) -> Dict[str, Any]:
        return self.raw.get("input", {})

    @property
    def depth(self) -> Dict[str, Any]:
        return self.raw.get("depth", {})

    @property
    def fusion(self) -> Dict[str, Any]:
        return self.raw.get("fusion", {})

    @property
    def output(self) -> Dict[str, Any]:
        return self.raw.get("output", {})

    @property
    def mesh(self) -> Dict[str, Any]:
        return self.raw.get("mesh", {})

    @property
    def texture(self) -> Dict[str, Any]:
        return self.raw.get("texture", {})

    @property
    def viewer(self) -> Dict[str, Any]:
        return self.raw.get("viewer", {})

    @property
    def system(self) -> Dict[str, Any]:
        return self.raw.get("system", {})


def load_config(config_path: str) -> Config:
    """
    YAML 설정 파일을 로드하여 Config 객체로 반환합니다.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # TODO: 필요하면 여기서 간단한 검증/기본값 채우기 추가
    return Config(raw=raw)
