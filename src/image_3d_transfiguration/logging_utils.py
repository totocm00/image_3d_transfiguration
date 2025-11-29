import logging
from typing import Optional


def setup_logging(level: str = "INFO") -> None:
    """
    공통 로깅 설정 함수.
    config.system.log_level 값을 받아서 호출하면 됩니다.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    모듈별 logger 생성용 헬퍼.
    """
    return logging.getLogger(name or __name__)