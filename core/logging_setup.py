"""structlog 기반 구조화된 로깅 설정."""

import logging
import sys

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """애플리케이션 전역 로깅을 설정한다."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # 표준 logging 설정
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """모듈별 로거를 반환한다."""
    return structlog.get_logger(name)
