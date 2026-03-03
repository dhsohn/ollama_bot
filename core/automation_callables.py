"""내장 자동화 callable 등록."""

from __future__ import annotations

from core.automation_callables_impl.common import (
    CONSOLIDATION_MERGE_SCHEMA as _CONSOLIDATION_MERGE_SCHEMA,
    DAILY_SUMMARY_SCHEMA as _DAILY_SUMMARY_SCHEMA,
    FEEDBACK_ANALYSIS_SCHEMA as _FEEDBACK_ANALYSIS_SCHEMA,
    MEMORY_HYGIENE_SCHEMA as _MEMORY_HYGIENE_SCHEMA,
    PREFERENCES_SCHEMA as _PREFERENCES_SCHEMA,
    STALE_EVALUATION_SCHEMA as _STALE_EVALUATION_SCHEMA,
    TRIAGE_SCHEMA as _TRIAGE_SCHEMA,
)
from core.automation_callables_impl.memory_consolidation import (
    build_memory_consolidation_callable,
)
from core.automation_callables_impl.memory_hygiene import (
    build_memory_hygiene_callable,
)
from core.automation_callables_impl.observability import (
    build_error_log_triage_callable,
    build_health_check_callable,
)
from core.automation_callables_impl.rag import (
    build_rag_reindex_callable,
)
from core.automation_callables_impl.summary import (
    build_daily_summary_callable,
    build_extract_preferences_callable,
)
from core.engine import Engine
from core.feedback_manager import FeedbackManager
from core.logging_setup import get_logger
from core.memory import MemoryManager


def register_builtin_callables(
    scheduler,
    engine: Engine,
    memory: MemoryManager,
    allowed_users: list[int],
    data_dir: str = "data",
    feedback: FeedbackManager | None = None,
    dft_index: object | None = None,
    kb_dirs: list[str] | None = None,
) -> None:
    """내장 자동화 callable을 스케줄러에 등록한다."""
    logger = get_logger("automation_callables")

    scheduler.register_callable(
        "daily_summary",
        build_daily_summary_callable(
            engine=engine,
            memory=memory,
            allowed_users=allowed_users,
            logger=logger,
        ),
    )
    scheduler.register_callable(
        "extract_preferences",
        build_extract_preferences_callable(
            engine=engine,
            memory=memory,
            allowed_users=allowed_users,
            logger=logger,
        ),
    )
    scheduler.register_callable(
        "error_log_triage",
        build_error_log_triage_callable(
            engine=engine,
            data_dir=data_dir,
            logger=logger,
        ),
    )
    scheduler.register_callable(
        "health_check",
        build_health_check_callable(
            engine=engine,
            memory=memory,
            data_dir=data_dir,
            logger=logger,
        ),
    )
    scheduler.register_callable(
        "memory_consolidation",
        build_memory_consolidation_callable(
            engine=engine,
            memory=memory,
            allowed_users=allowed_users,
            logger=logger,
        ),
    )
    scheduler.register_callable(
        "memory_hygiene",
        build_memory_hygiene_callable(
            engine=engine,
            memory=memory,
            allowed_users=allowed_users,
            logger=logger,
        ),
    )
    scheduler.register_callable(
        "rag_reindex",
        build_rag_reindex_callable(
            engine=engine,
            logger=logger,
        ),
    )

    # 피드백 분석 callable
    if feedback is not None:
        from core.automation_callables_impl.feedback_analysis import build_feedback_analysis_callable
        scheduler.register_callable(
            "feedback_analysis",
            build_feedback_analysis_callable(engine, memory, feedback, allowed_users, logger),
        )
    else:
        async def _feedback_analysis_noop(**kwargs) -> str:
            return ""
        scheduler.register_callable("feedback_analysis", _feedback_analysis_noop)

    # DFT 모니터 callable
    if dft_index is not None:
        from core.automation_callables_impl.dft_monitor import build_dft_monitor_callable
        scheduler.register_callable(
            "dft_monitor",
            build_dft_monitor_callable(
                dft_index=dft_index,
                kb_dirs=kb_dirs or [],
                logger=logger,
            ),
        )
    else:
        async def _dft_monitor_noop(**kwargs) -> str:
            return ""
        scheduler.register_callable("dft_monitor", _dft_monitor_noop)

    # KTO 파인튜닝 데이터 내보내기 callable
    if feedback is not None:
        from core.automation_callables_impl.export_training_data import build_export_training_data_callable
        scheduler.register_callable(
            "export_training_data",
            build_export_training_data_callable(feedback, data_dir, logger),
        )
    else:
        async def _export_training_data_noop(**kwargs) -> str:
            return ""
        scheduler.register_callable("export_training_data", _export_training_data_noop)


__all__ = [
    "_CONSOLIDATION_MERGE_SCHEMA",
    "_DAILY_SUMMARY_SCHEMA",
    "_FEEDBACK_ANALYSIS_SCHEMA",
    "_MEMORY_HYGIENE_SCHEMA",
    "_PREFERENCES_SCHEMA",
    "_STALE_EVALUATION_SCHEMA",
    "_TRIAGE_SCHEMA",
    "register_builtin_callables",
]
