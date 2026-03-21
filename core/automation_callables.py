"""Registration helpers for built-in automation callables."""

from __future__ import annotations

from core.automation_callables_impl.common import (
    CONSOLIDATION_MERGE_SCHEMA as _CONSOLIDATION_MERGE_SCHEMA,
)
from core.automation_callables_impl.common import (
    DAILY_SUMMARY_SCHEMA as _DAILY_SUMMARY_SCHEMA,
)
from core.automation_callables_impl.common import (
    MEMORY_HYGIENE_SCHEMA as _MEMORY_HYGIENE_SCHEMA,
)
from core.automation_callables_impl.common import (
    PREFERENCES_SCHEMA as _PREFERENCES_SCHEMA,
)
from core.automation_callables_impl.common import (
    STALE_EVALUATION_SCHEMA as _STALE_EVALUATION_SCHEMA,
)
from core.automation_callables_impl.common import (
    TRIAGE_SCHEMA as _TRIAGE_SCHEMA,
)
from core.automation_callables_impl.memory_consolidation import (
    build_memory_consolidation_callable,
)
from core.automation_callables_impl.memory_hygiene import (
    build_memory_hygiene_callable,
)
from core.automation_callables_impl.observability import (
    build_health_check_callable,
    build_log_triage_callable,
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
) -> None:
    """Register built-in automation callables with the scheduler."""
    logger = get_logger("automation_callables")
    scheduler_config = getattr(scheduler, "_config", None)
    scheduler_bot = getattr(scheduler_config, "bot", None)
    default_language = getattr(scheduler_bot, "language", "ko")

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
        "health_check",
        build_health_check_callable(
            engine=engine,
            memory=memory,
            data_dir=data_dir,
            logger=logger,
            default_language=default_language,
        ),
    )
    scheduler.register_callable(
        "log_triage",
        build_log_triage_callable(
            engine=engine,
            data_dir=data_dir,
            logger=logger,
            default_language=default_language,
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


__all__ = [
    "_CONSOLIDATION_MERGE_SCHEMA",
    "_DAILY_SUMMARY_SCHEMA",
    "_MEMORY_HYGIENE_SCHEMA",
    "_PREFERENCES_SCHEMA",
    "_STALE_EVALUATION_SCHEMA",
    "_TRIAGE_SCHEMA",
    "register_builtin_callables",
]
