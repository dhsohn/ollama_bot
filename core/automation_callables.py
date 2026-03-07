"""내장 자동화 callable 등록."""

from __future__ import annotations

from pathlib import Path

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
from core.automation_callables_impl.memory_consolidation import (
    build_memory_consolidation_callable,
)
from core.automation_callables_impl.memory_hygiene import (
    build_memory_hygiene_callable,
)
from core.automation_callables_impl.observability import (
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
    sim_scheduler: object | None = None,
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

    # DFT 모니터 callable
    if dft_index is not None:
        from core.automation_callables_impl.dft_monitor import (
            build_dft_monitor_callable,
        )

        get_external_dirs = None
        if sim_scheduler is not None:
            get_external_dirs = _build_get_external_dirs(sim_scheduler)

        scheduler.register_callable(
            "dft_monitor",
            build_dft_monitor_callable(
                dft_index=dft_index,
                kb_dirs=kb_dirs or [],
                logger=logger,
                state_file=str(Path(data_dir) / "automation" / "dft_monitor_state.json"),
                get_external_dirs=get_external_dirs,
            ),
        )
    else:
        async def _dft_monitor_noop(**kwargs) -> str:
            return ""
        scheduler.register_callable("dft_monitor", _dft_monitor_noop)



def _build_get_external_dirs(sim_scheduler: object):
    """SimJobScheduler에서 외부 실행 중인 시뮬레이션의 작업 디렉토리를 반환하는 콜백을 생성한다."""

    async def get_external_dirs() -> list[str]:
        jobs = await sim_scheduler.get_external_running_jobs()  # type: ignore[attr-defined]
        dirs: list[str] = []
        for job in jobs:
            input_file = job.get("input_file", "")
            if input_file and input_file != "-":
                p = Path(input_file).expanduser()
                candidate = p if p.is_dir() else p.parent
                if candidate.is_dir():
                    dirs.append(str(candidate))
                    continue

            # input_file이 없는 경우 (예: crest): PID의 CWD로 폴백
            pid = job.get("pid")
            if pid and isinstance(pid, int) and pid > 0:
                try:
                    cwd = Path(f"/proc/{pid}/cwd").resolve()
                    if cwd.is_dir():
                        dirs.append(str(cwd))
                except OSError:
                    pass
        return dirs

    return get_external_dirs


__all__ = [
    "_CONSOLIDATION_MERGE_SCHEMA",
    "_DAILY_SUMMARY_SCHEMA",
    "_MEMORY_HYGIENE_SCHEMA",
    "_PREFERENCES_SCHEMA",
    "_STALE_EVALUATION_SCHEMA",
    "register_builtin_callables",
]
