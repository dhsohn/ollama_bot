"""헬스체크 자동화 callable 구현."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from core.engine import Engine
from core.memory import MemoryManager

from .common import (
    count_recent_errors_async,
)


def build_health_check_callable(
    engine: Engine,
    memory: MemoryManager,
    data_dir: str,
    logger: Any,
):
    async def health_check(
        disk_warn_pct: int = 85,
        error_hours_back: int = 1,
        model: str | None = None,
        model_role: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """시스템 상태를 점검한다. 이상 발견 시에만 보고서를 반환한다."""
        _ = (model, model_role, temperature, max_tokens)
        if not (1 <= disk_warn_pct <= 99):
            raise ValueError("disk_warn_pct must be between 1 and 99")
        if error_hours_back <= 0:
            raise ValueError("error_hours_back must be > 0")

        has_issue = False
        lines: list[str] = ["🏥 시스템 상태 점검\n"]

        try:
            status = await engine.get_status()
            llm_info = status.get("llm", {})
            if llm_info.get("status") == "ok":
                models_count = llm_info.get("models_count", 0)
                default_available = llm_info.get(
                    "default_model_available", False,
                )
                if not default_available:
                    has_issue = True
                    lines.append(
                        f"⚠️ LLM: 기본 모델 사용 불가 "
                        f"(모델 {models_count}개)"
                    )
                else:
                    lines.append(
                        f"✅ LLM: 정상 "
                        f"(모델 {models_count}개, 기본 모델 사용 가능)"
                    )
            else:
                has_issue = True
                error_msg = llm_info.get("error", "알 수 없는 오류")
                lines.append(f"🔴 LLM: 오류 ({error_msg})")
        except Exception as exc:
            has_issue = True
            lines.append(f"🔴 LLM: 연결 실패 ({exc})")

        try:
            ok = await memory.ping()
            if ok:
                lines.append("✅ 데이터베이스: 정상")
            else:
                has_issue = True
                lines.append("🔴 데이터베이스: 응답 이상")
        except Exception as exc:
            has_issue = True
            lines.append(f"🔴 데이터베이스: 오류 ({exc})")

        try:
            usage = shutil.disk_usage(data_dir)
            used_pct = int((usage.used / usage.total) * 100)
            if used_pct >= disk_warn_pct:
                has_issue = True
                lines.append(
                    f"⚠️ 디스크: {used_pct}% 사용 "
                    f"(경고 임계값 {disk_warn_pct}%)"
                )
            else:
                lines.append(f"✅ 디스크: {used_pct}% 사용")
        except Exception as exc:
            has_issue = True
            lines.append(f"🔴 디스크: 확인 실패 ({exc})")

        log_path = Path(data_dir) / "logs"
        _, error_count, _ = await count_recent_errors_async(
            log_path, error_hours_back, max_entries=0,
        )
        if error_count > 0:
            has_issue = True
            lines.append(
                f"⚠️ 오류 로그: 최근 {error_hours_back}시간 "
                f"오류 {error_count}건"
            )
        else:
            lines.append(
                f"✅ 오류 로그: 최근 {error_hours_back}시간 오류 0건"
            )

        if not has_issue:
            logger.info("health_check_all_ok")
            return ""

        return "\n".join(lines)

    return health_check
