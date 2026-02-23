"""ollama_bot 진입점.

모든 모듈을 의존성 순서대로 초기화하고
텔레그램 폴링과 자동화 스케줄러를 실행한다.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path

from core.auto_scheduler import AutoScheduler
from core.config import load_config
from core.engine import Engine
from core.logging_setup import get_logger, setup_logging
from core.memory import MemoryManager
from core.ollama_client import OllamaClient
from core.orca_monitor import generate_orca_progress_report
from core.security import SecurityManager
from core.skill_manager import SkillManager
from core.telegram_handler import TelegramHandler

_MEMORY_MAINTENANCE_INTERVAL_SECONDS = 6 * 60 * 60


def _is_running_in_container() -> bool:
    """현재 프로세스가 컨테이너 내부에서 실행 중인지 판단한다."""
    if Path("/.dockerenv").exists():
        return True

    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8")
    except OSError:
        return False

    markers = ("docker", "containerd", "kubepods", "podman")
    return any(marker in cgroup for marker in markers)


async def _memory_maintenance_loop(
    memory: MemoryManager,
    logger,
    interval_seconds: int = _MEMORY_MAINTENANCE_INTERVAL_SECONDS,
) -> None:
    """주기적으로 오래된 대화 데이터를 정리한다."""
    while True:
        try:
            deleted = await memory.prune_old_conversations()
            logger.debug("memory_retention_pruned", deleted=deleted)
        except Exception as exc:
            logger.error("memory_retention_prune_failed", error=str(exc))
        await asyncio.sleep(interval_seconds)


async def async_main() -> None:
    """비동기 메인 루프."""
    if not _is_running_in_container():
        print(
            "오류: 이 애플리케이션은 Docker 컨테이너에서만 실행됩니다.\n"
            "실행 방법: docker compose up --build -d",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── 설정 로드 ──
    try:
        config = load_config()
    except ValueError as exc:
        print(
            f"오류: 설정값이 잘못되었습니다. {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    setup_logging(config.log_level)
    logger = get_logger("main")
    logger.info("starting", version="0.1.0")

    if not config.telegram_bot_token or config.telegram_bot_token == "your_telegram_bot_token_here":
        logger.error("telegram_bot_token_not_set")
        print(
            "오류: TELEGRAM_BOT_TOKEN이 설정되지 않았습니다.\n"
            ".env 파일에 유효한 텔레그램 봇 토큰을 입력하세요.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not config.security.allowed_users:
        logger.error("allowed_users_not_set")
        print(
            "오류: ALLOWED_TELEGRAM_USERS가 비어 있습니다.\n"
            "private chat에서 허용할 사용자 ID를 .env에 설정하세요.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── 모듈 초기화 (의존성 순서) ──

    # 1. 보안 (의존성 없음)
    security = SecurityManager(config.security)
    logger.info("security_initialized", allowed_users=len(config.security.allowed_users))

    # 2. 메모리 (config만 의존)
    memory = MemoryManager(
        config=config.memory,
        data_dir=config.data_dir,
        max_conversation_length=config.bot.max_conversation_length,
    )
    await memory.initialize()
    try:
        pruned = await memory.prune_old_conversations()
        logger.info("memory_retention_pruned_on_start", deleted=pruned)
    except Exception as exc:
        logger.error("memory_retention_prune_failed_on_start", error=str(exc))

    # 3. Ollama 클라이언트 (config만 의존)
    ollama = OllamaClient(config.ollama)
    try:
        await ollama.initialize()
    except Exception as exc:
        logger.error("ollama_init_failed", error=str(exc))
        await memory.close()
        print(
            f"오류: Ollama 초기화 실패 ({config.ollama.host})\n"
            f"{exc}\n"
            "Ollama 실행 상태와 기본 모델 준비 상태를 확인하세요. 봇 시작을 중단합니다.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 4. 스킬 매니저 (security 의존)
    skills = SkillManager(security=security, skills_dir="skills")
    skill_count = await skills.load_skills()
    logger.info("skills_loaded", count=skill_count)

    # 5. 엔진 (ollama, memory, skills 의존)
    engine = Engine(
        config=config,
        ollama=ollama,
        memory=memory,
        skills=skills,
    )

    # 6. 텔레그램 핸들러 (engine, security 의존)
    telegram = TelegramHandler(
        config=config,
        engine=engine,
        security=security,
    )

    # 7. 자동화 스케줄러 (engine, telegram 의존)
    scheduler = AutoScheduler(
        config=config,
        security=security,
        auto_dir="auto",
    )
    scheduler.set_dependencies(engine=engine, telegram=telegram)
    scheduler.register_callable("orca_progress_report", generate_orca_progress_report)
    telegram.set_scheduler(scheduler)
    auto_count = await scheduler.load_automations()
    logger.info("automations_loaded", count=auto_count)

    # ── 텔레그램 Application 초기화 ──
    app = await telegram.initialize()

    # ── 실행 ──
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("shutdown_signal_received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows에서는 signal handler가 제한적
            pass

    async with app:
        await app.start()
        scheduler.start()
        memory_maintenance_task = asyncio.create_task(
            _memory_maintenance_loop(memory, logger),
            name="memory_maintenance",
        )

        logger.info(
            "bot_running",
            model=config.ollama.model,
            skills=skill_count,
            automations=auto_count,
        )

        await app.updater.start_polling(
            poll_interval=config.telegram.polling_interval,
            drop_pending_updates=True,
        )

        # 종료 시그널 대기
        await stop_event.wait()

        # ── 정리 (역순) ──
        logger.info("shutting_down")
        scheduler.stop()
        memory_maintenance_task.cancel()
        await asyncio.gather(memory_maintenance_task, return_exceptions=True)
        await app.updater.stop()
        await app.stop()
        await memory.close()
        await ollama.close()
        logger.info("shutdown_complete")


def main() -> None:
    """동기 진입점."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
