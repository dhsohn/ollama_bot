"""ollama_bot 진입점.

모든 모듈을 의존성 순서대로 초기화하고
텔레그램 폴링과 자동화 스케줄러를 실행한다.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
import os
import random
import signal
import sys
from pathlib import Path
from typing import Any

import aiosqlite

from core.auto_evaluator import AutoEvaluator
from core.automation_callables import register_builtin_callables
from core.auto_scheduler import AutoScheduler
from core.config import AppSettings, load_config
from core.context_compressor import ContextCompressor
from core.engine import Engine
from core.feedback_manager import FeedbackManager
from core.instant_responder import InstantResponder
from core.logging_setup import get_logger, setup_logging
from core.memory import MemoryManager
from core.ollama_client import OllamaClient
from core.security import SecurityManager
from core.skill_manager import SkillManager
from core.telegram_handler import TelegramHandler

_MEMORY_MAINTENANCE_INTERVAL_SECONDS = 6 * 60 * 60
_OLLAMA_RECOVERY_INTERVAL_SECONDS = 60
_CACHE_PRUNE_INTERVAL_SECONDS = 3600
_MEMORY_MAINTENANCE_JITTER_RATIO = 0.1
_ALLOW_LOCAL_RUN_ENV = "ALLOW_LOCAL_RUN"


class StartupError(RuntimeError):
    """초기화 실패 시 사용자 노출용 메시지를 전달한다."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass
class RuntimeState:
    """초기화된 런타임 의존성을 묶어 관리한다."""

    config: AppSettings
    logger: Any
    memory: MemoryManager
    ollama: OllamaClient
    app: Any
    scheduler: AutoScheduler
    skill_count: int
    auto_count: int
    cleanup_stack: AsyncExitStack
    feedback: FeedbackManager | None = None
    auto_evaluator: AutoEvaluator | None = None
    semantic_cache: Any = None  # SemanticCache | None


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


def _is_truthy(value: str | None) -> bool:
    """환경변수 문자열을 bool로 변환한다."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


async def _open_sqlite_db(path: Path) -> aiosqlite.Connection:
    """WAL 모드 SQLite 연결을 생성한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(str(path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.commit()
    return db


async def _memory_maintenance_loop(
    memory: MemoryManager,
    logger,
    interval_seconds: int = _MEMORY_MAINTENANCE_INTERVAL_SECONDS,
    jitter_ratio: float = _MEMORY_MAINTENANCE_JITTER_RATIO,
    feedback: FeedbackManager | None = None,
    feedback_retention_days: int | None = None,
    semantic_cache: Any = None,
) -> None:
    """주기적으로 오래된 대화/피드백/캐시 데이터를 정리한다."""
    while True:
        try:
            deleted = await memory.prune_old_conversations()
            logger.debug("memory_retention_pruned", deleted=deleted)
        except Exception as exc:
            logger.error("memory_retention_prune_failed", error=str(exc))

        if feedback is not None and feedback_retention_days is not None:
            try:
                fb_deleted = await feedback.prune_old_feedback(feedback_retention_days)
                logger.debug("feedback_retention_pruned", deleted=fb_deleted)
            except Exception as exc:
                logger.error("feedback_retention_prune_failed", error=str(exc))

            try:
                eval_deleted = await feedback.prune_old_auto_evaluations(feedback_retention_days)
                if eval_deleted:
                    logger.debug("auto_eval_retention_pruned", deleted=eval_deleted)
            except Exception as exc:
                logger.error("auto_eval_retention_prune_failed", error=str(exc))

        # 시맨틱 캐시 TTL 정리
        if semantic_cache is not None:
            try:
                cache_pruned = await semantic_cache.prune_expired()
                if cache_pruned:
                    logger.debug("semantic_cache_pruned", deleted=cache_pruned)
            except Exception as exc:
                logger.error("semantic_cache_prune_failed", error=str(exc))

        base_interval = max(1, interval_seconds)
        jitter = random.uniform(0.0, base_interval * max(0.0, jitter_ratio))
        await asyncio.sleep(base_interval + jitter)


async def _ollama_recovery_loop(
    ollama: OllamaClient,
    logger,
    interval_seconds: int = _OLLAMA_RECOVERY_INTERVAL_SECONDS,
) -> None:
    """Ollama 상태를 주기 점검하고 연결 장애 시 재연결을 시도한다."""
    while True:
        try:
            health = await ollama.health_check(attempt_recovery=False)
            if health.get("status") != "ok":
                recovered = await ollama.recover_connection(force=True)
                if recovered:
                    logger.info("ollama_recovered_by_loop")
                else:
                    logger.warning(
                        "ollama_still_unhealthy",
                        error=health.get("error"),
                    )
        except Exception as exc:
            logger.error("ollama_recovery_loop_failed", error=str(exc))

        await asyncio.sleep(interval_seconds)


def _validate_required_settings(config: AppSettings, logger: Any) -> None:
    """필수 런타임 설정을 검사한다."""
    if (
        not config.telegram_bot_token
        or config.telegram_bot_token == "your_telegram_bot_token_here"
    ):
        logger.error("telegram_bot_token_not_set")
        raise StartupError(
            "오류: TELEGRAM_BOT_TOKEN이 설정되지 않았습니다.\n"
            ".env 파일에 유효한 텔레그램 봇 토큰을 입력하세요."
        )

    if not config.security.allowed_users:
        logger.error("allowed_users_not_set")
        raise StartupError(
            "오류: ALLOWED_TELEGRAM_USERS가 비어 있습니다.\n"
            "private chat에서 허용할 사용자 ID를 .env에 설정하세요."
        )


async def _build_runtime(config: AppSettings, logger: Any) -> RuntimeState:
    """의존성 순서대로 모듈을 초기화하고 런타임 상태를 반환한다."""
    cleanup_stack = AsyncExitStack()
    try:
        # 1. 보안
        security = SecurityManager(config.security)
        logger.info(
            "security_initialized",
            allowed_users=len(config.security.allowed_users),
        )

        # 2. 메모리
        memory = MemoryManager(
            config=config.memory,
            data_dir=config.data_dir,
            max_conversation_length=config.bot.max_conversation_length,
            archive_enabled=config.context_compressor.enabled and config.context_compressor.archive_enabled,
        )
        await memory.initialize()
        cleanup_stack.push_async_callback(memory.close)
        try:
            pruned = await memory.prune_old_conversations()
            logger.info("memory_retention_pruned_on_start", deleted=pruned)
        except Exception as exc:
            logger.error("memory_retention_prune_failed_on_start", error=str(exc))

        # 2.5. 피드백 매니저
        feedback: FeedbackManager | None = None
        if config.feedback.enabled:
            feedback_db_path = Path(config.data_dir) / "memory" / "feedback.db"
            feedback_db = await _open_sqlite_db(feedback_db_path)
            cleanup_stack.push_async_callback(feedback_db.close)
            feedback = FeedbackManager(feedback_db)
            await feedback.initialize_schema()
            try:
                fb_pruned = await feedback.prune_old_feedback(config.feedback.retention_days)
                if fb_pruned:
                    logger.info("feedback_pruned", count=fb_pruned)
            except Exception as exc:
                logger.error("feedback_prune_failed", error=str(exc))

        # 3. Ollama
        ollama = OllamaClient(config.ollama)
        try:
            await ollama.initialize()
        except Exception as exc:
            logger.error("ollama_init_failed", error=str(exc))
            raise StartupError(
                f"오류: Ollama 초기화 실패 ({config.ollama.host})\n"
                f"{exc}\n"
                "Ollama 실행 상태와 기본 모델 준비 상태를 확인하세요. 봇 시작을 중단합니다."
            ) from exc
        cleanup_stack.push_async_callback(ollama.close)

        # 4. 스킬 매니저
        skills = SkillManager(security=security, skills_dir="skills")
        try:
            skill_count = await skills.load_skills(strict=True)
        except Exception as exc:
            logger.error("skills_init_failed", error=str(exc))
            raise StartupError(
                f"오류: 스킬 로드 실패\n{exc}\n"
                "중복 이름/트리거 또는 YAML 형식을 확인하세요."
            ) from exc
        logger.info("skills_loaded", count=skill_count)
        skill_load_errors = skills.get_last_load_errors()
        if skill_load_errors:
            logger.warning(
                "skills_loaded_with_partial_failures",
                error_count=len(skill_load_errors),
                sample=skill_load_errors[:3],
            )

        # 4.5. 속도 최적화 컴포넌트 초기화
        instant_responder = None
        semantic_cache = None
        intent_router = None
        context_compressor = None

        # 즉시 응답
        if config.instant_responder.enabled:
            instant_responder = InstantResponder(
                rules_path=config.instant_responder.rules_path,
            )
            logger.info("instant_responder_initialized", rules=instant_responder.rules_count)

        # 시맨틱 캐시
        if config.semantic_cache.enabled:
            cache_db: aiosqlite.Connection | None = None
            try:
                from core.semantic_cache import SemanticCache

                cache_db_path = Path(config.data_dir) / "memory" / "cache.db"
                cache_db = await _open_sqlite_db(cache_db_path)
                semantic_cache = SemanticCache(
                    db=cache_db,
                    model_name=config.semantic_cache.model_name,
                    embedding_device=config.semantic_cache.embedding_device,
                    similarity_threshold=config.semantic_cache.similarity_threshold,
                    max_entries=config.semantic_cache.max_entries,
                    ttl_hours=config.semantic_cache.ttl_hours,
                    min_query_chars=config.semantic_cache.min_query_chars,
                    exclude_patterns=config.semantic_cache.exclude_patterns,
                )
                await semantic_cache.initialize()
                cleanup_stack.push_async_callback(cache_db.close)
                logger.info("semantic_cache_initialized", enabled=semantic_cache.enabled)
            except Exception as exc:
                if cache_db is not None:
                    await cache_db.close()
                logger.warning("semantic_cache_init_failed", error=str(exc))
                semantic_cache = None

        # 인텐트 라우터
        if config.intent_router.enabled:
            try:
                from core.intent_router import IntentRouter

                shared_encoder = (
                    semantic_cache.encoder
                    if semantic_cache is not None and semantic_cache.enabled
                    else None
                )
                intent_router = await asyncio.to_thread(
                    IntentRouter,
                    routes_path=config.intent_router.routes_path,
                    encoder_model=config.intent_router.encoder_model,
                    min_confidence=config.intent_router.min_confidence,
                    encoder=shared_encoder,
                )
                logger.info(
                    "intent_router_initialized",
                    enabled=intent_router.enabled,
                    routes=intent_router.routes_count,
                )
            except Exception as exc:
                logger.warning("intent_router_init_failed", error=str(exc))
                intent_router = None

        # 컨텍스트 압축
        if config.context_compressor.enabled:
            context_compressor = ContextCompressor(
                ollama=ollama,
                memory=memory,
                recent_keep=config.context_compressor.recent_keep,
                summary_refresh_interval=config.context_compressor.summary_refresh_interval,
                summary_max_tokens=config.context_compressor.summary_max_tokens,
                summarize_concurrency=config.context_compressor.summarize_concurrency,
            )
            logger.info("context_compressor_initialized")

        # 5. 엔진
        engine = Engine(
            config=config,
            ollama=ollama,
            memory=memory,
            skills=skills,
            feedback_manager=feedback,
            instant_responder=instant_responder,
            semantic_cache=semantic_cache,
            intent_router=intent_router,
            context_compressor=context_compressor,
        )

        # 5.5. 자동 평가
        auto_evaluator: AutoEvaluator | None = None
        if config.auto_evaluation.enabled and feedback is not None:
            auto_evaluator = AutoEvaluator(
                config=config.auto_evaluation,
                ollama=ollama,
                feedback_manager=feedback,
                timezone_name=config.scheduler.timezone,
            )
            logger.info("auto_evaluator_initialized")

        # 6. 텔레그램
        telegram = TelegramHandler(
            config=config,
            engine=engine,
            security=security,
            feedback=feedback,
            auto_evaluator=auto_evaluator,
            semantic_cache=semantic_cache,
        )

        # 7. 자동화
        try:
            scheduler = AutoScheduler(
                config=config,
                security=security,
                auto_dir="auto",
            )
        except Exception as exc:
            logger.error("scheduler_init_failed", error=str(exc))
            raise StartupError(
                f"오류: 자동화 스케줄러 초기화 실패\n{exc}\n"
                "SCHEDULER_TIMEZONE 설정을 확인하세요."
            ) from exc

        # 순환 의존 방지를 위해 런타임 주입 순서를 고정한다.
        # 1) scheduler.set_dependencies
        # 2) register_builtin_callables
        # 3) telegram.set_scheduler
        scheduler.set_dependencies(engine=engine, telegram=telegram)
        assert scheduler.dependencies_ready(), (
            "Scheduler dependencies must be wired before automation loading."
        )
        register_builtin_callables(
            scheduler=scheduler,
            engine=engine,
            memory=memory,
            allowed_users=config.security.allowed_users,
            data_dir=config.data_dir,
            feedback=feedback,
        )
        telegram.set_scheduler(scheduler)
        assert telegram.has_scheduler(), (
            "Telegram handler must receive scheduler before initialization."
        )

        try:
            auto_count = await scheduler.load_automations(strict=True)
        except Exception as exc:
            logger.error("automations_init_failed", error=str(exc))
            raise StartupError(
                f"오류: 자동화 로드 실패\n{exc}\n"
                "중복 이름 또는 YAML 형식을 확인하세요."
            ) from exc
        logger.info("automations_loaded", count=auto_count)
        auto_load_errors = scheduler.get_last_load_errors()
        if auto_load_errors:
            logger.warning(
                "automations_loaded_with_partial_failures",
                error_count=len(auto_load_errors),
                sample=auto_load_errors[:3],
            )

        app = await telegram.initialize()

        return RuntimeState(
            config=config,
            logger=logger,
            memory=memory,
            ollama=ollama,
            app=app,
            scheduler=scheduler,
            skill_count=skill_count,
            auto_count=auto_count,
            cleanup_stack=cleanup_stack,
            feedback=feedback,
            auto_evaluator=auto_evaluator,
            semantic_cache=semantic_cache,
        )
    except Exception:
        await cleanup_stack.aclose()
        raise


async def _shutdown_runtime(
    runtime: RuntimeState,
    memory_maintenance_task: asyncio.Task | None,
    ollama_recovery_task: asyncio.Task | None,
    scheduler_started: bool,
    app_started: bool,
    updater_started: bool,
) -> None:
    """실행 중 자원을 역순으로 정리한다."""
    logger = runtime.logger
    app = runtime.app
    logger.info("shutting_down")

    if scheduler_started:
        try:
            runtime.scheduler.stop()
        except Exception as exc:
            logger.error("scheduler_stop_failed", error=str(exc))

    if runtime.auto_evaluator is not None:
        try:
            await runtime.auto_evaluator.shutdown()
        except Exception as exc:
            logger.error("auto_evaluator_shutdown_failed", error=str(exc))

    if memory_maintenance_task is not None:
        memory_maintenance_task.cancel()
        await asyncio.gather(memory_maintenance_task, return_exceptions=True)

    if ollama_recovery_task is not None:
        ollama_recovery_task.cancel()
        await asyncio.gather(ollama_recovery_task, return_exceptions=True)

    if updater_started:
        try:
            await app.updater.stop()
        except Exception as exc:
            logger.error("updater_stop_failed", error=str(exc))

    if app_started:
        try:
            await app.stop()
        except Exception as exc:
            logger.error("app_stop_failed", error=str(exc))

    try:
        await runtime.cleanup_stack.aclose()
    except Exception as exc:
        logger.error("resource_cleanup_failed", error=str(exc))

    logger.info("shutdown_complete")


async def async_main() -> None:
    """비동기 메인 루프."""
    if not _is_running_in_container() and not _is_truthy(os.environ.get(_ALLOW_LOCAL_RUN_ENV)):
        print(
            "오류: 이 애플리케이션은 Docker 컨테이너에서만 실행됩니다.\n"
            "실행 방법: docker compose up --build -d\n"
            f"로컬 실행 우회: {_ALLOW_LOCAL_RUN_ENV}=1",
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

    log_dir = str(Path(config.data_dir) / "logs")
    setup_logging(config.log_level, log_dir=log_dir)
    logger = get_logger("main")
    logger.info("starting", version="0.1.0")

    try:
        _validate_required_settings(config, logger)
        runtime = await _build_runtime(config, logger)
    except StartupError as exc:
        print(exc.message, file=sys.stderr)
        sys.exit(1)

    # ── 실행 ──
    stop_event = asyncio.Event()
    app = runtime.app

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

    memory_maintenance_task: asyncio.Task | None = None
    ollama_recovery_task: asyncio.Task | None = None
    scheduler_started = False
    app_started = False
    updater_started = False

    async with app:
        try:
            await app.start()
            app_started = True

            runtime.scheduler.start()
            scheduler_started = True

            memory_maintenance_task = asyncio.create_task(
                _memory_maintenance_loop(
                    runtime.memory,
                    logger,
                    feedback=runtime.feedback,
                    feedback_retention_days=runtime.config.feedback.retention_days,
                    semantic_cache=runtime.semantic_cache,
                ),
                name="memory_maintenance",
            )
            ollama_recovery_task = asyncio.create_task(
                _ollama_recovery_loop(runtime.ollama, logger),
                name="ollama_recovery",
            )

            logger.info(
                "bot_running",
                model=runtime.config.ollama.model,
                skills=runtime.skill_count,
                automations=runtime.auto_count,
            )

            await app.updater.start_polling(
                poll_interval=runtime.config.telegram.polling_interval,
                drop_pending_updates=True,
            )
            updater_started = True

            # 종료 시그널 대기
            await stop_event.wait()
        finally:
            await _shutdown_runtime(
                runtime=runtime,
                memory_maintenance_task=memory_maintenance_task,
                ollama_recovery_task=ollama_recovery_task,
                scheduler_started=scheduler_started,
                app_started=app_started,
                updater_started=updater_started,
            )


def main() -> None:
    """동기 진입점."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
