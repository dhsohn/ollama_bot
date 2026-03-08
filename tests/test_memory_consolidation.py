"""메모리 통합(압축) 자동화 callable 테스트."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from core.auto_scheduler import AutoAction, AutoDefinition, AutoScheduler
from core.automation_callables import (
    _CONSOLIDATION_MERGE_SCHEMA,
    register_builtin_callables,
)
from core.config import (
    AppSettings,
    BotConfig,
    LemonadeConfig,
    MemoryConfig,
    SecurityConfig,
    TelegramConfig,
)
from core.memory import MemoryManager
from core.security import SecurityManager

# ── fixtures ──


@pytest.fixture
def app_settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        data_dir=str(tmp_path),
        bot=BotConfig(),
        lemonade=LemonadeConfig(),
        security=SecurityConfig(allowed_users=[111]),
        memory=MemoryConfig(),
        telegram=TelegramConfig(bot_token="test_token"),
    )


@pytest.fixture
def security_manager() -> SecurityManager:
    return SecurityManager(SecurityConfig(allowed_users=[111], rate_limit=30))


@pytest_asyncio.fixture
async def memory_manager(tmp_path: Path):
    mm = MemoryManager(MemoryConfig(), str(tmp_path), max_conversation_length=20)
    await mm.initialize()
    yield mm
    await mm.close()


@pytest.fixture
def scheduler(
    app_settings: AppSettings,
    security_manager: SecurityManager,
    tmp_path: Path,
) -> AutoScheduler:
    auto_dir = tmp_path / "auto"
    (auto_dir / "_builtin").mkdir(parents=True)
    (auto_dir / "custom").mkdir(parents=True)
    return AutoScheduler(
        config=app_settings,
        security=security_manager,
        auto_dir=str(auto_dir),
    )


# ── helpers ──


def _setup(scheduler, engine, memory_manager, app_settings):
    scheduler.set_dependencies(engine=engine, telegram=AsyncMock())
    register_builtin_callables(
        scheduler=scheduler,
        engine=engine,
        memory=memory_manager,
        allowed_users=app_settings.security.allowed_users,
        data_dir=app_settings.data_dir,
    )


def _auto(**overrides) -> AutoDefinition:
    params = {
        "min_entries_per_category": 5,
        "max_llm_calls": 3,
        "max_entries_per_merge": 8,
    }
    params.update(overrides)
    return AutoDefinition(
        name="memory_consolidation",
        description="메모리 통합",
        schedule="0 4 * * sun",
        action=AutoAction(
            type="callable",
            target="memory_consolidation",
            parameters=params,
        ),
    )


class TestMemoryConsolidation:
    # ── 1. 메모리 없음 ──

    @pytest.mark.asyncio
    async def test_no_memories_returns_empty(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        engine = AsyncMock()
        _setup(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_auto())

        assert result == ""
        engine.process_prompt.assert_not_awaited()

    # ── 2. 임계값 미만 ──

    @pytest.mark.asyncio
    async def test_below_threshold_skips(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        for i in range(3):
            await memory_manager.store_memory(111, f"k{i}", f"v{i}", "preferences")

        engine = AsyncMock()
        _setup(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_auto())

        assert result == ""
        engine.process_prompt.assert_not_awaited()

    # ── 3. 정상 통합 ──

    @pytest.mark.asyncio
    async def test_consolidates_related_entries(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        await memory_manager.store_memory(111, "food_pizza", "피자를 좋아함", "preferences")
        await memory_manager.store_memory(111, "food_pasta", "파스타를 좋아함", "preferences")
        await memory_manager.store_memory(111, "food_italian", "이탈리안 음식 선호", "preferences")
        await memory_manager.store_memory(111, "color_blue", "파란색 좋아함", "preferences")
        await memory_manager.store_memory(111, "pet_cat", "고양이를 키움", "preferences")

        llm_response = json.dumps([{
            "merge_keys": ["food_pizza", "food_pasta", "food_italian"],
            "new_key": "food_preferences",
            "new_value": "이탈리안 음식 선호 (피자, 파스타 등)",
        }], ensure_ascii=False)

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value=llm_response)
        _setup(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_auto())

        assert "통합 그룹: 1건" in result
        assert "제거된 항목: 3건" in result
        assert "생성된 항목: 1건" in result
        assert "순감소: 2건" in result

        remaining = await memory_manager.recall_memory(111)
        keys = {m["key"] for m in remaining}
        assert "food_preferences" in keys
        assert "food_pizza" not in keys
        assert "food_pasta" not in keys
        assert "food_italian" not in keys
        assert "color_blue" in keys
        assert "pet_cat" in keys
        assert len(remaining) == 3

    # ── 4. 병합 대상 외 항목 보존 ──

    @pytest.mark.asyncio
    async def test_preserves_non_merged_entries(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        for i in range(6):
            await memory_manager.store_memory(111, f"key_{i}", f"val_{i}", "preferences")

        llm_response = json.dumps([{
            "merge_keys": ["key_0", "key_1"],
            "new_key": "merged_01",
            "new_value": "val_0 + val_1",
        }], ensure_ascii=False)

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value=llm_response)
        _setup(scheduler, engine, memory_manager, app_settings)

        await scheduler._run_action(_auto())

        remaining = await memory_manager.recall_memory(111)
        keys = {m["key"] for m in remaining}
        assert "merged_01" in keys
        for i in range(2, 6):
            assert f"key_{i}" in keys
        assert len(remaining) == 5  # 6 - 2 + 1

    # ── 5. max_llm_calls=0 ──

    @pytest.mark.asyncio
    async def test_zero_llm_budget_skips(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        for i in range(5):
            await memory_manager.store_memory(111, f"k{i}", f"v{i}", "preferences")

        engine = AsyncMock()
        _setup(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_auto(max_llm_calls=0))

        assert result == ""
        engine.process_prompt.assert_not_awaited()

    # ── 6. JSON 파싱 실패 ──

    @pytest.mark.asyncio
    async def test_json_parse_failure_no_crash(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        for i in range(5):
            await memory_manager.store_memory(111, f"k{i}", f"v{i}", "preferences")

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value="not valid json")
        _setup(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_auto())

        assert result == ""
        remaining = await memory_manager.recall_memory(111)
        assert len(remaining) == 5

    # ── 7. 존재하지 않는 merge_key ──

    @pytest.mark.asyncio
    async def test_nonexistent_merge_key_ignored(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        for i in range(5):
            await memory_manager.store_memory(111, f"k{i}", f"v{i}", "preferences")

        llm_response = json.dumps([{
            "merge_keys": ["k0", "nonexistent"],
            "new_key": "merged",
            "new_value": "merged value",
        }], ensure_ascii=False)

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value=llm_response)
        _setup(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_auto())

        assert result == ""
        remaining = await memory_manager.recall_memory(111)
        assert len(remaining) == 5

    # ── 8. merge_keys 1개 ──

    @pytest.mark.asyncio
    async def test_single_merge_key_ignored(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        for i in range(5):
            await memory_manager.store_memory(111, f"k{i}", f"v{i}", "preferences")

        llm_response = json.dumps([{
            "merge_keys": ["k0"],
            "new_key": "merged",
            "new_value": "merged value",
        }], ensure_ascii=False)

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value=llm_response)
        _setup(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_auto())

        assert result == ""
        remaining = await memory_manager.recall_memory(111)
        assert len(remaining) == 5

    # ── 9. 카테고리 2개, 예산 1 ──

    @pytest.mark.asyncio
    async def test_budget_exhaustion_stops_processing(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        for i in range(5):
            await memory_manager.store_memory(111, f"a{i}", f"va{i}", "alpha")
        for i in range(5):
            await memory_manager.store_memory(111, f"b{i}", f"vb{i}", "beta")

        llm_response = json.dumps([{
            "merge_keys": ["a0", "a1"],
            "new_key": "a_merged",
            "new_value": "merged alpha",
        }], ensure_ascii=False)

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value=llm_response)
        _setup(scheduler, engine, memory_manager, app_settings)

        await scheduler._run_action(_auto(max_llm_calls=1))

        engine.process_prompt.assert_awaited_once()
        alpha = await memory_manager.recall_memory(111, category="alpha")
        beta = await memory_manager.recall_memory(111, category="beta")
        assert any(m["key"] == "a_merged" for m in alpha)
        assert len(beta) == 5

    # ── 10. new_key가 기존 비병합 키와 충돌 ──

    @pytest.mark.asyncio
    async def test_new_key_collision_with_non_merge_key_skips(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        await memory_manager.store_memory(111, "important", "중요한 데이터", "preferences")
        await memory_manager.store_memory(111, "k1", "v1", "preferences")
        await memory_manager.store_memory(111, "k2", "v2", "preferences")
        await memory_manager.store_memory(111, "k3", "v3", "preferences")
        await memory_manager.store_memory(111, "k4", "v4", "preferences")

        llm_response = json.dumps([{
            "merge_keys": ["k1", "k2"],
            "new_key": "important",
            "new_value": "덮어쓰기 시도",
        }], ensure_ascii=False)

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value=llm_response)
        _setup(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_auto())

        assert result == ""
        remaining = await memory_manager.recall_memory(111)
        keys = {m["key"] for m in remaining}
        assert "important" in keys
        assert "k1" in keys
        assert "k2" in keys
        original = next(m for m in remaining if m["key"] == "important")
        assert original["value"] == "중요한 데이터"

    # ── 11. 겹치는 merge 그룹 ──

    @pytest.mark.asyncio
    async def test_overlapping_merge_groups_only_first_applied(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        for i in range(5):
            await memory_manager.store_memory(111, f"k{i}", f"v{i}", "preferences")

        llm_response = json.dumps([
            {
                "merge_keys": ["k0", "k1"],
                "new_key": "merged_first",
                "new_value": "first merge",
            },
            {
                "merge_keys": ["k1", "k2"],
                "new_key": "merged_second",
                "new_value": "second merge",
            },
        ], ensure_ascii=False)

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value=llm_response)
        _setup(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_auto())

        assert "통합 그룹: 1건" in result
        remaining = await memory_manager.recall_memory(111)
        keys = {m["key"] for m in remaining}
        assert "merged_first" in keys
        assert "merged_second" not in keys
        assert "k2" in keys

    # ── 12. category 보존 ──

    @pytest.mark.asyncio
    async def test_merged_entry_preserves_category(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        for i in range(5):
            await memory_manager.store_memory(
                111, f"item{i}", f"val{i}", "my_category",
            )

        llm_response = json.dumps([{
            "merge_keys": ["item0", "item1"],
            "new_key": "item_merged",
            "new_value": "merged value",
        }], ensure_ascii=False)

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value=llm_response)
        _setup(scheduler, engine, memory_manager, app_settings)

        await scheduler._run_action(_auto())

        remaining = await memory_manager.recall_memory(111, category="my_category")
        merged = next((m for m in remaining if m["key"] == "item_merged"), None)
        assert merged is not None
        assert merged["category"] == "my_category"

    # ── 13. 파라미터 검증 ──

    @pytest.mark.asyncio
    async def test_invalid_min_entries_raises(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        engine = AsyncMock()
        _setup(scheduler, engine, memory_manager, app_settings)

        with pytest.raises(ValueError, match="min_entries_per_category"):
            await scheduler._run_action(_auto(min_entries_per_category=1))

    @pytest.mark.asyncio
    async def test_invalid_max_llm_calls_raises(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        engine = AsyncMock()
        _setup(scheduler, engine, memory_manager, app_settings)

        with pytest.raises(ValueError, match="max_llm_calls"):
            await scheduler._run_action(_auto(max_llm_calls=-1))

    @pytest.mark.asyncio
    async def test_invalid_max_entries_per_merge_raises(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        engine = AsyncMock()
        _setup(scheduler, engine, memory_manager, app_settings)

        with pytest.raises(ValueError, match="max_entries_per_merge"):
            await scheduler._run_action(_auto(max_entries_per_merge=1))

    # ── 14. schema 전달 확인 ──

    @pytest.mark.asyncio
    async def test_schema_format_passed_to_engine(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        for i in range(5):
            await memory_manager.store_memory(111, f"k{i}", f"v{i}", "preferences")

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value="[]")
        _setup(scheduler, engine, memory_manager, app_settings)

        await scheduler._run_action(_auto())

        engine.process_prompt.assert_awaited_once()
        call_kwargs = engine.process_prompt.await_args.kwargs
        assert call_kwargs["response_format"] is _CONSOLIDATION_MERGE_SCHEMA
        assert call_kwargs["max_tokens"] == 768
        assert call_kwargs["temperature"] == 0.2

    # ── 15. new_key가 기존 merge_key인 경우 생성 카운트 미증가 + 결과 출력 ──

    @pytest.mark.asyncio
    async def test_existing_new_key_counts_as_update_not_creation(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        await memory_manager.store_memory(111, "food_pizza", "피자", "preferences")
        await memory_manager.store_memory(111, "food_pasta", "파스타", "preferences")
        await memory_manager.store_memory(111, "food_italian", "이탈리안", "preferences")
        await memory_manager.store_memory(111, "k3", "v3", "preferences")
        await memory_manager.store_memory(111, "k4", "v4", "preferences")

        llm_response = json.dumps([{
            "merge_keys": ["food_pizza", "food_pasta", "food_italian"],
            "new_key": "food_pizza",
            "new_value": "이탈리안 선호(피자, 파스타)",
        }], ensure_ascii=False)

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value=llm_response)
        _setup(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_auto())

        assert "통합 그룹: 1건" in result
        assert "제거된 항목: 2건" in result
        assert "생성된 항목: 0건" in result
        assert "순감소: 2건" in result

        remaining = await memory_manager.recall_memory(111)
        keys = {m["key"] for m in remaining}
        assert "food_pizza" in keys
        assert "food_pasta" not in keys
        assert "food_italian" not in keys

    # ── 16. 코드펜스 JSON 파싱 ──

    @pytest.mark.asyncio
    async def test_code_fenced_json_is_parsed(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        for i in range(5):
            await memory_manager.store_memory(111, f"k{i}", f"v{i}", "preferences")

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(
            return_value='```json\n[{"merge_keys":["k0","k1"],"new_key":"merged","new_value":"v0+v1"}]\n```',
        )
        _setup(scheduler, engine, memory_manager, app_settings)

        result = await scheduler._run_action(_auto())

        assert "통합 그룹: 1건" in result
        remaining = await memory_manager.recall_memory(111)
        keys = {m["key"] for m in remaining}
        assert "merged" in keys
        assert "k0" not in keys
        assert "k1" not in keys

    # ── 17. 저장 실패 시 삭제 미실행 ──

    @pytest.mark.asyncio
    async def test_store_failure_does_not_delete_original_entries(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        for i in range(5):
            await memory_manager.store_memory(111, f"k{i}", f"v{i}", "preferences")

        llm_response = json.dumps([{
            "merge_keys": ["k0", "k1"],
            "new_key": "merged",
            "new_value": "v0+v1",
        }], ensure_ascii=False)

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value=llm_response)
        _setup(scheduler, engine, memory_manager, app_settings)

        original_store = memory_manager.store_memory

        async def _failing_store(chat_id, key, value, category="general"):
            if key == "merged":
                raise RuntimeError("store failed")
            return await original_store(chat_id, key, value, category)

        memory_manager.store_memory = AsyncMock(side_effect=_failing_store)  # type: ignore[method-assign]
        try:
            with pytest.raises(RuntimeError, match="store failed"):
                await scheduler._run_action(_auto())
        finally:
            memory_manager.store_memory = original_store  # type: ignore[method-assign]

        remaining = await memory_manager.recall_memory(111)
        keys = {m["key"] for m in remaining}
        assert keys == {"k0", "k1", "k2", "k3", "k4"}

    # ── 18. 긴 key는 truncate 없이 프롬프트에 전달 ──

    @pytest.mark.asyncio
    async def test_long_key_is_not_truncated_in_prompt(
        self, scheduler, app_settings, memory_manager,
    ) -> None:
        long_key = "k" * 120
        await memory_manager.store_memory(111, long_key, "v0", "preferences")
        for i in range(1, 5):
            await memory_manager.store_memory(111, f"k{i}", f"v{i}", "preferences")

        engine = AsyncMock()
        engine.process_prompt = AsyncMock(return_value="[]")
        _setup(scheduler, engine, memory_manager, app_settings)

        await scheduler._run_action(_auto())

        call_kwargs = engine.process_prompt.await_args.kwargs
        prompt = call_kwargs["prompt"]
        assert f"key={long_key}" in prompt
