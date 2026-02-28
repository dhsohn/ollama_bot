"""여러 Lemonade 인스턴스를 하나의 클라이언트로 라우팅한다."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from core.config import LemonadeConfig, LemonadeInstanceConfig, OllamaConfig
from core.lemonade_client import LemonadeClient, LemonadeClientError
from core.logging_setup import get_logger
from core.llm_types import ChatResponse, ChatStreamState

_PRIMARY_INSTANCE = "primary"


class LemonadeMultiClient:
    """모델/역할 기준으로 Lemonade 인스턴스를 분기하는 래퍼."""

    def __init__(
        self,
        config: LemonadeConfig,
        *,
        fallback_ollama: OllamaConfig | None = None,
    ) -> None:
        self._logger = get_logger("lemonade_multi_client")
        self._default_instance = _PRIMARY_INSTANCE
        self._default_model = ""
        self._clients: dict[str, LemonadeClient] = {
            _PRIMARY_INSTANCE: LemonadeClient(
                config,
                fallback_ollama=fallback_ollama,
            )
        }
        self._instance_labels: dict[str, str] = {_PRIMARY_INSTANCE: "primary"}
        self._ready_instances: set[str] = set()
        self._static_model_routes: dict[str, str] = {}
        self._dynamic_model_routes: dict[str, str] = {}
        # 임베딩/리랭커는 기본적으로 primary 인스턴스에 고정한다.
        self._role_routes: dict[str, str] = {
            "embedding": _PRIMARY_INSTANCE,
            "reranker": _PRIMARY_INSTANCE,
        }

        for instance in config.instances:
            key = self._normalize_instance_key(instance.name)
            if key == _PRIMARY_INSTANCE:
                raise ValueError("lemonade instance name 'primary' is reserved")

            instance_config = LemonadeConfig(
                host=instance.host,
                api_key=instance.api_key,
                base_path=instance.base_path,
                timeout_seconds=instance.timeout_seconds,
                model_load_timeout_seconds=instance.model_load_timeout_seconds,
                heavy_model_load_timeout_seconds=instance.heavy_model_load_timeout_seconds,
            )
            self._clients[key] = LemonadeClient(
                instance_config,
                fallback_ollama=fallback_ollama,
            )
            self._instance_labels[key] = instance.name
            self._register_static_routes(key, instance)

    @staticmethod
    def _normalize_instance_key(name: str) -> str:
        return name.strip().lower()

    @staticmethod
    def _normalize_role(role: str | None) -> str:
        if role is None:
            return ""
        return role.strip().lower()

    @property
    def default_model(self) -> str:
        return self._default_model

    @default_model.setter
    def default_model(self, model: str) -> None:
        self._default_model = model.strip()

    @property
    def system_prompt(self) -> str:
        return self._clients[self._default_instance].system_prompt

    @property
    def host(self) -> str:
        hosts: list[str] = []
        for key, client in self._clients.items():
            label = self._instance_labels.get(key, key)
            hosts.append(f"{label}:{client.host}")
        return ",".join(hosts)

    def _register_static_routes(
        self,
        instance_key: str,
        instance: LemonadeInstanceConfig,
    ) -> None:
        model_names = list(instance.route_models)
        if instance.model:
            model_names.append(instance.model)
        for name in model_names:
            model = name.strip()
            if not model:
                continue
            previous = self._static_model_routes.get(model)
            if previous is not None and previous != instance_key:
                self._logger.warning(
                    "lemonade_route_model_overridden",
                    model=model,
                    previous=self._instance_labels.get(previous, previous),
                    current=self._instance_labels.get(instance_key, instance_key),
                )
            self._static_model_routes[model] = instance_key

        for role_name in instance.route_roles:
            role = self._normalize_role(role_name)
            if not role:
                continue
            previous = self._role_routes.get(role)
            if previous is not None and previous != instance_key:
                self._logger.warning(
                    "lemonade_route_role_overridden",
                    role=role,
                    previous=self._instance_labels.get(previous, previous),
                    current=self._instance_labels.get(instance_key, instance_key),
                )
            self._role_routes[role] = instance_key

    async def initialize(self) -> None:
        """모든 인스턴스 초기화. 추가 인스턴스 실패는 경고 후 유지한다."""
        primary = self._clients[self._default_instance]
        await primary.initialize()
        self._ready_instances.add(self._default_instance)

        for key, client in self._clients.items():
            if key == self._default_instance:
                continue
            try:
                await client.initialize()
                self._ready_instances.add(key)
            except Exception as exc:
                self._ready_instances.discard(key)
                self._logger.warning(
                    "lemonade_instance_init_failed",
                    instance=self._instance_labels.get(key, key),
                    host=client.host,
                    error=str(exc),
                )

        await self._refresh_dynamic_model_routes()
        self._logger.info(
            "lemonade_multi_initialized",
            instances=len(self._clients),
            ready_instances=len(self._ready_instances),
            routed_models=len(self._static_model_routes) + len(self._dynamic_model_routes),
        )

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()
        self._ready_instances.clear()
        self._dynamic_model_routes.clear()

    async def _ensure_instance_ready(self, instance_key: str) -> bool:
        """요청 대상 인스턴스가 준비되지 않은 경우 재초기화를 시도한다."""
        if instance_key in self._ready_instances:
            return True

        client = self._clients[instance_key]
        label = self._instance_labels.get(instance_key, instance_key)
        try:
            await client.initialize()
            self._ready_instances.add(instance_key)
            await self._update_dynamic_routes_for_instance(instance_key)
            self._logger.info("lemonade_instance_recovered", instance=label)
            return True
        except Exception as exc:
            self._ready_instances.discard(instance_key)
            self._logger.warning(
                "lemonade_instance_recovery_failed",
                instance=label,
                host=client.host,
                error=str(exc),
            )
            return False

    def _resolve_route_key(self, model: str | None, role: str | None) -> str:
        role_key = self._normalize_role(role)
        if role_key:
            mapped = self._role_routes.get(role_key)
            if mapped is not None:
                return mapped

        if model:
            mapped = self._static_model_routes.get(model)
            if mapped is not None:
                return mapped
            mapped = self._dynamic_model_routes.get(model)
            if mapped is not None:
                return mapped
        return self._default_instance

    async def _select_client(
        self,
        *,
        model: str | None = None,
        role: str | None = None,
    ) -> tuple[LemonadeClient, str | None, str]:
        target_model = (model or "").strip() or self._default_model or None
        route_key = self._resolve_route_key(target_model, role)
        is_model_explicit = bool((model or "").strip())

        if route_key not in self._clients:
            route_key = self._default_instance

        if route_key not in self._ready_instances:
            ready = await self._ensure_instance_ready(route_key)
            if not ready:
                label = self._instance_labels.get(route_key, route_key)
                if is_model_explicit:
                    raise LemonadeClientError(
                        f"Routed Lemonade instance '{label}' is unavailable for model '{target_model}'."
                    )
                self._logger.warning(
                    "lemonade_instance_fallback_to_primary",
                    requested_instance=label,
                    fallback=self._instance_labels.get(
                        self._default_instance,
                        self._default_instance,
                    ),
                )
                route_key = self._default_instance
                if route_key not in self._ready_instances:
                    default_ready = await self._ensure_instance_ready(route_key)
                    if not default_ready:
                        raise LemonadeClientError(
                            "Default Lemonade instance is unavailable."
                        )

        return self._clients[route_key], target_model, route_key

    async def _update_dynamic_routes_for_instance(self, instance_key: str) -> None:
        client = self._clients[instance_key]
        try:
            models = await client.list_models()
        except Exception as exc:
            self._logger.debug(
                "dynamic_route_update_failed",
                instance=self._instance_labels.get(instance_key, instance_key),
                error=str(exc),
            )
            return
        for model in models:
            name = model.get("name")
            if not isinstance(name, str) or not name:
                continue
            if name in self._static_model_routes:
                continue
            self._dynamic_model_routes[name] = instance_key

    async def _refresh_dynamic_model_routes(self) -> None:
        self._dynamic_model_routes.clear()
        for key in self._ready_instances:
            await self._update_dynamic_routes_for_instance(key)

    async def prepare_model(
        self,
        *,
        model: str | None = None,
        role: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        client, target_model, _ = await self._select_client(model=model, role=role)
        await client.prepare_model(
            model=target_model,
            role=role,
            timeout_seconds=timeout_seconds,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int = 60,
        response_format: str | dict | None = None,
    ) -> ChatResponse:
        client, target_model, route_key = await self._select_client(model=model)
        try:
            return await client.chat(
                messages=messages,
                model=target_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                response_format=response_format,
            )
        except Exception as exc:
            if route_key == self._default_instance:
                raise
            fallback_client = self._clients[self._default_instance]
            fallback_model = self._default_model or target_model
            self._logger.warning(
                "lemonade_chat_fallback_to_primary",
                failed_instance=self._instance_labels.get(route_key, route_key),
                failed_model=target_model,
                fallback_instance=self._instance_labels.get(
                    self._default_instance,
                    self._default_instance,
                ),
                fallback_model=fallback_model,
                error=str(exc),
            )
            return await fallback_client.chat(
                messages=messages,
                model=fallback_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                response_format=response_format,
            )

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int = 60,
        stream_state: ChatStreamState | None = None,
    ) -> AsyncGenerator[str, None]:
        client, target_model, route_key = await self._select_client(model=model)
        try:
            async for chunk in client.chat_stream(
                messages=messages,
                model=target_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                stream_state=stream_state,
            ):
                yield chunk
            return
        except Exception as exc:
            if route_key == self._default_instance:
                raise
            fallback_client = self._clients[self._default_instance]
            fallback_model = self._default_model or target_model
            self._logger.warning(
                "lemonade_chat_stream_fallback_to_primary",
                failed_instance=self._instance_labels.get(route_key, route_key),
                failed_model=target_model,
                fallback_instance=self._instance_labels.get(
                    self._default_instance,
                    self._default_instance,
                ),
                fallback_model=fallback_model,
                error=str(exc),
            )
            fallback_state = ChatStreamState()
            async for chunk in fallback_client.chat_stream(
                messages=messages,
                model=fallback_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                stream_state=fallback_state,
            ):
                yield chunk
            if stream_state is not None:
                stream_state.usage = fallback_state.usage

    async def list_models(self) -> list[dict]:
        merged: dict[str, dict] = {}
        for key, client in self._clients.items():
            if not await self._ensure_instance_ready(key):
                continue
            try:
                models = await client.list_models()
            except Exception as exc:
                self._ready_instances.discard(key)
                self._logger.warning(
                    "lemonade_instance_list_models_failed",
                    instance=self._instance_labels.get(key, key),
                    error=str(exc),
                )
                continue
            for model in models:
                name = model.get("name")
                if isinstance(name, str) and name and name not in merged:
                    merged[name] = model
        await self._refresh_dynamic_model_routes()
        return sorted(merged.values(), key=lambda item: str(item.get("name", "")))

    async def get_model_info(self, model: str) -> dict:
        client, target_model, _ = await self._select_client(model=model)
        if target_model is None:
            raise LemonadeClientError("Model name is required")
        return await client.get_model_info(target_model)

    async def health_check(self, *, attempt_recovery: bool = False) -> dict:
        details: dict[str, dict[str, Any]] = {}
        all_models: set[str] = set()
        degraded: list[str] = []

        for key, client in self._clients.items():
            label = self._instance_labels.get(key, key)
            if attempt_recovery:
                await self._ensure_instance_ready(key)
            if key not in self._ready_instances:
                details[label] = {
                    "status": "error",
                    "host": client.host,
                    "error": "instance_not_initialized",
                }
                if key != self._default_instance:
                    degraded.append(label)
                continue
            try:
                health = await client.health_check(attempt_recovery=attempt_recovery)
            except Exception as exc:
                health = {
                    "status": "error",
                    "host": client.host,
                    "error": str(exc),
                }
            details[label] = health
            if health.get("status") == "ok":
                self._ready_instances.add(key)
                for model in health.get("models", []):
                    if isinstance(model, str):
                        all_models.add(model)
            else:
                self._ready_instances.discard(key)
                if key != self._default_instance:
                    degraded.append(label)

        default_label = self._instance_labels.get(
            self._default_instance,
            self._default_instance,
        )
        default_health = details.get(default_label, {})
        default_ok = default_health.get("status") == "ok"
        status = "ok" if default_ok else "error"
        return {
            "status": status,
            "host": self.host,
            "models_count": len(all_models),
            "models": sorted(all_models),
            "default_model": self._default_model,
            "default_model_available": (
                True if not self._default_model else self._default_model in all_models
            ),
            "instances": details,
            "degraded_instances": degraded,
        }

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        timeout: int | None = None,
    ) -> list[list[float]]:
        client, target_model, _ = await self._select_client(
            model=model,
            role="embedding",
        )
        return await client.embed(texts, model=target_model, timeout=timeout)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        model: str | None = None,
        top_n: int | None = None,
        timeout: int | None = None,
    ) -> list[dict[str, Any]]:
        client, target_model, _ = await self._select_client(
            model=model,
            role="reranker",
        )
        return await client.rerank(
            query=query,
            documents=documents,
            model=target_model,
            top_n=top_n,
            timeout=timeout,
        )

    async def check_model_availability(self, model_names: list[str]) -> dict[str, bool]:
        availability = {name: False for name in model_names}
        if not model_names:
            return availability

        for model_name in model_names:
            try:
                client, target_model, _ = await self._select_client(model=model_name)
                if target_model is None:
                    continue
                result = await client.check_model_availability([target_model])
                availability[model_name] = bool(result.get(target_model, False))
            except LemonadeClientError as exc:
                self._logger.debug(
                    "model_availability_check_client_error",
                    model=model_name,
                    error=str(exc),
                )
                availability[model_name] = False
            except Exception as exc:
                self._logger.warning(
                    "model_availability_check_unexpected_error",
                    model=model_name,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                availability[model_name] = False
        return availability

    async def recover_connection(self, *, force: bool = False) -> bool:
        default_ok = False
        for key, client in self._clients.items():
            try:
                if key in self._ready_instances:
                    recovered = await client.recover_connection(force=force)
                else:
                    recovered = await self._ensure_instance_ready(key)
            except Exception:
                recovered = False
            if recovered:
                self._ready_instances.add(key)
            else:
                self._ready_instances.discard(key)
            if key == self._default_instance:
                default_ok = recovered or key in self._ready_instances
        if default_ok:
            await self._refresh_dynamic_model_routes()
        return default_ok


def build_lemonade_client(
    config: LemonadeConfig,
    *,
    fallback_ollama: OllamaConfig | None = None,
) -> LemonadeClient | LemonadeMultiClient:
    """설정에 따라 단일/다중 Lemonade 클라이언트를 생성한다."""
    if config.instances:
        return LemonadeMultiClient(
            config,
            fallback_ollama=fallback_ollama,
        )
    return LemonadeClient(
        config,
        fallback_ollama=fallback_ollama,
    )
