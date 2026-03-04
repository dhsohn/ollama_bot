"""시뮬레이션 리소스 매니저 — CPU 코어 및 메모리 할당 추적."""

from __future__ import annotations

import asyncio
from typing import Any


class ResourceManager:
    """할당된 CPU 코어와 메모리를 추적하여 리소스 초과를 방지한다."""

    def __init__(
        self,
        total_cores: int,
        total_memory_mb: int,
        max_concurrent: int,
    ) -> None:
        self._total_cores = total_cores
        self._total_memory_mb = total_memory_mb
        self._max_concurrent = max_concurrent
        self._allocated_cores = 0
        self._allocated_memory_mb = 0
        self._running_count = 0
        self._lock = asyncio.Lock()

    async def can_allocate(self, cores: int, memory_mb: int) -> bool:
        """요청된 리소스를 할당할 수 있는지 확인한다."""
        async with self._lock:
            return (
                self._running_count < self._max_concurrent
                and self._allocated_cores + cores <= self._total_cores
                and self._allocated_memory_mb + memory_mb <= self._total_memory_mb
            )

    async def allocate(self, cores: int, memory_mb: int) -> bool:
        """리소스를 예약한다. 부족하면 False를 반환한다."""
        async with self._lock:
            if (
                self._running_count >= self._max_concurrent
                or self._allocated_cores + cores > self._total_cores
                or self._allocated_memory_mb + memory_mb > self._total_memory_mb
            ):
                return False
            self._allocated_cores += cores
            self._allocated_memory_mb += memory_mb
            self._running_count += 1
            return True

    async def release(self, cores: int, memory_mb: int) -> None:
        """작업 완료 후 리소스를 해제한다."""
        async with self._lock:
            self._allocated_cores = max(0, self._allocated_cores - cores)
            self._allocated_memory_mb = max(0, self._allocated_memory_mb - memory_mb)
            self._running_count = max(0, self._running_count - 1)

    async def get_status(self) -> dict[str, Any]:
        """현재 할당 현황을 반환한다."""
        async with self._lock:
            return {
                "total_cores": self._total_cores,
                "allocated_cores": self._allocated_cores,
                "available_cores": self._total_cores - self._allocated_cores,
                "total_memory_mb": self._total_memory_mb,
                "allocated_memory_mb": self._allocated_memory_mb,
                "available_memory_mb": self._total_memory_mb - self._allocated_memory_mb,
                "running_jobs": self._running_count,
                "max_concurrent": self._max_concurrent,
            }

    async def sync_from_db(self, running_jobs: list[dict[str, Any]]) -> None:
        """DB에서 실행 중인 작업을 기반으로 할당 상태를 복구한다."""
        async with self._lock:
            self._allocated_cores = sum(j["cores"] for j in running_jobs)
            self._allocated_memory_mb = sum(j["memory_mb"] for j in running_jobs)
            self._running_count = len(running_jobs)
