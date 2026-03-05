"""시뮬레이션 동시 실행 슬롯 매니저."""

from __future__ import annotations

import asyncio
from typing import Any


class ResourceManager:
    """동시 실행 슬롯만 추적하는 경량 매니저."""

    def __init__(self, max_concurrent: int) -> None:
        self._max_concurrent = max_concurrent
        self._running_count = 0
        self._lock = asyncio.Lock()

    async def has_slot(self) -> bool:
        """실행 슬롯이 남아 있는지 확인한다."""
        async with self._lock:
            return self._running_count < self._max_concurrent

    async def acquire(self) -> bool:
        """슬롯을 하나 확보한다. 부족하면 False."""
        async with self._lock:
            if self._running_count >= self._max_concurrent:
                return False
            self._running_count += 1
            return True

    async def release(self) -> None:
        """슬롯을 하나 반환한다."""
        async with self._lock:
            self._running_count = max(0, self._running_count - 1)

    async def get_status(self) -> dict[str, Any]:
        """현재 실행 현황을 반환한다."""
        async with self._lock:
            return {
                "running_jobs": self._running_count,
                "max_concurrent": self._max_concurrent,
            }

    async def sync_from_db(self, running_jobs: list[dict[str, Any]]) -> None:
        """DB에서 실행 중인 작업 수를 기반으로 상태를 복구한다."""
        async with self._lock:
            self._running_count = len(running_jobs)
