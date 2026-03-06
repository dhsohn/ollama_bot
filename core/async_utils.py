"""비동기 유틸리티."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar

_T = TypeVar("_T")


async def run_in_thread(
    func: Callable[..., _T],
    /,
    *args: Any,
    **kwargs: Any,
) -> _T:
    """Python 버전과 무관하게 동기 함수를 스레드에서 실행한다."""
    to_thread = getattr(asyncio, "to_thread", None)
    if to_thread is not None:
        return await to_thread(func, *args, **kwargs)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))
