"""Shared lifecycle state for runtime startup and shutdown."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass
class RuntimeTaskHandles:
    """Track started runtime tasks and application lifecycle flags."""

    memory_maintenance_task: asyncio.Task[Any] | None = None
    llm_recovery_task: asyncio.Task[Any] | None = None
    scheduler_started: bool = False
    app_started: bool = False
    updater_started: bool = False
