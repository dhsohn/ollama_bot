"""DFT monitor callable 회귀 테스트."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.automation_callables_impl.dft_monitor import build_dft_monitor_callable


_COMPLETED_OUT = "\n".join([
    "! B3LYP def2-SVP Opt",
    "* xyz 0 1",
    "C 0.0 0.0 0.0",
    "H 0.0 0.0 1.0",
    "*",
    "",
    "FINAL SINGLE POINT ENERGY      -100.123456789",
    "",
    "CARTESIAN COORDINATES (ANGSTROEM)",
    "----------------------------",
    " C    0.000000    0.000000    0.000000",
    " H    0.000000    0.000000    1.000000",
    "",
    "                             ****ORCA TERMINATED NORMALLY****",
    "TOTAL RUN TIME: 0 days 0 hours 1 minutes 2 seconds 3 msec",
])


@pytest.mark.asyncio
async def test_baseline_seed_prevents_restart_spam(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(parents=True)
    out_file = kb_dir / "calc.out"
    out_file.write_text(_COMPLETED_OUT, encoding="utf-8")

    state_file = tmp_path / "automation" / "dft_monitor_state.json"

    dft_index = AsyncMock()
    dft_index.upsert_single = AsyncMock(return_value=True)
    logger = MagicMock()

    monitor_1 = build_dft_monitor_callable(
        dft_index=dft_index,
        kb_dirs=[str(kb_dir)],
        logger=logger,
        state_file=str(state_file),
    )

    # 첫 실행은 기존 파일 baseline만 기록하고 알림하지 않는다.
    first_result = await monitor_1()
    assert first_result == ""
    dft_index.upsert_single.assert_not_awaited()
    assert state_file.is_file()

    # 재시작(새 callable 인스턴스) 후에도 동일 파일 재알림이 없어야 한다.
    monitor_2 = build_dft_monitor_callable(
        dft_index=dft_index,
        kb_dirs=[str(kb_dir)],
        logger=logger,
        state_file=str(state_file),
    )
    second_result = await monitor_2()
    assert second_result == ""
    dft_index.upsert_single.assert_not_awaited()

    # 파일이 실제로 변경되면 알림이 발생해야 한다.
    out_file.write_text(_COMPLETED_OUT + "\n# changed\n", encoding="utf-8")
    mtime = os.path.getmtime(out_file)
    os.utime(out_file, (mtime + 5.0, mtime + 5.0))

    third_result = await monitor_2()
    assert "DFT Monitor: 1건의 새 계산 감지" in third_result
    assert "OK" in third_result
    assert dft_index.upsert_single.await_count == 1
