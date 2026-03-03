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


# ---------------------------------------------------------------------------
# running 최적화 계산 progress 알림 테스트
# ---------------------------------------------------------------------------

_RUNNING_OPT_OUT = "\n".join([
    "! B3LYP def2-SVP Opt",
    "* xyz 0 1",
    "C 0.0 0.0 0.0",
    "H 0.0 0.0 1.0",
    "*",
    "",
    "CARTESIAN COORDINATES (ANGSTROEM)",
    "----------------------------",
    " C    0.000000    0.000000    0.000000",
    " H    0.000000    0.000000    1.000000",
    "",
    "---------------------------------------------------",
    "| Geometry Optimization Cycle   1                 |",
    "---------------------------------------------------",
    "",
    "FINAL SINGLE POINT ENERGY      -100.100000000",
    "",
    "---------------------------------------------------",
    "| Geometry Optimization Cycle   2                 |",
    "---------------------------------------------------",
    "",
    "FINAL SINGLE POINT ENERGY      -100.120000000",
    "",
    "                         *************************************",
    "                         *  GEOMETRY CONVERGENCE              *",
    "                         *************************************",
    "Item                Value     Tolerance   Converged",
    "Energy change      -0.020000  5.0000e-06    NO",
    "MAX gradient        0.005000  3.0000e-04    NO",
    "RMS gradient        0.002000  1.0000e-04    NO",
    "MAX step            0.010000  4.0000e-03    NO",
    "RMS step            0.004000  2.0000e-03    NO",
])


@pytest.mark.asyncio
async def test_running_opt_generates_progress_notification(tmp_path: Path) -> None:
    """진행 중인 최적화 계산은 OPT Progress 알림을 생성한다."""
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(parents=True)
    out_file = kb_dir / "opt_run.out"
    out_file.write_text(_RUNNING_OPT_OUT, encoding="utf-8")

    state_file = tmp_path / "automation" / "state.json"

    dft_index = AsyncMock()
    dft_index.upsert_single = AsyncMock(return_value=True)
    logger = MagicMock()

    monitor = build_dft_monitor_callable(
        dft_index=dft_index,
        kb_dirs=[str(kb_dir)],
        logger=logger,
        state_file=str(state_file),
    )

    # 첫 실행: baseline
    await monitor()

    # 파일 변경 시뮬레이션
    out_file.write_text(_RUNNING_OPT_OUT + "\n# updated\n", encoding="utf-8")
    mtime = os.path.getmtime(out_file)
    os.utime(out_file, (mtime + 5.0, mtime + 5.0))

    result = await monitor()
    assert "OPT Progress" in result
    assert "RUNNING" in result
    assert "Step" in result


@pytest.mark.asyncio
async def test_running_opt_does_not_upsert_to_index(tmp_path: Path) -> None:
    """진행 중인 최적화 계산은 인덱스에 upsert하지 않는다."""
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(parents=True)
    out_file = kb_dir / "opt_run.out"
    out_file.write_text(_RUNNING_OPT_OUT, encoding="utf-8")

    state_file = tmp_path / "automation" / "state.json"

    dft_index = AsyncMock()
    dft_index.upsert_single = AsyncMock(return_value=True)
    logger = MagicMock()

    monitor = build_dft_monitor_callable(
        dft_index=dft_index,
        kb_dirs=[str(kb_dir)],
        logger=logger,
        state_file=str(state_file),
    )

    # baseline
    await monitor()

    # 파일 변경
    out_file.write_text(_RUNNING_OPT_OUT + "\n# updated\n", encoding="utf-8")
    mtime = os.path.getmtime(out_file)
    os.utime(out_file, (mtime + 5.0, mtime + 5.0))

    await monitor()
    dft_index.upsert_single.assert_not_awaited()


@pytest.mark.asyncio
async def test_completed_and_running_mixed(tmp_path: Path) -> None:
    """완료된 파일과 진행 중인 파일이 동시에 있을 때 각각 올바른 알림을 생성한다."""
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(parents=True)

    completed_file = kb_dir / "done.out"
    completed_file.write_text(_COMPLETED_OUT, encoding="utf-8")

    running_file = kb_dir / "running.out"
    running_file.write_text(_RUNNING_OPT_OUT, encoding="utf-8")

    state_file = tmp_path / "automation" / "state.json"

    dft_index = AsyncMock()
    dft_index.upsert_single = AsyncMock(return_value=True)
    logger = MagicMock()

    monitor = build_dft_monitor_callable(
        dft_index=dft_index,
        kb_dirs=[str(kb_dir)],
        logger=logger,
        state_file=str(state_file),
    )

    # baseline
    await monitor()

    # 두 파일 모두 변경
    completed_file.write_text(_COMPLETED_OUT + "\n# changed\n", encoding="utf-8")
    mtime_c = os.path.getmtime(completed_file)
    os.utime(completed_file, (mtime_c + 5.0, mtime_c + 5.0))

    running_file.write_text(_RUNNING_OPT_OUT + "\n# changed\n", encoding="utf-8")
    mtime_r = os.path.getmtime(running_file)
    os.utime(running_file, (mtime_r + 5.0, mtime_r + 5.0))

    result = await monitor()
    assert "DFT Monitor: 1건의 새 계산 감지" in result
    assert "OPT Progress: 1건의 진행 중인 최적화" in result


# ---------------------------------------------------------------------------
# LLM 한줄 해석 테스트
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ai_comment_included_when_engine_provided(tmp_path: Path) -> None:
    """engine이 있으면 AI 한줄 해석이 progress 알림에 포함된다."""
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(parents=True)
    out_file = kb_dir / "opt_run.out"
    out_file.write_text(_RUNNING_OPT_OUT, encoding="utf-8")

    state_file = tmp_path / "automation" / "state.json"

    dft_index = AsyncMock()
    dft_index.upsert_single = AsyncMock(return_value=True)
    logger = MagicMock()

    mock_engine = AsyncMock()
    mock_engine.process_prompt = AsyncMock(
        return_value="에너지가 단조 감소 중이며 수렴에 근접하고 있습니다."
    )

    monitor = build_dft_monitor_callable(
        dft_index=dft_index,
        kb_dirs=[str(kb_dir)],
        logger=logger,
        state_file=str(state_file),
        engine=mock_engine,
    )

    # baseline
    await monitor()

    # 파일 변경
    out_file.write_text(_RUNNING_OPT_OUT + "\n# updated\n", encoding="utf-8")
    mtime = os.path.getmtime(out_file)
    os.utime(out_file, (mtime + 5.0, mtime + 5.0))

    result = await monitor()
    assert "AI: " in result
    assert "단조 감소" in result
    mock_engine.process_prompt.assert_awaited_once()


@pytest.mark.asyncio
async def test_ai_comment_graceful_on_engine_failure(tmp_path: Path) -> None:
    """engine.process_prompt 실패 시에도 progress 알림은 정상 전송된다."""
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(parents=True)
    out_file = kb_dir / "opt_run.out"
    out_file.write_text(_RUNNING_OPT_OUT, encoding="utf-8")

    state_file = tmp_path / "automation" / "state.json"

    dft_index = AsyncMock()
    dft_index.upsert_single = AsyncMock(return_value=True)
    logger = MagicMock()

    mock_engine = AsyncMock()
    mock_engine.process_prompt = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    monitor = build_dft_monitor_callable(
        dft_index=dft_index,
        kb_dirs=[str(kb_dir)],
        logger=logger,
        state_file=str(state_file),
        engine=mock_engine,
    )

    # baseline
    await monitor()

    # 파일 변경
    out_file.write_text(_RUNNING_OPT_OUT + "\n# updated\n", encoding="utf-8")
    mtime = os.path.getmtime(out_file)
    os.utime(out_file, (mtime + 5.0, mtime + 5.0))

    result = await monitor()
    # AI 코멘트 없이 progress 테이블은 정상 출력
    assert "OPT Progress" in result
    assert "Step" in result
    assert "AI: " not in result
