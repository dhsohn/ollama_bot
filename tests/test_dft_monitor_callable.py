"""DFT monitor callable 회귀 테스트."""

from __future__ import annotations

import json
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
    (kb_dir / "run_state.json").write_text('{"status": "completed"}', encoding="utf-8")

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
    assert "DFT Monitor: 1건의 계산 업데이트" in third_result
    assert "✅" in third_result
    assert "<b>" in third_result
    assert "<pre>" in third_result
    assert dft_index.upsert_single.await_count == 1


@pytest.mark.asyncio
async def test_detects_run_state_only_output_update(tmp_path: Path) -> None:
    """run_state만 있을 때도 selected_inp 기반 out 변경을 감지해야 한다."""
    kb_dir = tmp_path / "kb"
    run_dir = kb_dir / "run_state_only"
    run_dir.mkdir(parents=True)

    inp_file = run_dir / "input.inp"
    out_file = run_dir / "input.out"
    inp_file.write_text("* xyz 0 1\n*", encoding="utf-8")
    out_file.write_text(_COMPLETED_OUT, encoding="utf-8")

    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "reaction_dir": str(run_dir),
                "selected_inp": str(inp_file),
                "attempts": [],
                "final_result": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

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

    await monitor()  # baseline

    out_file.write_text(_COMPLETED_OUT + "\n# changed\n", encoding="utf-8")
    mtime = os.path.getmtime(out_file)
    os.utime(out_file, (mtime + 5.0, mtime + 5.0))

    result = await monitor()
    assert "DFT Monitor: 1건의 계산 업데이트" in result
    assert "input.out" in result
    dft_index.upsert_single.assert_awaited_once_with(str(out_file))


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
    (kb_dir / "run_state.json").write_text('{"status": "running"}', encoding="utf-8")

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
    assert "계산 업데이트" in result
    assert "🔄" in result
    assert "<b>" in result
    assert "<pre>" in result
    # running 계산은 인덱스에 upsert하지 않는다
    dft_index.upsert_single.assert_not_awaited()


@pytest.mark.asyncio
async def test_completed_and_running_mixed(tmp_path: Path) -> None:
    """완료된 파일과 진행 중인 파일이 동시에 있을 때 각각 올바른 알림을 생성한다."""
    kb_dir = tmp_path / "kb"

    done_dir = kb_dir / "done_dir"
    done_dir.mkdir(parents=True)
    completed_file = done_dir / "done.out"
    completed_file.write_text(_COMPLETED_OUT, encoding="utf-8")
    (done_dir / "run_state.json").write_text('{"status": "completed"}', encoding="utf-8")

    run_dir = kb_dir / "run_dir"
    run_dir.mkdir(parents=True)
    running_file = run_dir / "running.out"
    running_file.write_text(_RUNNING_OPT_OUT, encoding="utf-8")
    (run_dir / "run_state.json").write_text('{"status": "running"}', encoding="utf-8")

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
    assert "DFT Monitor: 2건의 계산 업데이트" in result
    assert "✅" in result
    assert "🔄" in result


@pytest.mark.asyncio
async def test_running_progress_deduplicates_symlink_alias_paths(tmp_path: Path) -> None:
    """같은 output을 alias 경로와 실제 경로로 동시에 스캔해도 1건만 표시한다."""
    kb_dir = tmp_path / "kb"
    run_dir = kb_dir / "run_dir"
    run_dir.mkdir(parents=True)
    alias_dir = tmp_path / "run_alias"
    alias_dir.symlink_to(run_dir, target_is_directory=True)

    out_file = run_dir / "running.out"
    out_file.write_text(_RUNNING_OPT_OUT, encoding="utf-8")
    (run_dir / "run_state.json").write_text('{"status": "running"}', encoding="utf-8")

    state_file = tmp_path / "automation" / "state.json"
    dft_index = AsyncMock()
    dft_index.upsert_single = AsyncMock(return_value=True)
    logger = MagicMock()

    monitor = build_dft_monitor_callable(
        dft_index=dft_index,
        kb_dirs=[str(kb_dir)],
        logger=logger,
        state_file=str(state_file),
        get_external_dirs=AsyncMock(return_value=[str(alias_dir)]),
    )

    await monitor()  # baseline

    out_file.write_text(_RUNNING_OPT_OUT + "\n# changed\n", encoding="utf-8")
    mtime = os.path.getmtime(out_file)
    os.utime(out_file, (mtime + 5.0, mtime + 5.0))

    result = await monitor()
    assert "DFT Monitor: 1건의 계산 업데이트" in result
    assert "🔄" in result


@pytest.mark.asyncio
async def test_state_file_normalizes_path_aliases_across_restart(tmp_path: Path) -> None:
    """이전 실행이 alias 경로를 저장했어도 재시작 후 실제 경로를 중복 새 계산으로 보지 않는다."""
    kb_dir = tmp_path / "kb"
    run_dir = kb_dir / "run_dir"
    run_dir.mkdir(parents=True)
    alias_dir = tmp_path / "run_alias"
    alias_dir.symlink_to(run_dir, target_is_directory=True)

    out_file = run_dir / "running.out"
    out_file.write_text(_RUNNING_OPT_OUT, encoding="utf-8")
    (run_dir / "run_state.json").write_text('{"status": "running"}', encoding="utf-8")

    state_file = tmp_path / "automation" / "state.json"
    dft_index = AsyncMock()
    dft_index.upsert_single = AsyncMock(return_value=True)
    logger = MagicMock()

    monitor_alias = build_dft_monitor_callable(
        dft_index=dft_index,
        kb_dirs=[str(alias_dir)],
        logger=logger,
        state_file=str(state_file),
    )
    assert await monitor_alias() == ""

    monitor_real = build_dft_monitor_callable(
        dft_index=dft_index,
        kb_dirs=[str(run_dir)],
        logger=logger,
        state_file=str(state_file),
    )
    assert await monitor_real() == ""


@pytest.mark.asyncio
async def test_running_dedup_when_mtime_changes_between_scans(tmp_path: Path) -> None:
    """external_dirs가 kb_dirs의 하위 디렉토리일 때, 파일 mtime이 스캔 도중 변해도 1건만 보고한다."""
    kb_dir = tmp_path / "orca_runs"
    sub_dir = kb_dir / "products"
    sub_dir.mkdir(parents=True)

    out_file = sub_dir / "products.out"
    out_file.write_text(_RUNNING_OPT_OUT, encoding="utf-8")
    (sub_dir / "run_state.json").write_text('{"status": "running"}', encoding="utf-8")

    state_file = tmp_path / "automation" / "state.json"
    dft_index = AsyncMock()
    dft_index.upsert_single = AsyncMock(return_value=True)
    logger = MagicMock()

    # external_dirs가 kb_dir의 하위 디렉토리를 반환 (겹치는 디렉토리)
    monitor = build_dft_monitor_callable(
        dft_index=dft_index,
        kb_dirs=[str(kb_dir)],
        logger=logger,
        state_file=str(state_file),
        get_external_dirs=AsyncMock(return_value=[str(sub_dir)]),
    )

    await monitor()  # baseline

    out_file.write_text(_RUNNING_OPT_OUT + "\n# changed\n", encoding="utf-8")
    mtime = os.path.getmtime(out_file)
    os.utime(out_file, (mtime + 5.0, mtime + 5.0))

    result = await monitor()
    assert "DFT Monitor: 1건의 계산 업데이트" in result
    assert "🔄" in result


# ---------------------------------------------------------------------------
# run_state.json status가 parser status를 보완하는 테스트
# ---------------------------------------------------------------------------

# NEB 계산: 각 이미지 SP마다 ORCA TERMINATED NORMALLY가 출력되어
# parser는 "completed"로 판정하지만 run_state는 "running"인 케이스
_NEB_RUNNING_OUT = "\n".join([
    "! NEB B3LYP def2-SVP",
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
    "FINAL SINGLE POINT ENERGY      -100.123456789",
    "",
    "                             ****ORCA TERMINATED NORMALLY****",
    "TOTAL RUN TIME: 0 days 0 hours 0 minutes 30 seconds 0 msec",
])


@pytest.mark.asyncio
async def test_neb_running_uses_run_state_status(tmp_path: Path) -> None:
    """NEB 계산: parser는 completed이지만 run_state가 running이면 RUNNING Progress로 간다."""
    kb_dir = tmp_path / "kb"
    run_dir = kb_dir / "neb_job"
    run_dir.mkdir(parents=True)
    out_file = run_dir / "neb.out"
    out_file.write_text(_NEB_RUNNING_OUT, encoding="utf-8")
    (run_dir / "run_state.json").write_text('{"status": "running"}', encoding="utf-8")

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

    # 파일 변경 (NEB가 다음 iteration 진행)
    out_file.write_text(_NEB_RUNNING_OUT + "\n# next iteration\n", encoding="utf-8")
    mtime = os.path.getmtime(out_file)
    os.utime(out_file, (mtime + 5.0, mtime + 5.0))

    result = await monitor()
    assert "계산 업데이트" in result
    assert "🔄" in result
    # running 계산은 인덱스에 upsert하지 않아야 한다
    dft_index.upsert_single.assert_not_awaited()
