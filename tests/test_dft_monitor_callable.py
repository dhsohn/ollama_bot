"""DFT monitor callable 회귀 테스트."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.automation_callables_impl.dft_monitor import (
    _extract_comment,
    _is_thinking_line,
    build_dft_monitor_callable,
)

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
    (kb_dir / "run_state.json").write_text('{"status": "running"}', encoding="utf-8")

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
                "status": "running",
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
    assert "DFT Monitor: 1건의 새 계산 감지" in result
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


def _build_running_special_out(calc_keyword: str) -> str:
    """OPT step 테이블 없이 running 상태를 만들기 위한 ORCA 출력 템플릿."""
    return "\n".join([
        f"! B3LYP def2-SVP {calc_keyword}",
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
        "FINAL SINGLE POINT ENERGY      -100.200000000",
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
    assert "OPT Progress" in result
    assert "RUNNING" in result
    assert "Step" in result
    assert "<b>" in result
    assert "<pre>" in result


@pytest.mark.asyncio
async def test_running_opt_does_not_upsert_to_index(tmp_path: Path) -> None:
    """진행 중인 최적화 계산은 인덱스에 upsert하지 않는다."""
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
    assert "DFT Monitor: 1건의 새 계산 감지" in result
    assert "RUNNING Progress: 1건의 진행 중인 계산" in result


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
    assert "RUNNING Progress: 1건의 진행 중인 계산" in result
    assert result.count("OPT Progress") == 1


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
    assert "RUNNING Progress: 1건의 진행 중인 계산" in result
    assert result.count("OPT Progress") == 1


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
    (kb_dir / "run_state.json").write_text('{"status": "running"}', encoding="utf-8")

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
    assert "💬" in result
    assert "<i>" in result
    assert "에너지가 단조 감소" in result
    mock_engine.process_prompt.assert_awaited_once()


@pytest.mark.asyncio
async def test_ai_comment_graceful_on_engine_failure(tmp_path: Path) -> None:
    """engine.process_prompt 실패 시에도 progress 알림은 정상 전송된다."""
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(parents=True)
    out_file = kb_dir / "opt_run.out"
    out_file.write_text(_RUNNING_OPT_OUT, encoding="utf-8")
    (kb_dir / "run_state.json").write_text('{"status": "running"}', encoding="utf-8")

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
    assert "💬" not in result


@pytest.mark.asyncio
async def test_ai_comment_budget_prevents_global_timeout(tmp_path: Path) -> None:
    """잡 timeout 예산이 작을 때 AI 코멘트를 제한해 전체 작업 timeout을 방지한다."""
    kb_dir = tmp_path / "kb"
    run_a = kb_dir / "run_a"
    run_b = kb_dir / "run_b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    out_a = run_a / "opt_a.out"
    out_b = run_b / "opt_b.out"
    out_a.write_text(_RUNNING_OPT_OUT, encoding="utf-8")
    out_b.write_text(_RUNNING_OPT_OUT, encoding="utf-8")
    (run_a / "run_state.json").write_text('{"status": "running"}', encoding="utf-8")
    (run_b / "run_state.json").write_text('{"status": "running"}', encoding="utf-8")

    state_file = tmp_path / "automation" / "state.json"
    dft_index = AsyncMock()
    dft_index.upsert_single = AsyncMock(return_value=True)
    logger = MagicMock()

    async def _slow_comment(**kwargs: object) -> str:
        _ = kwargs
        await asyncio.sleep(10)
        return "느린 응답"

    mock_engine = AsyncMock()
    mock_engine.process_prompt = AsyncMock(side_effect=_slow_comment)

    monitor = build_dft_monitor_callable(
        dft_index=dft_index,
        kb_dirs=[str(kb_dir)],
        logger=logger,
        state_file=str(state_file),
        engine=mock_engine,
    )

    await monitor()  # baseline

    out_a.write_text(_RUNNING_OPT_OUT + "\n# updated\n", encoding="utf-8")
    mtime_a = os.path.getmtime(out_a)
    os.utime(out_a, (mtime_a + 5.0, mtime_a + 5.0))

    out_b.write_text(_RUNNING_OPT_OUT + "\n# updated\n", encoding="utf-8")
    mtime_b = os.path.getmtime(out_b)
    os.utime(out_b, (mtime_b + 5.0, mtime_b + 5.0))

    result = await asyncio.wait_for(monitor(timeout=17), timeout=18)
    assert "RUNNING Progress: 2건의 진행 중인 계산" in result
    assert "💬" not in result
    assert mock_engine.process_prompt.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("calc_keyword", "expected_calc_type"),
    [
        ("OPTTS", "ts"),
        ("NEB", "neb"),
        ("IRC", "irc"),
    ],
)
async def test_running_ts_neb_irc_include_ai_comment(
    tmp_path: Path,
    calc_keyword: str,
    expected_calc_type: str,
) -> None:
    """running TS/NEB/IRC 계산도 AI 한줄 코멘트를 포함해 알림한다."""
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(parents=True)
    out_file = kb_dir / f"{expected_calc_type}_run.out"
    out_file.write_text(_build_running_special_out(calc_keyword), encoding="utf-8")
    (kb_dir / "run_state.json").write_text('{"status": "running"}', encoding="utf-8")

    state_file = tmp_path / "automation" / "state.json"

    dft_index = AsyncMock()
    dft_index.upsert_single = AsyncMock(return_value=True)
    logger = MagicMock()

    mock_engine = AsyncMock()
    mock_engine.process_prompt = AsyncMock(
        return_value="진행 경향은 안정적이며 다음 스텝에서 수렴 여부를 확인하세요."
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
    out_file.write_text(
        _build_running_special_out(calc_keyword) + "\n# updated\n",
        encoding="utf-8",
    )
    mtime = os.path.getmtime(out_file)
    os.utime(out_file, (mtime + 5.0, mtime + 5.0))

    result = await monitor()
    assert "RUNNING Progress" in result
    assert expected_calc_type in result
    assert "💬" in result
    assert "<i>" in result
    dft_index.upsert_single.assert_not_awaited()
    mock_engine.process_prompt.assert_awaited_once()


# ---------------------------------------------------------------------------
# _extract_comment 사고 과정 필터링 테스트
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # 정상 한줄 코멘트 → 그대로 반환
        ("에너지가 단조 감소 중입니다.", "에너지가 단조 감소 중입니다."),
        # 사고 과정 + 코멘트 → 사고 과정 건너뛰기
        (
            "분석: dE가 감소하고 있으므로\n에너지가 단조 감소 중입니다.",
            "에너지가 단조 감소 중입니다.",
        ),
        # 여러 줄에서 사고 과정 제거 후 나머지 합침
        (
            "먼저 데이터를 살펴보면\n확인 결과\n수렴이 안정적으로 진행 중입니다.",
            "수렴이 안정적으로 진행 중입니다.",
        ),
        # 영어 사고 과정 접두어
        (
            "Let me analyze the data.\nThe optimization is converging well.",
            "The optimization is converging well.",
        ),
        # 빈 응답
        ("", ""),
        ("  \n  ", ""),
        # 사고 과정만 있는 경우 → 빈 문자열 반환 (사고 유출 방지)
        ("분석: 데이터 확인\n검토: 수렴 패턴", ""),
        # 영어 메타 추론 유출 → 빈 문자열 반환
        (
            'The user says: "아래 ORCA" They want a comment in Korean.',
            "",
        ),
        (
            "We need to produce a comment based on the given data.\n"
            "수렴이 안정적으로 진행 중입니다.",
            "수렴이 안정적으로 진행 중입니다.",
        ),
        # 중국어 사고 과정 + 한국어 코멘트 → 사고 과정 건너뛰기
        (
            "首先分析一下数据\uFF0C能量在下降\n에너지가 안정적으로 수렴 중입니다.",
            "에너지가 안정적으로 수렴 중입니다.",
        ),
        # 중국어 사고 과정만 → 빈 문자열
        ("用户要求写一个韩语评论。根据数据来看计算正在进行。", ""),
        # 긴 코멘트도 유지 (글자수 제한 없음)
        ("A" * 201, "A" * 201),
        # 정상 코멘트 → 유지
        ("wB97X-D3/def2-TZVP 수준의 구조 최적화가 진행 중이며, 에너지가 안정화되고 있습니다.", "wB97X-D3/def2-TZVP 수준의 구조 최적화가 진행 중이며, 에너지가 안정화되고 있습니다."),
    ],
)
def test_extract_comment_filters_thinking(raw: str, expected: str) -> None:
    """_extract_comment가 사고 과정을 필터링하고 최종 코멘트를 추출한다."""
    assert _extract_comment(raw) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        # 영어 메타 추론 감지
        ('The user says: "코멘트를 작성하세요"', True),
        ("They want a comment in Korean about the data.", True),
        ("We need to produce a comment based on the given data.", True),
        ("We don't know next check point from data.", True),
        ("Probably just mention that the calculation is running.", True),
        ("This seems to be missing data.", True),
        # 중국어 사고 과정 접두어
        ("首先分析一下数据", True),
        ("让我看看这个计算结果", True),
        ("用户要求写一个韩语评论", True),
        ("我们需要根据数据生成评论", True),
        ("根据给定的ORCA数据来看", True),
        ("这意味着优化正在进行中", True),
        # 한국어 사고 과정 접두어
        ("분석: dE가 감소하고 있으므로", True),
        ("먼저 데이터를 살펴보면", True),
        # 정상 코멘트 → False
        ("에너지가 단조 감소 중입니다.", False),
        ("wB97X-D3 수준에서 구조 최적화가 진행 중입니다.", False),
    ],
)
def test_is_thinking_line(line: str, expected: bool) -> None:
    """_is_thinking_line이 사고 과정 줄을 정확히 감지한다."""
    assert _is_thinking_line(line) == expected
