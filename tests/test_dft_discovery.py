"""DFT 대상 파일 탐색 테스트."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.dft_discovery import discover_orca_targets


def test_discover_uses_only_last_out_when_run_report_exists(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    run_dir = kb_dir / "run_a"
    run_dir.mkdir(parents=True)

    failed_out = run_dir / "job.retry01.out"
    final_out = run_dir / "job.retry02.out"
    failed_out.write_text("failed", encoding="utf-8")
    final_out.write_text("completed", encoding="utf-8")

    report = {
        "final_result": {
            "last_out_path": str(final_out),
        },
    }
    (run_dir / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False),
        encoding="utf-8",
    )

    targets = discover_orca_targets(kb_dir, max_bytes=1024 * 1024)
    assert [str(p) for p in targets] == [str(final_out)]


def test_discover_falls_back_to_filename_when_report_path_root_differs(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    run_dir = kb_dir / "run_b"
    run_dir.mkdir(parents=True)

    final_out = run_dir / "final.out"
    final_out.write_text("completed", encoding="utf-8")

    # 다른 루트(/home/...)로 기록된 경로도 같은 run 디렉토리 파일명으로 해석한다.
    report = {
        "final_result": {
            "last_out_path": "/home/someone/orca_outputs/run_b/final.out",
        },
    }
    (run_dir / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False),
        encoding="utf-8",
    )

    targets = discover_orca_targets(kb_dir, max_bytes=1024 * 1024)
    assert [str(p) for p in targets] == [str(final_out)]


def test_discover_can_merge_report_and_legacy_targets(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    run_dir = kb_dir / "run_c"
    run_dir.mkdir(parents=True)

    retry1 = run_dir / "job.retry01.out"
    retry2 = run_dir / "job.retry02.out"
    retry1.write_text("retry1", encoding="utf-8")
    retry2.write_text("retry2", encoding="utf-8")

    report = {
        "final_result": {
            "last_out_path": str(retry2),
        },
    }
    (run_dir / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False),
        encoding="utf-8",
    )

    targets = discover_orca_targets(
        kb_dir,
        max_bytes=1024 * 1024,
        include_legacy_when_report_exists=True,
    )
    assert [str(p) for p in targets] == [str(retry1), str(retry2)]


def test_discover_uses_run_state_last_out_when_report_missing(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    run_dir = kb_dir / "run_state_only"
    run_dir.mkdir(parents=True)

    out_file = run_dir / "input.out"
    out_file.write_text("completed", encoding="utf-8")
    run_state = {
        "status": "completed",
        "final_result": {
            "last_out_path": str(out_file),
        },
    }
    (run_dir / "run_state.json").write_text(
        json.dumps(run_state, ensure_ascii=False),
        encoding="utf-8",
    )

    targets = discover_orca_targets(kb_dir, max_bytes=1024 * 1024)
    assert [str(p) for p in targets] == [str(out_file)]


def test_discover_infers_out_from_run_state_selected_inp(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    run_dir = kb_dir / "run_infer"
    run_dir.mkdir(parents=True)

    inp_path = run_dir / "reactants.inp"
    out_file = run_dir / "reactants.out"
    inp_path.write_text("* xyz 0 1\n*", encoding="utf-8")
    out_file.write_text("running", encoding="utf-8")

    run_state = {
        "status": "running",
        "reaction_dir": str(run_dir),
        "selected_inp": str(inp_path),
        "attempts": [],
        "final_result": {},
    }
    (run_dir / "run_state.json").write_text(
        json.dumps(run_state, ensure_ascii=False),
        encoding="utf-8",
    )

    targets = discover_orca_targets(kb_dir, max_bytes=1024 * 1024)
    assert [str(p) for p in targets] == [str(out_file)]


def test_orca_runs_tracks_latest_out_for_running_state(tmp_path: Path) -> None:
    kb_dir = tmp_path / "orca_runs"
    run_dir = kb_dir / "job_a"
    run_dir.mkdir(parents=True)

    old_out = run_dir / "job.retry01.out"
    new_out = run_dir / "job.retry02.out"
    old_out.write_text("old", encoding="utf-8")
    new_out.write_text("new", encoding="utf-8")

    old_mtime = old_out.stat().st_mtime
    new_mtime = new_out.stat().st_mtime
    if new_mtime <= old_mtime:
        new_out.touch()

    run_state = {
        "status": "running",
        "reaction_dir": str(run_dir),
        "selected_inp": str(run_dir / "job.inp"),
        "final_result": {},
    }
    (run_dir / "run_state.json").write_text(
        json.dumps(run_state, ensure_ascii=False),
        encoding="utf-8",
    )

    targets = discover_orca_targets(kb_dir, max_bytes=1024 * 1024)
    assert [str(p) for p in targets] == [str(new_out)]


def test_orca_runs_tracks_latest_out_for_failed_state(tmp_path: Path) -> None:
    kb_dir = tmp_path / "orca_runs"
    run_dir = kb_dir / "job_b"
    run_dir.mkdir(parents=True)

    old_out = run_dir / "job.retry01.out"
    new_out = run_dir / "job.retry02.out"
    old_out.write_text("old", encoding="utf-8")
    new_out.write_text("new", encoding="utf-8")

    old_mtime = old_out.stat().st_mtime
    new_mtime = new_out.stat().st_mtime
    if new_mtime <= old_mtime:
        new_out.touch()

    run_state = {
        "status": "failed",
        "reaction_dir": str(run_dir),
        "selected_inp": str(run_dir / "job.inp"),
        "final_result": {},
    }
    (run_dir / "run_state.json").write_text(
        json.dumps(run_state, ensure_ascii=False),
        encoding="utf-8",
    )

    targets = discover_orca_targets(kb_dir, max_bytes=1024 * 1024)
    assert [str(p) for p in targets] == [str(new_out)]


def test_orca_outputs_tracks_only_last_out_path(tmp_path: Path) -> None:
    kb_dir = tmp_path / "orca_outputs"
    run_dir = kb_dir / "run_ok"
    run_dir.mkdir(parents=True)

    final_out = run_dir / "final.out"
    newer_retry = run_dir / "final.retry01.out"
    final_out.write_text("final", encoding="utf-8")
    newer_retry.write_text("retry", encoding="utf-8")

    final_mtime = final_out.stat().st_mtime
    newer_mtime = newer_retry.stat().st_mtime
    if newer_mtime <= final_mtime:
        newer_retry.touch()

    run_state = {
        "status": "completed",
        "final_result": {
            "last_out_path": str(final_out),
        },
    }
    (run_dir / "run_state.json").write_text(
        json.dumps(run_state, ensure_ascii=False),
        encoding="utf-8",
    )

    targets = discover_orca_targets(kb_dir, max_bytes=1024 * 1024)
    assert [str(p) for p in targets] == [str(final_out)]


def test_orca_outputs_includes_recent_completed_with_completed_at(tmp_path: Path) -> None:
    kb_dir = tmp_path / "orca_outputs"
    run_dir = kb_dir / "run_recent"
    run_dir.mkdir(parents=True)

    final_out = run_dir / "final.out"
    final_out.write_text("final", encoding="utf-8")

    completed_at = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    run_state = {
        "status": "completed",
        "final_result": {
            "status": "completed",
            "completed_at": completed_at,
            "last_out_path": str(final_out),
        },
    }
    (run_dir / "run_state.json").write_text(
        json.dumps(run_state, ensure_ascii=False),
        encoding="utf-8",
    )

    targets = discover_orca_targets(
        kb_dir,
        max_bytes=1024 * 1024,
        recent_completed_window_minutes=60,
    )
    assert [str(p) for p in targets] == [str(final_out)]


def test_orca_outputs_excludes_old_completed_with_completed_at(tmp_path: Path) -> None:
    kb_dir = tmp_path / "orca_outputs"
    run_dir = kb_dir / "run_old"
    run_dir.mkdir(parents=True)

    final_out = run_dir / "final.out"
    final_out.write_text("final", encoding="utf-8")

    completed_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    run_state = {
        "status": "completed",
        "final_result": {
            "status": "completed",
            "completed_at": completed_at,
            "last_out_path": str(final_out),
        },
    }
    (run_dir / "run_state.json").write_text(
        json.dumps(run_state, ensure_ascii=False),
        encoding="utf-8",
    )

    targets = discover_orca_targets(
        kb_dir,
        max_bytes=1024 * 1024,
        recent_completed_window_minutes=60,
    )
    assert targets == []


def test_orca_outputs_uses_mtime_when_completed_at_missing(tmp_path: Path) -> None:
    kb_dir = tmp_path / "orca_outputs"
    run_dir = kb_dir / "run_no_completed_at"
    run_dir.mkdir(parents=True)

    final_out = run_dir / "final.out"
    final_out.write_text("final", encoding="utf-8")

    old_mtime = (datetime.now(timezone.utc) - timedelta(hours=2)).timestamp()
    os.utime(final_out, (old_mtime, old_mtime))

    run_state = {
        "status": "completed",
        "final_result": {
            "status": "completed",
            "last_out_path": str(final_out),
        },
    }
    (run_dir / "run_state.json").write_text(
        json.dumps(run_state, ensure_ascii=False),
        encoding="utf-8",
    )

    targets = discover_orca_targets(
        kb_dir,
        max_bytes=1024 * 1024,
        recent_completed_window_minutes=60,
    )
    assert targets == []
