"""DFT 대상 파일 탐색 테스트."""

from __future__ import annotations

import json
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
