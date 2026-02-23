"""ORCA 모니터 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.orca_monitor import (
    _dir_label,
    _elapsed_human,
    _load_state_file,
    generate_orca_progress_report,
)


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def _make_state(
    name: str,
    status: str = "running",
    attempts: int = 0,
    max_retries: int = 5,
    final_result: dict | None = None,
) -> dict:
    return {
        "run_id": f"run_test_{name}",
        "reaction_dir": f"/orca_runs/{name}",
        "selected_inp": f"/orca_runs/{name}/{name}.inp",
        "max_retries": max_retries,
        "status": status,
        "started_at": "2026-02-23T10:00:00+00:00",
        "updated_at": "2026-02-23T12:00:00+00:00",
        "attempts": [
            {
                "index": i + 1,
                "analyzer_status": "error_scf" if i < attempts - 1 else "completed",
                "analyzer_reason": "SCF convergence failure" if i < attempts - 1 else "",
                "started_at": "2026-02-23T10:00:00+00:00",
                "ended_at": "2026-02-23T11:00:00+00:00",
            }
            for i in range(attempts)
        ],
        "final_result": final_result,
    }


class TestLoadStateFile:
    def test_valid_state(self, tmp_path: Path) -> None:
        state = _make_state("test", status="running")
        sf = tmp_path / "run_state.json"
        _write_state(sf, state)
        result = _load_state_file(sf)
        assert result is not None
        assert result["run_id"] == "run_test_test"

    def test_invalid_json(self, tmp_path: Path) -> None:
        sf = tmp_path / "run_state.json"
        sf.write_text("not json", encoding="utf-8")
        assert _load_state_file(sf) is None

    def test_missing_run_id(self, tmp_path: Path) -> None:
        sf = tmp_path / "run_state.json"
        _write_state(sf, {"status": "running"})
        assert _load_state_file(sf) is None

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        sf = tmp_path / "nonexistent.json"
        assert _load_state_file(sf) is None


class TestDirLabel:
    def test_full_path(self) -> None:
        assert _dir_label("/home/user/orca_runs/reactants") == "reactants"

    def test_none(self) -> None:
        assert _dir_label(None) == "unknown"

    def test_empty(self) -> None:
        assert _dir_label("") == "unknown"


class TestElapsedHuman:
    def test_none_input(self) -> None:
        assert _elapsed_human(None) == "?"

    def test_invalid_format(self) -> None:
        assert _elapsed_human("not-a-date") == "?"


class TestGenerateReport:
    @pytest.mark.asyncio
    async def test_missing_directory(self) -> None:
        result = await generate_orca_progress_report("/nonexistent/path")
        assert "찾을 수 없습니다" in result

    @pytest.mark.asyncio
    async def test_empty_directory(self, tmp_path: Path) -> None:
        result = await generate_orca_progress_report(str(tmp_path))
        assert "파일이 없습니다" in result

    @pytest.mark.asyncio
    async def test_mixed_statuses(self, tmp_path: Path) -> None:
        _write_state(
            tmp_path / "reactants" / "run_state.json",
            _make_state("reactants", status="running", attempts=1),
        )
        _write_state(
            tmp_path / "products" / "run_state.json",
            _make_state("products", status="completed", attempts=2),
        )
        _write_state(
            tmp_path / "ts" / "run_state.json",
            _make_state(
                "ts", status="failed", attempts=5,
                final_result={"reason": "max retries exceeded"},
            ),
        )

        result = await generate_orca_progress_report(str(tmp_path))

        assert "Total jobs: 3" in result
        assert "Completed: 1" in result
        assert "Failed:    1" in result
        assert "Running:   1" in result
        assert "Active Jobs" in result
        assert "reactants" in result
        assert "Recent Completions" in result
        assert "products" in result
        assert "Failed Jobs" in result
        assert "ts" in result

    @pytest.mark.asyncio
    async def test_corrupt_file_counted(self, tmp_path: Path) -> None:
        _write_state(
            tmp_path / "good" / "run_state.json",
            _make_state("good", status="completed"),
        )
        corrupt = tmp_path / "bad" / "run_state.json"
        corrupt.parent.mkdir(parents=True, exist_ok=True)
        corrupt.write_text("{invalid", encoding="utf-8")

        result = await generate_orca_progress_report(str(tmp_path))
        assert "Total jobs: 1" in result
        assert "Parse errors: 1" in result

    @pytest.mark.asyncio
    async def test_all_completed(self, tmp_path: Path) -> None:
        for name in ("r1", "r2", "r3"):
            _write_state(
                tmp_path / name / "run_state.json",
                _make_state(name, status="completed", attempts=1),
            )
        result = await generate_orca_progress_report(str(tmp_path))
        assert "100.0%" in result
        assert "Active Jobs" not in result
        assert "Failed Jobs" not in result
