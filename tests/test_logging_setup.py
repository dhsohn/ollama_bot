"""로깅 설정 테스트."""

from __future__ import annotations

import logging
from pathlib import Path

from core.logging_setup import setup_logging


class TestSetupLogging:
    def test_stdout_only_no_log_dir(self) -> None:
        """log_dir 없이 호출하면 파일 핸들러가 생성되지 않는다."""
        setup_logging("DEBUG")
        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers if hasattr(h, "baseFilename")
        ]
        assert len(file_handlers) == 0

    def test_file_handler_created_with_log_dir(self, tmp_path: Path) -> None:
        """log_dir를 지정하면 파일 핸들러가 생성된다."""
        log_dir = str(tmp_path / "logs")
        setup_logging("DEBUG", log_dir=log_dir)
        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers if hasattr(h, "baseFilename")
        ]
        assert len(file_handlers) == 1
        assert (tmp_path / "logs" / "app.log").exists()

    def test_log_dir_created_if_not_exists(self, tmp_path: Path) -> None:
        """로그 디렉토리가 없으면 자동 생성된다."""
        log_dir = str(tmp_path / "new_logs")
        setup_logging("INFO", log_dir=log_dir)
        assert (tmp_path / "new_logs").is_dir()
