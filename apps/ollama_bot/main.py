"""ollama_bot 앱 진입점."""

from __future__ import annotations

from core.app_runtime import run_app


def main() -> None:
    """Dual-Provider 고정 구성(lemonade chat + ollama retrieval)으로 봇을 실행한다."""
    run_app(app_name="ollama_bot")


if __name__ == "__main__":
    main()
