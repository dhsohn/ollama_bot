"""ollama_bot 앱 진입점."""

from __future__ import annotations

from core.app_runtime import run_app


def main() -> None:
    """LLM_PROVIDER 설정(ollama/lemonade)에 따라 봇을 실행한다."""
    run_app(app_name="ollama_bot")


if __name__ == "__main__":
    main()
