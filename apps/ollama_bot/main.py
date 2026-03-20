"""ollama_bot application entrypoint."""

from __future__ import annotations

from core.app_runtime import run_app


def main() -> None:
    """Run the bot on the Ollama-only stack."""
    run_app(app_name="ollama_bot")


if __name__ == "__main__":
    main()
