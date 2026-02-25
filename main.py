"""레거시 호환 진입점.

현재 기본 실행 경로는 `apps/ollama_bot/main.py` 이다.
기존 `python main.py` 사용자를 위해 유지한다.
"""

from apps.ollama_bot.main import main


if __name__ == "__main__":
    main()
