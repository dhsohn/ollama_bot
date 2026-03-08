"""Lightweight i18n string catalog.

Simple key-based translation with ``t(key, lang, **kwargs)`` lookup.
Supports Korean (ko) and English (en).
"""

from __future__ import annotations

_STRINGS: dict[str, dict[str, str]] = {
    # ── /start & onboarding ──
    "welcome": {
        "ko": (
            "안녕하세요! {bot_name} 입니다.\n\n"
            "AI 어시스턴트입니다.\n"
            "자유롭게 대화하거나, /help 명령으로 도움말을 확인하세요."
        ),
        "en": (
            "Hello! I'm {bot_name}.\n\n"
            "I'm an AI assistant.\n"
            "Chat freely, or type /help to see available commands."
        ),
    },
    "onboard_welcome": {
        "ko": (
            "반갑습니다! {bot_name}에 오신 것을 환영합니다.\n\n"
            "먼저 언어를 선택해주세요:"
        ),
        "en": (
            "Welcome to {bot_name}!\n\n"
            "Please select your language:"
        ),
    },
    "onboard_lang_set": {
        "ko": "언어가 한국어로 설정되었습니다.",
        "en": "Language has been set to English.",
    },
    "onboard_done": {
        "ko": (
            "설정이 완료되었습니다! 자유롭게 대화를 시작하세요.\n"
            "/help 명령으로 도움말을 확인할 수 있습니다."
        ),
        "en": (
            "Setup complete! Feel free to start chatting.\n"
            "Type /help to see available commands."
        ),
    },
    # ── /help ──
    "help_title": {
        "ko": "사용 가능한 명령어",
        "en": "Available Commands",
    },
    "help_header_cmd": {
        "ko": "명령어",
        "en": "Command",
    },
    "help_header_desc": {
        "ko": "설명",
        "en": "Description",
    },
    "help_chat_mode": {
        "ko": "대화 모드",
        "en": "Chat Mode",
    },
    "help_chat_desc": {
        "ko": "명령어 없이 자유롭게 대화하세요.",
        "en": "Chat freely without any commands.",
    },
    "help_skill_mode": {
        "ko": "스킬 모드",
        "en": "Skill Mode",
    },
    "help_skill_desc": {
        "ko": (
            "스킬 트리거 키워드를 사용하면 전문 기능이 활성화됩니다.\n"
            "/skills 명령으로 스킬 목록을 확인하세요."
        ),
        "en": (
            "Using skill trigger keywords activates specialized features.\n"
            "Type /skills to see the skill list."
        ),
    },
    # ── command descriptions (for BotCommand + /help table) ──
    "cmd_start": {"ko": "봇 시작", "en": "Start bot"},
    "cmd_help": {"ko": "도움말", "en": "Help"},
    "cmd_skills": {"ko": "스킬 목록", "en": "Skills"},
    "cmd_auto": {"ko": "자동화 관리", "en": "Automations"},
    "cmd_memory": {"ko": "메모리 관리", "en": "Memory"},
    "cmd_status": {"ko": "시스템 상태", "en": "System status"},
    "cmd_continue": {"ko": "긴 답변 이어보기", "en": "Continue long response"},
    "cmd_feedback": {"ko": "피드백 통계", "en": "Feedback stats"},
    "cmd_settings": {"ko": "설정", "en": "Settings"},
    # ── /skills ──
    "skills_title": {"ko": "사용 가능한 스킬", "en": "Available Skills"},
    "skills_empty": {"ko": "등록된 스킬이 없습니다.", "en": "No skills registered."},
    "skills_reloaded": {
        "ko": "스킬을 다시 로드했습니다: {count}개",
        "en": "Skills reloaded: {count}",
    },
    "skills_reload_failed": {
        "ko": "스킬 로드 실패: {error}",
        "en": "Skill reload failed: {error}",
    },
    "skills_header_name": {"ko": "스킬", "en": "Skill"},
    "skills_header_desc": {"ko": "설명", "en": "Description"},
    "skills_header_trigger": {"ko": "트리거", "en": "Trigger"},
    # ── /status ──
    "status_title": {"ko": "시스템 상태", "en": "System Status"},
    "status_uptime": {"ko": "가동 시간", "en": "Uptime"},
    "status_llm_backend": {"ko": "LLM 백엔드", "en": "LLM Backend"},
    "status_llm_ok": {"ko": "정상", "en": "OK"},
    "status_llm_error": {"ko": "오류", "en": "Error"},
    "status_model": {"ko": "모델", "en": "Model"},
    "status_skills": {"ko": "로드된 스킬", "en": "Loaded Skills"},
    "status_degraded": {"ko": "Degraded", "en": "Degraded"},
    "status_degraded_none": {"ko": "없음", "en": "None"},
    "status_automations": {"ko": "자동화", "en": "Automations"},
    "status_automations_active": {
        "ko": "{enabled}/{total}개 활성",
        "en": "{enabled}/{total} active",
    },
    "status_feedback": {"ko": "피드백", "en": "Feedback"},
    "status_header_item": {"ko": "항목", "en": "Item"},
    "status_header_value": {"ko": "값", "en": "Value"},
    "status_count_suffix": {"ko": "{count}개", "en": "{count}"},
    "status_seconds_suffix": {"ko": "{seconds}초", "en": "{seconds}s"},
    # ── /memory ──
    "memory_title": {"ko": "메모리 상태", "en": "Memory Status"},
    "memory_conversations": {"ko": "대화 기록", "en": "Conversations"},
    "memory_long_term": {"ko": "장기 메모리", "en": "Long-term Memory"},
    "memory_oldest": {"ko": "가장 오래된 대화", "en": "Oldest Conversation"},
    "memory_none": {"ko": "없음", "en": "None"},
    "memory_cleared": {
        "ko": "대화 기록 {count}건이 삭제되었습니다.",
        "en": "{count} conversation(s) cleared.",
    },
    "memory_exported": {
        "ko": "대화 기록이 내보내기되었습니다: ",
        "en": "Conversation exported: ",
    },
    "memory_usage": {"ko": "사용법: /memory [clear|export]", "en": "Usage: /memory [clear|export]"},
    "memory_count": {"ko": "{count}건", "en": "{count}"},
    # ── /auto ──
    "auto_title": {"ko": "자동화 목록", "en": "Automations"},
    "auto_no_scheduler": {
        "ko": "자동화 스케줄러가 초기화되지 않았습니다.",
        "en": "Automation scheduler not initialized.",
    },
    "auto_empty": {"ko": "등록된 자동화가 없습니다.", "en": "No automations registered."},
    "auto_header_name": {"ko": "이름", "en": "Name"},
    "auto_header_schedule": {"ko": "스케줄", "en": "Schedule"},
    "auto_header_desc": {"ko": "설명", "en": "Description"},
    "auto_disabled": {
        "ko": "'{name}' 자동화가 비활성화되었습니다.",
        "en": "Automation '{name}' disabled.",
    },
    "auto_not_found": {
        "ko": "'{name}' 자동화를 찾을 수 없습니다.",
        "en": "Automation '{name}' not found.",
    },
    "auto_reloaded": {
        "ko": "자동화를 다시 로드했습니다: {count}개",
        "en": "Automations reloaded: {count}",
    },
    "auto_reload_failed": {
        "ko": "자동화 로드 실패: {error}",
        "en": "Automation reload failed: {error}",
    },
    "auto_run_success": {
        "ko": "'{name}' 자동화를 수동 실행했습니다.",
        "en": "Automation '{name}' executed.",
    },
    "auto_run_failed": {
        "ko": "'{name}' 자동화 실행에 실패했습니다. 로그를 확인하세요.",
        "en": "Automation '{name}' failed. Check logs.",
    },
    "auto_is_disabled": {
        "ko": "'{name}' 자동화는 비활성화 상태입니다.",
        "en": "Automation '{name}' is disabled.",
    },
    "auto_usage": {
        "ko": "사용법: /auto [list|disable <이름>|run <이름>|reload]",
        "en": "Usage: /auto [list|disable <name>|run <name>|reload]",
    },
    # ── /feedback ──
    "feedback_title": {"ko": "피드백 통계", "en": "Feedback Stats"},
    "feedback_disabled": {
        "ko": "피드백 기능이 비활성화되어 있습니다.",
        "en": "Feedback feature is disabled.",
    },
    "feedback_total": {"ko": "전체", "en": "Total"},
    "feedback_positive": {"ko": "긍정", "en": "Positive"},
    "feedback_negative": {"ko": "부정", "en": "Negative"},
    "feedback_satisfaction": {"ko": "만족도", "en": "Satisfaction"},
    "feedback_thanks": {"ko": "피드백 감사합니다!", "en": "Thanks for your feedback!"},
    "feedback_updated": {
        "ko": "피드백을 업데이트했어요.",
        "en": "Feedback updated.",
    },
    "feedback_invalid": {
        "ko": "잘못된 피드백 요청입니다.",
        "en": "Invalid feedback request.",
    },
    "feedback_unsupported": {
        "ko": "지원하지 않는 피드백 값입니다.",
        "en": "Unsupported feedback value.",
    },
    "feedback_reason_ask": {
        "ko": "어떤 점이 아쉬웠나요? 사유를 입력해주세요.\n건너뛰려면 /skip 을 입력하세요.",
        "en": "What could be improved? Please provide a reason.\nType /skip to skip.",
    },
    "feedback_reason_replaced": {
        "ko": "이전 사유 입력 요청은 자동 만료되어 최신 요청으로 교체되었어요.",
        "en": "Previous reason request expired and replaced with the latest one.",
    },
    "feedback_reason_expired": {
        "ko": "사유 입력 시간이 만료되었습니다.",
        "en": "Reason input time expired.",
    },
    "feedback_reason_min_chars": {
        "ko": "사유는 최소 {min_chars}자 이상 입력해주세요. 건너뛰려면 /skip",
        "en": "Reason must be at least {min_chars} characters. Type /skip to skip.",
    },
    "feedback_reason_saved": {
        "ko": "사유가 기록되었습니다. 감사합니다!",
        "en": "Reason recorded. Thank you!",
    },
    "feedback_reason_not_found": {
        "ko": "사유를 저장할 대상 피드백을 찾지 못했습니다.",
        "en": "Could not find the target feedback to save the reason.",
    },
    "feedback_reason_skipped": {
        "ko": "사유 입력을 건너뛰었습니다.",
        "en": "Reason input skipped.",
    },
    "feedback_reason_no_pending": {
        "ko": "건너뛸 사유 요청이 없습니다.",
        "en": "No pending reason request to skip.",
    },
    # ── common ──
    "private_chat_only": {
        "ko": "이 봇은 private chat에서만 동작합니다.",
        "en": "This bot only works in private chats.",
    },
    "rate_limited": {
        "ko": "요청이 너무 많습니다. 잠시 후 다시 시도해주세요.",
        "en": "Too many requests. Please try again later.",
    },
    "concurrency_limited": {
        "ko": "현재 요청이 많습니다. 잠시 후 다시 시도해주세요.",
        "en": "Currently busy. Please try again shortly.",
    },
    "reload_warnings": {
        "ko": "\n\n⚠️ 일부 항목 로드 실패({count}건)",
        "en": "\n\n⚠️ Some items failed to load ({count})",
    },
    "reload_more": {
        "ko": "- ... 외 {count}건",
        "en": "- ... and {count} more",
    },
    # ── menu ──
    "menu_title": {"ko": "메뉴", "en": "Menu"},
    "menu_btn_skills": {"ko": "스킬", "en": "Skills"},
    "menu_btn_memory": {"ko": "메모리", "en": "Memory"},
    "menu_btn_status": {"ko": "상태", "en": "Status"},
    "menu_btn_help": {"ko": "도움말", "en": "Help"},
    "menu_btn_settings": {"ko": "설정", "en": "Settings"},
    "menu_btn_auto": {"ko": "자동화", "en": "Automations"},
    # ── settings ──
    "settings_title": {"ko": "설정", "en": "Settings"},
    "settings_language": {"ko": "언어 설정", "en": "Language Setting"},
    "settings_select_language": {
        "ko": "언어를 선택하세요:",
        "en": "Select your language:",
    },
}

# Default language for the bot
_DEFAULT_LANG = "ko"


def t(key: str, lang: str = "ko", **kwargs: object) -> str:
    """Translate a string key to the given language.

    Falls back to Korean, then to the raw key if not found.
    """
    strings = _STRINGS.get(key)
    if strings is None:
        return key
    text = strings.get(lang) or strings.get(_DEFAULT_LANG) or key
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError):
            return text
    return text
