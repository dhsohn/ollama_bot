"""TelegramHandler 시뮬레이션 큐 명령 구현."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

from telegram.constants import ParseMode
from telegram.ext import ContextTypes

if TYPE_CHECKING:
    from datetime import datetime

    from telegram import Update

    from core.sim_scheduler import SimJobScheduler
    from core.telegram_handler import TelegramHandler


async def cmd_sim(
    self: TelegramHandler,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = update.effective_message
    if message is None:
        return

    if self._sim_scheduler is None:
        await message.reply_text("시뮬레이션 큐가 활성화되어 있지 않습니다.")
        return

    args = context.args or []
    if not args:
        await message.reply_text(
            "사용법: /sim [submit|list|status|info|cancel|priority|retry|tools]"
        )
        return

    subcmd = args[0].lower()
    handlers = {
        "submit": self._sim_submit,
        "list": self._sim_list,
        "status": self._sim_status,
        "info": self._sim_info,
        "cancel": self._sim_cancel,
        "priority": self._sim_priority,
        "retry": self._sim_retry,
        "tools": self._sim_tools,
        "clear": self._sim_clear,
    }
    handler = handlers.get(subcmd)
    if handler is None:
        await message.reply_text(f"알 수 없는 서브커맨드: {subcmd}")
        return
    await handler(update, args[1:])


async def get_sim_scheduler(
    self: TelegramHandler,
    update: Update,
) -> SimJobScheduler | None:
    sim_scheduler = self._sim_scheduler
    if sim_scheduler is None:
        message = update.effective_message
        if message is not None:
            await message.reply_text("시뮬레이션 큐가 활성화되어 있지 않습니다.")
        return None
    return sim_scheduler


async def sim_submit(self: TelegramHandler, update: Update, args: list[str]) -> None:
    """Usage: /sim submit <tool> <input_file> [--priority N] [--label TEXT]"""
    sim_scheduler = await self._get_sim_scheduler(update)
    if sim_scheduler is None:
        return

    if len(args) < 2:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "사용법: /sim submit <tool> <input_file> [--priority N] [--label TEXT]"
        )
        return

    tool = args[0]
    input_file = args[1]
    priority = 100
    label = ""

    i = 2
    while i < len(args):
        if args[i] == "--priority" and i + 1 < len(args):
            with suppress(ValueError):
                priority = int(args[i + 1])
            i += 2
        elif args[i] == "--label" and i + 1 < len(args):
            label = args[i + 1]
            i += 2
        else:
            i += 1

    try:
        result = await sim_scheduler.submit_job(
            tool=tool,
            input_file=input_file,
            submitted_by=update.effective_chat.id,  # type: ignore[union-attr]
            priority=priority,
            label=label,
        )
        display_job_id = (
            result.job_id if result.job_id.startswith("ext-") else result.job_id[:8]
        )
        msg = f"작업 등록 완료: {display_job_id}\n도구: {tool} | 우선순위: {priority}"
        if result.cancelled_job_id:
            msg += f"\n(기존 대기 작업 {result.cancelled_job_id[:8]} 자동 취소됨)"
        await update.effective_message.reply_text(msg)  # type: ignore[union-attr]
    except (ValueError, FileNotFoundError) as exc:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"등록 실패: {exc}"
        )


def sim_elapsed_text(job: dict[str, Any]) -> str:
    """작업의 총 경과 시간을 사람이 읽기 좋은 형태로 반환한다."""
    elapsed = int(job.get("elapsed_seconds", 0))
    if not elapsed and job.get("started_at"):
        from datetime import UTC, datetime

        try:
            started = datetime.fromisoformat(str(job["started_at"]))
            if started.tzinfo is None:
                started = started.replace(tzinfo=UTC)
            elapsed = max(0, int((datetime.now(UTC) - started).total_seconds()))
        except (ValueError, TypeError):
            elapsed = 0
    if elapsed <= 0:
        return "-"
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


async def sim_list(self: TelegramHandler, update: Update, args: list[str]) -> None:
    sim_scheduler = await self._get_sim_scheduler(update)
    if sim_scheduler is None:
        return

    status_filter = args[0] if args else None
    if status_filter == "external":
        jobs = [
            job
            for job in await sim_scheduler.list_jobs(limit=50)
            if (
                str(job.get("status")) == "running"
                and str(job.get("cli_command") or "").startswith("delegated:")
            )
        ]
    else:
        jobs = await sim_scheduler.list_jobs(status=status_filter)

    external_jobs: list[dict[str, Any]] = []
    if status_filter in (None, "all", "running", "external"):
        external_jobs = await sim_scheduler.get_external_running_jobs()
        tracked_external_pids = {
            int(job["pid"])
            for job in jobs
            if isinstance(job.get("pid"), int)
            and (
                str(job.get("status")) == "running"
                and str(job.get("cli_command") or "").startswith("delegated:")
            )
        }
        if tracked_external_pids:
            external_jobs = [
                job
                for job in external_jobs
                if not (
                    isinstance((pid_val := job.get("pid")), int)
                    and pid_val in tracked_external_pids
                )
            ]

    if not jobs and not external_jobs:
        await update.effective_message.reply_text("등록된 작업이 없습니다.")  # type: ignore[union-attr]
        return

    table_rows: list[str] = []
    for job in jobs:
        status = str(job.get("status") or "")
        tool = str(job.get("tool") or "")
        elapsed_text = self._sim_elapsed_text(job)
        table_rows.append(
            f"{job['job_id'][:8]:<10s} {status:<9s} {tool:<8s} {elapsed_text}"
        )

    for job in external_jobs:
        ext_status = str(job.get("status") or "running")
        tool = str(job.get("tool") or "")
        elapsed_text = self._sim_elapsed_text(job)
        display_id = str(job.get("job_id") or "")[:10]
        table_rows.append(
            f"{display_id:<10s} {ext_status:<9s} {tool:<8s} {elapsed_text}"
        )

    header = "ID         상태      도구     경과시간"
    sep = "─" * len(header)
    table = chr(10).join([header, sep, *table_rows])

    text = f"<b>시뮬레이션 작업 목록</b>\n\n<pre>{table}</pre>"
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        text,
        parse_mode=ParseMode.HTML,
    )


async def sim_clear(self: TelegramHandler, update: Update, args: list[str]) -> None:
    _ = args
    sim_scheduler = await self._get_sim_scheduler(update)
    if sim_scheduler is None:
        return
    count = await sim_scheduler.clear_finished()
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        f"완료/실패/취소 작업 {count}건 삭제됨."
    )


async def sim_status(self: TelegramHandler, update: Update, args: list[str]) -> None:
    _ = args
    sim_scheduler = await self._get_sim_scheduler(update)
    if sim_scheduler is None:
        return

    status = await sim_scheduler.get_queue_status()
    queue_running = status.get("running", 0)
    running_total = status.get("running_total", queue_running)
    detected_running = max(0, int(running_total) - int(queue_running))
    running_detail = ""
    if detected_running:
        running_detail = f"  (감지:{detected_running})"

    q_rows: list[tuple[str, str]] = [
        ("상태", "건수"),
        ("─" * 10, "─" * 6),
        ("대기", str(status.get("queued", 0))),
        ("실행 중", f"{running_total}{running_detail}"),
    ]
    completed = int(status.get("completed", 0))
    failed = int(status.get("failed", 0))
    if completed:
        q_rows.append(("완료", str(completed)))
    if failed:
        q_rows.append(("실패", str(failed)))
    q_table = chr(10).join(f"{row[0]:<10s} {row[1]}" for row in q_rows)

    footer = f"동시실행: {running_total}/{status.get('max_concurrent', 0)}"
    text = f"<b>시뮬레이션 큐 현황</b>\n\n<pre>{q_table}\n\n{footer}</pre>"
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        text,
        parse_mode=ParseMode.HTML,
    )


async def sim_info(self: TelegramHandler, update: Update, args: list[str]) -> None:
    sim_scheduler = await self._get_sim_scheduler(update)
    if sim_scheduler is None:
        return

    if not args:
        await update.effective_message.reply_text("사용법: /sim info <job_id>")  # type: ignore[union-attr]
        return

    job_id_prefix = args[0]
    jobs = await sim_scheduler.list_jobs(limit=50)
    matched = [job for job in jobs if job["job_id"].startswith(job_id_prefix)]
    if not matched:
        external_jobs = await sim_scheduler.get_external_running_jobs()
        matched_external = [
            job for job in external_jobs if str(job.get("job_id", "")).startswith(job_id_prefix)
        ]
        if not matched_external and job_id_prefix.isdigit():
            matched_external = [
                job for job in external_jobs if str(job.get("pid", "")).startswith(job_id_prefix)
            ]

        if matched_external:
            ext = matched_external[0]
            elapsed_seconds = int(ext.get("elapsed_seconds", 0))
            elapsed_h = elapsed_seconds // 3600
            elapsed_m = (elapsed_seconds % 3600) // 60
            elapsed_s = elapsed_seconds % 60
            if elapsed_h:
                elapsed_text = f"{elapsed_h}h {elapsed_m}m {elapsed_s}s"
            elif elapsed_m:
                elapsed_text = f"{elapsed_m}m {elapsed_s}s"
            else:
                elapsed_text = f"{elapsed_s}s"

            cmd = str(ext.get("cli_command") or "").strip()
            if len(cmd) > 1000:
                cmd = f"{cmd[:1000]}..."

            rows: list[tuple[str, str]] = [
                ("유형", "detected"),
                ("도구", self._escape_html(str(ext.get("tool", "-")))),
                ("상태", self._escape_html(str(ext.get("status", "running")))),
                ("PID", str(ext.get("pid", "-"))),
                ("경과", elapsed_text),
                ("입력", self._escape_html(str(ext.get("input_file") or "-"))),
                ("타임스탬프", "감지 프로세스라 추적 불가"),
            ]
            if cmd:
                rows.append(("명령", self._escape_html(cmd)))

            lw = max(len(row[0]) for row in rows)
            table_lines = [f"{'항목':<{lw}s}  값", "─" * (lw + 20)]
            for label, value in rows:
                table_lines.append(f"{label:<{lw}s}  {value}")
            table = chr(10).join(table_lines)
            text = f"<b>감지 작업 상세: {ext['job_id']}</b>\n\n<pre>{table}</pre>"
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                text,
                parse_mode=ParseMode.HTML,
            )
            return

        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"작업을 찾을 수 없음: {job_id_prefix}"
        )
        return

    job = matched[0]
    rows = [
        ("도구", self._escape_html(job["tool"])),
        ("상태", str(job["status"])),
        ("입력", self._escape_html(job["input_file"])),
        ("우선순위", str(job["priority"])),
        ("재시도", f"{job['retry_count']}/{job['max_retries']}"),
        ("제출", str(job["submitted_at"] or "-")),
        ("시작", str(job["started_at"] or "-")),
        ("완료", str(job["completed_at"] or "-")),
    ]
    if job.get("label"):
        rows.append(("라벨", self._escape_html(job["label"])))
    if job.get("exit_code") is not None:
        rows.append(("종료코드", str(job["exit_code"])))
    if job.get("error_message"):
        rows.append(("오류", self._escape_html(job["error_message"])))
    if job.get("output_file"):
        rows.append(("출력", self._escape_html(job["output_file"])))

    lw = max(len(row[0]) for row in rows)
    table_lines = [f"{'항목':<{lw}s}  값", "─" * (lw + 20)]
    for label, value in rows:
        table_lines.append(f"{label:<{lw}s}  {value}")
    table = chr(10).join(table_lines)
    text = f"<b>작업 상세: {job['job_id'][:8]}</b>\n\n<pre>{table}</pre>"
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        text,
        parse_mode=ParseMode.HTML,
    )


async def sim_cancel(self: TelegramHandler, update: Update, args: list[str]) -> None:
    sim_scheduler = await self._get_sim_scheduler(update)
    if sim_scheduler is None:
        return

    if not args:
        await update.effective_message.reply_text("사용법: /sim cancel <job_id>")  # type: ignore[union-attr]
        return

    job_id_prefix = args[0]
    jobs = await sim_scheduler.list_jobs(limit=50)
    matched = [job for job in jobs if job["job_id"].startswith(job_id_prefix)]
    if matched:
        success = await sim_scheduler.cancel_job(matched[0]["job_id"])
        if success:
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                f"작업 {matched[0]['job_id'][:8]} 취소 완료"
            )
        else:
            await update.effective_message.reply_text(  # type: ignore[union-attr]
                f"작업 {matched[0]['job_id'][:8]} 취소 불가 (이미 완료/실패/취소됨)"
            )
        return

    external_jobs = await sim_scheduler.get_external_running_jobs()
    matched_external = [
        job for job in external_jobs if str(job.get("job_id", "")).startswith(job_id_prefix)
    ]
    if not matched_external and job_id_prefix.isdigit():
        matched_external = [
            job for job in external_jobs if str(job.get("pid", "")).startswith(job_id_prefix)
        ]

    if not matched_external:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"작업을 찾을 수 없음: {job_id_prefix}"
        )
        return

    target = matched_external[0]
    pid = target.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "외부 작업 PID를 확인할 수 없어 취소할 수 없습니다."
        )
        return

    success = await sim_scheduler.cancel_external_job(pid)
    if success:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"감지 작업 {target['job_id']} (PID:{pid}) 종료 완료"
        )
    else:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"감지 작업 {target['job_id']} 종료 불가 (이미 종료됨/권한 없음)"
        )


async def sim_priority(self: TelegramHandler, update: Update, args: list[str]) -> None:
    sim_scheduler = await self._get_sim_scheduler(update)
    if sim_scheduler is None:
        return

    if len(args) < 2:
        await update.effective_message.reply_text("사용법: /sim priority <job_id> <priority>")  # type: ignore[union-attr]
        return

    job_id_prefix = args[0]
    try:
        new_priority = int(args[1])
    except ValueError:
        await update.effective_message.reply_text("우선순위는 정수여야 합니다.")  # type: ignore[union-attr]
        return

    jobs = await sim_scheduler.list_jobs(limit=50)
    matched = [job for job in jobs if job["job_id"].startswith(job_id_prefix)]
    if not matched:
        await update.effective_message.reply_text(f"작업을 찾을 수 없음: {job_id_prefix}")  # type: ignore[union-attr]
        return

    success = await sim_scheduler.reprioritize(matched[0]["job_id"], new_priority)
    if success:
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            f"작업 {matched[0]['job_id'][:8]} 우선순위 → {new_priority}"
        )
    else:
        await update.effective_message.reply_text("우선순위 변경 불가 (대기 상태가 아님)")  # type: ignore[union-attr]


async def sim_retry(self: TelegramHandler, update: Update, args: list[str]) -> None:
    sim_scheduler = await self._get_sim_scheduler(update)
    if sim_scheduler is None:
        return

    if not args:
        await update.effective_message.reply_text("사용법: /sim retry <job_id>")  # type: ignore[union-attr]
        return

    job_id_prefix = args[0]
    jobs = await sim_scheduler.list_jobs(limit=50)
    matched = [job for job in jobs if job["job_id"].startswith(job_id_prefix)]
    if not matched:
        await update.effective_message.reply_text(f"작업을 찾을 수 없음: {job_id_prefix}")  # type: ignore[union-attr]
        return

    old_job = matched[0]
    if old_job["status"] not in ("failed", "cancelled"):
        await update.effective_message.reply_text(  # type: ignore[union-attr]
            "실패 또는 취소된 작업만 재시도할 수 있습니다."
        )
        return

    try:
        result = await sim_scheduler.submit_job(
            tool=old_job["tool"],
            input_file=old_job["input_file"],
            submitted_by=update.effective_chat.id,  # type: ignore[union-attr]
            priority=old_job["priority"],
            label=old_job.get("label", ""),
        )
        msg = f"재시도 등록: {result.job_id[:8]} (원본: {old_job['job_id'][:8]})"
        if result.cancelled_job_id:
            msg += f"\n(기존 대기 작업 {result.cancelled_job_id[:8]} 자동 취소됨)"
        await update.effective_message.reply_text(msg)  # type: ignore[union-attr]
    except (ValueError, FileNotFoundError) as exc:
        await update.effective_message.reply_text(f"재시도 실패: {exc}")  # type: ignore[union-attr]


async def sim_tools(self: TelegramHandler, update: Update, args: list[str]) -> None:
    _ = args
    sim_scheduler = await self._get_sim_scheduler(update)
    if sim_scheduler is None:
        return

    tools = sim_scheduler.get_tools()
    if not tools:
        await update.effective_message.reply_text("설정된 시뮬레이션 도구가 없습니다.")  # type: ignore[union-attr]
        return

    name_w = max(4, max(len(name) for name in tools))
    header = f"{'도구':<{name_w}s}  상태    실행파일"
    sep = "─" * max(len(header), 40)
    table_lines = [header, sep]
    for name, info in tools.items():
        status = "활성" if info["enabled"] else "비활성"
        table_lines.append(
            f"{self._escape_html(name):<{name_w}s}  {status:<6s}  {self._escape_html(info['executable'])}"
        )

    table = chr(10).join(table_lines)
    text = f"<b>시뮬레이션 도구 목록</b>\n\n<pre>{table}</pre>"
    await update.effective_message.reply_text(  # type: ignore[union-attr]
        text,
        parse_mode=ParseMode.HTML,
    )
