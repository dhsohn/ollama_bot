"""DFT 계산 파일 변경 감지 및 자동 인덱싱.

kb_dirs를 주기적으로 스캔하여 새로 완료된 ORCA 계산을 감지하고,
DFT 인덱스에 등록 후 텔레그램으로 알림을 전송한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from core.orca_parser import parse_orca_output

_ORCA_EXTENSIONS = {".out", ".log"}


def build_dft_monitor_callable(
    dft_index: Any,
    kb_dirs: list[str],
    logger: Any,
):
    """DFT 모니터 callable을 빌드한다.

    Args:
        dft_index: DFTIndex 인스턴스
        kb_dirs: 모니터링할 디렉토리 목록
        logger: 로거

    Returns:
        dft_monitor async callable
    """
    # 마지막 스캔 시점의 mtime 캐시
    _last_mtimes: dict[str, float] = {}

    async def dft_monitor(
        max_file_size_mb: int = 64,
        model: str | None = None,
        model_role: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """kb_dirs에서 새로/변경된 ORCA 파일을 감지하여 인덱싱한다.

        Returns:
            새 계산이 있으면 알림 텍스트, 없으면 빈 문자열
        """
        max_bytes = max_file_size_mb * 1024 * 1024
        new_results: list[dict[str, str]] = []

        for kb_dir in kb_dirs:
            kb_path = Path(kb_dir)
            if not kb_path.is_dir():
                continue

            for ext in _ORCA_EXTENSIONS:
                for fpath in kb_path.rglob(f"*{ext}"):
                    if not fpath.is_file():
                        continue
                    if fpath.stat().st_size > max_bytes:
                        continue

                    spath = str(fpath)
                    current_mtime = os.path.getmtime(spath)
                    last_mtime = _last_mtimes.get(spath, 0.0)

                    if current_mtime <= last_mtime:
                        continue

                    # 새 파일 또는 변경된 파일 발견
                    try:
                        result = parse_orca_output(spath)
                        success = await dft_index.upsert_single(spath)
                        if success:
                            _last_mtimes[spath] = current_mtime

                            # 알림 정보 수집
                            energy_str = (
                                f"E = {result.energy_hartree:.6f} Eh"
                                if result.energy_hartree is not None
                                else "E = N/A"
                            )
                            method_basis = result.method
                            if result.basis_set:
                                method_basis += f"/{result.basis_set}"

                            status_emoji = {
                                "completed": "OK",
                                "failed": "FAIL",
                                "running": "RUNNING",
                            }.get(result.status, result.status)

                            notes: list[str] = []
                            if result.opt_converged is False:
                                notes.append("NOT CONVERGED")
                            if result.has_imaginary_freq:
                                notes.append("imaginary freq")

                            note_str = f" ({', '.join(notes)})" if notes else ""

                            new_results.append({
                                "formula": result.formula or "unknown",
                                "method_basis": method_basis or "unknown",
                                "energy": energy_str,
                                "status": status_emoji,
                                "calc_type": result.calc_type,
                                "path": _short_path(spath),
                                "note": note_str,
                            })

                            logger.info(
                                "dft_monitor_new_calc",
                                path=spath,
                                formula=result.formula,
                                method=result.method,
                                status=result.status,
                            )
                    except Exception as exc:
                        logger.warning(
                            "dft_monitor_parse_error",
                            path=spath,
                            error=str(exc),
                        )

        if not new_results:
            return ""

        # 알림 메시지 포맷
        lines = [f"DFT Monitor: {len(new_results)}건의 새 계산 감지\n"]
        for r in new_results:
            lines.append(
                f"  {r['formula']} | {r['calc_type']} | "
                f"{r['method_basis']} | {r['energy']} | "
                f"{r['status']}{r['note']}"
            )
            lines.append(f"    -> {r['path']}")

        return "\n".join(lines)

    return dft_monitor


def _short_path(path: str) -> str:
    """긴 경로를 마지막 3개 세그먼트로 축약한다."""
    parts = path.replace("\\", "/").split("/")
    if len(parts) <= 3:
        return path
    return "/".join(parts[-3:])
