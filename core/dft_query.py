"""DFT 자연어 질의 엔진 — 사용자 질문을 구조화 쿼리로 변환한다.

자연어 패턴 매칭으로 의도를 파악하고, DFTIndex를 통해 SQL 검색 후
마크다운 테이블로 포맷하여 LLM 컨텍스트에 주입한다.
"""

from __future__ import annotations

import re
from typing import Any

from core.dft_index import DFTIndex
from core.logging_setup import get_logger
from core.orca_parser import HARTREE_TO_KCALMOL

# ---------------------------------------------------------------------------
# 의도 인식 패턴
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    # 통계/현황
    (re.compile(r"통계|현황|summary|stats|몇\s*개|얼마나|전체", re.IGNORECASE), "stats"),
    # 실패한 계산
    (re.compile(r"실패|fail|수렴.{0,4}안|not.*converg|error", re.IGNORECASE), "failed"),
    # 에너지 비교 / 상대 에너지
    (re.compile(r"비교|compare|상대\s*에너지|relative", re.IGNORECASE), "compare"),
    # 에너지 낮은 구조
    (re.compile(r"에너지.{0,8}낮|lowest.*energy|가장.{0,6}안정|most\s*stable|minimum", re.IGNORECASE), "lowest_energy"),
    # 최근 계산
    (re.compile(r"최근|recent|마지막|latest|새로운|new", re.IGNORECASE), "recent"),
    # imaginary frequency
    (re.compile(r"imaginary|음의\s*주파수|허수|im.*freq", re.IGNORECASE), "imaginary_freq"),
    # calc_type 필터
    (re.compile(r"\bfreq\b|진동|vibra|주파수", re.IGNORECASE), "by_calctype_freq"),
    (re.compile(r"\bopt\b|최적화|구조\s*최적|geometry\s*opt", re.IGNORECASE), "by_calctype_opt"),
    (re.compile(r"\bts\b|전이\s*상태|transition\s*state|saddle", re.IGNORECASE), "by_calctype_ts"),
]

# 메서드 추출 패턴 — 한국어 조사(로, 을, 의 등) 바로 뒤에서도 매칭되도록
# \b 대신 lookbehind/lookahead 사용
_METHOD_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(B3LYP|PBE0|PBE|BP86|TPSS|M06-2X|M06|HF|MP2|"
    r"CCSD\(?T?\)?|DLPNO-CCSD\(T\)|"
    r"[wω]B97[XM]?[-\w]*|"
    r"B2PLYP|BLYP|CAM-B3LYP|r2SCAN-3c|B97-3c|PBEh-3c|REVPBE)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

# 기저함수 추출 패턴
_BASIS_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(def2-(?:QZVPP?|TZVPP?|SVP|SV\(P\))|"
    r"ma-def2-\w+|"
    r"(?:aug-)?cc-pV[DTQ5]Z|"
    r"6-31[1+]*G[\w(),*]*|"
    r"STO-3G)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

# 화학식 패턴 (최소 2글자, 대문자 시작)
_FORMULA_RE = re.compile(r"\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*){0,20})\b")

# DFT 관련 트리거 키워드 — 이 키워드가 없으면 DFT 쿼리를 실행하지 않음
_DFT_TRIGGER_RE = re.compile(
    r"dft|orca|계산|에너지|energy|결과|freq|opt|수렴|converge|기저|basis|"
    r"hartree|gibbs|enthalpy|분자|molecule|구조|structure|"
    r"비교|compare|method|메서드",
    re.IGNORECASE,
)


class DFTQueryEngine:
    """자연어 질의를 DFT 인덱스 쿼리로 변환하여 실행한다."""

    def __init__(self, dft_index: DFTIndex) -> None:
        self._index = dft_index
        self._logger = get_logger("dft_query")

    async def process_query(self, text: str) -> str | None:
        """사용자 메시지를 분석하여 DFT 쿼리 결과를 반환한다.

        DFT 관련 질의가 아니면 None을 반환한다 (RAG fallthrough).
        """
        if not _DFT_TRIGGER_RE.search(text):
            return None

        intent = self._detect_intent(text)
        method = self._extract_method(text)
        basis = self._extract_basis(text)
        formula = self._extract_formula(text)

        self._logger.info(
            "dft_query_detected",
            intent=intent,
            method=method,
            basis=basis,
            formula=formula,
        )

        try:
            if intent == "stats":
                return await self._format_stats()
            if intent == "failed":
                return await self._format_query_results(
                    await self._index.query({"status": "failed", "limit": 20}),
                    title="실패한 DFT 계산",
                )
            if intent == "compare":
                return await self._format_comparison(formula=formula, method=method)
            if intent == "lowest_energy":
                return await self._format_query_results(
                    await self._index.get_lowest_energy(formula=formula, limit=10),
                    title="에너지가 낮은 계산 결과",
                )
            if intent == "recent":
                return await self._format_query_results(
                    await self._index.get_recent(limit=10),
                    title="최근 DFT 계산",
                )
            if intent == "imaginary_freq":
                return await self._format_query_results(
                    await self._index.query({"has_imaginary_freq": True, "limit": 20}),
                    title="Imaginary frequency가 있는 계산",
                )
            if intent == "by_calctype_freq":
                filters: dict[str, Any] = {"calc_type": "freq", "limit": 20}
                if formula:
                    filters["formula"] = formula
                results = await self._index.query(filters)
                if not results:
                    filters["calc_type"] = "opt+freq"
                    results = await self._index.query(filters)
                return await self._format_query_results(results, title="Frequency 계산")
            if intent == "by_calctype_opt":
                filters = {"calc_type": "opt", "limit": 20}
                if formula:
                    filters["formula"] = formula
                results = await self._index.query(filters)
                if not results:
                    filters["calc_type"] = "opt+freq"
                    results = await self._index.query(filters)
                return await self._format_query_results(results, title="최적화 계산")
            if intent == "by_calctype_ts":
                filters = {"calc_type": "ts", "limit": 20}
                results = await self._index.query(filters)
                if not results:
                    filters["calc_type"] = "ts+freq"
                    results = await self._index.query(filters)
                return await self._format_query_results(results, title="전이상태 계산")

            # 기본: 필터 조합 검색
            filters = {"limit": 20}
            if method:
                filters["method_like"] = method
            if basis:
                filters["basis_set"] = basis
            if formula:
                filters["formula"] = formula
            results = await self._index.query(filters)
            if not results and formula:
                results = await self._index.search_by_formula(formula)
            if results:
                return await self._format_query_results(results, title="DFT 계산 결과")

            # 결과 없으면 전체 통계라도 반환
            stats = await self._index.get_stats()
            if stats.get("total", 0) > 0:
                return await self._format_stats()

        except Exception as exc:
            self._logger.warning("dft_query_error", error=str(exc))

        return None

    # ------------------------------------------------------------------
    # 의도 / 필터 추출
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_intent(text: str) -> str:
        for pattern, intent in _INTENT_PATTERNS:
            if pattern.search(text):
                return intent
        return "general"

    @staticmethod
    def _extract_method(text: str) -> str | None:
        m = _METHOD_RE.search(text)
        return m.group(1) if m else None

    @staticmethod
    def _extract_basis(text: str) -> str | None:
        m = _BASIS_RE.search(text)
        return m.group(1) if m else None

    @staticmethod
    def _extract_formula(text: str) -> str | None:
        # 화학식 후보 추출 — 너무 짧거나 일반 단어면 제외
        _common_words = {
            "DFT", "ORCA", "SCF", "HF", "MP", "TS", "SP", "MD", "NEB",
            "IRC", "OK", "IT", "IS", "IN", "ON", "OR", "AN", "AT", "BY",
            "DO", "GO", "IF", "NO", "SO", "TO", "UP", "WE",
        }
        candidates = _FORMULA_RE.findall(text)
        for c in candidates:
            if len(c) < 2:
                continue
            if c.upper() in _common_words:
                continue
            # 최소 하나의 숫자 또는 2종류 이상의 원소가 포함되어야 함
            has_digit = any(ch.isdigit() for ch in c)
            upper_count = sum(1 for ch in c if ch.isupper())
            if has_digit or upper_count >= 2:
                return c
        return None

    # ------------------------------------------------------------------
    # 포맷팅
    # ------------------------------------------------------------------

    async def _format_stats(self) -> str:
        stats = await self._index.get_stats()
        lines = [f"총 {stats['total']}건의 DFT 계산이 인덱싱되어 있습니다.\n"]

        if stats.get("by_status"):
            lines.append("**상태별:**")
            for status, cnt in stats["by_status"].items():
                lines.append(f"- {status}: {cnt}건")

        if stats.get("by_method"):
            lines.append("\n**메서드별 (상위 10):**")
            for method, cnt in stats["by_method"].items():
                label = method if method else "(unknown)"
                lines.append(f"- {label}: {cnt}건")

        if stats.get("by_calc_type"):
            lines.append("\n**계산 유형별:**")
            for ct, cnt in stats["by_calc_type"].items():
                lines.append(f"- {ct}: {cnt}건")

        if stats.get("top_formulas"):
            lines.append("\n**주요 분자 (상위 10):**")
            for formula, cnt in stats["top_formulas"].items():
                label = formula if formula else "(unknown)"
                lines.append(f"- {label}: {cnt}건")

        return "\n".join(lines)

    async def _format_query_results(
        self,
        results: list[dict[str, Any]],
        *,
        title: str = "DFT 계산 결과",
    ) -> str:
        if not results:
            return f"{title}: 결과 없음"

        lines = [f"**{title}** ({len(results)}건)\n"]
        lines.append("| # | 파일 | Formula | Method/Basis | E (Eh) | Status | 비고 |")
        lines.append("|---|------|---------|-------------|--------|--------|------|")

        for i, r in enumerate(results, 1):
            path = _short_path(r.get("source_path", ""))
            formula = r.get("formula", "")
            method = r.get("method", "")
            basis = r.get("basis_set", "")
            mb = f"{method}/{basis}" if basis else method
            energy = r.get("energy_hartree")
            e_str = f"{energy:.6f}" if energy is not None else "-"
            status = r.get("status", "")

            notes: list[str] = []
            if r.get("opt_converged") == 0:
                notes.append("NOT CONVERGED")
            if r.get("has_imaginary_freq") == 1:
                freq = r.get("lowest_freq_cm1")
                notes.append(f"imag: {freq:.1f} cm-1" if freq else "imag freq")
            note_str = ", ".join(notes)

            lines.append(f"| {i} | {path} | {formula} | {mb} | {e_str} | {status} | {note_str} |")

        return "\n".join(lines)

    async def _format_comparison(
        self,
        formula: str | None = None,
        method: str | None = None,
    ) -> str:
        results = await self._index.get_for_comparison(formula=formula, method=method)
        if not results:
            return "비교할 계산 결과가 없습니다."

        # 최저 에너지 기준 상대 에너지
        ref_energy = None
        for r in results:
            if r.get("energy_hartree") is not None:
                ref_energy = r["energy_hartree"]
                break

        lines = ["**DFT 에너지 비교**\n"]
        lines.append("| # | 파일 | Formula | Method/Basis | E (Eh) | ΔE (kcal/mol) | 비고 |")
        lines.append("|---|------|---------|-------------|--------|--------------|------|")

        for i, r in enumerate(results, 1):
            path = _short_path(r.get("source_path", ""))
            formula_val = r.get("formula", "")
            method_val = r.get("method", "")
            basis = r.get("basis_set", "")
            mb = f"{method_val}/{basis}" if basis else method_val
            energy = r.get("energy_hartree")
            e_str = f"{energy:.6f}" if energy is not None else "-"

            if energy is not None and ref_energy is not None:
                delta = (energy - ref_energy) * HARTREE_TO_KCALMOL
                delta_str = f"{delta:.2f}"
            else:
                delta_str = "-"

            notes: list[str] = []
            if i == 1 and energy is not None:
                notes.append("lowest")
            if r.get("has_imaginary_freq") == 1:
                notes.append("imag freq")
            if r.get("opt_converged") == 0:
                notes.append("NOT CONVERGED")

            gibbs = r.get("gibbs_energy")
            if gibbs is not None:
                notes.append(f"G={gibbs:.6f}")

            note_str = ", ".join(notes)
            lines.append(f"| {i} | {path} | {formula_val} | {mb} | {e_str} | {delta_str} | {note_str} |")

        return "\n".join(lines)


def _short_path(path: str) -> str:
    """긴 경로를 파일명 + 부모 디렉토리로 축약한다."""
    if not path:
        return ""
    parts = path.replace("\\", "/").split("/")
    if len(parts) <= 2:
        return path
    return "/".join(parts[-2:])
