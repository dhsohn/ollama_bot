"""CREST 출력 파싱 모듈.

CREST 작업 디렉토리에서 상태, 배좌이성질체 수, 최저 에너지 등을 추출한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CrestResult:
    """CREST 계산 결과."""

    status: str  # "completed", "failed", "running"
    n_conformers: int | None = None
    best_energy_hartree: float | None = None
    n_atoms: int | None = None
    method: str = "GFN2-xTB"
    source_path: str = ""
    error_message: str | None = None


_CREST_HEADER_RE = re.compile(r"C\s*R\s*E\s*S\s*T")
_N_ATOMS_RE = re.compile(r"number\s+of\s+atoms\s*[:\s]\s*(\d+)", re.IGNORECASE)
_N_CONFORMERS_RE = re.compile(
    r"(\d+)\s+(?:unique\s+)?conformer",
    re.IGNORECASE,
)
_BEST_ENERGY_RE = re.compile(
    r"(?:best|lowest)\s+(?:conformer\s+)?energy\s*[:\s=]\s*([-]?\d+\.\d+)",
    re.IGNORECASE,
)
_NORMAL_TERM_RE = re.compile(r"CREST\s+terminated\s+normally", re.IGNORECASE)
_ERROR_RE = re.compile(
    r"(?:\bERROR\b|abnormal\s+termination|\bFAILED\b)",
    re.IGNORECASE,
)
_METHOD_RE = re.compile(r"(GFN\d?-xTB|GFN-FF)", re.IGNORECASE)


def is_crest_output(path: str | Path) -> bool:
    """파일이 CREST 출력인지 확인한다 (첫 3 KB에서 헤더 탐색)."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(3000)
        return bool(_CREST_HEADER_RE.search(head))
    except OSError:
        return False


def parse_crest_output(path: str) -> CrestResult:
    """CREST 출력 파일(또는 crest_best.xyz)을 파싱한다.

    path가 .out 파일이면 내용을 파싱하고,
    그 외 파일이면 부모 디렉토리에서 .out과 마커 파일을 수집한다.
    """
    file_path = Path(path)
    work_dir = file_path.parent

    result = CrestResult(status="running", source_path=path)

    # .out 파일 파싱
    out_content = ""
    if file_path.suffix == ".out" and file_path.is_file():
        out_content = _read_text_safe(file_path)
    else:
        out_file = _find_out_file(work_dir)
        if out_file:
            out_content = _read_text_safe(out_file)

    if out_content:
        _parse_out_content(out_content, result)

    # crest_best.xyz에서 보완 정보 추출
    best_xyz = work_dir / "crest_best.xyz"
    if best_xyz.is_file():
        _parse_best_xyz(best_xyz, result)
        if result.status == "running":
            if out_content and _NORMAL_TERM_RE.search(out_content):
                result.status = "completed"
            elif not out_content:
                result.status = "completed"

    # crest_conformers.xyz에서 conformer 수 보완
    if result.n_conformers is None:
        conformers_xyz = work_dir / "crest_conformers.xyz"
        if conformers_xyz.is_file():
            n = _count_xyz_structures(conformers_xyz)
            if n is not None:
                result.n_conformers = n

    return result


# ── 내부 헬퍼 ──


def _read_text_safe(path: Path, max_bytes: int = 5 * 1024 * 1024) -> str:
    try:
        if path.stat().st_size > max_bytes:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _find_out_file(directory: Path) -> Path | None:
    """디렉토리에서 CREST .out 파일을 찾는다 (가장 최근 파일 우선)."""
    candidates: list[tuple[float, Path]] = []
    for out in directory.glob("*.out"):
        if not out.is_file():
            continue
        if is_crest_output(out):
            try:
                candidates.append((out.stat().st_mtime, out))
            except OSError:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _parse_out_content(content: str, result: CrestResult) -> None:
    """CREST .out 파일 내용에서 주요 정보를 추출한다."""
    m = _N_ATOMS_RE.search(content)
    if m:
        result.n_atoms = int(m.group(1))

    m = _METHOD_RE.search(content)
    if m:
        result.method = m.group(1).upper()

    # conformer 수 (마지막 매칭 사용 — 최종 결과 반영)
    for m in _N_CONFORMERS_RE.finditer(content):
        result.n_conformers = int(m.group(1))

    # 최저 에너지
    for m in _BEST_ENERGY_RE.finditer(content):
        try:
            result.best_energy_hartree = float(m.group(1))
        except ValueError:
            pass

    # 종료 상태
    if _NORMAL_TERM_RE.search(content):
        result.status = "completed"
    elif _ERROR_RE.search(content):
        result.status = "failed"
        for line in content.splitlines():
            if re.search(r"\bERROR\b", line, re.IGNORECASE):
                result.error_message = line.strip()[:300]
                break


def _parse_best_xyz(path: Path, result: CrestResult) -> None:
    """crest_best.xyz에서 원자 수와 에너지를 추출한다."""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return
    if len(lines) < 2:
        return

    try:
        n_atoms = int(lines[0].strip())
        if result.n_atoms is None:
            result.n_atoms = n_atoms
    except ValueError:
        pass

    # 둘째 줄: 에너지 (예: "energy: -15.123456" 또는 단순 숫자)
    comment = lines[1].strip()
    energy_match = re.search(r"([-]?\d+\.\d{4,})", comment)
    if energy_match and result.best_energy_hartree is None:
        try:
            result.best_energy_hartree = float(energy_match.group(1))
        except ValueError:
            pass


def _count_xyz_structures(path: Path) -> int | None:
    """multi-structure xyz 파일의 구조 개수를 센다."""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None

    count = 0
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            n_atoms = int(line)
        except ValueError:
            i += 1
            continue
        if n_atoms <= 0:
            i += 1
            continue
        count += 1
        i += n_atoms + 2  # 원자 수 줄 + 코멘트 줄 + 좌표 줄들
    return count if count > 0 else None
