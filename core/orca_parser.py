"""ORCA 양자화학 출력 파일(.out) 파서.

ORCA 계산 결과에서 에너지, 메서드, 기저함수, 수렴 여부, 좌표 등
핵심 메타데이터를 추출한다.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCALMOL = 627.5094740631

# 원소 기호 → 원자번호 순서 (화학식 정렬용)
_ELEMENT_ORDER: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56,
    "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63,
    "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77,
    "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84,
    "At": 85, "Rn": 86,
}

# ---------------------------------------------------------------------------
# 정규식 패턴
# ---------------------------------------------------------------------------

# 입력 라인: "! B3LYP def2-TZVP Opt Freq ..." 또는 "|  1> ! B3LYP ..."
_INPUT_LINE_RE = re.compile(r"^(?:\s*\|\s*\d+>\s*)?!\s*(.+)$", re.MULTILINE)

# 에너지
_ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+([-\d.]+)")

# 최적화 수렴
_OPT_CONVERGED_RE = re.compile(r"THE OPTIMIZATION HAS CONVERGED")
_OPT_NOT_CONVERGED_RE = re.compile(
    r"ORCA GEOMETRY OPTIMIZATION.*(?:DID NOT CONVERGE|NOT CONVERGED)|"
    r"The optimization did not converge",
    re.IGNORECASE,
)

# 좌표 섹션 (원소 + xyz)
_COORD_SECTION_RE = re.compile(
    r"CARTESIAN COORDINATES \(ANGSTROEM\)\s*\n"
    r"-+\s*\n"
    r"((?:\s*[A-Z][a-z]?\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\n)+)",
)
_COORD_LINE_RE = re.compile(r"^\s*([A-Z][a-z]?)\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+", re.MULTILINE)

# 진동 주파수
_FREQ_SECTION_RE = re.compile(
    r"VIBRATIONAL FREQUENCIES\s*\n"
    r"-+\s*\n"
    r"([\s\S]*?)(?:\n\s*\n|\n-{20,})",
)
_FREQ_VALUE_RE = re.compile(r"^\s*\d+:\s+([-\d.]+)\s+cm\*\*-1", re.MULTILINE)

# 열역학
_ENTHALPY_RE = re.compile(r"Total (?:E|e)nthalpy\s*\.{3,}\s*([-\d.]+)\s*Eh")
_GIBBS_RE = re.compile(r"Final Gibbs free energy\s*\.{3,}\s*([-\d.]+)\s*Eh")

# 실행 시간
_RUNTIME_RE = re.compile(
    r"TOTAL RUN TIME:\s*(\d+)\s*days?\s+(\d+)\s*hours?\s+"
    r"(\d+)\s*minutes?\s+(\d+)\s*seconds?",
)

# charge / multiplicity: "* xyz 0 1" 또는 "|  2> * xyz 0 1"
_CHARGE_MULT_RE = re.compile(r"(?:\|\s*\d+>\s*)?\*\s*xyz\s+([-\d]+)\s+(\d+)")

# 정상 종료 마커
_NORMAL_TERMINATION_RE = re.compile(r"ORCA TERMINATED NORMALLY")

# 알려진 계산 유형 키워드 (입력 라인에서 검색)
_CALC_TYPE_KEYWORDS: dict[str, str] = {
    "OPTTS": "ts",
    "TS": "ts",
    "OPT": "opt",
    "FREQ": "freq",
    "MD": "md",
    "COPT": "opt",
    "NEB": "neb",
    "SCAN": "scan",
    "IRC": "irc",
}

# 알려진 메서드 키워드
_METHOD_KEYWORDS: list[str] = [
    "CCSD(T)", "CCSD", "MP2", "RI-MP2", "DLPNO-CCSD(T)",
    "B3LYP", "PBE0", "PBE", "BP86", "TPSS", "M06-2X", "M06",
    "ωB97X-D3", "wB97X-D3", "ωB97X-D", "wB97X-D", "ωB97X", "wB97X",
    "ωB97M-V", "wB97M-V", "ωB97M-D4", "wB97M-D4",
    "B2PLYP", "REVPBE", "BLYP",
    "CAM-B3LYP", "LC-BLYP", "BHandHLYP",
    "HF", "RHF", "UHF", "ROHF",
    "CASSCF", "NEVPT2", "MRCI",
    "B97-3c", "r2SCAN-3c", "PBEh-3c",
]

# 알려진 기저함수 키워드
_BASIS_KEYWORDS: list[str] = [
    "def2-QZVPP", "def2-QZVP", "def2-TZVPP", "def2-TZVP",
    "def2-SVP", "def2-SV(P)",
    "ma-def2-TZVPP", "ma-def2-TZVP", "ma-def2-SVP",
    "cc-pVQZ", "cc-pVTZ", "cc-pVDZ",
    "aug-cc-pVQZ", "aug-cc-pVTZ", "aug-cc-pVDZ",
    "6-311++G(d,p)", "6-311+G(d,p)", "6-311G(d,p)",
    "6-31++G(d,p)", "6-31+G(d,p)", "6-31G(d,p)", "6-31G*", "6-31G**",
    "STO-3G",
]


# ---------------------------------------------------------------------------
# 결과 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class OrcaResult:
    """ORCA 출력 파일에서 추출한 계산 결과."""

    source_path: str
    calc_type: str = ""
    method: str = ""
    basis_set: str = ""
    charge: int = 0
    multiplicity: int = 1
    formula: str = ""
    n_atoms: int = 0
    energy_hartree: float | None = None
    energy_ev: float | None = None
    energy_kcalmol: float | None = None
    opt_converged: bool | None = None
    has_imaginary_freq: bool | None = None
    lowest_freq_cm1: float | None = None
    enthalpy: float | None = None
    gibbs_energy: float | None = None
    wall_time_seconds: int | None = None
    status: str = "completed"
    file_hash: str = ""
    mtime: float = 0.0
    input_line: str = ""
    elements: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 파서 함수
# ---------------------------------------------------------------------------

def _build_formula(elements: list[str]) -> str:
    """원소 기호 목록에서 Hill system 화학식을 생성한다."""
    counts = Counter(elements)
    if not counts:
        return ""

    # Hill system: C 먼저, H 다음, 나머지 알파벳 순
    parts: list[str] = []
    for sym in ("C", "H"):
        if sym in counts:
            parts.append(sym if counts[sym] == 1 else f"{sym}{counts[sym]}")
            del counts[sym]

    for sym in sorted(counts, key=lambda s: _ELEMENT_ORDER.get(s, 999)):
        parts.append(sym if counts[sym] == 1 else f"{sym}{counts[sym]}")

    return "".join(parts)


def _parse_input_line(text: str) -> tuple[str, str, str, list[str]]:
    """입력 라인에서 calc_type, method, basis_set을 추출한다.

    Returns:
        (calc_type, method, basis_set, all_input_tokens)
    """
    matches = _INPUT_LINE_RE.findall(text)
    if not matches:
        return ("sp", "", "", [])

    # 여러 입력 라인이 있을 수 있음 — 합치기
    all_tokens: list[str] = []
    for line in matches:
        all_tokens.extend(line.strip().split())

    tokens_upper = [t.upper() for t in all_tokens]

    # calc_type 결정
    calc_types: list[str] = []
    for token_upper in tokens_upper:
        for kw, ct in _CALC_TYPE_KEYWORDS.items():
            if token_upper == kw:
                calc_types.append(ct)
    if not calc_types:
        calc_type = "sp"
    elif "opt" in calc_types and "freq" in calc_types:
        calc_type = "opt+freq"
    elif "ts" in calc_types and "freq" in calc_types:
        calc_type = "ts+freq"
    else:
        calc_type = calc_types[0]

    # method 결정 — 대소문자 보존
    method = ""
    for mk in _METHOD_KEYWORDS:
        for token in all_tokens:
            if token.upper() == mk.upper():
                method = mk
                break
        if method:
            break

    # basis_set 결정 — 대소문자 보존
    basis_set = ""
    for bk in _BASIS_KEYWORDS:
        for token in all_tokens:
            if token.upper() == bk.upper():
                basis_set = bk
                break
        if basis_set:
            break

    return (calc_type, method, basis_set, all_tokens)


def _parse_coordinates(text: str) -> tuple[list[str], int]:
    """좌표 섹션에서 원소 기호를 추출한다.

    Returns:
        (elements, n_atoms)
    """
    # 마지막 좌표 섹션 사용 (최적화 후 최종 좌표)
    sections = list(_COORD_SECTION_RE.finditer(text))
    if not sections:
        return ([], 0)

    last_section = sections[-1].group(1)
    elements = _COORD_LINE_RE.findall(last_section)
    return (elements, len(elements))


def _parse_frequencies(text: str) -> tuple[bool | None, float | None]:
    """진동 주파수 섹션에서 imaginary 여부와 최저 주파수를 추출한다.

    Returns:
        (has_imaginary_freq, lowest_freq_cm1)
    """
    section_match = _FREQ_SECTION_RE.search(text)
    if section_match is None:
        return (None, None)

    section = section_match.group(1)
    freq_values = [float(v) for v in _FREQ_VALUE_RE.findall(section)]

    if not freq_values:
        return (None, None)

    # 0.0 cm^-1 근처의 병진/회전 모드 제외 (< 10 cm^-1 절대값)
    real_freqs = [f for f in freq_values if abs(f) > 10.0]
    if not real_freqs:
        return (False, None)

    lowest = min(real_freqs)
    has_imaginary = lowest < 0.0
    return (has_imaginary, lowest)


def _parse_wall_time(text: str) -> int | None:
    """실행 시간을 초 단위로 변환한다."""
    m = _RUNTIME_RE.search(text)
    if m is None:
        return None
    days, hours, minutes, seconds = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _compute_file_hash(file_path: str) -> str:
    """파일의 SHA-256 해시 앞 16자를 반환한다."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def parse_orca_output(file_path: str) -> OrcaResult:
    """ORCA .out 파일을 파싱하여 OrcaResult를 반환한다.

    Args:
        file_path: ORCA 출력 파일 경로

    Returns:
        추출된 계산 결과

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        UnicodeDecodeError: 파일 인코딩 문제
    """
    with open(file_path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    result = OrcaResult(source_path=file_path)
    result.mtime = os.path.getmtime(file_path)
    result.file_hash = _compute_file_hash(file_path)

    # 입력 라인 파싱
    calc_type, method, basis_set, input_tokens = _parse_input_line(text)
    result.calc_type = calc_type
    result.method = method
    result.basis_set = basis_set
    result.input_line = " ".join(input_tokens)

    # charge / multiplicity
    cm_match = _CHARGE_MULT_RE.search(text)
    if cm_match:
        result.charge = int(cm_match.group(1))
        result.multiplicity = int(cm_match.group(2))

    # 좌표 → 원소 → 화학식
    elements, n_atoms = _parse_coordinates(text)
    result.elements = elements
    result.n_atoms = n_atoms
    result.formula = _build_formula(elements)

    # 에너지 (마지막 값 사용 — 최적화 시 여러 번 출력됨)
    energy_matches = _ENERGY_RE.findall(text)
    if energy_matches:
        energy = float(energy_matches[-1])
        result.energy_hartree = energy
        result.energy_ev = energy * HARTREE_TO_EV
        result.energy_kcalmol = energy * HARTREE_TO_KCALMOL

    # 최적화 수렴
    if _OPT_CONVERGED_RE.search(text):
        result.opt_converged = True
    elif _OPT_NOT_CONVERGED_RE.search(text):
        result.opt_converged = False

    # 진동 주파수
    has_imag, lowest = _parse_frequencies(text)
    result.has_imaginary_freq = has_imag
    result.lowest_freq_cm1 = lowest

    # 열역학
    enthalpy_match = _ENTHALPY_RE.search(text)
    if enthalpy_match:
        result.enthalpy = float(enthalpy_match.group(1))

    gibbs_match = _GIBBS_RE.search(text)
    if gibbs_match:
        result.gibbs_energy = float(gibbs_match.group(1))

    # 실행 시간
    result.wall_time_seconds = _parse_wall_time(text)

    # 상태 판별
    if _NORMAL_TERMINATION_RE.search(text):
        if result.opt_converged is False:
            result.status = "failed"
        else:
            result.status = "completed"
    elif result.wall_time_seconds is not None:
        # TOTAL RUN TIME은 있지만 TERMINATED NORMALLY 없음
        result.status = "failed"
    else:
        result.status = "running"

    return result
