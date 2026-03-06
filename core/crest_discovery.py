"""CREST 출력 파일 탐색 유틸.

디렉토리를 스캔하여 CREST 출력(마커 파일 + .out)을 찾는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.crest_parser import is_crest_output


def discover_crest_targets(
    kb_path: Path,
    *,
    max_bytes: int,
    logger: Any | None = None,
    exclude_paths: set[str] | None = None,
) -> list[Path]:
    """모니터링 대상 CREST 출력 파일 목록을 반환한다.

    탐색 전략:
    1) crest_best.xyz 마커 → 해당 디렉토리의 .out 또는 마커 자체
    2) crest_conformers.xyz 마커 → 동일
    3) .out 파일 직접 스캔 (마커 없는 초기 실행)
    """
    exclude = exclude_paths or set()
    targets: dict[str, Path] = {}
    seen_dirs: set[Path] = set()

    # 1. crest_best.xyz 마커로 완료된 CREST 디렉토리 탐색
    for marker in kb_path.rglob("crest_best.xyz"):
        directory = marker.parent
        if directory in seen_dirs:
            continue
        seen_dirs.add(directory)
        target = _pick_target(directory, max_bytes, exclude)
        if target:
            targets[str(target)] = target

    # 2. crest_conformers.xyz 마커 (crest_best.xyz 없는 진행 중 케이스)
    for marker in kb_path.rglob("crest_conformers.xyz"):
        directory = marker.parent
        if directory in seen_dirs:
            continue
        seen_dirs.add(directory)
        target = _pick_target(directory, max_bytes, exclude)
        if target:
            targets[str(target)] = target

    # 3. .out 파일 직접 스캔 (마커 파일 없는 초기 실행 단계)
    for out_path in kb_path.rglob("*.out"):
        spath = str(out_path)
        if spath in targets or spath in exclude:
            continue
        directory = out_path.parent
        if directory in seen_dirs:
            continue
        if not out_path.is_file():
            continue
        try:
            size = out_path.stat().st_size
            if size > max_bytes or size < 100:
                continue
        except OSError:
            continue
        if is_crest_output(out_path):
            seen_dirs.add(directory)
            targets[spath] = out_path

    return sorted(targets.values(), key=lambda p: str(p))


def _pick_target(
    directory: Path,
    max_bytes: int,
    exclude: set[str],
) -> Path | None:
    """디렉토리에서 추적할 대표 CREST 파일을 선택한다.

    우선순위: .out 파일 > crest_best.xyz
    """
    # .out 파일 우선 (진행 상황 포함)
    for out in directory.glob("*.out"):
        spath = str(out)
        if spath in exclude:
            continue
        if not out.is_file():
            continue
        try:
            if out.stat().st_size > max_bytes:
                continue
        except OSError:
            continue
        if is_crest_output(out):
            return out

    # .out 없으면 crest_best.xyz
    best = directory / "crest_best.xyz"
    if best.is_file() and str(best) not in exclude:
        try:
            if best.stat().st_size <= max_bytes:
                return best
        except OSError:
            pass

    return None
