"""AMD NPU 실행 프로파일별 런타임 환경변수 보정."""

from __future__ import annotations

from collections.abc import MutableMapping
import os

_PROFILE_DEFAULTS: dict[str, dict[str, str]] = {
    "balanced": {
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
    },
    "throughput": {
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "TOKENIZERS_PARALLELISM": "false",
    },
    "latency": {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
    },
}


def apply_npu_profile(
    profile: str | None,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """AMD NPU 프로파일에 맞춰 누락된 환경변수 기본값을 적용한다.

    이미 설정된 값은 덮어쓰지 않는다.
    """
    normalized = (profile or "").strip().lower()
    if not normalized:
        return {}

    if normalized not in _PROFILE_DEFAULTS:
        allowed = ", ".join(sorted(_PROFILE_DEFAULTS))
        raise ValueError(f"Invalid AMD_NPU_PROFILE '{profile}'. Allowed: {allowed}")

    target = environ if environ is not None else os.environ
    applied: dict[str, str] = {}
    for key, value in _PROFILE_DEFAULTS[normalized].items():
        if not target.get(key):
            target[key] = value
            applied[key] = value

    return applied
