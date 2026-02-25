"""AMD NPU 프로파일 적용 테스트."""

from __future__ import annotations

import pytest

from packages.hw_amd_npu.runtime_env import apply_npu_profile


def test_apply_profile_sets_defaults_when_missing() -> None:
    env: dict[str, str] = {}
    applied = apply_npu_profile("balanced", env)

    assert env["OMP_NUM_THREADS"] == "4"
    assert env["MKL_NUM_THREADS"] == "4"
    assert env["OPENBLAS_NUM_THREADS"] == "4"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert applied["OMP_NUM_THREADS"] == "4"


def test_apply_profile_does_not_override_existing_values() -> None:
    env = {
        "OMP_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "16",
        "OPENBLAS_NUM_THREADS": "16",
    }
    applied = apply_npu_profile("latency", env)

    assert env["OMP_NUM_THREADS"] == "16"
    assert env["MKL_NUM_THREADS"] == "16"
    assert env["OPENBLAS_NUM_THREADS"] == "16"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert "OMP_NUM_THREADS" not in applied


def test_invalid_profile_raises() -> None:
    with pytest.raises(ValueError, match="Invalid AMD_NPU_PROFILE"):
        apply_npu_profile("unknown", {})
