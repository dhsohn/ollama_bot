"""ORCA parser 회귀 테스트."""

from __future__ import annotations

from pathlib import Path

from core.orca_parser import parse_orca_output


def test_error_termination_is_classified_as_failed(tmp_path: Path) -> None:
    out_file = tmp_path / "error_case.out"
    out_file.write_text(
        "\n".join([
            "! B3LYP def2-SVP Opt",
            "* xyz 0 1",
            "C 0.0 0.0 0.0",
            "H 0.0 0.0 1.0",
            "*",
            "",
            "ORCA finished by error termination in SCF gradient",
            "[file orca_tools/qcmsg.cpp, line 394]:",
            "  .... aborting the run",
        ]),
        encoding="utf-8",
    )

    result = parse_orca_output(str(out_file))

    assert result.status == "failed"


def test_utf16_completed_output_is_parsed(tmp_path: Path) -> None:
    out_file = tmp_path / "utf16_completed.out"
    out_file.write_text(
        "\n".join([
            "! B3LYP def2-SVP Opt",
            "* xyz 0 1",
            "C 0.0 0.0 0.0",
            "H 0.0 0.0 1.0",
            "*",
            "FINAL SINGLE POINT ENERGY      -100.123456",
            "                             ****ORCA TERMINATED NORMALLY****",
            "TOTAL RUN TIME: 0 days 0 hours 1 minutes 2 seconds 3 msec",
        ]),
        encoding="utf-16",
    )

    result = parse_orca_output(str(out_file))

    assert result.status == "completed"
    assert result.method == "B3LYP"
