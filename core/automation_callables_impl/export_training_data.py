"""KTO 파인튜닝 데이터 내보내기 callable."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.pii import redact_pii


def build_export_training_data_callable(
    feedback: Any,
    data_dir: str,
    logger: Any,
):
    """KTO 데이터 내보내기 callable 팩토리."""

    async def export_training_data(
        min_preview_length: int = 20,
    ) -> str:
        """피드백 데이터를 KTO 형식으로 내보낸다."""
        dataset = await feedback.export_kto_dataset(
            min_preview_length=min_preview_length,
        )

        if not dataset:
            return ""

        # PII 마스킹
        for item in dataset:
            item["prompt"] = redact_pii(item["prompt"])
            item["completion"] = redact_pii(item["completion"])

        # 원자적 쓰기
        output_dir = Path(data_dir) / "training"
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"kto_dataset_{ts}.jsonl"
        filepath = output_dir / filename
        tmp_path = filepath.with_suffix(".tmp")

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            os.replace(str(tmp_path), str(filepath))
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        # 메타데이터 사이드카
        meta = {
            "created_at": ts,
            "total_samples": len(dataset),
            "positive_samples": sum(1 for d in dataset if d["label"]),
            "negative_samples": sum(1 for d in dataset if not d["label"]),
            "format": "kto",
        }
        meta_path = filepath.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(
            "training_data_exported",
            path=str(filepath),
            samples=len(dataset),
        )

        return (
            f"## KTO 데이터 내보내기 완료\n\n"
            f"- 파일: {filepath.name}\n"
            f"- 총 {len(dataset)}건 "
            f"(긍정 {meta['positive_samples']}, 부정 {meta['negative_samples']})"
        )

    return export_training_data
