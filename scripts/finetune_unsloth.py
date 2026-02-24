#!/usr/bin/env python3
"""Unsloth KTO 파인튜닝 스크립트.

사용법:
    python scripts/finetune_unsloth.py --dataset data/training/kto_dataset_*.jsonl --output models/finetuned

요구사항:
    pip install unsloth datasets trl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_kto_dataset(path: str) -> list[dict]:
    """JSONL 파일에서 KTO 데이터셋을 로드한다."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsloth KTO Fine-tuning")
    parser.add_argument("--dataset", required=True, help="KTO JSONL 데이터셋 경로")
    parser.add_argument("--base-model", default="unsloth/llama-3-8b-bnb-4bit", help="베이스 모델")
    parser.add_argument("--output", default="models/finetuned", help="출력 디렉터리")
    parser.add_argument("--epochs", type=int, default=1, help="학습 에포크 수")
    parser.add_argument("--batch-size", type=int, default=4, help="배치 크기")
    parser.add_argument("--lr", type=float, default=5e-5, help="학습률")
    parser.add_argument("--max-length", type=int, default=1024, help="최대 시퀀스 길이")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"오류: 데이터셋 파일을 찾을 수 없습니다: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    records = load_kto_dataset(args.dataset)
    if not records:
        print("오류: 데이터셋이 비어 있습니다.", file=sys.stderr)
        sys.exit(1)

    print(f"데이터셋 로드 완료: {len(records)}건")
    print(f"  긍정: {sum(1 for r in records if r.get('label'))}건")
    print(f"  부정: {sum(1 for r in records if not r.get('label'))}건")

    try:
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import KTOConfig, KTOTrainer
    except ImportError as exc:
        print(
            f"오류: 필요한 패키지가 설치되지 않았습니다: {exc}\n"
            "pip install unsloth datasets trl",
            file=sys.stderr,
        )
        sys.exit(1)

    # 모델 로드
    print(f"모델 로드 중: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=args.lora_r,
        lora_dropout=0,
        bias="none",
    )

    # 데이터셋 변환
    kto_data = {
        "prompt": [r["prompt"] for r in records],
        "completion": [r["completion"] for r in records],
        "label": [r["label"] for r in records],
    }
    dataset = Dataset.from_dict(kto_data)

    # 학습 설정
    training_args = KTOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        logging_steps=10,
        save_steps=100,
        fp16=True,
    )

    trainer = KTOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("학습 시작...")
    trainer.train()

    # 저장
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(output_dir), tokenizer, save_method="merged_16bit")
    print(f"모델 저장 완료: {output_dir}")


if __name__ == "__main__":
    main()
