"""
CulturaRAG — LoRA Fine-tuning Utility
Provides a reusable script for PEFT/LoRA fine-tuning of a base LLM
on cultural Q&A preference pairs collected from the RLHF feedback pipeline.

Usage:
    python -m app.utils.lora_finetune --help
    python -m app.utils.lora_finetune --dataset ./data/feedback_log.jsonl --epochs 3

This module is intentionally standalone so it can be run as a batch job
separately from the live API server.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

from loguru import logger


def load_preference_dataset(jsonl_path: str) -> List[Dict]:
    """Load RLHF preference pairs from feedback JSONL."""
    pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("corrected_answer") and entry.get("rating", 0) >= 4:
                pairs.append(entry)
    logger.info(f"Loaded {len(pairs)} preference pairs from {jsonl_path}")
    return pairs


def build_hf_dataset(pairs: List[Dict]):
    """Convert pairs to a HuggingFace Dataset for SFT/DPO training."""
    from datasets import Dataset

    records = []
    for p in pairs:
        # Format as instruction-following pairs
        records.append({
            "instruction": "Answer the cultural question accurately and respectfully.",
            "input": f"Query ID: {p['query_id']}",
            "output": p["corrected_answer"],
        })

    return Dataset.from_list(records)


def run_lora_finetune(
    base_model: str,
    dataset_path: str,
    output_dir: str,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    num_epochs: int = 3,
    batch_size: int = 4,
):
    """
    Fine-tune a base model using LoRA (PEFT) on cultural preference data.

    Args:
        base_model: HuggingFace model ID (e.g. 'google/gemma-2b')
        dataset_path: Path to feedback JSONL file
        output_dir: Where to save LoRA adapter weights
        lora_rank: LoRA rank (r) — controls adapter capacity
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout on LoRA layers
        num_epochs: Training epochs
        batch_size: Per-device training batch size
    """
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Install with: pip install transformers peft")
        raise

    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],   # Typical for LLaMA/Gemma architectures
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Prepare Dataset ───────────────────────────────────────────────────────
    pairs = load_preference_dataset(dataset_path)
    if not pairs:
        logger.warning("No training pairs found. Collect more RLHF feedback first.")
        return

    dataset = build_hf_dataset(pairs)

    def tokenize(examples):
        prompt = (
            f"### Instruction:\n{examples['instruction']}\n\n"
            f"### Input:\n{examples['input']}\n\n"
            f"### Response:\n{examples['output']}"
        )
        return tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    # ── Training Arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    logger.info("Starting LoRA fine-tuning …")
    trainer.train()

    # ── Save Adapter ──────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"LoRA adapter saved to: {output_dir}")


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CulturaRAG LoRA Fine-tuning")
    parser.add_argument("--base-model", default="google/gemma-2b", help="Base model ID")
    parser.add_argument("--dataset", default="./data/feedback_log.jsonl", help="RLHF JSONL path")
    parser.add_argument("--output-dir", default="./data/lora_adapters", help="Adapter output dir")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    run_lora_finetune(
        base_model=args.base_model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )
