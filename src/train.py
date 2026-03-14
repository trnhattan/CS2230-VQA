"""
Training script cho ViTextVQA — hỗ trợ nhiều model qua adapter pattern.

Cách dùng:
    python -m src.train --config configs/qwen2vl_2b.yaml
    python -m src.train --config configs/smolvlm_500m.yaml
    python -m src.train --config configs/smolvlm_2b.yaml
    python -m src.train --config configs/internvl2_2b.yaml
"""

import argparse
import os
import sys

import torch
import yaml
from transformers import Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.adapters import get_adapter
from src.dataset import ViTextVQADataset, VQADataCollator


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(config_path: str):
    cfg = load_config(config_path)
    t_cfg = cfg["training"]
    torch.manual_seed(t_cfg.get("seed", 42))

    # Load model qua adapter
    adapter = get_adapter(cfg["model"]["type"])
    adapter.load(cfg)

    # Datasets
    data_cfg = cfg["data"]
    train_dataset = ViTextVQADataset(data_cfg["train_file"], data_cfg["image_dir"])
    dev_dataset = ViTextVQADataset(data_cfg["dev_file"], data_cfg["image_dir"])
    print(f"Train: {len(train_dataset):,} | Dev: {len(dev_dataset):,}")

    collator = VQADataCollator(
        adapter, max_length=t_cfg["max_length"], training=True
    )

    training_args = TrainingArguments(
        output_dir=t_cfg["output_dir"],
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        weight_decay=t_cfg.get("weight_decay", 0.01),
        warmup_ratio=t_cfg.get("warmup_ratio", 0.03),
        logging_steps=t_cfg.get("logging_steps", 50),
        eval_strategy="steps",
        eval_steps=t_cfg.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=t_cfg.get("save_steps", 500),
        save_total_limit=t_cfg.get("save_total_limit", 2),
        load_best_model_at_end=t_cfg.get("load_best_model_at_end", True),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=t_cfg.get("dataloader_num_workers", 2),
        report_to=t_cfg.get("report_to", "none"),
        remove_unused_columns=False,
        seed=t_cfg.get("seed", 42),
    )

    trainer = Trainer(
        model=adapter.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
    )

    print("Bắt đầu training...")
    trainer.train()

    best_path = os.path.join(t_cfg["output_dir"], "best_model")
    os.makedirs(best_path, exist_ok=True)
    adapter.model.save_pretrained(best_path)
    if hasattr(adapter, "tokenizer") and adapter.tokenizer is not None:
        adapter.tokenizer.save_pretrained(best_path)
    elif adapter.processor is not None:
        adapter.processor.save_pretrained(best_path)
    print(f"Đã lưu checkpoint tại: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    train(args.config)
