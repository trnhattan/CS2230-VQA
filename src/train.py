"""
Training script cho ViTextVQA — hỗ trợ nhiều model qua adapter pattern.

Cách dùng:
    python -m src.train --config configs/qwen2vl_2b.yaml
    python -m src.train --config configs/smolvlm_500m.yaml
    python -m src.train --config configs/smolvlm_2b.yaml
"""

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.adapters import get_adapter
from src.dataset import ViTextVQADataset, VQADataCollator
from src.metrics import compute_metrics


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class VQATrainer(Trainer):
    """
    Custom Trainer: evaluate bằng model.generate() → tính ANLS thực sự.
    HuggingFace Trainer mặc định dùng loss để eval — không phù hợp với VQA.
    """

    def __init__(self, *args, adapter=None, dev_dataset_raw=None, eval_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._adapter = adapter
        self._dev_dataset_raw = dev_dataset_raw
        self._eval_cfg = eval_cfg or {}

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if self._dev_dataset_raw is None:
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        self.model.eval()
        batch_size = self._eval_cfg.get("eval_batch_size", 2)
        max_new_tokens = self._eval_cfg.get("max_new_tokens", 64)
        max_length = self._eval_cfg.get("max_length", 512)

        collator = VQADataCollator(self._adapter, max_length=max_length, training=False)
        loader = DataLoader(
            self._dev_dataset_raw,
            batch_size=batch_size,
            collate_fn=lambda x: x,
            num_workers=0,
            shuffle=False,
        )

        predictions, ground_truths = [], []

        for raw_batch in tqdm(loader, desc="Eval (dev)"):
            ground_truths.extend(item["all_answers"] for item in raw_batch)

            inputs = collator(raw_batch)
            inputs = {
                k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            with torch.no_grad():
                preds = self._adapter.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                )
            predictions.extend(preds)

        scores = compute_metrics(predictions, ground_truths)
        output = {
            f"{metric_key_prefix}_anls": scores["anls"],
            f"{metric_key_prefix}_exact_match": scores["exact_match"],
            f"{metric_key_prefix}_num_samples": scores["num_samples"],
        }
        self.log(output)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output
        )
        return output


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

    train_collator = VQADataCollator(
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
        metric_for_best_model=t_cfg.get("metric_for_best_model", "eval_anls"),
        greater_is_better=t_cfg.get("greater_is_better", True),
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=t_cfg.get("dataloader_num_workers", 2),
        report_to=t_cfg.get("report_to", "none"),
        remove_unused_columns=False,
        seed=t_cfg.get("seed", 42),
    )

    trainer = VQATrainer(
        model=adapter.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,  # Dùng để Trainer không báo lỗi thiếu eval_dataset
        data_collator=train_collator,
        adapter=adapter,
        dev_dataset_raw=dev_dataset,
        eval_cfg={
            "max_new_tokens": cfg["evaluation"].get("max_new_tokens", 64),
            "eval_batch_size": t_cfg["per_device_eval_batch_size"],
            "max_length": t_cfg["max_length"],
        },
    )

    print("Bắt đầu training...")
    trainer.train()

    # Lưu LoRA adapter + processor
    best_path = os.path.join(t_cfg["output_dir"], "best_model")
    adapter.model.save_pretrained(best_path)
    adapter.processor.save_pretrained(best_path)
    print(f"Đã lưu checkpoint tại: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    train(args.config)
