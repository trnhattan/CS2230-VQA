"""
Inference + Evaluation trên test set ViTextVQA — hỗ trợ nhiều model.

Cách dùng:
    # Zero-shot (base model)
    python -m src.evaluate --config configs/qwen2vl_2b.yaml

    # Fine-tuned model
    python -m src.evaluate --config configs/qwen2vl_2b.yaml \
                           --checkpoint checkpoints/qwen2vl_2b/best_model

    # Chỉ định output file
    python -m src.evaluate --config configs/smolvlm_500m.yaml \
                           --results_file results/smolvlm_500m_test.json
"""

import argparse
import json
import os
import sys

import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.adapters import get_adapter
from src.dataset import ViTextVQADataset, VQADataCollator
from src.metrics import compute_metrics


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_inference(adapter, dataset: ViTextVQADataset, eval_cfg: dict, max_length: int) -> list[dict]:
    batch_size = eval_cfg.get("batch_size", 4)
    max_new_tokens = eval_cfg.get("max_new_tokens", 64)
    num_beams = eval_cfg.get("num_beams", 1)

    collator = VQADataCollator(adapter, max_length=max_length, training=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda x: x,
        num_workers=2,
        shuffle=False,
    )

    device = next(adapter.model.parameters()).device
    results = []

    for raw_batch in tqdm(loader, desc="Inference"):
        inputs = collator(raw_batch)
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        predictions = adapter.generate(inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)

        for item, pred in zip(raw_batch, predictions):
            results.append({
                "question_id": item["question_id"],
                "image_id": item["image_id"],
                "question": item["question"],
                "prediction": pred,
                "ground_truths": item["all_answers"],
            })

    return results


def evaluate(config_path: str, checkpoint: str | None = None, results_file: str | None = None):
    cfg = load_config(config_path)
    eval_cfg = cfg["evaluation"]

    checkpoint = checkpoint or eval_cfg.get("checkpoint")
    results_file = results_file or eval_cfg.get("results_file", "results/test_predictions.json")

    # Load model qua adapter
    adapter = get_adapter(cfg["model"]["type"])
    adapter.load_for_inference(cfg, checkpoint)

    # Test dataset
    data_cfg = cfg["data"]
    test_dataset = ViTextVQADataset(data_cfg["test_file"], data_cfg["image_dir"])
    print(f"Test samples: {len(test_dataset):,}")

    results = run_inference(
        adapter,
        test_dataset,
        eval_cfg,
        max_length=cfg["training"]["max_length"],
    )

    # Tính metrics
    preds = [r["prediction"] for r in results]
    gts = [r["ground_truths"] for r in results]
    metrics = compute_metrics(preds, gts)

    print("\n" + "=" * 55)
    print(f"  Model  : {cfg['model']['name']}")
    mode = f"Fine-tuned  → {checkpoint}" if checkpoint else "Base model (zero-shot)"
    print(f"  Mode   : {mode}")
    print("=" * 55)
    print(f"  ANLS         : {metrics['anls']*100:.2f}%")
    print(f"  Exact Match  : {metrics['exact_match']*100:.2f}%")
    print(f"  Num samples  : {metrics['num_samples']:,}")
    print("=" * 55)

    os.makedirs(os.path.dirname(results_file) or ".", exist_ok=True)
    output = {
        "model": cfg["model"]["name"],
        "checkpoint": checkpoint,
        "metrics": metrics,
        "predictions": results,
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nPredictions saved → {results_file}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--results_file", type=str, default=None)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.results_file)
