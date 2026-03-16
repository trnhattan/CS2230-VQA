from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
EDGE_PUNCT_RE = re.compile(r"^[\s\.,!?;:'\"“”‘’`~_]+|[\s\.,!?;:'\"“”‘’`~_]+$")
DASH_TRANSLATION = str.maketrans({"–": "-", "—": "-", "−": "-", "‐": "-"})
SIMPLE_NUMBER_WORDS = {
    "không": "0",
    "một": "1",
    "hai": "2",
    "ba": "3",
    "bốn": "4",
    "tư": "4",
    "năm": "5",
    "lăm": "5",
    "sáu": "6",
    "bảy": "7",
    "bẩy": "7",
    "tám": "8",
    "chín": "9",
    "mười": "10",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _canonicalize_number(text: str) -> str:
    compact = text.replace(" ", "")
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", compact):
        return re.sub(r"[.,]", "", compact)
    if re.fullmatch(r"\d+,\d+", compact) and not re.fullmatch(
        r"\d{1,3}(,\d{3})+", compact
    ):
        return compact.replace(",", ".")
    return text


def normalize_answer(text: str) -> str:
    """Chuẩn hóa text trước khi so sánh."""
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.translate(DASH_TRANSLATION)
    text = ZERO_WIDTH_RE.sub("", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = EDGE_PUNCT_RE.sub("", text)
    text = re.sub(r"[^\w\s\u00C0-\u024F\u1E00-\u1EFF\-./,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text in SIMPLE_NUMBER_WORDS:
        text = SIMPLE_NUMBER_WORDS[text]

    text = _canonicalize_number(text)
    text = re.sub(r"^[\.,!?;:]+|[\.,!?;:]+$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match_score(prediction: str, ground_truths: list[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(normalize_answer(gt) == pred for gt in ground_truths))


def _token_f1_single(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction: str, ground_truths: list[str]) -> float:
    return max((_token_f1_single(prediction, gt) for gt in ground_truths), default=0.0)


def compute_metrics(predictions: list[str], ground_truths: list[list[str]]) -> dict[str, Any]:
    assert len(predictions) == len(ground_truths)
    if not predictions:
        return {"exact_match": 0.0, "f1": 0.0, "num_samples": 0}

    total_em = sum(
        exact_match_score(pred, gts) for pred, gts in zip(predictions, ground_truths)
    )
    total_f1 = sum(f1_score(pred, gts) for pred, gts in zip(predictions, ground_truths))
    n = len(predictions)
    return {
        "exact_match": total_em / n,
        "f1": total_f1 / n,
        "num_samples": n,
    }


def load_annotations(json_path: str | Path) -> list[dict]:
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["annotations", "data", "items", "samples"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"Unsupported annotation format: {json_path}")


def extract_question(ann: dict) -> str:
    for key in ["question", "query", "text"]:
        if ann.get(key):
            return str(ann[key])
    raise KeyError(f"Cannot find question field in sample: {ann}")


def extract_ground_truths(ann: dict) -> list[str]:
    if isinstance(ann.get("answers"), list):
        return [str(x) for x in ann["answers"]]
    if isinstance(ann.get("all_answers"), list):
        return [str(x) for x in ann["all_answers"]]
    if ann.get("answer") is not None:
        return [str(ann["answer"])]
    if ann.get("label") is not None:
        if isinstance(ann["label"], list):
            return [str(x) for x in ann["label"]]
        return [str(ann["label"])]
    return [""]


def extract_image_identifier(ann: dict) -> str:
    for key in ["image_id", "image", "image_name", "file_name", "filename"]:
        value = ann.get(key)
        if value is not None:
            return str(value)
    raise KeyError(f"Cannot find image identifier field in sample: {ann}")


@lru_cache(maxsize=100_000)
def resolve_image_path(
    image_identifier: str,
    image_dir: str | Path,
    image_exts: tuple[str, ...],
) -> Path:
    image_dir = Path(image_dir)
    candidate = Path(image_identifier)

    if candidate.suffix:
        direct = image_dir / candidate.name
        if direct.exists():
            return direct
        if candidate.exists():
            return candidate

    stem = candidate.stem if candidate.suffix else candidate.name
    for ext in image_exts:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path

    matches = sorted(image_dir.glob(f"{stem}.*"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Cannot find image for id '{image_identifier}' in {image_dir}")


def build_prompt_for_log(question: str, prompt_template: str) -> str:
    return prompt_template.format(question=question)


def build_multimodal_user_text(question: str, prompt_template: str) -> str:
    prompt = build_prompt_for_log(question, prompt_template).strip()
    prompt = re.sub(r"^\s*<image>\s*\n?", "", prompt, count=1)
    prompt = re.sub(r"^\s*User:\s*", "", prompt, count=1)
    prompt = re.sub(r"\nAssistant:\s*$", "", prompt)
    return prompt.strip()


def build_internvl_question(question: str, prompt_template: str) -> str:
    prompt = build_multimodal_user_text(question, prompt_template)
    return "<image>\n" + prompt


def save_run_outputs(
    stats_path: str | Path,
    predictions_path: str | Path,
    stats: dict[str, Any],
    results: list[dict[str, Any]],
    skipped: list[dict[str, Any]] | None = None,
) -> None:
    stats_path = Path(stats_path)
    predictions_path = Path(predictions_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "stats": stats,
        "predictions": results,
        "skipped": skipped or [],
    }
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
