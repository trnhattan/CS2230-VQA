"""
Metrics cho ViTextVQA:
- ANLS (Average Normalized Levenshtein Similarity): metric chính của TextVQA
- Exact Match (EM): metric phụ
"""

import re
import unicodedata
from Levenshtein import distance as levenshtein_distance


def normalize_answer(text: str) -> str:
    """Chuẩn hóa text trước khi so sánh."""
    # Unicode normalization (quan trọng cho tiếng Việt)
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    # Xóa dấu chấm câu thừa ở đầu/cuối
    text = re.sub(r"[^\w\s\u00C0-\u024F\u1E00-\u1EFF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def anls_score(prediction: str, ground_truths: list[str], threshold: float = 0.5) -> float:
    """
    Tính ANLS cho 1 câu hỏi.
    ANLS = max similarity với tất cả ground truth answers.
    Nếu similarity < threshold thì tính = 0 (penalize các câu trả lời sai hoàn toàn).
    """
    pred = normalize_answer(prediction)

    max_sim = 0.0
    for gt in ground_truths:
        gt = normalize_answer(gt)
        nl = max(len(pred), len(gt))
        if nl == 0:
            sim = 1.0
        else:
            edit_dist = levenshtein_distance(pred, gt)
            sim = 1.0 - edit_dist / nl
        max_sim = max(max_sim, sim)

    return max_sim if max_sim >= threshold else 0.0


def exact_match_score(prediction: str, ground_truths: list[str]) -> float:
    """Exact match sau khi chuẩn hóa."""
    pred = normalize_answer(prediction)
    return float(any(normalize_answer(gt) == pred for gt in ground_truths))


def compute_metrics(predictions: list[str], ground_truths: list[list[str]]) -> dict:
    """
    Tính ANLS và EM trên toàn bộ tập dữ liệu.

    Args:
        predictions: danh sách câu trả lời dự đoán
        ground_truths: danh sách các câu trả lời đúng (mỗi câu hỏi có thể có nhiều đáp án)

    Returns:
        {"anls": float, "exact_match": float, "num_samples": int}
    """
    assert len(predictions) == len(ground_truths), (
        f"Số lượng predictions ({len(predictions)}) "
        f"không khớp ground_truths ({len(ground_truths)})"
    )

    total_anls = 0.0
    total_em = 0.0

    for pred, gts in zip(predictions, ground_truths):
        total_anls += anls_score(pred, gts)
        total_em += exact_match_score(pred, gts)

    n = len(predictions)
    return {
        "anls": total_anls / n if n > 0 else 0.0,
        "exact_match": total_em / n if n > 0 else 0.0,
        "num_samples": n,
    }
