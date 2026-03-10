import json
import os
from typing import TYPE_CHECKING

from torch.utils.data import Dataset
from PIL import Image

if TYPE_CHECKING:
    from src.adapters.base import BaseAdapter


class ViTextVQADataset(Dataset):
    """Dataset cho ViTextVQA. Mỗi annotation = 1 sample."""

    def __init__(self, annotation_file: str, image_dir: str):
        with open(annotation_file, encoding="utf-8") as f:
            data = json.load(f)

        self.image_dir = image_dir
        self.annotations = data["annotations"]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        ann = self.annotations[idx]
        image_path = os.path.join(self.image_dir, f"{ann['image_id']}.jpg")
        image = Image.open(image_path).convert("RGB")

        return {
            "image": image,
            "question": ann["question"],
            "answer": ann["answers"][0],
            "all_answers": ann["answers"],
            "question_id": ann["id"],
            "image_id": ann["image_id"],
        }


class VQADataCollator:
    """
    Collator model-agnostic: delegate toàn bộ xử lý cho adapter.
    Hoạt động với cả training (trả về labels) và inference (không có labels).
    """

    def __init__(self, adapter: "BaseAdapter", max_length: int = 512, training: bool = True):
        self.adapter = adapter
        self.max_length = max_length
        self.training = training

    def __call__(self, batch: list[dict]) -> dict:
        return self.adapter.process_batch(batch, self.max_length, self.training)
