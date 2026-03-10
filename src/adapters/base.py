"""
Abstract base adapter — định nghĩa interface chung cho tất cả models.
Mỗi model implement lớp này để xử lý:
  - Load model + processor (training / inference mode)
  - process_batch: xử lý list[dict] raw items → batched tensors cho model
  - generate: chạy model.generate() và decode output
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
from transformers import BitsAndBytesConfig


class BaseAdapter(ABC):
    model: Any = None
    processor: Any = None
    pad_token_id: int = 0

    # ------------------------------------------------------------------ #
    #  Interface bắt buộc                                                 #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load(self, cfg: dict) -> None:
        """Load model + processor cho training (với QLoRA)."""
        ...

    @abstractmethod
    def load_for_inference(self, cfg: dict, checkpoint: str | None = None) -> None:
        """Load model + processor cho inference (full precision, tuỳ chọn LoRA)."""
        ...

    @abstractmethod
    def process_batch(
        self,
        items: list[dict],
        max_length: int = 512,
        training: bool = True,
    ) -> dict:
        """
        Nhận list raw items (có keys: image, question, answer, ...).
        Trả về batched tensors sẵn sàng đưa vào model.forward() / model.generate().
        Khi training=True: phải có key 'labels' trong output.
        """
        ...

    @abstractmethod
    def generate(
        self,
        inputs: dict,
        max_new_tokens: int = 64,
        num_beams: int = 1,
    ) -> list[str]:
        """Chạy model.generate() và decode, trả về list[str] predictions."""
        ...

    # ------------------------------------------------------------------ #
    #  Helpers dùng chung                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_bnb_config(quant_cfg: dict) -> BitsAndBytesConfig:
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        return BitsAndBytesConfig(
            load_in_4bit=quant_cfg["load_in_4bit"],
            bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=dtype_map[quant_cfg["bnb_4bit_compute_dtype"]],
            bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        )

    @staticmethod
    def _apply_qlora(model, lora_cfg: dict):
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg["bias"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    @staticmethod
    def _pad_1d(sequences: list[torch.Tensor], pad_value: int, max_len: int) -> torch.Tensor:
        """Pad list of 1-D tensors to same length và stack."""
        padded = []
        for s in sequences:
            pad_len = max_len - s.size(0)
            padded.append(
                torch.cat([s, torch.full((pad_len,), pad_value, dtype=s.dtype)])
            )
        return torch.stack(padded)

    @staticmethod
    def _compute_labels(
        input_ids: torch.Tensor,
        full_text: str,
        prompt_text: str,
        tokenizer,
    ) -> torch.Tensor:
        """
        Tạo labels: mask prompt với -100, chỉ tính loss trên phần trả lời.
        Dùng cách cắt chuỗi để xác định độ dài answer mà không cần gọi processor thêm lần nữa.
        """
        answer_part = full_text[len(prompt_text):]
        answer_ids = tokenizer.encode(answer_part, add_special_tokens=False)
        prompt_len = max(0, input_ids.size(0) - len(answer_ids))
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        return labels
