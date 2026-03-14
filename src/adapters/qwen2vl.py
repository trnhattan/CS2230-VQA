"""
Adapter cho Qwen2-VL (2B, 7B).
Supported models:
  - Qwen/Qwen2-VL-2B-Instruct
  - Qwen/Qwen2-VL-7B-Instruct

Đặc điểm:
  - Dynamic resolution: ảnh được chia patches linh hoạt theo min/max_pixels
  - pixel_values được concatenate (không stack) vì mỗi ảnh có số patches khác nhau
  - image_grid_thw lưu layout (temporal, height, width) của từng ảnh
"""

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from .base import BaseAdapter


class Qwen2VLAdapter(BaseAdapter):

    def _build_processor(self, model_name: str, model_cfg: dict) -> AutoProcessor:
        min_pixels = model_cfg.get("min_pixels", 256) * 28 * 28
        max_pixels = model_cfg.get("max_pixels", 1280) * 28 * 28
        return AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels, max_pixels=max_pixels
        )

    # ------------------------------------------------------------------ #
    #  Load                                                                #
    # ------------------------------------------------------------------ #

    def load(self, cfg: dict) -> None:
        """Training mode: QLoRA."""
        model_name = cfg["model"]["name"]
        print(f"[Qwen2-VL] Loading processor: {model_name}")
        self.processor = self._build_processor(model_name, cfg["model"])
        self.pad_token_id = self.processor.tokenizer.pad_token_id or 0

        print(f"[Qwen2-VL] Loading model (4-bit): {model_name}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=self._build_bnb_config(cfg["quantization"]),
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model = self._apply_qlora(model, cfg["lora"])

    def load_for_inference(self, cfg: dict, checkpoint: str | None = None) -> None:
        """Inference mode: full precision, optional LoRA merge."""
        from peft import PeftModel

        model_name = cfg["model"]["name"]
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        proc_src = checkpoint or model_name
        print(f"[Qwen2-VL] Loading processor từ: {proc_src}")
        self.processor = self._build_processor(proc_src, cfg["model"])
        self.pad_token_id = self.processor.tokenizer.pad_token_id or 0

        print(f"[Qwen2-VL] Loading base model: {model_name}")
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
        if checkpoint:
            print(f"[Qwen2-VL] Merging LoRA từ: {checkpoint}")
            self.model = PeftModel.from_pretrained(base, checkpoint).merge_and_unload()
        else:
            self.model = base
        self.model.eval()

    # ------------------------------------------------------------------ #
    #  process_batch                                                       #
    # ------------------------------------------------------------------ #

    def process_batch(
        self,
        items: list[dict],
        max_length: int = 512,
        training: bool = True,
    ) -> dict:
        all_input_ids, all_masks, all_labels = [], [], []
        all_pixel_values, all_grid_thw = [], []

        for item in items:
            image = item["image"]
            question = item["question"]
            answer = item.get("answer", "")

            messages_user = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            prompt_text = self.processor.apply_chat_template(
                messages_user, tokenize=False, add_generation_prompt=True
            )

            if training:
                messages_full = messages_user + [{"role": "assistant", "content": answer}]
                full_text = self.processor.apply_chat_template(
                    messages_full, tokenize=False, add_generation_prompt=False
                )
            else:
                full_text = prompt_text

            image_inputs, _ = process_vision_info(messages_user)
            enc = self.processor(
                text=[full_text],
                images=image_inputs,
                padding=False,
                return_tensors="pt",
            )

            ids_full = enc.input_ids[0]
            mask_full = enc.attention_mask[0]

            # Tính labels trên sequence ĐẦY ĐỦ trước khi truncate,
            # rồi truncate cùng lúc với ids/mask để tránh lệch offset.
            if training:
                labels_full = self._compute_labels(
                    ids_full, full_text, prompt_text, self.processor.tokenizer
                )
                all_labels.append(labels_full)

            # Không truncate input_ids vì sẽ cắt image placeholder tokens
            # → mismatch với pixel_values. Kiểm soát độ dài qua max_pixels trong config.
            all_input_ids.append(ids_full)
            all_masks.append(mask_full)
            all_pixel_values.append(enc.pixel_values)
            all_grid_thw.append(enc.image_grid_thw)

        max_len = max(t.size(0) for t in all_input_ids)
        result = {
            "input_ids": self._pad_1d(all_input_ids, self.pad_token_id, max_len),
            "attention_mask": self._pad_1d(all_masks, 0, max_len),
            "pixel_values": torch.cat(all_pixel_values, dim=0),
            "image_grid_thw": torch.cat(all_grid_thw, dim=0),
        }
        if all_labels:
            result["labels"] = self._pad_1d(all_labels, -100, max_len)
        return result

    # ------------------------------------------------------------------ #
    #  Generate                                                            #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        inputs: dict,
        max_new_tokens: int = 64,
        num_beams: int = 1,
    ) -> list[str]:
        input_len = inputs["input_ids"].shape[1]
        gen_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        generated = self.model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            pad_token_id=self.pad_token_id,
        )
        return [
            self.processor.tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
            for seq in generated
        ]
