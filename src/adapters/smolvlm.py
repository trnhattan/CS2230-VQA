"""
Adapter cho SmolVLM (500M, 2B) và SmolVLM2 (2.2B).
Supported models:
  - HuggingFaceTB/SmolVLM-500M-Instruct
  - HuggingFaceTB/SmolVLM-Instruct        (2B)
  - HuggingFaceTB/SmolVLM2-2.2B-Instruct

Đặc điểm:
  - Dùng AutoModelForVision2Seq (tương thích tất cả phiên bản SmolVLM)
  - pixel_values có shape phức tạp (image splitting) → để processor tự pad khi batch
  - Batch processing: gọi processor 1 lần cho cả batch thay vì từng sample
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from .base import BaseAdapter


class SmolVLMAdapter(BaseAdapter):

    # ------------------------------------------------------------------ #
    #  Load                                                                #
    # ------------------------------------------------------------------ #

    def load(self, cfg: dict) -> None:
        """Training mode: QLoRA."""
        model_name = cfg["model"]["name"]
        print(f"[SmolVLM] Loading processor: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.pad_token_id = (
            self.processor.tokenizer.pad_token_id
            if self.processor.tokenizer.pad_token_id is not None
            else self.processor.tokenizer.eos_token_id
        )

        print(f"[SmolVLM] Loading model (4-bit): {model_name}")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            quantization_config=self._build_bnb_config(cfg["quantization"]),
            device_map="auto",
            torch_dtype=torch.bfloat16,
            _attn_implementation="eager",  # flash_attn optional
        )
        self.model = self._apply_qlora(model, cfg["lora"])

    def load_for_inference(self, cfg: dict, checkpoint: str | None = None) -> None:
        """Inference mode: full precision, optional LoRA merge."""
        from peft import PeftModel

        model_name = cfg["model"]["name"]
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        proc_src = checkpoint or model_name
        print(f"[SmolVLM] Loading processor từ: {proc_src}")
        self.processor = AutoProcessor.from_pretrained(proc_src)
        self.pad_token_id = (
            self.processor.tokenizer.pad_token_id
            if self.processor.tokenizer.pad_token_id is not None
            else self.processor.tokenizer.eos_token_id
        )

        print(f"[SmolVLM] Loading base model: {model_name}")
        base = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            _attn_implementation="eager",
        )
        if checkpoint:
            print(f"[SmolVLM] Merging LoRA từ: {checkpoint}")
            self.model = PeftModel.from_pretrained(base, checkpoint).merge_and_unload()
        else:
            self.model = base
        self.model.eval()

    # ------------------------------------------------------------------ #
    #  process_batch                                                       #
    # ------------------------------------------------------------------ #

    def _build_texts(
        self, items: list[dict], training: bool
    ) -> tuple[list[str], list[str], list]:
        """Trả về (full_texts, prompt_texts, images)."""
        full_texts, prompt_texts, images = [], [], []

        for item in items:
            messages_user = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": item["question"]},
                    ],
                }
            ]
            prompt_text = self.processor.apply_chat_template(
                messages_user, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(prompt_text)
            images.append(item["image"])

            if training:
                messages_full = messages_user + [
                    {"role": "assistant", "content": item.get("answer", "")}
                ]
                full_text = self.processor.apply_chat_template(
                    messages_full, tokenize=False, add_generation_prompt=False
                )
            else:
                full_text = prompt_text
            full_texts.append(full_text)

        return full_texts, prompt_texts, images

    def process_batch(
        self,
        items: list[dict],
        max_length: int = 512,
        training: bool = True,
    ) -> dict:
        full_texts, prompt_texts, images = self._build_texts(items, training)

        # Gọi processor 1 lần cho cả batch — processor tự handle pixel_values padding
        enc = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        result = dict(enc)

        if training:
            labels = enc.input_ids.clone()
            for i, (full_text, prompt_text) in enumerate(zip(full_texts, prompt_texts)):
                answer_ids = self.processor.tokenizer.encode(
                    full_text[len(prompt_text):], add_special_tokens=False
                )
                # Dùng real_len (không tính padding) thay vì total padded length
                # để tránh tính sai prompt_len khi sequence ngắn hơn max_length
                real_len = int(enc.attention_mask[i].sum().item())
                prompt_len = max(0, real_len - len(answer_ids))
                labels[i, :prompt_len] = -100
            # Mask padding tokens trong labels
            labels[enc.input_ids == self.pad_token_id] = -100
            result["labels"] = labels

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
