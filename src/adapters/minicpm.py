"""
Adapter cho MiniCPM-Llama3-V 2.5.
Supported models:
  - openbmb/MiniCPM-Llama3-V-2_5

Đặc điểm:
  - Dùng AutoModel/AutoProcessor với trust_remote_code=True
  - Processor xử lý từng sample, sau đó pad thủ công theo batch
  - Model custom có forward(data, **kwargs) nên cần wrapper để Trainer dùng
  - image_bound phải được offset lại sau khi left-pad input_ids
"""

import torch
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from .base import BaseAdapter


class _MiniCPMWrapper(nn.Module):
    """Chuẩn hóa MiniCPM về interface giống các model HF thông thường."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: list[list[torch.Tensor]],
        tgt_sizes: list[torch.Tensor],
        image_bound: list[torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        data = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "tgt_sizes": tgt_sizes,
            "image_bound": image_bound,
            "position_ids": position_ids,
        }
        return self.base_model(
            data=data,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: list[list[torch.Tensor]],
        tgt_sizes: list[torch.Tensor],
        image_bound: list[torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        tokenizer=None,
        **kwargs,
    ):
        model_inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "tgt_sizes": tgt_sizes,
            "image_bound": image_bound,
            "position_ids": position_ids,
        }
        return self.base_model.generate(
            model_inputs,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            **kwargs,
        )

    def save_pretrained(self, *args, **kwargs):
        return self.base_model.save_pretrained(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


class MiniCPMAdapter(BaseAdapter):

    @staticmethod
    def _apply_qlora(model, lora_cfg: dict):
        """MiniCPM có forward custom nên không dùng task_type=CAUSAL_LM."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg["bias"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    @staticmethod
    def _build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        return position_ids

    @staticmethod
    def _left_pad_1d(
        sequences: list[torch.Tensor], pad_value: int, max_len: int
    ) -> torch.Tensor:
        padded = []
        for seq in sequences:
            pad_len = max_len - seq.size(0)
            padded.append(
                torch.cat(
                    [torch.full((pad_len,), pad_value, dtype=seq.dtype), seq],
                    dim=0,
                )
            )
        return torch.stack(padded)

    def _encode_sample(self, image, text: str, max_length: int) -> dict:
        enc = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
            max_length=max_length,
        )
        input_ids = enc["input_ids"]
        image_bound = enc.get("image_bound") or []
        if image_bound:
            image_bound = image_bound[0]
        else:
            image_bound = torch.empty((0, 2), dtype=torch.long)
        return {
            "input_ids": input_ids[0],
            "pixel_values": enc["pixel_values"],
            "tgt_sizes": enc["tgt_sizes"],
            "image_bound": image_bound.long(),
        }

    def _build_texts(self, item: dict, training: bool) -> tuple[str, str]:
        image_prefix = "(<image>./</image>)\n"
        prompt_messages = [
            {
                "role": "user",
                "content": image_prefix + item["question"],
            }
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if training:
            full_messages = prompt_messages + [
                {"role": "assistant", "content": item.get("answer", "")}
            ]
            full_text = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            full_text = prompt_text
        return full_text, prompt_text

    def _wrap_model(self, model):
        return _MiniCPMWrapper(model)

    # ------------------------------------------------------------------ #
    #  Load                                                                #
    # ------------------------------------------------------------------ #

    def load(self, cfg: dict) -> None:
        model_name = cfg["model"]["name"]
        use_lora = "lora" in cfg
        use_quant = "quantization" in cfg

        print(f"[MiniCPM] Loading tokenizer/processor: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

        load_kwargs = dict(
            device_map=self._get_device_map(),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        if use_quant:
            print(f"[MiniCPM] Loading model (4-bit): {model_name}")
            load_kwargs["quantization_config"] = self._build_bnb_config(cfg["quantization"])
        else:
            print(f"[MiniCPM] Loading model (bf16): {model_name}")

        model = AutoModel.from_pretrained(model_name, **load_kwargs)

        if use_lora:
            model = self._apply_qlora(model, cfg["lora"])
        else:
            model.train()
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"trainable params: {trainable:,} || all params: {total:,} "
                f"|| trainable%: {trainable / total * 100:.4f}"
            )

        self.model = self._wrap_model(model)

    def load_for_inference(self, cfg: dict, checkpoint: str | None = None) -> None:
        from peft import PeftModel

        model_name = cfg["model"]["name"]
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        proc_src = checkpoint or model_name
        print(f"[MiniCPM] Loading tokenizer/processor từ: {proc_src}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            proc_src, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            proc_src, trust_remote_code=True
        )
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

        print(f"[MiniCPM] Loading base model: {model_name}")
        base = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=self._get_device_map(),
        )
        if checkpoint:
            print(f"[MiniCPM] Merging LoRA từ: {checkpoint}")
            base = PeftModel.from_pretrained(base, checkpoint).merge_and_unload()
        base.eval()
        self.model = self._wrap_model(base)
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
        encoded_items = []
        labels_list = []

        for item in items:
            full_text, prompt_text = self._build_texts(item, training)
            encoded = self._encode_sample(item["image"], full_text, max_length)
            encoded_items.append(encoded)

            if training:
                labels = self._compute_labels(
                    encoded["input_ids"].long(),
                    full_text,
                    prompt_text,
                    self.tokenizer,
                )
                labels[encoded["input_ids"] == self.pad_token_id] = -100
                labels_list.append(labels)

        max_len = max(x["input_ids"].size(0) for x in encoded_items)
        input_ids = self.processor.pad(
            encoded_items,
            padding_side="left",
            padding_value=self.pad_token_id,
            max_length=max_len,
            key="input_ids",
        ).long()
        attention_mask = input_ids.ne(self.pad_token_id).long()
        position_ids = self._build_position_ids(attention_mask)

        image_bound = []
        pixel_values = []
        tgt_sizes = []
        for encoded in encoded_items:
            pad_offset = max_len - encoded["input_ids"].size(0)
            if encoded["image_bound"].numel() > 0:
                image_bound.append(encoded["image_bound"] + pad_offset)
            else:
                image_bound.append(encoded["image_bound"])
            pixel_values.append(encoded["pixel_values"])
            tgt_sizes.append(encoded["tgt_sizes"])

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "tgt_sizes": tgt_sizes,
            "image_bound": image_bound,
        }
        if training:
            result["labels"] = self._left_pad_1d(labels_list, -100, max_len).long()
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
        gen_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        generated = self.model.generate(
            **gen_inputs,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            decode_text=True,
        )
        return [text.strip() for text in generated]
