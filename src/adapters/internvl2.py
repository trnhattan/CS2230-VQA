"""
Adapter cho InternVL2 (1B, 2B, 4B, 8B).
Supported models:
  - OpenGVLab/InternVL2-1B
  - OpenGVLab/InternVL2-2B
  - OpenGVLab/InternVL2-4B
  - OpenGVLab/InternVL2-8B

Đặc điểm:
  - Dynamic resolution: ảnh được chia tiles 448×448 linh hoạt
  - Dùng trust_remote_code=True để load model class từ HuggingFace
  - Mỗi tile → 256 image tokens (pixel shuffle downsample 2×)
  - Cần tự expand <image> → <img><IMG_CONTEXT>*N</img> trước khi tokenize
  - Model tự handle image embedding injection trong forward()
"""

from contextlib import contextmanager

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from .base import BaseAdapter

# --- Patch transformers v5 compat cho InternVL2 trust_remote_code ---
# InternVL2 custom model thiếu `all_tied_weights_keys` mà quantizer v5 cần.
try:
    import transformers.quantizers.base as _qbase

    _orig_get_keys = _qbase.get_keys_to_not_convert

    def _patched_get_keys(model):
        if not hasattr(model, "all_tied_weights_keys"):
            model.all_tied_weights_keys = {}
        return _orig_get_keys(model)

    _qbase.get_keys_to_not_convert = _patched_get_keys
except (ImportError, AttributeError):
    pass

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def _build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_ar)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_ar = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_ar[0]
    target_height = image_size * target_ar[1]
    blocks = target_ar[0] * target_ar[1]

    resized_img = image.resize((target_width, target_height))
    processed = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed.append(resized_img.crop(box))

    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


def _preprocess_image(image, input_size=448, max_num=6):
    transform = _build_transform(input_size)
    images = _dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    return torch.stack([transform(img) for img in images])


@contextmanager
def _patch_meta_linspace():
    """InternVL2 custom code gọi torch.linspace(...).item() trong __init__,
    nhưng device_map='auto' dùng meta tensors → .item() crash.
    Patch tạm để linspace trả về CPU tensor thay vì meta tensor."""
    _orig = torch.linspace

    def _safe(*args, **kwargs):
        t = _orig(*args, **kwargs)
        return t.to("cpu") if t.is_meta else t

    torch.linspace = _safe
    try:
        yield
    finally:
        torch.linspace = _orig


class InternVL2Adapter(BaseAdapter):

    _max_num_tiles: int = 6
    _num_image_token: int = 256
    _im_end_id: int = 0

    @staticmethod
    def _apply_qlora(model, lora_cfg: dict):
        """Override: không dùng task_type vì InternVL2 custom forward
        không chấp nhận inputs_embeds mà PeftModelForCausalLM tự thêm."""
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

    # ------------------------------------------------------------------ #
    #  Load                                                                #
    # ------------------------------------------------------------------ #

    def load(self, cfg: dict) -> None:
        """Training mode: full fine-tune hoặc QLoRA tuỳ config."""
        model_name = cfg["model"]["name"]
        self._max_num_tiles = cfg["model"].get("max_num_tiles", 6)
        use_lora = "lora" in cfg
        use_quant = "quantization" in cfg

        print(f"[InternVL2] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self._im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        load_kwargs = dict(
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        if use_quant:
            print(f"[InternVL2] Loading model (4-bit): {model_name}")
            load_kwargs["quantization_config"] = self._build_bnb_config(cfg["quantization"])
        else:
            print(f"[InternVL2] Loading model (bf16): {model_name}")

        with _patch_meta_linspace():
            model = AutoModel.from_pretrained(model_name, **load_kwargs)

        # forward() cần img_context_token_id để thay embedding
        model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self._num_image_token = model.num_image_token  # 256

        if use_lora:
            self.model = self._apply_qlora(model, cfg["lora"])
        else:
            model.train()
            self.model = model
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"trainable params: {trainable:,} || all params: {total:,} "
                f"|| trainable%: {trainable / total * 100:.4f}"
            )

    def load_for_inference(self, cfg: dict, checkpoint: str | None = None) -> None:
        from peft import PeftModel

        model_name = cfg["model"]["name"]
        self._max_num_tiles = cfg["model"].get("max_num_tiles", 6)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        print(f"[InternVL2] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self._im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        print(f"[InternVL2] Loading base model: {model_name}")
        with _patch_meta_linspace():
            base = AutoModel.from_pretrained(
                model_name,
                dtype=dtype,
                trust_remote_code=True,
                device_map="auto",
            )
        base.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self._num_image_token = base.num_image_token

        if checkpoint:
            print(f"[InternVL2] Merging LoRA từ: {checkpoint}")
            self.model = PeftModel.from_pretrained(base, checkpoint).merge_and_unload()
        else:
            self.model = base
        self.model.eval()

    # ------------------------------------------------------------------ #
    #  process_batch                                                       #
    # ------------------------------------------------------------------ #

    def _expand_image_placeholder(self, text: str, num_patches: int) -> str:
        """<image> → <img><IMG_CONTEXT>×N</img>"""
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * (self._num_image_token * num_patches)
            + IMG_END_TOKEN
        )
        return text.replace("<image>", image_tokens, 1)

    def process_batch(
        self,
        items: list[dict],
        max_length: int = 512,
        training: bool = True,
    ) -> dict:
        all_input_ids, all_masks, all_labels = [], [], []
        all_pixel_values = []

        for item in items:
            image = item["image"]
            question = item["question"]
            answer = item.get("answer", "")

            pixel_values = _preprocess_image(
                image, input_size=448, max_num=self._max_num_tiles
            ).to(torch.bfloat16)
            num_patches = pixel_values.shape[0]
            all_pixel_values.append(pixel_values)

            messages_user = [
                {"role": "user", "content": f"<image>\n{question}"}
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                messages_user, tokenize=False, add_generation_prompt=True
            )

            if training:
                messages_full = messages_user + [
                    {"role": "assistant", "content": answer}
                ]
                full_text = self.tokenizer.apply_chat_template(
                    messages_full, tokenize=False, add_generation_prompt=False
                )
            else:
                full_text = prompt_text

            prompt_text = self._expand_image_placeholder(prompt_text, num_patches)
            full_text = self._expand_image_placeholder(full_text, num_patches)

            enc = self.tokenizer(
                full_text, return_tensors="pt", add_special_tokens=False
            )
            ids = enc.input_ids[0]
            mask = enc.attention_mask[0]

            if training:
                labels = self._compute_labels(
                    ids, full_text, prompt_text, self.tokenizer
                )
                all_labels.append(labels)

            all_input_ids.append(ids)
            all_masks.append(mask)

        max_len = max(t.size(0) for t in all_input_ids)
        pixel_values_cat = torch.cat(all_pixel_values, dim=0)

        result = {
            "input_ids": self._pad_1d(all_input_ids, self.pad_token_id, max_len),
            "attention_mask": self._pad_1d(all_masks, 0, max_len),
            "pixel_values": pixel_values_cat,
            "image_flags": torch.ones(pixel_values_cat.shape[0], 1, dtype=torch.long),
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
        # InternVL2 custom generate() dùng inputs_embeds nội bộ,
        # output chỉ chứa NEW tokens (không bao gồm input).
        generated = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            eos_token_id=self._im_end_id,
            pad_token_id=self.pad_token_id,
        )
        return [
            self.tokenizer.decode(seq, skip_special_tokens=True).strip()
            for seq in generated
        ]
