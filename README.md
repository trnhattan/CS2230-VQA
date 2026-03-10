# ViTextVQA — Fine-tuning & Evaluation

Fine-tuning các Vision-Language Model nhẹ trên tập dữ liệu **ViTextVQA** (Vietnamese Text-based VQA).

## Models được hỗ trợ

| Config | Model | Params | VRAM (train) |
|--------|-------|--------|--------------|
| `configs/qwen2vl_2b.yaml` | Qwen/Qwen2-VL-2B-Instruct | 2B | ~8 GB |
| `configs/smolvlm_500m.yaml` | HuggingFaceTB/SmolVLM-500M-Instruct | 500M | ~5 GB |
| `configs/smolvlm_2b.yaml` | HuggingFaceTB/SmolVLM-Instruct | 2B | ~8 GB |
| `configs/smolvlm2_2b.yaml` | HuggingFaceTB/SmolVLM2-2.2B-Instruct | 2.2B | ~8 GB |

Tất cả dùng **QLoRA** (4-bit NF4 + LoRA r=16) để tối thiểu VRAM.

---

## Cấu trúc thư mục

```
ViTextVQA/
├── configs/                     # Config cho từng model
│   ├── qwen2vl_2b.yaml
│   ├── smolvlm_500m.yaml
│   ├── smolvlm_2b.yaml
│   └── smolvlm2_2b.yaml
├── data/
│   ├── ViTextVQA_train.json
│   ├── ViTextVQA_dev.json
│   ├── ViTextVQA_test_gt .json
│   └── st_images/               # Ảnh (*.jpg)
├── src/
│   ├── adapters/
│   │   ├── __init__.py          # Factory: get_adapter(type)
│   │   ├── base.py              # Abstract base adapter
│   │   ├── qwen2vl.py           # Adapter cho Qwen2-VL
│   │   └── smolvlm.py           # Adapter cho SmolVLM / SmolVLM2
│   ├── dataset.py               # ViTextVQADataset + VQADataCollator
│   ├── metrics.py               # ANLS + Exact Match
│   ├── train.py                 # Training script
│   └── evaluate.py              # Inference + Evaluation
├── results/                     # Predictions JSON output
├── checkpoints/                 # LoRA checkpoints
├── requirements.txt
└── README.md
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

> **Lưu ý:** `bitsandbytes` yêu cầu CUDA. Với GPU NVIDIA RTX 3080+ (8 GB VRAM) là đủ.
> Muốn nhanh hơn nên cài thêm `flash-attn`:
> ```bash
> pip install flash-attn --no-build-isolation
> ```

---

## Training

```bash
# Train Qwen2-VL-2B (khuyến nghị cho Vietnamese OCR-VQA)
python -m src.train --config configs/qwen2vl_2b.yaml

# Train SmolVLM-500M (nhẹ nhất, chạy được trên GPU 5GB)
python -m src.train --config configs/smolvlm_500m.yaml

# Train SmolVLM-2B
python -m src.train --config configs/smolvlm_2b.yaml

# Train SmolVLM2-2.2B (phiên bản mới nhất)
python -m src.train --config configs/smolvlm2_2b.yaml
```

Checkpoint tốt nhất theo ANLS trên dev set được lưu tại `checkpoints/<model>/best_model/`.

---

## Evaluation trên Test Set

```bash
# Zero-shot (base model, không fine-tune)
python -m src.evaluate --config configs/qwen2vl_2b.yaml

# Fine-tuned model
python -m src.evaluate \
    --config configs/qwen2vl_2b.yaml \
    --checkpoint checkpoints/qwen2vl_2b/best_model

# Chỉ định file output
python -m src.evaluate \
    --config configs/smolvlm_500m.yaml \
    --checkpoint checkpoints/smolvlm_500m/best_model \
    --results_file results/smolvlm_500m_test.json
```

Kết quả in ra terminal:
```
=======================================================
  Model  : Qwen/Qwen2-VL-2B-Instruct
  Mode   : Fine-tuned  → checkpoints/qwen2vl_2b/best_model
=======================================================
  ANLS         : 65.42%
  Exact Match  : 51.30%
  Num samples  : 4013
=======================================================
```

---

## Metrics

- **ANLS** (Average Normalized Levenshtein Similarity): metric chính của TextVQA, đo độ tương đồng giữa prediction và ground truth (threshold = 0.5).
- **Exact Match**: tỉ lệ khớp chính xác sau khi chuẩn hóa text (lowercase, bỏ dấu câu thừa).

---

## Tùy chỉnh config

### Giảm VRAM khi training

Trong file config, chỉnh các tham số:

```yaml
model:
  max_pixels: 512    # Giảm từ 1280 (Qwen2-VL: ảnh nhỏ hơn)

training:
  per_device_train_batch_size: 1     # Giữ ở 1
  gradient_accumulation_steps: 32   # Tăng lên (effective batch giữ nguyên)

lora:
  r: 8               # Giảm LoRA rank (ít params hơn)
```

### Thêm model mới

1. Tạo adapter mới tại `src/adapters/<model_type>.py` kế thừa `BaseAdapter`:

```python
from .base import BaseAdapter

class MyModelAdapter(BaseAdapter):
    def load(self, cfg): ...
    def load_for_inference(self, cfg, checkpoint=None): ...
    def process_batch(self, items, max_length=512, training=True): ...
    def generate(self, inputs, max_new_tokens=64, num_beams=1): ...
```

2. Đăng ký trong `src/adapters/__init__.py`:

```python
from .my_model import MyModelAdapter

_REGISTRY = {
    "qwen2vl": Qwen2VLAdapter,
    "smolvlm": SmolVLMAdapter,
    "mymodel": MyModelAdapter,   # thêm vào đây
}
```

3. Tạo config `configs/mymodel.yaml` với `model.type: "mymodel"`.

---

## Kỹ thuật tiết kiệm GPU

| Kỹ thuật | Tác dụng |
|----------|----------|
| **4-bit NF4 quantization** | Giảm ~75% VRAM cho base model weights |
| **LoRA r=16** | Chỉ train ~1% tham số, bỏ optimizer states cho 99% model |
| **Gradient checkpointing** | Giảm activation memory (đánh đổi ~20% tốc độ) |
| **batch_size=1 + grad_accum** | Peak VRAM thấp, effective batch size vẫn lớn |
| **bf16 / fp16** | Mixed precision, giảm ~50% memory cho activations |
