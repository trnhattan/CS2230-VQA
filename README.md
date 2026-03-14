# ViTextVQA — Fine-tuning & Evaluation

Fine-tuning các Vision-Language Model trên tập dữ liệu **ViTextVQA** (Vietnamese Text-based VQA).

## Models được hỗ trợ

| Config | Model | Params | Method | VRAM (train) |
|--------|-------|--------|--------|--------------|
| `configs/qwen2vl_2b.yaml` | Qwen/Qwen2-VL-2B-Instruct | 2B | QLoRA 4-bit | ~8 GB |
| `configs/internvl2_2b.yaml` | OpenGVLab/InternVL2-2B | 2.2B | QLoRA 4-bit | ~8 GB |
| `configs/smolvlm_500m.yaml` | HuggingFaceTB/SmolVLM-500M-Instruct | 500M | LoRA (bf16) | ~4 GB |
| `configs/smolvlm_2b.yaml` | HuggingFaceTB/SmolVLM-Instruct | 2B | QLoRA 4-bit | ~8 GB |
| `configs/smolvlm2_2b.yaml` | HuggingFaceTB/SmolVLM2-2.2B-Instruct | 2.2B | QLoRA 4-bit | ~8 GB |

Hỗ trợ 3 chế độ training: **QLoRA** (4-bit + LoRA), **LoRA** (bf16 + LoRA), và **Full fine-tune** — tùy theo các section có trong config yaml.

---

## Cấu trúc thư mục

```
CS2230-VQA/
├── configs/                     # Config cho từng model
│   ├── qwen2vl_2b.yaml
│   ├── internvl2_2b.yaml
│   ├── smolvlm_500m.yaml
│   ├── smolvlm_2b.yaml
│   └── smolvlm2_2b.yaml
├── data/
│   ├── ViTextVQA_train.json
│   ├── ViTextVQA_dev.json
│   ├── ViTextVQA_test_gt.json
│   └── st_images/               # Ảnh (*.jpg)
├── src/
│   ├── adapters/
│   │   ├── __init__.py          # Factory: get_adapter(type)
│   │   ├── base.py              # Abstract base adapter
│   │   ├── internvl2.py         # Adapter cho InternVL2
│   │   ├── qwen2vl.py           # Adapter cho Qwen2-VL
│   │   └── smolvlm.py           # Adapter cho SmolVLM / SmolVLM2
│   ├── dataset.py               # ViTextVQADataset + VQADataCollator
│   ├── metrics.py               # ANLS + Exact Match
│   ├── train.py                 # Training script
│   └── evaluate.py              # Inference + Evaluation
├── results/                     # Predictions JSON output
├── checkpoints/                 # Model checkpoints
├── requirements.txt
└── README.md
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

> **Lưu ý:** `bitsandbytes` yêu cầu CUDA. Google Colab T4 (16 GB VRAM) là đủ cho tất cả models.

---

## Training

```bash
# Qwen2-VL-2B — multilingual tốt, khuyến nghị cho ViTextVQA
python -m src.train --config configs/qwen2vl_2b.yaml

# InternVL2-2B — vision encoder mạnh cho OCR
python -m src.train --config configs/internvl2_2b.yaml

# SmolVLM-500M — nhẹ nhất
python -m src.train --config configs/smolvlm_500m.yaml

# SmolVLM-2B
python -m src.train --config configs/smolvlm_2b.yaml

# SmolVLM2-2.2B
python -m src.train --config configs/smolvlm2_2b.yaml
```

Checkpoint tốt nhất theo eval_loss trên dev set được lưu tại `checkpoints/<model>/best_model/`.

### Multi-GPU (Data Parallelism)

Hỗ trợ train trên nhiều GPU bằng `accelerate` hoặc `torchrun`. Code tự detect `LOCAL_RANK` để chuyển từ model parallelism sang data parallelism.

```bash
# 2 GPUs
accelerate launch --num_processes=2 -m src.train --config configs/qwen2vl_2b.yaml

# 4 GPUs
accelerate launch --num_processes=4 -m src.train --config configs/internvl2_2b.yaml

# Hoặc dùng torchrun
torchrun --nproc_per_node=2 -m src.train --config configs/qwen2vl_2b.yaml
```

> **Lưu ý:** Effective batch size = `per_device_train_batch_size × num_gpus × gradient_accumulation_steps`. Khi tăng số GPU, nên giảm `gradient_accumulation_steps` tương ứng để giữ effective batch size.

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
- **Exact Match**: tỉ lệ khớp chính xác sau khi chuẩn hóa text (lowercase, bỏ dấu câu thừa, Unicode NFC).

---

## Adapter pattern

Mỗi model family có một adapter riêng kế thừa `BaseAdapter`, implement 4 method:

| Method | Chức năng |
|--------|-----------|
| `load()` | Load model + tokenizer/processor cho training |
| `load_for_inference()` | Load cho inference (full precision, optional LoRA merge) |
| `process_batch()` | Xử lý raw items → batched tensors |
| `generate()` | Chạy model.generate() và decode output |

### Chế độ training

Config yaml quyết định chế độ training dựa trên sự có mặt của các section:

| Section trong config | Chế độ |
|---------------------|--------|
| `lora` + `quantization` | QLoRA (4-bit + LoRA) |
| `lora` (không `quantization`) | LoRA (bf16 + LoRA) |
| Không có cả hai | Full fine-tune (bf16) |

### Thêm model mới

1. Tạo `src/adapters/<model_type>.py` kế thừa `BaseAdapter`
2. Đăng ký trong `src/adapters/__init__.py`:

```python
_REGISTRY = {
    "internvl2": InternVL2Adapter,
    "qwen2vl": Qwen2VLAdapter,
    "smolvlm": SmolVLMAdapter,
    "mymodel": MyModelAdapter,   # thêm vào đây
}
```

3. Tạo config `configs/mymodel.yaml` với `model.type: "mymodel"`

---

## Tùy chỉnh config

### Giảm VRAM

```yaml
model:
  max_pixels: 512      # Qwen2-VL: giảm resolution ảnh
  max_num_tiles: 2     # InternVL2: giảm số tiles

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32

lora:
  r: 8                 # Giảm LoRA rank
```

### Tăng tốc training

```yaml
training:
  eval_steps: 2000               # Giảm tần suất eval (eval dùng generate, rất chậm)
  per_device_train_batch_size: 4  # Tăng batch nếu đủ VRAM
  gradient_accumulation_steps: 4
  dataloader_num_workers: 4
```

---

## Kỹ thuật tiết kiệm GPU

| Kỹ thuật | Tác dụng |
|----------|----------|
| **4-bit NF4 quantization** | Giảm ~75% VRAM cho base model weights |
| **LoRA r=16** | Chỉ train ~1% tham số, bỏ optimizer states cho 99% model |
| **Gradient checkpointing** | Giảm activation memory (đánh đổi ~20% tốc độ) |
| **batch_size=1 + grad_accum** | Peak VRAM thấp, effective batch size vẫn lớn |
| **bf16 / fp16** | Mixed precision, giảm ~50% memory cho activations |
