"""
Factory để tạo adapter tương ứng từ config.

Thêm model mới:
  1. Tạo file src/adapters/<model_type>.py kế thừa BaseAdapter
  2. Đăng ký vào _REGISTRY bên dưới
  3. Tạo config yaml trong configs/
"""

from .base import BaseAdapter
from .qwen2vl import Qwen2VLAdapter
from .smolvlm import SmolVLMAdapter

_REGISTRY: dict[str, type[BaseAdapter]] = {
    "qwen2vl": Qwen2VLAdapter,
    "smolvlm": SmolVLMAdapter,
}


def get_adapter(model_type: str) -> BaseAdapter:
    """
    Trả về adapter instance từ model type string.

    Args:
        model_type: một trong các key trong _REGISTRY
                    (lấy từ field `model.type` trong config.yaml)
    """
    model_type = model_type.lower()
    if model_type not in _REGISTRY:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[model_type]()
