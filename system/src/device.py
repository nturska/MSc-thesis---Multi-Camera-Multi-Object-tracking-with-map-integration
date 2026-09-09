"""Wybór urządzenia PyTorch: CUDA → MPS → CPU."""

from __future__ import annotations

import os

import torch


def get_torch_device() -> torch.device:
    """
    Preferuje NVIDIA CUDA na serwerach GPU, potem Apple MPS, w przeciwnym razie CPU.
    Ustawia też PYTORCH_ENABLE_MPS_FALLBACK=1 (bezpieczne na wszystkich platformach).
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
