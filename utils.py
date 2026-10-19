"""通用工具：设备、混合精度、随机种子（同 energy_llm_v2/utils.py）"""
import math
import random
import sys
from contextlib import nullcontext

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows 控制台中文
except Exception:
    pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name=None):
    if name:
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def amp_setup(device):
    """2080Ti（算力 7.5）不支持 bf16 → fp16 + GradScaler；30 系及以上 → bf16；CPU → 全精度"""
    if device.startswith("cuda"):
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype)
        scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
        print(f"[AMP] GPU 算力 {major}.x，使用 {dtype}")
        return ctx, scaler
    return nullcontext(), torch.amp.GradScaler("cpu", enabled=False)


def cosine_lr(step, max_steps, base_lr, min_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    ratio = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * ratio))


def print_gpu_mem():
    if torch.cuda.is_available():
        print(f"[显存] 峰值占用 {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
