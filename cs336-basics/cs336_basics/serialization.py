from __future__ import annotations

import os
import torch
from typing import BinaryIO, IO


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]) -> None:
    """保存模型与优化器状态字典，以及当前迭代计数"""
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(obj, out)


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """载入检查点并恢复模型与优化器，返回恢复后的起始迭代计数"""
    obj = torch.load(src, map_location="cpu")
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return int(obj["iteration"])
