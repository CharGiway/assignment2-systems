from __future__ import annotations

import numpy as np
import torch
import numpy.typing as npt


def get_batch(dataset: npt.NDArray[np.int_], batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """从一维 token 序列随机采样一个训练 batch

    说明：
    - 随机选择 `batch_size` 个起点，每个位置取长度为 `context_length` 的连续片段
    - 输入 `x` 为片段本身，目标 `y` 为右移一位的片段（下一 token 预测）
    参数：
    - dataset：形如 `[N]` 的 token 数组
    - batch_size：批大小
    - context_length：上下文长度
    - device：'cpu'/'cuda'/'mps' 等
    返回：
    - `(x, y)`，形状均为 `[batch_size, context_length]` 的 LongTensor
    """
    n = int(len(dataset))
    cl = int(context_length)
    num_possible = n - cl
    assert num_possible > 0
    starts = np.random.randint(0, num_possible, size=(batch_size,), dtype=np.int64)
    offsets = np.arange(cl, dtype=np.int64)[None, :]
    idx = starts[:, None] + offsets
    x_np = dataset[idx]
    y_np = dataset[idx + 1]
    x = torch.as_tensor(x_np, dtype=torch.long)
    y = torch.as_tensor(y_np, dtype=torch.long)
    return x.to(device), y.to(device)
