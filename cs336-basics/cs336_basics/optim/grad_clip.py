import torch
from typing import Iterable


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """按 L2 范数对梯度进行裁剪，避免更新步过大导致不稳定

    参数：
    - parameters：待裁剪的参数迭代器
    - max_l2_norm：允许的最大梯度范数
    - eps：数值稳定性项
    """
    grads = [p.grad for p in parameters if (p.grad is not None and p.requires_grad)]
    if not grads:
        return
    total_sq = torch.tensor(0.0, device=grads[0].device)
    for g in grads:
        total_sq = total_sq + g.float().pow(2).sum()
    total_norm = torch.sqrt(total_sq)
    if total_norm <= max_l2_norm:
        return
    scale = max_l2_norm / (total_norm + eps)
    with torch.no_grad():
        for g in grads:
            g.mul_(scale)
