"""推理解码与采样工具

提供基于温度与 top-p（nucleus）采样的下一 token 选择函数，以及按步生成的解码函数。
"""
import torch
from cs336_basics.nn.softmax import softmax


def _sample_next(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """从最后一步 logits 中采样一个下一 token id

    参数：
    - logits：形如 `[vocab_size]` 的未归一化对数概率
    - temperature：温度，>0 时进行缩放；=0 退化为贪心 argmax
    - top_p：nucleus 采样阈值（累积概率截断），≥1 时退化为全量采样
    返回：
    - 选中的 token id（int）
    """
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())
    scaled = logits / float(temperature)
    probs = softmax(scaled, dim=-1)
    if top_p is None or top_p >= 1.0:
        return int(torch.multinomial(probs, 1).item())
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    cutoff = torch.searchsorted(cdf, torch.tensor(float(top_p), device=logits.device))
    k = int(cutoff.item()) + 1
    k = max(1, k)
    nucleus_probs = sorted_probs[:k]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()
    pick = torch.multinomial(nucleus_probs, 1).item()
    return int(sorted_idx[pick].item())


def decode(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    *,
    eot_id: int,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str | torch.device | None = None,
    return_full_sequence: bool = False,
) -> torch.Tensor:
    """基于给定 prompt 进行自回归解码

    流程：
    - 将 prompt 送入模型，取最后位置的 logits
    - 按温度与 top-p 规则采样下一 token
    - 追加到序列，直到生成 `max_new_tokens` 或遇到 `eot_id`
    注意：
    - 若模型定义了 `context_length`，仅用末尾 `T` 个 token 作为上下文
    - `return_full_sequence=True` 时返回包含 prompt 的完整序列，否则仅返回新增片段
    """
    if device is None:
        device = next(model.parameters()).device
    x = prompt_ids.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    T = getattr(model, "context_length", x.shape[1])
    start_len = x.shape[1]
    for _ in range(int(max_new_tokens)):
        ctx = x[:, -T:]
        logits = model(ctx)[:, -1, :]
        next_id = _sample_next(logits[0], temperature=temperature, top_p=top_p)
        x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if next_id == int(eot_id):
            break
    out = x[0]
    if return_full_sequence:
        return out
    return out[start_len:]
