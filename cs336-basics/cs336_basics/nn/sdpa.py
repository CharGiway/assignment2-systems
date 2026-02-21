import math
import torch
from cs336_basics.nn.softmax import softmax
from contextlib import contextmanager


try:
    import torch.cuda.nvtx as _nvtx
    def nvtx_range(name: str):
        return _nvtx.range(name)
except Exception:
    @contextmanager
    def nvtx_range(name: str):
        yield


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """缩放点积注意力（支持可选掩码）"""
    with nvtx_range("scaled dot product attention"):
        q = Q.to(torch.float32)
        k = K.to(torch.float32)
        v = V.to(torch.float32)
        d_k = q.shape[-1]
        with nvtx_range("computing attention scores"):
            scores = torch.einsum("... q d, ... k d -> ... q k", q, k) / math.sqrt(d_k)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
        with nvtx_range("computing softmax"):
            probs = softmax(scores, dim=-1)
        with nvtx_range("final matmul"):
            out = torch.einsum("... q k, ... k d -> ... q d", probs, v)
        return out.to(V.dtype)
