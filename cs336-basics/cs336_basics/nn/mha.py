import math
import torch
from torch import nn
import einx
from einops import rearrange
from cs336_basics.nn.sdpa import scaled_dot_product_attention
from cs336_basics.nn.rope import RotaryPositionalEmbedding
from contextlib import contextmanager


try:
    import torch.cuda.nvtx as _nvtx
    def nvtx_range(name: str):
        return _nvtx.range(name)
except Exception:
    @contextmanager
    def nvtx_range(name: str):
        yield


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力（可选 RoPE 位置编码）

    权重为手动注册的 `nn.Parameter`，前向中进行矩阵乘、重排、可选 RoPE，再经 SDPA 得到每头输出并合并。
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_rope: bool = False,
        max_seq_len: int | None = None,
        theta: float = 10000.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_head = self.d_model // self.num_heads
        self.use_rope = bool(use_rope)
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        # Store weights as (out_features, in_features)
        self.q_proj = nn.Parameter(torch.empty((self.d_model, self.d_model), **factory_kwargs))
        self.k_proj = nn.Parameter(torch.empty((self.d_model, self.d_model), **factory_kwargs))
        self.v_proj = nn.Parameter(torch.empty((self.d_model, self.d_model), **factory_kwargs))
        self.o_proj = nn.Parameter(torch.empty((self.d_model, self.d_model), **factory_kwargs))

        def _init(param: torch.Tensor):
            sigma = math.sqrt(2.0 / (self.d_model + self.d_model))
            torch.nn.init.trunc_normal_(param, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)

        for p in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            _init(p)
        if self.use_rope:
            assert max_seq_len is not None
            self.rope = RotaryPositionalEmbedding(theta=float(theta), d_k=self.d_head, max_seq_len=int(max_seq_len), device=factory_kwargs.get("device"))
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """前向传播

        参数：
        - x：形状 `[batch, seq_len, d_model]`
        - token_positions：可选的 token 位置（用于 RoPE），形状 `[batch, seq_len]`
        返回：
        - y：形状 `[batch, seq_len, d_model]`
        """
        device = x.device
        seq_len = x.shape[-2]
        with nvtx_range("qkv projections"):
            q = torch.einsum("... t d, o d -> ... t o", x, self.q_proj)
            k = torch.einsum("... t d, o d -> ... t o", x, self.k_proj)
            v = torch.einsum("... t d, o d -> ... t o", x, self.v_proj)
        with nvtx_range("reshape heads"):
            q = rearrange(q, "... t (h d) -> ... h t d", h=self.num_heads)
            k = rearrange(k, "... t (h d) -> ... h t d", h=self.num_heads)
            v = rearrange(v, "... t (h d) -> ... h t d", h=self.num_heads)
        if self.rope is not None:
            with nvtx_range("rope"):
                if token_positions is None:
                    b = x.shape[0]
                    pos = torch.arange(seq_len, device=device, dtype=torch.long)
                    ones = torch.ones((b,), device=device, dtype=torch.long)
                    token_positions = einx.multiply("b, t -> b t", ones, pos)
                q = self.rope(q, token_positions)
                k = self.rope(k, token_positions)
        causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        with nvtx_range("sdpa"):
            out_heads = scaled_dot_product_attention(q, k, v, mask=causal)
        with nvtx_range("merge heads"):
            out = rearrange(out_heads, "... h t d -> ... t (h d)")
        with nvtx_range("output projection"):
            y = torch.einsum("... t d, o d -> ... t o", out, self.o_proj)
        return y
