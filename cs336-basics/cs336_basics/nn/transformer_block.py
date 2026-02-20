import torch
from torch import nn
import einx
from cs336_basics.nn.rmsnorm import RMSNorm
from cs336_basics.nn.mha import MultiHeadSelfAttention
from cs336_basics.nn.swiglu import SwiGLU
from cs336_basics.nn.silu_ffn import SiLUFFN


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        norm_style: str = "pre",
        ffn_style: str = "swiglu",
        ffn_match_params: bool = False,
        dropout_p: float = 0.0,
        max_seq_len: int | None = None,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)
        self.use_rope = bool(use_rope)
        self.max_seq_len = None if max_seq_len is None else int(max_seq_len)
        self.theta = float(theta)
        self.norm_style = "pre" if norm_style not in ("pre", "post") else norm_style

        self.ln1 = RMSNorm(d_model=self.d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        self.attn = MultiHeadSelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            device=device,
            dtype=dtype,
            use_rope=self.use_rope,
            max_seq_len=self.max_seq_len,
            theta=self.theta,
        )
        self.ln2 = RMSNorm(d_model=self.d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        if ffn_style == "silu":
            width = int(round(1.5 * self.d_ff)) if ffn_match_params else self.d_ff
            self.ffn = SiLUFFN(d_model=self.d_model, d_ff=width, device=device, dtype=dtype)
        else:
            self.ffn = SwiGLU(d_model=self.d_model, d_ff=self.d_ff, device=device, dtype=dtype)
        self.attn_dropout = nn.Dropout(p=float(dropout_p)) if dropout_p and dropout_p > 0.0 else nn.Identity()
        self.ffn_dropout = nn.Dropout(p=float(dropout_p)) if dropout_p and dropout_p > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        t = x.shape[-2]
        if self.use_rope and token_positions is None:
            pos = torch.arange(t, device=x.device, dtype=torch.long)
            ones = torch.ones((b,), device=x.device, dtype=torch.long)
            token_positions = einx.multiply("b, t -> b t", ones, pos)
        if self.norm_style == "pre":
            h = self.ln1(x)
            h = self.attn(h, token_positions=token_positions)
            h = self.attn_dropout(h)
            x = x + h
            h2 = self.ln2(x)
            h2 = self.ffn(h2)
            h2 = self.ffn_dropout(h2)
            y = x + h2
            return y
        else:
            h = self.attn(x, token_positions=token_positions)
            h = self.attn_dropout(h)
            x_attn = x + h
            y1 = self.ln1(x_attn)
            f = self.ffn(y1)
            f = self.ffn_dropout(f)
            y2 = y1 + f
            y = self.ln2(y2)
            return y
