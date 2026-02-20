import math
import torch
from torch import nn
import einx


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        assert d_k % 2 == 0
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)
        inv = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, dtype=torch.float32) / float(self.d_k)))
        pos = torch.arange(0, self.max_seq_len, dtype=torch.float32)
        ang = einx.multiply("p, h -> p h", pos, inv)
        cos = torch.cos(ang)
        sin = torch.sin(ang)
        if device is not None:
            cos = cos.to(device)
            sin = sin.to(device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    # cos 的形状为 (max_seq_len, d_k // 2)
    # sin 的形状为 (max_seq_len, d_k // 2)
    # token_positions 的形状为 (batch_size, seq_len)
    # self.cos[token_positions] 的形状为 (batch_size, seq_len, d_k // 2)
    # self.sin[token_positions] 的形状为 (batch_size, seq_len, d_k // 2)
    # x 的形状为 (batch_size, seq_len, d_k)
    # x2 的形状为 (batch_size, seq_len, d_k // 2, 2)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x2 = einx.rearrange("... seq (h two) -> ... seq h two", x, two=2)
        xe = x2[..., 0]
        xo = x2[..., 1]
        while cos.dim() < xe.dim():
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        ye = xe * cos - xo * sin
        yo = xo * cos + xe * sin
        y = torch.stack((ye, yo), dim=-1)
        return einx.rearrange("... seq h two -> ... seq (h two)", y, two=2)
