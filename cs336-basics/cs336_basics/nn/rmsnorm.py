import torch
from torch import nn
import einx


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = int(d_model)
        self.eps = float(eps)
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        self.weight = nn.Parameter(torch.ones((self.d_model,), **factory_kwargs))

    # input shape (batch_size, sequence_length, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)
        m = einx.mean("... d -> ... 1", x32 * x32)
        denom = torch.sqrt(m + self.eps)
        y = einx.multiply("... d, d -> ... d", x32 / denom, self.weight)
        return y.to(orig_dtype)
