import torch


def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    m = in_features.max(dim=dim, keepdim=True).values
    z = torch.exp(in_features - m)
    return z / z.sum(dim=dim, keepdim=True)

