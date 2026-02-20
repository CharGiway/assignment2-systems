import torch
import einx


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    m = logits.max(dim=-1, keepdim=True).values
    z = torch.exp(logits - m)
    s = torch.sum(z, dim=-1, keepdim=True)
    logsumexp = m + torch.log(s)
    t = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1))
    loss = logsumexp - t
    return loss.mean()
