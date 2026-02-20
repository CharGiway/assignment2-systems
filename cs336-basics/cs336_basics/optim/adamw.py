import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3, # alpha
        betas: tuple[float, float] = (0.9, 0.999), # beta1 and beta2
        eps: float = 1e-8,  # epsilon
        weight_decay: float = 0.0,  # lambda
    ):
        # 中文说明：
        # - 本实现遵循 AdamW 的“解耦权重衰减（decoupled weight decay）”：
        #   p ← p - lr * weight_decay * p
        # - 一阶与二阶动量按照经典 Adam 更新：
        #   m_t = β1 * m_{t-1} + (1 - β1) * g_t
        #   v_t = β2 * v_{t-1} + (1 - β2) * g_t ⊙ g_t
        # - 偏置修正（bias correction）：
        #   m̂_t = m_t / (1 - β1^t), v̂_t = v_t / (1 - β2^t)
        # - 归一化步长（step size）：
        #   step_size = lr * sqrt(1 - β2^t) / (1 - β1^t)
        # - 参数更新：
        #   p ← p - step_size * m_t / (sqrt(v_t) + eps)  （其中 step_size 已含偏置修正）
        defaults = dict(lr=float(lr), betas=betas, eps=float(eps), weight_decay=float(weight_decay))
        super().__init__(params, defaults)

    def step(self):
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                wd = group["weight_decay"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    state["step"] += 1
                    step = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    if wd != 0.0:
                        # 解耦权重衰减：p ← p - lr * wd * p
                        p.add_(p, alpha=-lr * wd)

                    # 一阶动量：m_t = β1 m_{t-1} + (1 - β1) g_t
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    # 二阶动量：v_t = β2 v_{t-1} + (1 - β2) g_t ⊙ g_t
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    # 偏置修正系数：m̂_t = m_t / (1 - β1^t), v̂_t = v_t / (1 - β2^t)
                    # 这里采用等价形式：step_size = lr * sqrt(1 - β2^t) / (1 - β1^t)
                    bias_c1 = 1.0 - beta1 ** step
                    bias_c2 = 1.0 - beta2 ** step
                    step_size = lr * math.sqrt(bias_c2) / bias_c1

                    # 归一化与更新：p ← p - step_size * m_t / (sqrt(v_t) + eps)
                    denom = exp_avg_sq.sqrt().add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-step_size)
