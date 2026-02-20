import math


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """线性 warmup → 余弦退火 学习率调度

    说明：
    - 在 `warmup_iters` 前，学习率线性从 0 升至 `max_learning_rate`
    - 在 `warmup_iters..cosine_cycle_iters` 范围内，按余弦曲线从 `max_learning_rate` 退火至 `min_learning_rate`
    - 之后保持为 `min_learning_rate`
    """
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    if it < cosine_cycle_iters:
        span = cosine_cycle_iters - warmup_iters
        x = (it - warmup_iters) / span
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1.0 + math.cos(math.pi * x))
    return min_learning_rate
