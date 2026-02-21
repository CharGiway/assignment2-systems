# 基准测试脚本（forward / forward+backward）：
# - 根据给定超参数或预设尺寸构建 Transformer 语言模型
# - 随机生成一个批次的 token 数据
# - 先运行若干预热步，再进行若干计时步
# - 如在 CUDA 上，步后调用 torch.cuda.synchronize() 保证计时准确
import argparse
import timeit
import numpy as np
import torch
from cs336_basics.nn.transformer_lm import TransformerLM
from cs336_basics.nn.cross_entropy import cross_entropy
from contextlib import contextmanager


# 预设模型规格表（与作业要求一致，便于快速切换规模）
SIZE_PRESETS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def build_model(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, device, dtype):
    # 构建基础 Transformer 语言模型（不含优化器，仅用于前后向计时）
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        device=device,
        dtype=dtype,
    )
    return model


def make_batch(batch_size, context_length, vocab_size, device, dtype):
    # 生成随机整数 token，形状为 [batch_size, context_length]
    # 注意：取值范围 [0, vocab_size)，用于构造交叉熵目标
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=device, dtype=torch.long)
    y = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=device, dtype=torch.long)
    return x, y


try:
    import torch.cuda.nvtx as _nvtx
    def nvtx_range(name: str):
        return _nvtx.range(name)
except Exception:
    @contextmanager
    def nvtx_range(name: str):
        yield


def time_steps(model, x, y, steps, mode, device, tag: str = "measure"):
    # 执行 steps 次计算并统计每步耗时：
    # - mode == "fwd"：仅前向（评估/推理常用）
    # - 否则：前向 + 反向（训练常用）
    times = []
    for i in range(steps):
        step_name = f"{tag}/step_{i+1}"
        start = timeit.default_timer()
        with nvtx_range(step_name):
            if mode == "fwd":
                with nvtx_range("forward"):
                    with torch.no_grad():
                        _ = model(x)
            else:
                with nvtx_range("forward"):
                    logits = model(x)
                    loss = cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
                with nvtx_range("backward"):
                    model.zero_grad(set_to_none=True)
                    loss.backward()
        if device.type == "cuda":
            # CUDA 上需同步以消除异步调度对计时的影响
            torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end - start)
    return np.array(times, dtype=np.float64)


def main():
    # 命令行参数定义（与外层 benchmark.sh 的参数一一对应）
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=str, default=None, choices=[None, "small", "medium", "large", "xl", "2.7B"])
    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--d_ff", type=int, default=None)
    ap.add_argument("--num_layers", type=int, default=None)
    ap.add_argument("--num_heads", type=int, default=None)
    ap.add_argument("--vocab_size", type=int, default=10000)  # 作业推荐 vocab_size=10000
    ap.add_argument("--context_length", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--mode", type=str, default="fwd_bwd", choices=["fwd", "fwd_bwd"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", type=str, default="float32")
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    # 若指定了 size，则优先采用预设表中的超参数；否则使用显式传入或默认值
    if args.size:
        p = SIZE_PRESETS[args.size]
        d_model = args.d_model if args.d_model is not None else p["d_model"]
        d_ff = args.d_ff if args.d_ff is not None else p["d_ff"]
        num_layers = args.num_layers if args.num_layers is not None else p["num_layers"]
        num_heads = args.num_heads if args.num_heads is not None else p["num_heads"]
    else:
        d_model = args.d_model or 768
        d_ff = args.d_ff or 3072
        num_layers = args.num_layers or 12
        num_heads = args.num_heads or 12

    # 固定随机数种子，减少一次运行内的波动
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设备选择（cuda/mps/cpu），并在 CUDA/MPS 不可用时报错
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 模型构建与数据类型设定
    dtype = getattr(torch, args.dtype)
    model = build_model(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        device=device,
        dtype=dtype,
    )
    model.to(device=device, dtype=dtype)
    model.train(mode=(args.mode == "fwd_bwd"))  # 仅前向时可关闭训练模式

    # 随机生成一个 batch，形状 [batch_size, context_length]
    x, y = make_batch(args.batch_size, args.context_length, args.vocab_size, device, dtype)

    # 预热若干步，随后计时 steps 次
    _ = time_steps(model, x, y, args.warmup, args.mode, device, tag="warmup")
    times = time_steps(model, x, y, args.steps, args.mode, device, tag="measure")
    mean = float(times.mean())
    std = float(times.std(ddof=1)) if times.size > 1 else 0.0

    # 打印结果，并可选写入 JSON 文件
    mode_name = "forward" if args.mode == "fwd" else "forward+backward"
    print(f"size={args.size or 'custom'} d_model={d_model} d_ff={d_ff} layers={num_layers} heads={num_heads}")
    print(f"batch={args.batch_size} seq={args.context_length} vocab={args.vocab_size} device={device.type} dtype={args.dtype}")
    print(f"{mode_name}: steps={args.steps} warmup={args.warmup} mean={mean:.6f}s std={std:.6f}s")
    if args.out_json:
        import json, os
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        payload = {
            "size": args.size or "custom",
            "d_model": d_model,
            "d_ff": d_ff,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "vocab_size": args.vocab_size,
            "device": device.type,
            "dtype": args.dtype,
            "mode": mode_name,
            "warmup": args.warmup,
            "steps": args.steps,
            "mean_s": mean,
            "std_s": std,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f)


if __name__ == "__main__":
    main()
