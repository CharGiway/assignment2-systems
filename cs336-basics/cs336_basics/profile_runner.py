import argparse
import numpy as np
import torch
from cs336_basics.benchmark import SIZE_PRESETS, build_model, make_batch
from cs336_basics.nn.cross_entropy import cross_entropy
from cs336_basics.optim.adamw import AdamW


def nvtx_push(name: str):
    if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
        torch.cuda.nvtx.range_push(name)


def nvtx_pop():
    if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
        torch.cuda.nvtx.range_pop()


def get_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=str, default="small", choices=["small", "medium", "large", "xl", "2.7B"])
    ap.add_argument("--vocab_size", type=int, default=10000)
    ap.add_argument("--context_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float32")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", type=str, default="fwd", choices=["fwd", "fwd_bwd", "train"])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    args = ap.parse_args()

    p = SIZE_PRESETS[args.size]
    d_model, d_ff, num_layers, num_heads = p["d_model"], p["d_ff"], p["num_layers"], p["num_heads"]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
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

    x, y = make_batch(args.batch_size, args.context_length, args.vocab_size, device, dtype)

    optimizer = None
    if args.mode == "train":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model.train(True)
    else:
        model.train(args.mode == "fwd_bwd")

    # NVTX ranges for nsys capture
    nvtx_push("run")
    # forward
    nvtx_push("forward")
    if args.mode == "fwd":
        with torch.no_grad():
            logits = model(x)
            loss = cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
    else:
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
    if device.type == "cuda":
        torch.cuda.synchronize()
    nvtx_pop()  # forward

    # backward
    if args.mode in ("fwd_bwd", "train"):
        nvtx_push("backward")
        model.zero_grad(set_to_none=True)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        nvtx_pop()  # backward

    # optimizer step
    if args.mode == "train":
        nvtx_push("optimizer_step")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        nvtx_pop()  # optimizer_step

    nvtx_pop()  # run

    # Print a brief summary for logs
    mode_name = {"fwd": "forward", "fwd_bwd": "forward+backward", "train": "train_step"}[args.mode]
    print(f"nsys-profile-ready size={args.size} seq={args.context_length} batch={args.batch_size} mode={mode_name} device={device.type}")


if __name__ == "__main__":
    main()

