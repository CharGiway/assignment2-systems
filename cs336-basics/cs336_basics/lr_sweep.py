import argparse
import os
import time
import json
import numpy as np
import torch
from cs336_basics.nn.transformer_lm import TransformerLM
from cs336_basics.optim.adamw import AdamW
from cs336_basics.optim.lr_schedule import get_lr_cosine_schedule
from cs336_basics.optim.grad_clip import clip_gradients
from cs336_basics.serialization import save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.nn.cross_entropy import cross_entropy
from cs336_basics.exp_logger import ExperimentLogger


def train_once(
    *,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    max_steps: int,
    cosine_cycle_iters: int,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int | None,
    rope_theta: float,
    device: str,
    train_ds: np.ndarray,
    valid_ds: np.ndarray,
    grad_clip_norm: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    log_dir: str,
    eval_every: int,
    eval_iters: int,
    compile_backend: str | None,
) -> dict:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.jsonl")
    ckpt_path = os.path.join(log_dir, "last.ckpt")
    logger = ExperimentLogger(log_path)

    d_ff_val = d_ff if d_ff is not None else 4 * d_model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff_val,
        rope_theta=rope_theta,
        device=torch.device(device),
        dtype=torch.float32,
    )
    model.to(device)
    if compile_backend is not None:
        model = torch.compile(model, backend=compile_backend)
    optimizer = AdamW(model.parameters(), lr=max_lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)

    start_time = time.time()
    last_val = None
    diverged = False
    for it in range(max_steps):
        lr = get_lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_ds, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        if not torch.isfinite(loss):
            diverged = True
            break
        optimizer.zero_grad()
        loss.backward()
        clip_gradients(model.parameters(), grad_clip_norm)
        optimizer.step()
        logger.log(it + 1, lr=lr, train_loss=float(loss.item()))
        if (it + 1) % eval_every == 0:
            # lightweight validation
            model.eval()
            with torch.no_grad():
                v_losses = []
                for _ in range(eval_iters):
                    vx, vy = get_batch(valid_ds, batch_size, context_length, device)
                    v_logits = model(vx)
                    v_loss = cross_entropy(v_logits.view(-1, v_logits.shape[-1]), vy.view(-1))
                    v_losses.append(float(v_loss.item()))
            model.train()
            last_val = float(sum(v_losses) / len(v_losses))
            logger.log(it + 1, val_loss=last_val)
    tokens = batch_size * context_length * (it + 1)
    save_checkpoint(model, optimizer, it + 1, ckpt_path)
    logger.close()
    return {
        "max_lr": max_lr,
        "min_lr": min_lr,
        "warmup_iters": warmup_iters,
        "max_steps": max_steps,
        "cosine_cycle_iters": cosine_cycle_iters,
        "final_step": it + 1,
        "final_train_tokens": tokens,
        "final_val_loss": last_val,
        "diverged": diverged,
        "log_dir": log_dir,
        "elapsed_sec": time.time() - start_time,
    }


def parse_lr_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tokens", type=str, required=True)
    ap.add_argument("--valid_tokens", type=str, required=True)
    ap.add_argument("--tokens_dtype", type=str, default="uint16")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--vocab_size", type=int, required=True)
    ap.add_argument("--context_length", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--num_layers", type=int, default=12)
    ap.add_argument("--num_heads", type=int, default=12)
    ap.add_argument("--d_ff", type=int, default=None)
    ap.add_argument("--rope_theta", type=float, default=10000.0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--warmup_iters", type=int, default=500)
    ap.add_argument("--cosine_cycle_iters", type=int, default=None)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    ap.add_argument("--lr_list", type=str, default="1e-4,2e-4,3e-4,5e-4,8e-4")
    ap.add_argument("--grad_clip_norm", type=float, default=1.0)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_iters", type=int, default=10)
    ap.add_argument("--out_dir", type=str, default="artifacts/lr_sweep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compile_backend", type=str, default=None, help='e.g., "aot_eager" for mps or "inductor" for cpu')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype = getattr(np, args.tokens_dtype)
    train_ds = np.load(args.train_tokens, mmap_mode="r") if args.train_tokens.endswith(".npy") else np.memmap(args.train_tokens, dtype=dtype, mode="r")
    valid_ds = np.load(args.valid_tokens, mmap_mode="r") if args.valid_tokens.endswith(".npy") else np.memmap(args.valid_tokens, dtype=dtype, mode="r")

    ts = time.strftime("%Y%m%d-%H%M%S")
    base_dir = os.path.join(args.out_dir, ts)
    os.makedirs(base_dir, exist_ok=True)
    cosine_cycle_iters = args.cosine_cycle_iters if args.cosine_cycle_iters is not None else args.max_steps
    results = []
    for max_lr in parse_lr_list(args.lr_list):
        run_dir = os.path.join(base_dir, f"lr_{max_lr}")
        res = train_once(
            max_lr=max_lr,
            min_lr=args.min_lr,
            warmup_iters=args.warmup_iters,
            max_steps=args.max_steps,
            cosine_cycle_iters=cosine_cycle_iters,
            batch_size=args.batch_size,
            context_length=args.context_length,
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            device=args.device,
            train_ds=train_ds,
            valid_ds=valid_ds,
            grad_clip_norm=args.grad_clip_norm,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.eps,
            weight_decay=args.weight_decay,
            log_dir=run_dir,
            eval_every=args.eval_every,
            eval_iters=args.eval_iters,
            compile_backend=args.compile_backend,
        )
        results.append(res)
    summary_path = os.path.join(base_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"runs": results}, f, ensure_ascii=False, indent=2)
    print(f"LR sweep finished. Summary at {summary_path}")


if __name__ == "__main__":
    main()
