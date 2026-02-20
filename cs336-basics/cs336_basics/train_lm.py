"""Transformer 语言模型训练脚本

负责：
- 加载编码好的训练/验证 token 数据
- 构建 `TransformerLM` 与 `AdamW` 优化器
- 使用线性 warmup → 余弦退火的学习率调度进行训练
- 记录训练/验证日志，周期性保存检查点，支持断点续训
"""
import argparse
import os
import math
import time
import numpy as np
import torch
from cs336_basics.nn.transformer_lm import TransformerLM
from cs336_basics.optim.adamw import AdamW
from cs336_basics.optim.lr_schedule import get_lr_cosine_schedule
from cs336_basics.optim.grad_clip import clip_gradients
from cs336_basics.serialization import save_checkpoint, load_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.nn.cross_entropy import cross_entropy
from cs336_basics.exp_logger import ExperimentLogger


def evaluate(model: torch.nn.Module, dataset: np.ndarray, batch_size: int, context_length: int, device: str, iters: int) -> float:
    """在验证集上评估若干次，返回平均 loss"""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(iters):
            x, y = get_batch(dataset, batch_size, context_length, device)
            logits = model(x)
            loss = cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
            losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / len(losses)) if losses else math.nan


def main():
    """命令行入口：解析参数、构建数据与模型、进入训练循环"""
    p = argparse.ArgumentParser()
    p.add_argument("--train_tokens", type=str, required=True)
    p.add_argument("--valid_tokens", type=str, required=True)
    p.add_argument("--tokens_dtype", type=str, default="uint16")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument("--context_length", type=int, default=1024)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--resume_path", type=str, default=None)
    p.add_argument("--log_path", type=str, default="artifacts/exp_log.jsonl")
    p.add_argument("--no_rmsnorm", action="store_true", default=False)
    p.add_argument("--norm_style", type=str, default="pre")
    p.add_argument("--no_pos_emb", action="store_true", default=False)
    p.add_argument("--ffn_style", type=str, default="swiglu")
    p.add_argument("--ffn_match_params", action="store_true", default=False)
    p.add_argument("--save_best_path", type=str, default=None)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--max_lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=1000)
    p.add_argument("--cosine_cycle_iters", type=int, default=100000)
    p.add_argument("--dropout_p", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    start_time = time.time()
    block_start_time = start_time
    dtype = getattr(np, args.tokens_dtype)
    if args.train_tokens.endswith(".npy"):
        train_ds = np.load(args.train_tokens, mmap_mode="r")
    else:
        train_ds = np.memmap(args.train_tokens, dtype=dtype, mode="r")
    if args.valid_tokens.endswith(".npy"):
        valid_ds = np.load(args.valid_tokens, mmap_mode="r")
    else:
        valid_ds = np.memmap(args.valid_tokens, dtype=dtype, mode="r")

    d_ff = args.d_ff if args.d_ff is not None else 4 * args.d_model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
        use_rope=(not args.no_pos_emb),
        use_rmsnorm=(not args.no_rmsnorm),
        norm_style=str(args.norm_style),
        ffn_style=str(args.ffn_style),
        ffn_match_params=bool(args.ffn_match_params),
        dropout_p=float(args.dropout_p),
        device=torch.device(args.device),
        dtype=torch.float32,
    )
    model.to(args.device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_it = 0
    if args.resume_path is not None and os.path.exists(args.resume_path):
        start_it = load_checkpoint(args.resume_path, model, optimizer)
        model.to(args.device)

    logger = ExperimentLogger(args.log_path) if args.log_path else None
    best_val = float("inf")
    no_improve = 0
    for it in range(start_it, args.max_steps):
        lr = get_lr_cosine_schedule(it, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_ds, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        clip_gradients(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        if (it + 1) % args.log_every == 0:
            print(f"iter={it+1} lr={lr:.6g} train_loss={float(loss.item()):.6g}")
            if logger:
                logger.log(it + 1, lr=lr, train_loss=float(loss.item()))
        if (it + 1) % args.eval_every == 0:
            val_loss = evaluate(model, valid_ds, args.batch_size, args.context_length, args.device, args.eval_iters)
            print(f"iter={it+1} val_loss={val_loss:.6g}")
            if logger:
                logger.log(it + 1, val_loss=val_loss)
            if val_loss + args.min_delta < best_val:
                best_val = float(val_loss)
                no_improve = 0
                if args.save_best_path:
                    save_checkpoint(model, optimizer, it + 1, args.save_best_path)
            else:
                no_improve += 1
                if args.patience > 0 and no_improve >= args.patience:
                    break
        if args.checkpoint_path and (it + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, it + 1, args.checkpoint_path)
        if (it + 1) % 100 == 0:
            elapsed_100 = time.time() - block_start_time
            it_per_sec = 100.0 / elapsed_100 if elapsed_100 > 0 else float("inf")
            tokens_per_sec = it_per_sec * args.batch_size * args.context_length
            print(f"iter={it+1} elapsed_100={elapsed_100:.3f}s it_per_sec={it_per_sec:.3f} tokens_per_sec={tokens_per_sec:.1f}")
            block_start_time = time.time()
    if logger:
        logger.close()
    total_elapsed = time.time() - start_time
    completed_steps = args.max_steps - start_it
    avg_it_per_sec = completed_steps / total_elapsed if total_elapsed > 0 else float("inf")
    avg_tokens_per_sec = avg_it_per_sec * args.batch_size * args.context_length
    print(f"final_elapsed_sec={total_elapsed:.3f} avg_it_per_sec={avg_it_per_sec:.3f} avg_tokens_per_sec={avg_tokens_per_sec:.1f}")


if __name__ == "__main__":
    main()
