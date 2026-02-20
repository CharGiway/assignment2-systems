import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch

from cs336_basics.tokenizer import Tokenizer


def _load_vocab_merges(vocab_path: Path, merges_path: Path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    with open(merges_path, "r", encoding="utf-8") as f:
        merges_json = json.load(f)
    vocab: dict[int, bytes] = {int(k): bytes(v) for k, v in vocab_json.items()}
    merges: list[tuple[bytes, bytes]] = [(bytes(a), bytes(b)) for a, b in merges_json]
    return vocab, merges


def _detect_device(device: str) -> str:
    if device == "auto":
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    return device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_path", type=str, required=True)
    ap.add_argument("--vocab_path", type=str, required=True)
    ap.add_argument("--merges_path", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--limit_tokens", type=int, default=40960000)  # 默认低资源目标
    ap.add_argument("--dtype", type=str, default="uint16")
    ap.add_argument("--device", type=str, default="auto")  # 用于 Tokenizer 的可能设备选择（目前未用到）
    args = ap.parse_args()

    text_path = Path(args.text_path)
    vocab_path = Path(args.vocab_path)
    merges_path = Path(args.merges_path)
    out_path = Path(args.out_path)
    limit = int(args.limit_tokens)
    dtype = getattr(np, args.dtype)
    device = _detect_device(args.device)

    vocab, merges = _load_vocab_merges(vocab_path, merges_path)
    tok = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    count = 0
    if limit > 0:
        tmp_mmap_path = out_path.with_suffix(out_path.suffix + ".tmp.mmap")
        arr = np.memmap(tmp_mmap_path, mode="w+", dtype=dtype, shape=(limit,))
        with open(text_path, "r", encoding="utf-8") as f:
            for tid in tok.encode_iterable(f):
                if count >= limit:
                    break
                arr[count] = tid
                count += 1
        np.save(out_path, np.asarray(arr[:count], dtype=dtype))
        try:
            os.remove(tmp_mmap_path)
        except Exception:
            pass
    else:
        chunks: list[np.ndarray] = []
        chunk_size = 1_000_000
        buf = np.empty((chunk_size,), dtype=dtype)
        i = 0
        with open(text_path, "r", encoding="utf-8") as f:
            for tid in tok.encode_iterable(f):
                buf[i] = tid
                i += 1
                count += 1
                if i == chunk_size:
                    chunks.append(buf.copy())
                    i = 0
        if i > 0:
            chunks.append(buf[:i].copy())
        out_arr = np.concatenate(chunks) if chunks else np.empty((0,), dtype=dtype)
        np.save(out_path, out_arr)
    print(json.dumps({"out_path": str(out_path), "count": int(count), "device": device}))


if __name__ == "__main__":
    main()
