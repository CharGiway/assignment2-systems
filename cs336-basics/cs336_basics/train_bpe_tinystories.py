import argparse
import json
import os
import time
from pathlib import Path
import resource
import sys
import cProfile
import pstats
from typing import Dict, Tuple, List

from cs336_basics.bpe import train_bpe


def _serialize_vocab_merges(
    out_dir: Path,
    vocab: Dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir / "tinystories_bpe_vocab.json"
    merges_path = out_dir / "tinystories_bpe_merges.json"
    vocab_json = {int(k): list(v) for k, v in vocab.items()}
    merges_json = [[list(a), list(b)] for (a, b) in merges]
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False)
    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(merges_json, f, ensure_ascii=False)
    return vocab_path, merges_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default=str(Path("data") / "TinyStoriesV2-GPT4-train.txt"),
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path("artifacts")),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=(os.cpu_count() or 1),
    )
    args = parser.parse_args()

    input_path = args.input_path
    vocab_size = args.vocab_size
    out_dir = Path(args.out_dir)
    special_tokens = ["<|endoftext|>"]

    start_time = time.perf_counter()
    rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        n_workers=int(args.n_workers),
    )

    if args.profile:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats_file = out_dir / "tinystories_bpe_profile.txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(stats_file, "w") as f:
            stats.stream = f
            stats.print_stats(30)

    elapsed_sec = time.perf_counter() - start_time
    rss_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_kb = max(rss_before_kb, rss_after_kb)
    if sys.platform == "darwin":
        peak_rss_gb = peak_rss_kb / (1024 ** 3)
    else:
        peak_rss_gb = peak_rss_kb / (1024 ** 2)

    vocab_path, merges_path = _serialize_vocab_merges(out_dir, vocab, merges)

    longest_token_id = max(vocab.keys(), key=lambda k: len(vocab[k]))
    longest_token_bytes = vocab[longest_token_id]
    longest_len = len(longest_token_bytes)
    try:
        longest_token_preview = longest_token_bytes.decode("utf-8", errors="ignore")
    except Exception:
        longest_token_preview = ""

    summary = {
        "elapsed_seconds": elapsed_sec,
        "peak_rss_gb": peak_rss_gb,
        "vocab_path": str(vocab_path),
        "merges_path": str(merges_path),
        "longest_token_id": int(longest_token_id),
        "longest_token_length_bytes": longest_len,
        "longest_token_preview_utf8": longest_token_preview[:80],
        "profile_path": str(out_dir / "tinystories_bpe_profile.txt") if args.profile else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
