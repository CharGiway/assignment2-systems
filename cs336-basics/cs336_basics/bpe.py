from __future__ import annotations

import io
from collections import Counter, defaultdict
import json
from typing import Dict, Iterable, List, Tuple
import time
from datetime import datetime, timezone

import regex as re


def _compile_gpt2_pretok_pattern() -> re.Pattern:
    """
    构造 GPT-2 风格的预分词正则表达式。

    规则说明（UTF-8 字节级）：
    - 英文常见缩写后缀：'s, 't, 're, 've, 'm, 'll, 'd
    - 以可选空格开头的字母块或数字块：` ?\p{L}+`、` ?\p{N}+`
    - 其它非空白的符号块：` ?[^ \p{L}\p{N}\s]+`
    - 仅由空白组成且后面不跟非空白的结尾空白：`\s+(?!\S)`（用于捕获结尾换行/空格）
    """
    pat_str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    return re.compile(pat_str)


import cs336_basics.logutil as logutil


def _split_on_special(text: str, special_tokens: List[str]) -> Iterable[str]:
    """
    在特殊标记处切分文本（保持特殊标记原样、避免与邻近文本合并）。
    仅返回普通文本片段，特殊标记不计入频次统计与合并。
    """
    # 如果没有特殊标记，直接返回文本
    if not special_tokens:
        yield text
        return
    pat = re.compile("|".join(re.escape(t) for t in special_tokens))
    pos = 0
    for m in pat.finditer(text):
        if m.start() > pos:
            yield text[pos : m.start()]
        pos = m.end()
    if pos < len(text):
        yield text[pos:]


def _pretokenize(text: str, special_tokens: List[str]) -> Counter[bytes]:
    """
    预分词：
    - 先按特殊标记切分，保证特殊标记不参与 BPE 合并；
    - 对普通片段使用 GPT-2 正则进行分块，并以 UTF-8 编码成字节；
    - 统计每个片段的出现次数，作为初始“词”（字节序列）。
    """
    # 先取出 Pattern，避免重复编译
    pattern = _compile_gpt2_pretok_pattern()
    counts: Counter[bytes] = Counter()
    for segment in _split_on_special(text, special_tokens):
        for m in pattern.finditer(segment):
            s = m.group(0)
            if not s:
                continue
            counts[s.encode("utf-8")] += 1
    return counts

def _pretokenize_worker(segment: str) -> Counter[bytes]:
    pattern = _compile_gpt2_pretok_pattern()
    c: Counter[bytes] = Counter()
    for m in pattern.finditer(segment):
        s = m.group(0)
        if not s:
            continue
        c[s.encode("utf-8")] += 1
    return c

def _pretokenize_parallel(text: str, special_tokens: List[str], n_workers: int) -> Counter[bytes]:
    if n_workers <= 0:
        return _pretokenize(text, special_tokens)
    segments = list(_split_on_special(text, special_tokens))
    from concurrent.futures import ProcessPoolExecutor
    out: Counter[bytes] = Counter()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for c in ex.map(_pretokenize_worker, segments):
            out.update(c)
    return out

# 合并 word 中的 a,b 为 new_id，比如(aId, bIb, cId, aId, bId) -> (newId, cId, newId)
def _merge_word(word: Tuple[int, ...], a: int, b: int, new_id: int) -> Tuple[int, ...]:
    """
    在给定“词”中，将相邻对 (a,b) 替换为新 ID，返回合并后的“词”。
    """
    out: List[int] = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def _pairs(word: Tuple[int, ...]) -> List[Tuple[int, int]]:
    if len(word) < 2:
        return []
    return list(zip(word, word[1:]))

# train_bpe 从input_date中训练获取BPE模型，也就是 词汇ID -> bytes数组
def train_bpe(
    input_path: str | io.BytesIO,
    vocab_size: int,
    special_tokens: List[str],
    n_workers: int = 0,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    assert vocab_size > 0
    base_vocab_count = 256 + len(special_tokens)
    # 计算实际需要执行的合并次数，不能超过 vocab_size - base_vocab_count
    num_merges = max(0, vocab_size - base_vocab_count)
    
    text: str
    if isinstance(input_path, io.BytesIO):
        text = input_path.getvalue().decode("utf-8", errors="ignore")
    else:
        with open(input_path, "rb") as f:
            text = f.read().decode("utf-8", errors="ignore")

    logutil.info_kvs(
        "event",  "read_input",
        "input_path",  "BytesIO" if isinstance(input_path, io.BytesIO) else str(input_path),
        "text_len",   len(text),
        "vocab_size",  vocab_size,
        "specials",  len(special_tokens),
        "special_tokens",  json.dumps(special_tokens, ensure_ascii=False),
    )

    # 按照special_token 和默认pattern 切分文本，得到形如 {['h','e','l','l','o']->2次, ['w','o','r','l','d']->1次} 类似这种
    counts = _pretokenize_parallel(text, special_tokens, n_workers)
    sample_counts = dict(counts.most_common(100))
    logutil.info_kvs(
        "event", "pretokenize_summary",
        "unique_pieces", len(counts),
        "counts_sample", sample_counts,
    )
    # ⚠️⚠️⚠️ 这里最终转为 map，形如 ('a','b','c','e') -> 8
    words: Dict[Tuple[int, ...], int] = {}
    for piece_bytes, freq in counts.items():
        words[tuple(piece_bytes)] = freq

    # 这里产生一个简单的id 到 bytes 的转换
    id_to_bytes: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for t in special_tokens:
        id_to_bytes[len(id_to_bytes)] = t.encode("utf-8")

    # 简单打印一下这个 id_to_bytes 内容，检查是否符合预期
    logutil.info_kvs(
        "event", "id_to_bytes",
        "id_to_bytes", id_to_bytes,
    )

    # 返回结果，需要合并的内容
    merges: List[Tuple[bytes, bytes]] = []

    # ⚠️⚠️⚠️ pair_counts 也是map，形如 ('a','b')->8, ('b','c')->10 类似这种
    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    # ⚠️⚠️⚠️ pair_index 也是map，形如 ('a','b')->{('a','b','c','e'),('a','b','d','e')} 类似这种
    pair_index: Dict[Tuple[int, int], set[Tuple[int, ...]]] = defaultdict(set)
    # words 拿到的是一个 dict，key 是字节序列，就是单词，value 是出现的频次
    for w, f in words.items():
        for p in _pairs(w):
            pair_counts[p] += f
            pair_index[p].add(w)

    # 注意一下 words、id、bytes、pairs 之间的关系就行
    
    step_idx = 0
    for _ in range(num_merges):
        _step_start = time.perf_counter()
        # 从 本次的 pair_counts 里面，选择出现频次最高的 pair，
        # 然后更新 id_to_bytes 和 merges 数组
        if not pair_counts:
            break
        # 先比较出现的频次，再比较第一个 & 第二个的字典序
        best_pair = max(
            pair_counts.items(), # 形如 (('a','b'),8)
            key=lambda kv: (kv[1], id_to_bytes[kv[0][0]], id_to_bytes[kv[0][1]]),
        )[0]
        a, b = best_pair
        new_bytes = id_to_bytes[a] + id_to_bytes[b]
        new_id = len(id_to_bytes)
        id_to_bytes[new_id] = new_bytes
        merges.append((id_to_bytes[a], id_to_bytes[b]))

        # 更新维护的 pair_index 和 pair_counts
        impacted = pair_index.get(best_pair)
        if not impacted:
            logutil.info_kvs(
                "event", "bpe_merge_step",
                "step", step_idx,
                "num_merges", num_merges,
                "pair", [int(a), int(b)],
                "a_bytes", id_to_bytes[a],
                "b_bytes", id_to_bytes[b],
                "new_id", new_id,
                "new_len", len(new_bytes),
                "impacted", 0,
                "pair_counts_size", len(pair_counts),
                "words_size", len(words),
                "elapsed_ms", int((time.perf_counter() - _step_start) * 1000),
                "status", "skipped",
            )
            pair_index.pop(best_pair, None)
            step_idx += 1
            continue
        
        # 临时变量，用于存储合并后的单词和对应的频次
        new_words_acc: Dict[Tuple[int, ...], int] = {}
        # 临时变量，用于存储需要清理的 pair_counts 和 pair_index
        remove_counts_acc: Dict[Tuple[int, int], int] = defaultdict(int)
        remove_index_acc: Dict[Tuple[int, int], set[Tuple[int, ...]]] = defaultdict(set)
        # 临时变量，用于存储需要添加的 pair_counts 和 pair_index    
        add_counts_acc: Dict[Tuple[int, int], int] = defaultdict(int)
        add_index_acc: Dict[Tuple[int, int], set[Tuple[int, ...]]] = defaultdict(set)
        
        for word in list(impacted):
            freq = words.get(word)
            if freq is None:
                continue
            
            # 计算待删除的 pair_counts 和 pair_index
            rc, ri = _compute_remove_deltas(word, freq)
            for pair, dec in rc.items():
                remove_counts_acc[pair] += dec
            for pair, holders in ri.items():
                remove_index_acc[pair].update(holders)
                
            # 合并 word 中的 a,b 为 new_id，比如(aId, bIb, cId, aId, bId) -> (newId, cId, newId)
            merged = _merge_word(word, a, b, new_id)
            
            # 计算待添加的 pair_counts 和 pair_index
            ac, ai = _compute_add_deltas(merged, freq)
            for pair, inc in ac.items():
                add_counts_acc[pair] += inc
            for pair, holders in ai.items():
                add_index_acc[pair].update(holders)
            words.pop(word, None)
            new_words_acc[merged] = new_words_acc.get(merged, 0) + freq

        # 实际删除 pair_counts 中的内容
        for pair, dec in remove_counts_acc.items():
            current = pair_counts.get(pair)
            if current is None:
                continue
            new_val = current - dec
            if new_val <= 0:
                pair_counts.pop(pair, None)
            else:
                pair_counts[pair] = new_val
        # 实际删除 pair_index 中的内容
        for pair, holders_to_remove in remove_index_acc.items():
            holders = pair_index.get(pair)
            if holders is not None:
                holders.difference_update(holders_to_remove)
                if not holders:
                    pair_index.pop(pair, None)
        # 实际添加 pair_counts 中的内容
        for pair, inc in add_counts_acc.items():
            pair_counts[pair] = pair_counts.get(pair, 0) + inc
        # 实际添加 pair_index 中的内容  
        for pair, holders_to_add in add_index_acc.items():
            holders = pair_index.get(pair)
            if holders is None:
                pair_index[pair] = set(holders_to_add)
            else:
                holders.update(holders_to_add)

        # 实际更新 words 中的内容
        for mw, fsum in new_words_acc.items():
            words[mw] = words.get(mw, 0) + fsum
        pair_index.pop(best_pair, None)
        logutil.info_kvs(
            "event", "bpe_merge_step",
            "step", step_idx,
            "num_merges", num_merges,
            "pair", [int(a), int(b)],
            "a_bytes", id_to_bytes[a],
            "b_bytes", id_to_bytes[b],
            "new_id", new_id,
            "new_len", len(new_bytes),
            "impacted", len(impacted),
            "pair_counts_size", len(pair_counts),
            "words_size", len(words),
            "elapsed_ms", int((time.perf_counter() - _step_start) * 1000),
            "status", "merged",
        )
        step_idx += 1

    vocab: Dict[int, bytes] = {i: id_to_bytes[i] for i in range(len(id_to_bytes))}
    return vocab, merges

# 把这些 word 相关的 pair_count 都减去 freq
def _compute_remove_deltas(
    word_tokens: Tuple[int, ...],
    freq: int,
) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], set[Tuple[int, ...]]]]:
    counts_delta: Dict[Tuple[int, int], int] = defaultdict(int)
    index_delta: Dict[Tuple[int, int], set[Tuple[int, ...]]] = defaultdict(set)
    for pair in _pairs(word_tokens):
        counts_delta[pair] += freq
        index_delta[pair].add(word_tokens)
    return counts_delta, index_delta

def _compute_add_deltas(
    merged_tokens: Tuple[int, ...],
    freq: int,
) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], set[Tuple[int, ...]]]]:
    counts_delta: Dict[Tuple[int, int], int] = defaultdict(int)
    index_delta: Dict[Tuple[int, int], set[Tuple[int, ...]]] = defaultdict(set)
    for pair in _pairs(merged_tokens):
        counts_delta[pair] += freq
        index_delta[pair].add(merged_tokens)
    return counts_delta, index_delta
