from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Tuple
import json
import re

from cs336_basics.bpe import _compile_gpt2_pretok_pattern


class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] | None = None,
    ) -> None:
        self.id_to_bytes: Dict[int, bytes] = dict(vocab)
        self.bytes_to_id: Dict[bytes, int] = {v: k for k, v in self.id_to_bytes.items()}
        self.pair_ranks: Dict[Tuple[bytes, bytes], int] = {
            (a, b): i for i, (a, b) in enumerate(merges)
        }
        self._pattern = _compile_gpt2_pretok_pattern()
        # 如果 token 不在已有的词汇表中，就把他添加到维护的bytes_to_id、id_to_bytes、_special_tokens、_special_regex 中
        self._special_tokens: List[str] = list(special_tokens or [])
        if self._special_tokens:
            for tok in self._special_tokens:
                b = tok.encode("utf-8")
                if b not in self.bytes_to_id:
                    new_id = len(self.id_to_bytes)
                    self.id_to_bytes[new_id] = b
                    self.bytes_to_id[b] = new_id
            ordered = sorted(self._special_tokens, key=len, reverse=True)
            self._special_regex = re.compile("|".join(re.escape(t) for t in ordered))
        else:
            self._special_regex = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: List[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab: Dict[int, bytes] = {int(k): bytes(v) for k, v in raw_vocab.items()}
        with open(merges_filepath, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)
        merges: List[Tuple[bytes, bytes]] = [(bytes(a), bytes(b)) for a, b in raw_merges]
        return cls(vocab, merges, special_tokens=special_tokens)

    def _split_with_special(self, text: str) -> List[Tuple[str, str]]:
        if not self._special_regex:
            return [("text", text)]
        out: List[Tuple[str, str]] = []
        pos = 0
        for m in self._special_regex.finditer(text):
            if m.start() > pos:
                out.append(("text", text[pos : m.start()]))
            out.append(("special", m.group(0)))
            pos = m.end()
        if pos < len(text):
            out.append(("text", text[pos:]))
        return out

    def _bpe_encode_bytes(self, data: bytes) -> List[int]:
        if not data:
            return []
        # data 转为token先，每个字节作为一个token
        tokens: List[bytes] = [bytes([b]) for b in data]
        if len(tokens) == 1:
            return [self.bytes_to_id[tokens[0]]]
        while True:
            # 收集当前序列中所有“可合并的相邻对”（存在于训练得到的 pair_ranks）
            pairs = []
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                # 查找该相邻对在训练中的合并次序（rank 越小代表越早被合并）
                rank = self.pair_ranks.get(pair)
                if rank is not None:
                    # 记录 (rank, 位置 i, 相邻对) 作为候选
                    pairs.append((rank, i, pair))
            # 没有任何候选相邻对时，结束循环
            if not pairs:
                break
            # 选择 rank 最小的候选进行合并；若 rank 并列，按遍历顺序取最左的那个
            _, idx, pair = min(pairs, key=lambda x: x[0])
            # 把该位置的两个相邻 token 字节拼接成一个更长的字节片段
            merged = pair[0] + pair[1]
            # 用合并后的片段替换原来的两个位置，序列长度减少 1
            tokens = tokens[:idx] + [merged] + tokens[idx + 2 :]
        return [self.bytes_to_id[t] for t in tokens]

    def encode(self, text: str) -> List[int]:
        """将输入字符串编码为 token id 列表
        流程：
        1) 先按特殊符号与普通文本分段（_split_with_special）
        2) 对特殊符号直接查表得到 id
        3) 对普通文本用 GPT‑2 预分词模式（_pattern）逐片匹配，再用字节级 BPE 合并（_bpe_encode_bytes）
        """
        ids: List[int] = []
        for kind, segment in self._split_with_special(text):
            if kind == "special":
                tok_bytes = segment.encode("utf-8")
                ids.append(self.bytes_to_id[tok_bytes])
            else:
                for m in self._pattern.finditer(segment):
                    s = m.group(0)
                    if not s:
                        continue
                    # 将该片段按字节编码，再调用字节级 BPE 做相邻字节合并，最终映射到词表 id
                    ids.extend(self._bpe_encode_bytes(s.encode("utf-8")))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for _id in self.encode(chunk):
                yield _id

    def decode(self, ids: List[int]) -> str:
        data = b"".join(self.id_to_bytes[i] for i in ids)
        return data.decode("utf-8", errors="ignore")
