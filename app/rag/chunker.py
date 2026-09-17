"""递归字符分块（规划 10.2）。

不引 langchain，自己写一遍：按「段落 → 换行 → 中文句末标点 → 逗号 → 空格 → 字符」
逐级降级切分，尽量在语义边界断开；超长片段最后才硬切。

中文分块的两个细节：
1. 分隔符要带中文句末标点（。！？；），否则整段中文会被当成一个"词"硬切；
2. 切分时**标点留在前一块末尾**（用 lookbehind 切），不然引用片段读起来是断句。
"""

from __future__ import annotations

import re

from app.rag.types import Chunk, LoadedItem

DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "…",
    ". ",
    "! ",
    "? ",
    "; ",
    "，",
    ", ",
    " ",
    "",
)

# 需要「标点跟随前文」的分隔符
_KEEP_TRAILING = frozenset({"\n\n", "\n", "。", "！", "？", "；", "…"})


def split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
) -> list[str]:
    """把一段文本切成若干块，块长尽量不超过 `chunk_size`。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正数")
    overlap = max(0, min(chunk_overlap, chunk_size // 2))
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= chunk_size:
        return [stripped]
    return _split(stripped, chunk_size, overlap, separators)


def _split(text: str, size: int, overlap: int, separators: tuple[str, ...]) -> list[str]:
    if len(text) <= size:
        return [text.strip()] if text.strip() else []

    separator = separators[0]
    rest = separators[1:]
    pieces = _split_by(text, separator) if separator else [text]

    if len(pieces) <= 1:
        # 这一级分隔符切不动：降级到下一级；都试完了就硬切
        if rest:
            return _split(text, size, overlap, rest)
        return _hard_split(text, size, overlap)

    chunks: list[str] = []
    buffer = ""
    for piece in pieces:
        if len(piece) > size:
            if buffer.strip():
                chunks.append(buffer.strip())
                buffer = ""
            chunks.extend(_split(piece, size, overlap, separators[1:] if rest else separators))
            continue
        if buffer and len(buffer) + len(piece) > size:
            chunks.append(buffer.strip())
            buffer = _tail(buffer, overlap)
            if buffer and len(buffer) + len(piece) > size:
                buffer = ""  # 重叠会把这一块撑过上限时，宁可不重叠
        buffer += piece

    if buffer.strip():
        chunks.append(buffer.strip())
    return [c for c in chunks if c]


def _split_by(text: str, separator: str) -> list[str]:
    if separator in _KEEP_TRAILING:
        parts = re.split(f"(?<={re.escape(separator)})", text)
    else:
        parts = text.split(separator)
    return [p for p in parts if p]


def _tail(buffer: str, overlap: int) -> str:
    """取上一块的尾部做重叠，顺带在句末标点处对齐（重叠区不要从半句话开始）。"""
    if overlap <= 0:
        return ""
    tail = buffer[-overlap:]
    match = re.search(r"[。！？；\n]", tail)
    if match and match.end() < len(tail):
        return tail[match.end() :]
    return tail


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    return [text[i : i + size].strip() for i in range(0, len(text), step) if text[i : i + size].strip()]


def build_chunks(
    items: list[LoadedItem],
    *,
    source: str,
    source_type: str,
    doc_id: str,
    uploaded_at: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """条目 -> 块。`chunk_index` 在**整篇文档**范围内连续，便于「第 N 块」定位。"""
    chunks: list[Chunk] = []
    index = 0
    for item in items:
        for piece in split_text(item.text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(
                    # chunk_id 由 doc_id + 全局序号决定：同一份文件重复入库会覆盖同一批 id
                    chunk_id=f"{doc_id}:{index}",
                    text=piece,
                    source=source,
                    source_type=source_type,
                    doc_id=doc_id,
                    item_id=item.item_id,
                    chunk_index=index,
                    uploaded_at=uploaded_at,
                    page=item.meta.get("page"),
                )
            )
            index += 1
    return chunks
