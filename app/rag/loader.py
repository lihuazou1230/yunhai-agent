"""解析分流（规划 10.2）：按扩展名决定怎么读。

- pdf：pypdf 逐页抽文本，**一页一条条目**（引用能指到页码）
- md / txt：整篇一条条目，交给分块器递归切
- jsonl：一行一条**天然成块**（对话/问答语料最常见的格式）

这一层只负责「把文件变成条目」，不做分块、不碰向量——
解析器可替换，分块策略可调参，两边互不牵连。
"""

from __future__ import annotations

import io
import json
from typing import Any

from app.config import ALLOWED_EXTENSIONS, SOURCE_TYPES
from app.errors import EmptyDocument, UnsupportedFileType
from app.rag.types import LoadedItem


def source_type_for(filename: str) -> str:
    """扩展名 -> 来源类型；不在白名单里直接拒绝（规划 10.1 类型白名单）。"""
    lowered = filename.lower()
    for ext in ALLOWED_EXTENSIONS:
        if lowered.endswith(ext):
            return SOURCE_TYPES[ext]
    allowed = "、".join(sorted(e.lstrip(".") for e in ALLOWED_EXTENSIONS))
    raise UnsupportedFileType(f"不支持的文件类型，仅支持：{allowed}")


def decode_text(data: bytes) -> str:
    """中文语料常见的两种编码来回试；都失败就带替换字符硬解（不因一个坏字节丢整篇）。"""
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_items(filename: str, data: bytes, jsonl_fields: list[str] | None = None) -> list[LoadedItem]:
    """把文件字节解析成条目列表。"""
    source_type = source_type_for(filename)
    if source_type == "pdf":
        items = _load_pdf(data)
    elif source_type == "jsonl":
        items = _load_jsonl(data, jsonl_fields or [])
    else:
        text = decode_text(data)
        items = [LoadedItem(item_id="raw", text=text)] if text.strip() else []

    if not items:
        raise EmptyDocument("文件里没有可入库的文本（扫描版 PDF 需要先做 OCR）")
    # 全是空白也当空文档
    if all(not item.text.strip() for item in items):
        raise EmptyDocument("文件里没有可入库的文本（扫描版 PDF 需要先做 OCR）")
    return items


def _load_pdf(data: bytes) -> list[LoadedItem]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 依赖缺失时的明确提示
        raise EmptyDocument("缺少 pypdf 依赖，无法解析 PDF") from exc

    reader = PdfReader(io.BytesIO(data))
    items: list[LoadedItem] = []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - 单页损坏不该让整篇失败
            text = ""
        if text.strip():
            items.append(
                LoadedItem(item_id=f"p{page_no}", text=text, meta={"page": page_no})
            )
    return items


def _load_jsonl(data: bytes, fields: list[str]) -> list[LoadedItem]:
    """一行一条。

    有 `id` 字段就用它做 item_id（重跑解析仍稳定命中同一条），否则退回行号。
    """
    text = decode_text(data)
    items: list[LoadedItem] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        item_id = f"l{line_no}"
        content = line
        try:
            record: Any = json.loads(line)
        except json.JSONDecodeError:
            record = None  # 不是 JSON 就按纯文本行收下，不丢数据
        if isinstance(record, dict):
            if isinstance(record.get("id"), (str, int)):
                item_id = str(record["id"])
            picked = [str(record[f]) for f in fields if record.get(f) not in (None, "")]
            content = "\n".join(picked) if picked else json.dumps(record, ensure_ascii=False)
        elif record is not None:
            content = json.dumps(record, ensure_ascii=False)
        if content.strip():
            items.append(LoadedItem(item_id=item_id, text=content))
    return items
