"""解析器测试：类型白名单、编码兜底、PDF 分页、JSONL 一行一条。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.errors import EmptyDocument, UnsupportedFileType
from app.rag.loader import decode_text, load_items, source_type_for
from app.rag.tokenize import tokenize


def test_source_type_whitelist():
    assert source_type_for("手册.PDF") == "pdf"
    assert source_type_for("笔记.md") == "md"
    assert source_type_for("数据.jsonl") == "jsonl"
    with pytest.raises(UnsupportedFileType):
        source_type_for("图片.png")


def test_decode_gbk_fallback():
    assert decode_text("中文内容".encode("gb18030")) == "中文内容"


def test_markdown_is_one_item():
    items = load_items("笔记.md", "# 标题\n正文内容".encode())
    assert len(items) == 1
    assert items[0].item_id == "raw"
    assert "正文内容" in items[0].text


def test_empty_file_rejected():
    with pytest.raises(EmptyDocument):
        load_items("空.txt", b"   \n  ")


def test_jsonl_one_line_one_item_and_id_field():
    payload = (
        '{"id": "q1", "question": "分块多大？", "answer": "500 字符"}\n'
        '{"question": "阈值是多少？", "answer": "0.45"}\n'
        'not-json-line\n'
        '\n'
    ).encode()
    items = load_items("语料.jsonl", payload, ["question", "answer"])
    assert [item.item_id for item in items] == ["q1", "l2", "l3"]
    assert "分块多大？" in items[0].text and "500 字符" in items[0].text
    assert items[2].text == "not-json-line"


def test_jsonl_without_known_fields_keeps_json():
    items = load_items("x.jsonl", '{"a": 1, "b": "二"}'.encode(), ["text"])
    assert '"a": 1' in items[0].text


def test_pdf_items_are_pages(tmp_path: Path):
    """真造一个两页 PDF（fpdf2 + 系统字体），验证"一页一条 + 页码元数据"。"""
    fpdf = pytest.importorskip("fpdf")
    font = Path(r"C:\Windows\Fonts\simhei.ttf")
    if not font.exists():
        pytest.skip("缺少中文字体，跳过 PDF 解析用例")
    pdf = fpdf.FPDF()
    pdf.add_font("CJK", "", str(font))
    pdf.set_font("CJK", size=12)
    for index in (1, 2):
        pdf.add_page()
        pdf.multi_cell(0, 8, f"第 {index} 页：云海工作台的分块默认 500 字符。")
    target = tmp_path / "两页.pdf"
    pdf.output(str(target))

    items = load_items("两页.pdf", target.read_bytes())
    assert [item.item_id for item in items] == ["p1", "p2"]
    assert [item.meta["page"] for item in items] == [1, 2]
    assert "500" in items[0].text


def test_tokenize_keeps_cjk_bigrams():
    tokens = tokenize("云海工作台的分块")
    assert "云" in tokens
    assert "云海" in tokens  # 相邻双字
    assert "的" not in tokens  # 停用词
