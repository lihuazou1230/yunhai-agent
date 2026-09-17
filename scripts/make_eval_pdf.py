"""生成评测用的 50 页中文 PDF（验收清单里的「上传 50 页 PDF」）。

    .venv\\Scripts\\python scripts\\make_eval_pdf.py

做法：把手册与笔记正文、FAQ 语料，加上若干**确定性生成**的附录（变更记录、配置项参考、
接口字段说明）拼成一份长文档，用 fpdf2 + 系统黑体渲染成 PDF；
不足 50 页就继续追加附录条目，直到页数达标。因此这个脚本在任何机器上都能稳定产出 50+ 页。

字体：优先用 `C:\\Windows\\Fonts\\simhei.ttf`（中文不糊），没有就报错让人指定。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simkai.ttf"),
    Path(r"C:\Windows\Fonts\Deng.ttf"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
]


def plain_markdown(text: str) -> str:
    """去掉 markdown 记号：PDF 里不该出现 `##` 和 `**`。"""
    out: list[str] = []
    for line in text.splitlines():
        line = line.replace("**", "").replace("`", "")
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        out.append(line)
    return "\n".join(out)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_corpus() -> str:
    docs = ROOT / "eval" / "docs"
    sections: list[str] = []

    for name in ("云海工作台使用手册.md", "RAG技术笔记.md"):
        path = docs / name
        if path.exists():
            sections.append(plain_markdown(path.read_text(encoding="utf-8")))

    faq = docs / "常见问题.jsonl"
    if faq.exists():
        lines = ["常见问题汇编", ""]
        for index, row in enumerate(load_jsonl(faq), start=1):
            lines.append(f"第 {index} 问：{row['question']}")
            lines.append(f"答：{row['answer']}")
            lines.append("")
        sections.append("\n".join(lines))

    sections.append(_changelog(320))
    sections.append(_config_reference(240))
    sections.append(_api_fields(180))
    return "\n\n".join(sections)


def _changelog(count: int) -> str:
    lines = ["附录 A · 变更记录", ""]
    for index in range(1, count + 1):
        version = f"0.{index // 10}.{index % 10}"
        lines.append(
            f"{index:03d}. 版本 {version}（2026-{(index % 12) + 1:02d}-{(index % 28) + 1:02d}）："
            f"调整第 {index % 40 + 1} 项交互细节，修复统计页在范围切换后的一次重复请求，"
            f"补充 {index % 7 + 1} 条单元测试。"
        )
    return "\n".join(lines)


def _config_reference(count: int) -> str:
    lines = ["附录 B · 配置项参考", ""]
    keys = [
        "theme",
        "wallpaper",
        "tags",
        "links",
        "countdown-items",
        "earnings",
        "worklog",
        "reminder-settings",
        "search-engine",
        "dashboard-order",
    ]
    for index in range(1, count + 1):
        key = f"{keys[index % len(keys)]}-{index:03d}"
        lines.append(
            f"{index:03d}. 配置键 {key}：类型为字符串或对象，默认值为空，"
            f"属于{'随账号同步' if index % 3 else '仅本机保存'}的偏好项，"
            f"修改后 {index % 5 + 1} 秒内落盘。"
        )
    return "\n".join(lines)


def _api_fields(count: int) -> str:
    lines = ["附录 C · 接口字段说明", ""]
    fields = ["question", "session_id", "mode", "top_k", "threshold", "fallback_mode", "citation", "score"]
    for index in range(1, count + 1):
        field = f"{fields[index % len(fields)]}_{index:03d}"
        lines.append(
            f"{index:03d}. 字段 {field}：{'必填' if index % 2 else '可选'}，"
            f"取值范围见第 {index % 30 + 1} 节，缺失时按默认值处理，"
            f"服务端会在校验失败时返回 400 与中文错误说明。"
        )
    return "\n".join(lines)


def find_font(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SystemExit(f"字体不存在：{path}")
        return path
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise SystemExit("找不到中文字体，请用 --font 指定一个 .ttf/.ttc")


def render(text: str, output: Path, font: Path, min_pages: int) -> int:
    from fpdf import FPDF
    from pypdf import PdfReader

    def build(body: str) -> int:
        from fpdf.enums import XPos, YPos

        pdf = FPDF(format="A4")
        pdf.set_auto_page_break(auto=True, margin=16)
        pdf.add_font("CJK", "", str(font))
        pdf.set_font("CJK", size=12)
        pdf.add_page()
        for line in body.splitlines():
            if not line.strip():
                pdf.ln(3)
                continue
            # 必须显式回到左边距：multi_cell 默认把光标停在本行右侧，
            # 下一行再取「剩余宽度」就会无限窄 -> "Not enough horizontal space"
            pdf.multi_cell(0, 7, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.output(str(output))
        return len(PdfReader(str(output)).pages)

    body = text
    pages = build(body)
    extra = 1
    while pages < min_pages:
        # 页数不够就追加附录，保证「50 页 PDF」这条验收在任何机器上都成立
        body = body + "\n\n" + _changelog(200 + extra * 40)
        pages = build(body)
        extra += 1
        if extra > 12:
            break
    return pages


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 50 页评测 PDF")
    # 刻意放在 eval/pdf/ 而不是 eval/docs/：docs/ 是检索评测的语料目录，
    # 验收用的 50 页 PDF 与它同源，混在一起会让"库内问题该命中哪一份"变成两义。
    parser.add_argument("--out", default=str(ROOT / "eval" / "pdf" / "云海工作台产品说明（50页）.pdf"))
    parser.add_argument("--font", default=None)
    parser.add_argument("--min-pages", type=int, default=50)
    args = parser.parse_args()

    font = find_font(args.font)
    corpus = build_corpus()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)  # fpdf 不会替我们建目录
    pages = render(corpus, output, font, args.min_pages)
    size_kb = output.stat().st_size / 1024
    print(f"[ok] {output.name}：{pages} 页 / {size_kb:.0f} KB / 正文 {len(corpus)} 字（字体 {font.name}）")
    return 0 if pages >= args.min_pages else 1


if __name__ == "__main__":
    raise SystemExit(main())
