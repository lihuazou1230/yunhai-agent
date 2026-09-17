"""中文友好的分词（不引 jieba，保持依赖干净）。

做法：汉字按**单字 + 相邻双字（bigram）**入索引，英文/数字按整词入索引。
中文 bigram 是检索里性价比最高的一招——不需要词典，召回接近分词器，
对「云海工作台」这类自造词尤其稳（jieba 会把它切成"云海/工作台"，
一旦查询写成"云海工作台"，单字+bigram 反而不会漏）。
"""

from __future__ import annotations

import re

_CJK = r"\u4e00-\u9fff\u3400-\u4dbf"
_TOKEN_RE = re.compile(rf"[{_CJK}]|[a-zA-Z][a-zA-Z0-9_+#.-]*|\d+(?:\.\d+)?")

_STOPWORDS = frozenset(
    {
        "的",
        "了",
        "是",
        "在",
        "和",
        "与",
        "有",
        "我",
        "你",
        "他",
        "它",
        "这",
        "那",
        "什么",
        "怎么",
        "如何",
        "吗",
        "呢",
        "the",
        "a",
        "an",
        "of",
        "is",
        "are",
        "to",
        "and",
    }
)


def tokenize(text: str) -> list[str]:
    """返回用于倒排的词元列表（保留重复次数给 BM25 记词频）。"""
    raw = _TOKEN_RE.findall(text.lower())
    tokens: list[str] = []
    previous_cjk = ""
    for token in raw:
        if token in _STOPWORDS:
            previous_cjk = token if _is_cjk(token) else ""
            continue
        tokens.append(token)
        if _is_cjk(token):
            if previous_cjk:
                tokens.append(previous_cjk + token)
            previous_cjk = token
        else:
            previous_cjk = ""
    return tokens


def _is_cjk(token: str) -> bool:
    return len(token) == 1 and bool(re.match(rf"[{_CJK}]", token))
