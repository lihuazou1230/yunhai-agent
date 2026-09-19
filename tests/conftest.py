"""测试公共装置。

原则：**测试不依赖模型下载，也不依赖真实 LLM**。
- 向量用零依赖的 HashingEmbedder（EMBEDDER=hash），管线逻辑与模型无关；
- 生成用 StubLLM 逐块吐字，SSE 协议、引用、兜底三条路都能被真跑一遍。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.errors import LLMError  # noqa: E402
from app.main import create_app  # noqa: E402
from app.rag.generator import LLMClient  # noqa: E402
from app.runtime import Runtime, set_runtime  # noqa: E402


class StubLLM(LLMClient):
    """假 LLM：既能按预置片段逐块吐字，也能按**脚本**演工具调用。

    `script` 的每一项是一轮：`{"content": "...", "tool_calls": [{"name": ..., "arguments": {...}}], "tokens": N}`。
    脚本用完后回落到 `pieces`（纯文本）。第十一阶段的护栏用例靠它构造
    "反复调用同一个失败工具""token 爆预算"这类场景——真模型不配合的事情，假模型必须配合。
    """

    def __init__(
        self,
        pieces: tuple[str, ...] = ("分块默认 ", "500 ", "字符。", "[1]"),
        fail: str = "",
        script: list[dict] | None = None,
    ):
        self.pieces = list(pieces)
        self.fail = fail
        self.script = list(script or [])
        self.calls: list[dict] = []

    @property
    def configured(self) -> bool:  # type: ignore[override]
        return self.fail != "unconfigured"

    @property
    def model(self) -> str:  # type: ignore[override]
        return "stub-model"

    async def stream_chat(self, messages):  # type: ignore[override]
        self.calls.append({"messages": messages, "tools": None})
        if self.fail == "boom":
            raise LLMError("stub 生成失败")
        for piece in self.pieces:
            yield piece

    async def stream_with_tools(self, messages, tools, result):  # type: ignore[override]
        self.calls.append({"messages": messages, "tools": tools})
        if self.fail == "boom":
            raise LLMError("stub 生成失败")

        turn = self.script.pop(0) if self.script else {"content": "".join(self.pieces)}
        result.total_tokens = int(turn.get("tokens", 12))
        for piece in _chunks(str(turn.get("content", ""))):
            result.content += piece
            yield piece
        for index, call in enumerate(turn.get("tool_calls") or []):
            result.absorb_tool_call_delta(
                {
                    "index": index,
                    "id": call.get("id", f"call_{index}"),
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
                    },
                }
            )
        result.finish_tool_calls()
        result.finish_reason = "tool_calls" if result.tool_calls else "stop"


def _chunks(text: str, size: int = 6) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        embedder="hash",
        collection="test_knowledge",
        llm_api_key="test-key",
        llm_base_url="http://127.0.0.1:9/v1",
        llm_model="stub-model",
        chunk_size=120,
        chunk_overlap=20,
        top_k=3,
        score_threshold=0.3,
        lexical_score_threshold=0.3,
        fallback_mode="refuse",
    )


@pytest.fixture
def stub_llm() -> StubLLM:
    return StubLLM()


@pytest.fixture
def runtime(settings: Settings, stub_llm: StubLLM) -> Runtime:
    return Runtime.build(settings, llm=stub_llm)


@pytest.fixture
def client(runtime: Runtime):
    set_runtime(runtime)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    set_runtime(None)


SAMPLE_MD = """# 云海工作台使用手册

## 分块与检索

分块默认块长 500 个字符，重叠 80 个字符。分隔符按段落、换行、中文句末标点逐级降级。

## 向量化

默认模型是 bge-small-zh-v1.5，输出 512 维向量，做了 L2 归一化，存储用余弦距离。

## 阈值

语义检索的分数是余弦相似度，阈值校准结果是 0.45；BM25 需要一条独立阈值 0.35。
"""


@pytest.fixture
def sample_md() -> str:
    return SAMPLE_MD
