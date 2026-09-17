"""测试公共装置。

原则：**测试不依赖模型下载，也不依赖真实 LLM**。
- 向量用零依赖的 HashingEmbedder（EMBEDDER=hash），管线逻辑与模型无关；
- 生成用 StubLLM 逐块吐字，SSE 协议、引用、兜底三条路都能被真跑一遍。
"""

from __future__ import annotations

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
    """假 LLM：按预置片段逐块吐字，可切换成抛错。"""

    def __init__(self, pieces: tuple[str, ...] = ("分块默认 ", "500 ", "字符。", "[1]"), fail: str = ""):
        self.pieces = list(pieces)
        self.fail = fail
        self.calls: list[list[dict[str, str]]] = []

    @property
    def configured(self) -> bool:  # type: ignore[override]
        return self.fail != "unconfigured"

    @property
    def model(self) -> str:  # type: ignore[override]
        return "stub-model"

    async def stream_chat(self, messages):  # type: ignore[override]
        self.calls.append(messages)
        if self.fail == "boom":
            raise LLMError("stub 生成失败")
        for piece in self.pieces:
            yield piece


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
