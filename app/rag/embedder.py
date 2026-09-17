"""Embedding 层（规划 10.2：中文语义模型 vs 字面匹配，两条路都要能跑）。

三个实现，同一个接口：

| 实现 | 用途 | 依赖 |
|------|------|------|
| BGEEmbedder | 生产默认：bge-small-zh-v1.5 本地推理 | sentence-transformers + torch |
| HashingEmbedder | 零依赖兜底：字符 bigram 哈希，确定性、可离线、| 无（纯 Python） |
| ApiEmbedder | 不想装 torch 时走 OpenAI 兼容 /embeddings | httpx |

**query 与 document 用不同编码**：bge 中文模型的官方用法是给查询加一句指令前缀
（"为这个句子生成表示以用于检索相关文章："），文档侧不加。
忘了这一步，检索质量会明显掉——这也是题库里常见的一问。
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

import httpx

from app.config import Settings
from app.rag.tokenize import tokenize

# bge 系列查询侧指令（官方推荐写法）
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        """把一批文本编码成向量（已做 L2 归一化，方便直接用余弦）。"""
        ...


class HashingEmbedder:
    """零依赖兜底：词元 -> 固定桶，累加后 L2 归一化。

    它没有语义能力（"电脑"和"计算机"互不相干），但**确定性**且**不需要下载**，
    所以拿它跑单测和 CI：管线逻辑（分块、幂等、阈值、SSE）与模型无关，
    不该因为 CI 装不上 torch 就测不了。
    """

    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in tokenize(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[bucket] += sign
            vectors.append(_l2_normalize(vector))
        return vectors


class BGEEmbedder:
    """本地中文语义模型（bge-small-zh-v1.5，512 维）。"""

    name = "bge"

    def __init__(
        self,
        model: str = "BAAI/bge-small-zh-v1.5",
        local_dir: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
    ):
        self._model_name = model
        self._local_dir = local_dir
        self._device = device
        self._batch_size = batch_size
        self._model = None
        self._dim = 0

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        from pathlib import Path

        from sentence_transformers import SentenceTransformer

        source = self._model_name
        if self._local_dir and Path(self._local_dir).exists():
            source = self._local_dir  # 已按 hf-mirror 下载到本地就不再联网
        self._model = SentenceTransformer(source, device=self._device)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dim(self) -> int:
        if not self._dim:
            self._ensure_model()
        return self._dim

    def encode(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        payload = [BGE_QUERY_INSTRUCTION + t for t in texts] if is_query else texts
        vectors = model.encode(
            payload,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]


class ApiEmbedder:
    """OpenAI 兼容的 /embeddings（如 SiliconFlow 的 bge-m3）。

    给"不想在服务器上装 torch"的部署形态留的口子；Key 同样只在后端 .env。
    """

    name = "api"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self.dim = 0

    def encode(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        response = httpx.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": texts},
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()["data"]
        vectors = [item["embedding"] for item in data]
        self.dim = len(vectors[0]) if vectors else self.dim
        return [_l2_normalize(list(map(float, v))) for v in vectors]


def build_embedder(settings: Settings) -> Embedder:
    """按配置造 embedder。

    `bge` 需要本机有 torch/sentence-transformers——缺了不静默降级成 hash
    （那会让检索质量悄悄崩掉，用户还以为是模型不行），而是报明确的错。
    想在没模型的机器上跑，就把 `EMBEDDER=hash` 写进 .env，这是显式选择。
    """
    kind = settings.embedder.strip().lower()
    if kind == "hash":
        return HashingEmbedder(dim=settings.hash_embed_dim)
    if kind == "api":
        if not settings.embed_api_key:
            raise RuntimeError("EMBEDDER=api 需要 EMBED_API_KEY")
        return ApiEmbedder(settings.embed_api_base_url, settings.embed_api_key, settings.embed_api_model)
    if kind == "bge":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "EMBEDDER=bge 需要 sentence-transformers；"
                "没装请先 pip install -r requirements.txt，或把 .env 改成 EMBEDDER=hash"
            ) from exc
        return BGEEmbedder(
            model=settings.embed_model,
            local_dir=settings.embed_local_path,
            device=settings.embed_device,
            batch_size=settings.embed_batch_size,
        )
    raise RuntimeError(f"未知的 EMBEDDER 配置：{settings.embedder}")


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]
