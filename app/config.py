"""服务配置。

约定（规划 10.1）：**Key 只放后端 .env，前端永不碰密钥**。
前端只拿到一个基地址，所有厂商凭证都留在这一层。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# 文档类型白名单（规划 10.1 / 10.2）
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".md", ".markdown", ".txt", ".jsonl"})
SOURCE_TYPES: dict[str, str] = {
    ".pdf": "pdf",
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
    ".jsonl": "jsonl",
}

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB 硬上限
LARGE_FILE_BYTES = 2 * 1024 * 1024  # 超过它就走异步入库，先回任务 ID


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
    )

    # ---------- 生成侧（DeepSeek / 任何 OpenAI 兼容接口）----------
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_timeout_s: float = 60.0
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.2

    # ---------- 向量化（规划 10.2：语义 vs 字面 两条路都要能跑）----------
    # bge：本地中文语义模型；hash：零依赖兜底（测试/无模型环境）；api：OpenAI 兼容 /embeddings
    embedder: str = "bge"
    embed_model: str = "BAAI/bge-small-zh-v1.5"
    # 走 hf-mirror 预先下载到本地的目录；存在就直接用，不再联网
    embed_local_dir: str = str(BASE_DIR / "models" / "bge-small-zh-v1.5")
    embed_device: str = "cpu"
    embed_batch_size: int = 16
    embed_api_key: str = ""
    embed_api_base_url: str = "https://api.siliconflow.cn/v1"
    embed_api_model: str = "BAAI/bge-m3"
    hash_embed_dim: int = 512

    # ---------- 检索（规划 10.3）----------
    collection: str = "yunhai_knowledge"
    top_k: int = 4
    # 语义阈值：余弦相似度，绝对量纲，由 scripts/eval_retrieval.py 校准（当前语料：0.4）
    # 口径是**偏召回**：语义 0.4 能保住全部 16 道库内题，代价是 12 道库外题漏网 7 道。
    # 之所以敢这么选，是端到端实测（scripts/eval_answers.py）里漏网的题全部被模型自己
    # 说明「库里没有」，没有一条编造——阈值在这里的角色是省 token，不是最后一道防线。
    score_threshold: float = 0.4
    # 字面 BM25 用固定尺度压缩（raw/(raw+8)），与余弦不同量纲，故单独一条阈值（校准值 0.55，同样偏召回）
    lexical_score_threshold: float = 0.55
    # 命中不够上下文就拒答：低于该条数视为「检索不到」
    min_hits: int = 1
    # 兜底三模式：refuse（默认拒答）/ bare（裸答并标注）/ web（联网搜索，阶段后置）
    fallback_mode: str = "refuse"

    # ---------- Agent 内核（第十一阶段：ReAct 循环与护栏）----------
    # 护栏三件套：轮数上限 + token 预算 + 工具超时；外加"同一工具连续失败几次就收口"的熔断
    agent_max_rounds: int = 8
    agent_token_budget: int = 12000
    agent_tool_timeout_s: float = 10.0
    agent_run_timeout_s: float = 120.0
    agent_max_tool_failures: int = 3
    # 回填给模型的历史消息条数（**已剔除轻量闲聊**，见 11.4）
    agent_history_messages: int = 6
    # 一轮 LLM 调用失败（且一个字都没吐出来）时的重试次数与间隔：
    # provider 偶发 5xx/超时很常见，直接抛给用户看"生成失败"太脆
    agent_llm_retries: int = 1
    agent_llm_retry_delay_s: float = 0.8
    # 待续跑状态（等前端执行 client 工具）保留多久
    agent_run_ttl_s: int = 3600

    # ---------- 分块 ----------
    chunk_size: int = 500
    chunk_overlap: int = 80
    jsonl_text_fields: str = "text,content,question,answer,title,body,summary"

    # ---------- 存储 ----------
    data_dir: Path = BASE_DIR / "data"

    # ---------- HTTP ----------
    # 开发期只放行 localhost（+ Tauri WebView 的 origin），部署时用环境变量覆盖
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:4173,http://127.0.0.1:4173,"
        "http://tauri.localhost,tauri://localhost"
    )
    # 挂在子路径下时的前缀（如 IIS 子应用 /yhai）。默认空 = 不剥前缀，本地开发零影响。
    # 为什么需要它：IIS 的 HttpPlatformHandler 是把请求原样转给子进程的，
    # 路径里带着 /yhai 前缀，而本服务的路由是 /api/* —— 不剥就是一片 404。
    # 环境变量名 = `AGENT_URL_PREFIX`（pydantic-settings 按**字段名**取环境变量，
    # 字段叫 agent_url_prefix 才对得上，写成 url_prefix 会去找 URL_PREFIX）。
    agent_url_prefix: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def jsonl_fields(self) -> list[str]:
        return [f.strip() for f in self.jsonl_text_fields.split(",") if f.strip()]

    @property
    def embed_local_path(self) -> str:
        """本地模型目录：相对路径按仓库根解析（.env 里写 `models/...` 就不用管从哪启动）。"""
        raw = Path(self.embed_local_dir)
        return str(raw if raw.is_absolute() else BASE_DIR / raw)

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def lexical_path(self) -> Path:
        return self.data_dir / "lexical.json"

    @property
    def sessions_db(self) -> Path:
        return self.data_dir / "sessions.db"

    @property
    def agent_runs_db(self) -> Path:
        return self.data_dir / "agent_runs.db"

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key.strip())

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.chroma_dir, self.uploads_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
