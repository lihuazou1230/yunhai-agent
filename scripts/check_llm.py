"""LLM 冒烟测试：确认后端 .env 里的 Key 能真的流式出字。

    .venv\\Scripts\\python scripts\\check_llm.py

没配 Key 会给出明确提示；配了但报 401/402/429 会打印厂商返回的中文说明。
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.runtime import Runtime  # noqa: E402


async def main() -> int:
    runtime = Runtime.build()
    llm = runtime.llm
    if not llm.configured:
        print("[skip] 未配置 LLM_API_KEY（在 yunhai-agent/.env 里填上再跑）")
        return 1

    print(f"[llm] base_url={runtime.settings.llm_base_url} model={llm.model}")
    messages = [
        {"role": "system", "content": "你是知识库助手，只依据上下文回答，不足就明说。"},
        {"role": "user", "content": "【知识库片段】\n[1] 来源：自检（第 1 块）\n分块默认 500 字符，重叠 80。\n\n【用户问题】\n分块默认多大？"},
    ]
    started = time.perf_counter()
    first_token_at: float | None = None
    pieces: list[str] = []
    try:
        async for piece in llm.stream_chat(messages):
            if first_token_at is None:
                first_token_at = time.perf_counter() - started
            pieces.append(piece)
            print(piece, end="", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[fail] {exc}")
        return 2
    print()
    print(
        f"[ok] 首 token {first_token_at:.2f}s，总计 {time.perf_counter() - started:.2f}s，"
        f"{len(''.join(pieces))} 字"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
