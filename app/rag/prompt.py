"""生成侧约束（规划 10.3）。

Prompt 的三条硬规矩，对应验收清单里的「10 个库外问题 100% 不编造」：

1. **仅依据上下文回答**；
2. 上下文不足就**明说不足**，禁止用"一般来说""通常"这类话把缺口糊过去；
3. 结论必须能在上下文里指到出处（引用由后端按检索结果生成，不靠模型自报）。
"""

from __future__ import annotations

from typing import Any

from app.rag.types import Retrieved

SYSTEM_PROMPT = """你是「云海工作台」的知识库助手。你只能依据下面提供的知识库片段回答用户问题。

规则（必须遵守）：
1. 只使用【知识库片段】里出现的信息作答，不要使用片段之外的任何知识，也不要凭常识补充。
2. 如果片段不足以回答问题，直接说明"知识库里没有相关内容"，并指出你缺什么信息；不要猜测、不要编造数字或结论。
3. 回答要简洁务实：先给结论，再给依据。使用中文，不要输出 markdown 标题层级。
4. 不要提及"片段""上下文""prompt"这些词，直接说"根据《文件名》"。
5. 每个结论后面用 [编号] 标注它来自哪个片段，例如 [1]、[2][3]。"""

BARE_HINT = (
    "【重要】当前知识库里没有检索到与问题相关的内容。"
    "请先明确告诉用户「以下回答不基于知识库」，再尽量简短地用你自己的知识回答，"
    "并在结尾提醒用户该内容未经知识库核实。"
)

WEB_HINT = (
    "【重要】知识库未命中，且后端尚未接入联网搜索（规划 10.3 兜底模式 C 后置）。"
)


def build_context(hits: list[Retrieved], *, start_index: int = 1) -> str:
    """把命中块拼成带编号的上下文——编号就是后端发给前端的 citation.index。

    `start_index` 给 Agent 的多次检索用：第二轮检索的编号要接着上一轮往下排，
    否则模型写 [1] 时你分不清它指的是哪一次检索的片段。
    """
    blocks: list[str] = []
    for offset, hit in enumerate(hits):
        index = start_index + offset
        chunk = hit.chunk
        location = f"第 {chunk.page} 页" if chunk.page else f"第 {chunk.chunk_index + 1} 块"
        blocks.append(f"[{index}] 来源：{chunk.source}（{location}）\n{chunk.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[Retrieved], *, bare: bool = False) -> list[dict[str, str]]:
    """组装 messages。`bare=True` 是兜底模式 B：明确标注"不基于知识库"。"""
    system = SYSTEM_PROMPT
    if bare:
        system = system + "\n\n" + BARE_HINT
    context = build_context(hits) if hits else "（无）"
    user = f"【知识库片段】\n{context}\n\n【用户问题】\n{question}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def citations_of(hits: list[Retrieved], *, start_index: int = 1) -> list[dict[str, Any]]:
    """引用块（与 [编号] 一一对应）。"""
    return [hit.as_citation(start_index + offset) for offset, hit in enumerate(hits)]
