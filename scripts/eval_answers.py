"""端到端答案验收（规划第十阶段验收清单的第 1、2 条）。

    .venv\\Scripts\\python scripts\\ingest.py eval\\docs --reset
    .venv\\Scripts\\python scripts\\eval_answers.py            # 全部 16 + 12 题
    .venv\\Scripts\\python scripts\\eval_answers.py --limit 4  # 冒烟：只跑前 4 题

与 `eval_retrieval.py` 的分工：
- 那个脚本只测**检索**（不进生成，不需要 Key，可离线复现）；
- 这个脚本走**完整链路**（检索 → 阈值 → 生成 → 引用），所以会真的花 token，
  也因此才是"答案对不对"的证据。每题独立开一个会话，互不干扰。

判定口径（都写进报告，允许人工复核推翻）：
- 库内题：`fallback=kb` 且至少一条引用 → **通过**；再额外看答案里有没有 `hint` 里的关键词（数字/实词），
  这只是"看起来答对了"的廉价代理，最终由人看报告里的答案原文；
- 库外题：`fallback=refuse`（阈值拦住）→ 通过；若阈值没拦住但模型自己说"库里没有/不足" → 也算通过；
  两者都不是 → 标成**疑似编造**，必须人工看一眼。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.runtime import Runtime  # noqa: E402

# 模型"承认自己不知道"的措辞：命中任一即视为没有编造
INSUFFICIENT_MARKERS = (
    "没有",
    "不足",
    "无法",
    "未提及",
    "不包含",
    "不清楚",
    "未提供",
    "不基于知识库",
    "缺乏",
)

# 更严格的「库内题其实没答出来」信号：模型明说知识库里没有这块内容。
# 与上面的宽口径分开，是因为库内题的**正确答案**里也可能出现"没有"（比如"未配置薪资就不记录"），
# 用宽口径会把答对的题误判成拒答。
MISSED_MARKERS = (
    "知识库里没有",
    "知识库中没有",
    "没有检索到",
    "没有相关内容",
    "未找到相关内容",
    "现有资料只",
    "现有资料里",
)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def parse_frames(frames: list[str]) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for frame in frames:
        name = ""
        data = "{}"
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        if name:
            events.append((name, json.loads(data)))
    return events


def hint_tokens(hint: str) -> list[str]:
    """把 hint 拆成"数字型 token + 长度≥2 的实词"，用作期望事实的候选词。"""
    parts = hint.replace("、", " ").replace("，", " ").replace("：", " ").replace("(", " ").split()
    return [p for p in parts if p and (any(ch.isdigit() for ch in p) or len(p) >= 2)]


def keyword_hit(answer: str, hint: str) -> tuple[bool, list[str]]:
    """答案里是否出现期望事实（廉价代理，供人工复核）。"""
    tokens = hint_tokens(hint)
    hit = [token for token in tokens if token in answer]
    return bool(hit), hit


async def ask_once(runtime: Runtime, question: str, top_k: int | None) -> dict:
    started = time.perf_counter()
    frames = [frame async for frame in runtime.ask.stream(question, top_k=top_k)]
    events = parse_frames(frames)
    answer = "".join(data["text"] for name, data in events if name == "token")
    citations = [data for name, data in events if name == "citation"]
    done = next((data for name, data in events if name == "done"), {})
    error = next((data for name, data in events if name == "error"), None)
    return {
        "answer": answer.strip(),
        "citations": citations,
        "fallback": done.get("fallback", "?"),
        "hit_count": done.get("hit_count", 0),
        "latency_ms": done.get("latency_ms") or int((time.perf_counter() - started) * 1000),
        "session_id": done.get("session_id", ""),
        "error": error,
    }


def truncate(text: str, limit: int = 320) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


async def main() -> int:
    parser = argparse.ArgumentParser(description="端到端答案验收（需要 LLM Key）")
    parser.add_argument("--limit", type=int, default=0, help="每类只跑前 N 题（0 = 全部）")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.3, help="题间停顿（秒），避免触发限流")
    parser.add_argument("--report", default=str(ROOT / "eval" / "answer_report.md"))
    args = parser.parse_args()

    runtime = Runtime.build()
    if not runtime.llm.configured:
        print("[err] 未配置 LLM_API_KEY：本脚本要真的调模型，先在 .env 里填上")
        return 1
    stats = runtime.kb.stats()
    print(f"[env] embedder={runtime.embedder.name} model={runtime.llm.model} 库存 {stats['documents']} 篇 / {stats['chunks']} 块")
    if stats["chunks"] == 0:
        print("[err] 知识库是空的，先跑：python scripts/ingest.py eval/docs")
        return 1

    in_scope = load_jsonl(ROOT / "eval" / "in_scope.jsonl")
    out_scope = load_jsonl(ROOT / "eval" / "out_of_scope.jsonl")
    if args.limit:
        in_scope = in_scope[: args.limit]
        out_scope = out_scope[: args.limit]

    results: list[dict] = []

    print(f"\n=== 库内问题（{len(in_scope)} 题）：要求命中知识库并带引用 ===")
    for index, row in enumerate(in_scope, start=1):
        outcome = await ask_once(runtime, row["question"], args.top_k)
        ok, hits = keyword_hit(outcome["answer"], row.get("hint", ""))
        cited = outcome["fallback"] == "kb" and len(outcome["citations"]) > 0
        # 检索侧也独立量一次：命中块里到底有没有"期望事实"。
        # 这样能把「检索没找到」和「模型看到了却没用上」分开——两者的修法完全不同。
        retrieved = runtime.retriever.filtered(row["question"], top_k=args.top_k or None)
        tokens = hint_tokens(row.get("hint", ""))
        context_hit = any(token in hit.chunk.text for hit in retrieved for token in tokens)
        in_context = [token for token in tokens if any(token in hit.chunk.text for hit in retrieved)]
        results.append(
            {
                "kind": "库内",
                "question": row["question"],
                "expected": row.get("expected_source", ""),
                "hint": row.get("hint", ""),
                "paraphrase": bool(row.get("paraphrase")),
                "passed": ok,
                "cited": cited,
                "context_hit": context_hit,
                "context_tokens": in_context,
                "keyword_hit": ok,
                "keyword_hits": hits,
                "outcome": outcome,
            }
        )
        verdict = "答案含事实" if ok else ("检索未命中事实" if not context_hit else "检索到了但没答出")
        print(
            f"  [{index:>2}/{len(in_scope)}] {verdict} 引用 {len(outcome['citations'])} 条 · {outcome['latency_ms']}ms"
            f" · {row['question'][:20]}"
        )
        if args.delay:
            time.sleep(args.delay)

    print(f"\n=== 库外问题（{len(out_scope)} 题）：要求 100% 不编造 ===")
    for index, row in enumerate(out_scope, start=1):
        outcome = await ask_once(runtime, row["question"], args.top_k)
        refused = outcome["fallback"] == "refuse"
        admitted = any(marker in outcome["answer"] for marker in INSUFFICIENT_MARKERS)
        passed = refused or admitted
        results.append(
            {
                "kind": "库外",
                "question": row["question"],
                "expected": "",
                "hint": row.get("kind", ""),
                "passed": passed,
                "cited": False,
                "keyword_hit": False,
                "keyword_hits": [],
                "outcome": outcome,
            }
        )
        if refused:
            verdict = "拒答（阈值拦住，未调模型）"
        elif admitted:
            verdict = "模型自己说明不足（阈值放行但没编）"
        else:
            verdict = "疑似编造，需人工复核"
        print(f"  [{index:>2}/{len(out_scope)}] {'PASS' if passed else 'FAIL'} {verdict} · {row['question'][:22]}")
        if args.delay:
            time.sleep(args.delay)

    in_results = [r for r in results if r["kind"] == "库内"]
    out_results = [r for r in results if r["kind"] == "库外"]
    straight = [r for r in in_results if not r["paraphrase"]]
    paraphrase = [r for r in in_results if r["paraphrase"]]
    in_pass = sum(1 for r in in_results if r["passed"])
    in_cited = sum(1 for r in in_results if r["cited"])
    in_context = sum(1 for r in in_results if r["context_hit"])
    straight_pass = sum(1 for r in straight if r["passed"])
    paraphrase_pass = sum(1 for r in paraphrase if r["passed"])
    out_pass = sum(1 for r in out_results if r["passed"])
    out_refused = sum(1 for r in out_results if r["outcome"]["fallback"] == "refuse")

    print("\n=== 小结 ===")
    print(f"库内：带引用 {in_cited}/{len(in_results)}；检索到期望事实 {in_context}/{len(in_results)}；答案覆盖事实 {in_pass}/{len(in_results)}")
    print(f"      直问直答 {straight_pass}/{len(straight)}；口语化改写 {paraphrase_pass}/{len(paraphrase)}")
    print(f"库外：不编造 {out_pass}/{len(out_results)}（其中阈值直接拦住 {out_refused} 题）")

    lines: list[str] = []
    lines.append("# 端到端答案验收报告（第十阶段验收清单）\n\n")
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"- 模型：`{runtime.llm.model}`（{runtime.settings.llm_base_url}）\n")
    lines.append(
        f"- 检索：embedder=`{runtime.embedder.name}`，top-k={args.top_k or runtime.settings.top_k}，"
        f"语义阈值 {runtime.settings.score_threshold}\n"
    )
    lines.append(f"- 知识库：{stats['documents']} 篇 / {stats['chunks']} 块\n")
    lines.append("\n## 一、结论\n\n")
    lines.append(
        f"- **库内问题**：**{in_cited}/{len(in_results)} 命中知识库并带引用来源**"
        f"（验收要求 ≥8/10 正确且带引用）；把答案里是否出现期望事实作为「答对」的廉价代理，"
        f"则是 **{in_pass}/{len(in_results)}**——其中直问直答 **{straight_pass}/{len(straight)}**、"
        f"口语化改写 **{paraphrase_pass}/{len(paraphrase)}**\n"
    )
    lines.append(
        f"- **把「检索」和「生成」分开看**：检索到的块里含期望事实的有 **{in_context}/{len(in_results)}**，"
        f"答案覆盖事实的有 {in_pass}/{len(in_results)}。"
        "差额的两类问题修法完全不同：前者要调检索（换更强的 embedding / 查询改写），"
        "后者要调 prompt 或上下文组织\n"
    )
    lines.append(
        f"- **库外问题**：{out_pass}/{len(out_results)} 未编造；"
        f"其中 {out_refused} 题被阈值直接拦下、{out_pass - out_refused} 题由模型自己说明「库里没有」\n"
    )
    lines.append(
        "- **两个数为什么都要看**：检索没命中时，模型会如实说「知识库里没有」——"
        "这不算编造（库外题的判断标准），但对库内题就是**没答出**。"
        "只报「带引用」会把这类题算成成功，所以这里分开统计\n"
    )
    lines.append(
        "- 关键词命中是廉价代理（从 `hint` 里抽数字与实词做包含判断），"
        "**答案是否正确请以第四节的逐题原文为准**；"
        "它也会**漏判**——hint 的措辞和文档原话不一致时（例如 hint 写「内容哈希幂等」、"
        "文档里写「文档 ID 取 SHA-256 前 16 位，内容没变就直接跳过」），"
        "检索明明命中了、答案也答对了，但那一行仍会显示 ❌\n"
    )
    lines.append("\n## 二、逐题明细（库内）\n\n")
    lines.append(
        "| # | 问题 | 类型 | 带引用 | 检索到事实 | 答案含事实 | 引用条数 | 依据 | 耗时 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    for index, item in enumerate(in_results, start=1):
        outcome = item["outcome"]
        sources = "、".join(sorted({c["source"] for c in outcome["citations"]})) or "—"
        lines.append(
            f"| {index} | {item['question']} | {'改写' if item['paraphrase'] else '直问'} | "
            f"{'✅' if item['cited'] else '❌'} | {'✅' if item['context_hit'] else '❌'} | "
            f"{'✅' if item['keyword_hit'] else '❌'}"
            f"{'（' + '、'.join(item['keyword_hits']) + '）' if item['keyword_hits'] else ''} | "
            f"{len(outcome['citations'])} | {outcome['fallback']} / {sources} | {outcome['latency_ms']}ms |\n"
        )

    lines.append("\n## 三、逐题明细（库外）\n\n")
    lines.append("| # | 问题 | 结果 | 依据 | 判定 |\n| --- | --- | --- | --- | --- |\n")
    for index, item in enumerate(out_results, start=1):
        outcome = item["outcome"]
        if outcome["fallback"] == "refuse":
            verdict = "阈值拦住，未调模型"
        elif item["passed"]:
            verdict = "模型说明不足"
        else:
            verdict = "⚠️ 疑似编造"
        lines.append(
            f"| {index} | {item['question']} | {'✅' if item['passed'] else '❌'} | "
            f"{outcome['fallback']} / 命中 {outcome['hit_count']} 块 | {verdict} |\n"
        )

    lines.append("\n## 四、答案原文（供人工复核）\n")
    for index, item in enumerate(results, start=1):
        outcome = item["outcome"]
        lines.append(f"\n### {index}. [{item['kind']}] {item['question']}\n\n")
        if item["expected"]:
            lines.append(f"- 期望来源：{item['expected']}；hint：{item['hint']}\n")
        elif item["hint"]:
            lines.append(f"- 类型：{item['hint']}\n")
        lines.append(
            f"- 依据：`{outcome['fallback']}`，命中 {outcome['hit_count']} 块，"
            f"引用 {len(outcome['citations'])} 条，{outcome['latency_ms']}ms\n"
        )
        if outcome["citations"]:
            cited = "、".join(
                f"[{c['index']}] {c['source']}"
                + (f"（第 {c['page']} 页）" if c.get("page") else "")
                for c in outcome["citations"]
            )
            lines.append(f"- 引用：{cited}\n")
        if outcome["error"]:
            lines.append(f"- ⚠️ 错误事件：`{outcome['error']['code']}` {outcome['error']['message']}\n")
        lines.append(f"\n> {truncate(outcome['answer']) or '（无输出）'}\n")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"[done] 报告已写入 {report_path}")
    # 验收口径：直问直答的正确率 ≥80%（口语化改写单独看，它考的是模型上限而不是产品底线），
    # 且库外问题一条都不编造
    straight_ok = straight_pass >= max(1, int(len(straight) * 0.8))
    return 0 if straight_ok and out_pass == len(out_results) else 1


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))
