"""检索评测与阈值校准（规划 10.2 的「语义 vs 字面」对比 + 10.3 的阈值校准）。

跑法：

    .venv\\Scripts\\python scripts\\ingest.py eval/docs
    .venv\\Scripts\\python scripts\\eval_retrieval.py

输出：
- 控制台表格：两种策略的 recall@1 / recall@k / MRR，库内平均分 vs 库外最高分；
- `eval/report.md`：可直接贴进项目文档的完整报告（含逐题明细与阈值扫描过程）。

**为什么要有这个脚本**：规划里写的是"先对比字面匹配方案记录结论"。
结论不能靠印象——这个脚本就是那条结论的证据链。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.runtime import Runtime  # noqa: E402

MODES = ("semantic", "lexical")
MODE_LABEL = {"semantic": "语义（bge-small-zh）", "lexical": "字面（BM25）"}


@dataclass
class QuestionResult:
    question: str
    kind: str
    expected_source: str | None
    top_source: str
    top_score: float
    hit_rank: int | None  # 期望来源第一次出现的名次（1 起）；None = top-k 里都没有
    hits: list[tuple[str, float]]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def evaluate(runtime: Runtime, mode: str, questions: list[dict], top_k: int) -> list[QuestionResult]:
    results: list[QuestionResult] = []
    for row in questions:
        expected = row.get("expected_source")
        hits = runtime.retriever.search(row["question"], mode=mode, top_k=top_k)
        hit_rank = None
        if expected:
            for index, hit in enumerate(hits, start=1):
                if hit.chunk.source == expected:
                    hit_rank = index
                    break
        results.append(
            QuestionResult(
                question=row["question"],
                kind=row.get("kind") or ("改写" if row.get("paraphrase") else ("库内" if expected else "库外")),
                expected_source=expected,
                top_source=hits[0].chunk.source if hits else "",
                top_score=hits[0].score if hits else 0.0,
                hit_rank=hit_rank,
                hits=[(h.chunk.source, round(h.score, 4)) for h in hits],
            )
        )
    return results


def metrics(results: list[QuestionResult], top_k: int) -> dict:
    in_scope = [r for r in results if r.expected_source]
    out_scope = [r for r in results if not r.expected_source]
    paraphrase = [r for r in in_scope if r.kind == "改写"]
    ranks = [r.hit_rank for r in in_scope if r.hit_rank]
    return {
        "in_scope": len(in_scope),
        "out_scope": len(out_scope),
        "recall@1": round(sum(1 for r in in_scope if r.hit_rank == 1) / max(1, len(in_scope)), 3),
        f"recall@{top_k}": round(sum(1 for r in in_scope if r.hit_rank) / max(1, len(in_scope)), 3),
        "改写recall@1": (
            round(sum(1 for r in paraphrase if r.hit_rank == 1) / len(paraphrase), 3)
            if paraphrase
            else float("nan")
        ),
        "改写题数": len(paraphrase),
        "mrr": round(sum(1 / r for r in ranks) / max(1, len(in_scope)), 3),
        "库内平均分": round(statistics.mean([r.top_score for r in in_scope]) or 0.0, 3),
        "库内最低分": round(min([r.top_score for r in in_scope], default=0.0), 3),
        "库外平均分": round(statistics.mean([r.top_score for r in out_scope]) or 0.0, 3),
        "库外最高分": round(max([r.top_score for r in out_scope], default=0.0), 3),
    }


def scan_threshold(results: list[QuestionResult], step: float = 0.05) -> list[dict]:
    """在 [0.05, 0.95] 上扫阈值：既要库内尽量留下，又要库外尽量拦住。"""
    in_scope = [r for r in results if r.expected_source]
    out_scope = [r for r in results if not r.expected_source]
    rows: list[dict] = []
    value = step
    while value <= 0.95 + 1e-9:
        threshold = round(value, 2)
        kept = sum(1 for r in in_scope if r.top_score >= threshold)
        leaked = sum(1 for r in out_scope if r.top_score >= threshold)
        rows.append(
            {
                "threshold": threshold,
                "库内保留": kept,
                "库内保留率": round(kept / max(1, len(in_scope)), 3),
                "库外漏网": leaked,
                "库外拦截率": round(1 - leaked / max(1, len(out_scope)), 3),
            }
        )
        value += step
    return rows


def pick_threshold(rows: list[dict], in_scope_total: int, out_scope_total: int, leak_cap: float = 0.5) -> dict:
    """选择推荐阈值，规则写明、可复核：

    1. **库外零漏网且库内保留过半**（分数真的分得开）→ 零漏网前提下最大化库内保留；
    2. 分不开时（同领域但库里没有的问题会拿到中等分），退一步：
       **库外漏网不超过 `leak_cap`（默认 50%）** 的前提下，最大化库内保留；
    3. 连这个都做不到 → 取「库内保留 − 库外漏网」最大的一条，并把"无可行解"写进报告。

    第 1 步里那句「且库内保留过半」是评测当场补上的：只看"零漏网"会选出
    「阈值高到只剩 1 题能答」的荒谬解（BM25 的扫描表上真出现过 0.8 保留 1/16 却零漏网），
    那种阈值在生产上等于关掉知识库。
    """
    healthy = max(1, int(in_scope_total * 0.5))
    clean = [
        r for r in rows if r["库外漏网"] == 0 and r["库内保留"] >= healthy
    ]
    if clean:
        best = max(clean, key=lambda r: r["库内保留"])
        return {**best, "rule": f"库外零漏网且库内保留 ≥ {healthy} 题的前提下最大化库内保留"}

    capacity = int(out_scope_total * leak_cap)
    feasible = [r for r in rows if r["库外漏网"] <= capacity]
    if feasible:
        best = max(feasible, key=lambda r: (r["库内保留"], -r["库外漏网"]))
        return {
            **best,
            "rule": f"库外漏网 ≤ {int(leak_cap * 100)}%（{capacity} 题）前提下最大化库内保留",
        }

    best = max(rows, key=lambda r: (r["库内保留"] - r["库外漏网"], r["库内保留"]))
    return {**best, "rule": "无可行解：取「库内保留 − 库外漏网」最大的一条线"}


def main() -> int:
    parser = argparse.ArgumentParser(description="检索评测与阈值校准")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--report", default=str(ROOT / "eval" / "report.md"))
    parser.add_argument(
        "--leak-cap",
        type=float,
        default=0.5,
        help="库外漏网上限（比例）。放宽它 = 偏召回：保住更多库内题，代价是更多库外问题会带着上下文进生成侧。"
        "端到端实测（scripts/eval_answers.py）里漏网的题全部由模型自己说明「库里没有」，因此本项目生产默认用 0.6。",
    )
    args = parser.parse_args()

    runtime = Runtime.build()
    in_scope = load_jsonl(ROOT / "eval" / "in_scope.jsonl")
    out_scope = load_jsonl(ROOT / "eval" / "out_of_scope.jsonl")
    questions = in_scope + out_scope

    print(f"[env] embedder={runtime.embedder.name} 库内 {len(in_scope)} 题 / 库外 {len(out_scope)} 题")
    stats = runtime.kb.stats()
    print(f"[kb]  文档 {stats['documents']} 篇 / 向量 {stats['chunks']} 块")
    if stats["chunks"] == 0:
        print("[err] 知识库是空的，先跑：python scripts/ingest.py eval/docs")
        return 1

    sections: list[str] = []
    summary_rows: list[dict] = []
    details: dict[str, list[QuestionResult]] = {}
    calibrations: dict[str, dict] = {}

    for mode in MODES:
        results = evaluate(runtime, mode, questions, args.top_k)
        details[mode] = results
        row = metrics(results, args.top_k)
        row["mode"] = MODE_LABEL[mode]
        summary_rows.append(row)
        scan = scan_threshold(results)
        best = pick_threshold(scan, len(in_scope), len(out_scope), leak_cap=args.leak_cap)
        calibrations[mode] = {"scan": scan, "best": best}
        print(
            f"[{mode:8}] recall@1={row['recall@1']} recall@{args.top_k}={row[f'recall@{args.top_k}']} "
            f"MRR={row['mrr']} 库内均分={row['库内平均分']} 库内最低={row['库内最低分']} "
            f"库外最高={row['库外最高分']} → 阈值建议 {best['threshold']}"
            f"（库内留 {best['库内保留']}/{row['in_scope']}，库外漏 {best['库外漏网']}）"
        )

    sections.append("# 检索评测报告（第十阶段 10.2 / 10.3）\n")
    sections.append(
        f"- 生成时间：{_now()}\n- Embedding：`{runtime.embedder.name}`"
        f"（{runtime.settings.embed_model if runtime.embedder.name == 'bge' else '零依赖哈希' }）\n"
        f"- 分块：{runtime.settings.chunk_size} / 重叠 {runtime.settings.chunk_overlap}\n"
        f"- 知识库：{stats['documents']} 篇文档、{stats['chunks']} 块向量、{stats['lexical_chunks']} 块字面索引\n"
        f"- 题目：库内 {len(in_scope)} 题（其中 {sum(1 for q in in_scope if q.get('paraphrase'))} 题是口语化改写，"
        "用来暴露「只会字面匹配」的短板）、库外 12 题（6 个完全无关 + 6 个同领域但库里没有）\n"
        f"- top-k：{args.top_k}\n"
        f"- 阈值口径：库外漏网上限 {int(args.leak_cap * 100)}%（偏召回的线；"
        "端到端答案验收里漏网的题全部由模型自己说明「库里没有」，没有一条编造）\n"
    )

    sections.append("\n## 一、两种策略的整体表现\n")
    sections.append(
        f"| 策略 | recall@1 | recall@{args.top_k} | 改写题 recall@1 | MRR | "
        "库内平均分 | 库内最低分 | 库外平均分 | 库外最高分 |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    for row in summary_rows:
        paraphrase = row["改写recall@1"]
        paraphrase_text = "—" if isinstance(paraphrase, float) and paraphrase != paraphrase else paraphrase
        sections.append(
            f"| {row['mode']} | {row['recall@1']} | {row[f'recall@{args.top_k}']} | {paraphrase_text} | "
            f"{row['mrr']} | {row['库内平均分']} | {row['库内最低分']} | "
            f"{row['库外平均分']} | {row['库外最高分']} |\n"
        )

    semantic = summary_rows[0]
    lexical = summary_rows[1]
    sections.append("\n### 结论（自动生成，人工复核后再写进项目文档）\n")
    sections.append(_conclusion(semantic, lexical, calibrations, args.top_k))

    sections.append("\n## 二、阈值扫描\n")
    for mode in MODES:
        scan = calibrations[mode]["scan"]
        best = calibrations[mode]["best"]
        sections.append(f"\n### {MODE_LABEL[mode]}\n\n")
        sections.append("| 阈值 | 库内保留 | 库内保留率 | 库外漏网 | 库外拦截率 |\n| --- | --- | --- | --- | --- |\n")
        for row in scan:
            mark = " ✅" if row["threshold"] == best["threshold"] else ""
            sections.append(
                f"| {row['threshold']}{mark} | {row['库内保留']} | {row['库内保留率']} | "
                f"{row['库外漏网']} | {row['库外拦截率']} |\n"
            )
        sections.append(
            f"\n**建议阈值：`{best['threshold']}`**"
            f"（库内保留 {best['库内保留']}，库外漏网 {best['库外漏网']}；规则：{best['rule']}）\n"
        )

    sections.append("\n## 三、逐题明细\n")
    for mode in MODES:
        sections.append(f"\n### {MODE_LABEL[mode]}\n\n")
        sections.append("| 题目 | 类型 | 期望来源 | 首位来源 | 首位分数 | 命中名次 |\n| --- | --- | --- | --- | --- | --- |\n")
        for item in details[mode]:
            sections.append(
                f"| {_cell(item.question)} | {item.kind} | {item.expected_source or '—'} | "
                f"{item.top_source or '—'} | {round(item.top_score, 4)} | {item.hit_rank or '未命中'} |\n"
            )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(sections), encoding="utf-8")
    print(f"[done] 报告已写入 {report_path}")
    return 0


def _conclusion(semantic: dict, lexical: dict, calibrations: dict, top_k: int) -> str:
    lines: list[str] = []
    if semantic["recall@1"] > lexical["recall@1"]:
        lines.append(
            f"- **语义检索更准**：recall@1 {semantic['recall@1']} vs {lexical['recall@1']}，"
            f"recall@{top_k} {semantic[f'recall@{top_k}']} vs {lexical[f'recall@{top_k}']}，"
            f"MRR {semantic['mrr']} vs {lexical['mrr']}。"
        )
    elif semantic["recall@1"] < lexical["recall@1"]:
        lines.append(
            f"- **字面检索更准**：recall@1 {lexical['recall@1']} vs {semantic['recall@1']}，"
            f"MRR {lexical['mrr']} vs {semantic['mrr']}，recall@{top_k} "
            f"{lexical[f'recall@{top_k}']} vs {semantic[f'recall@{top_k}']}。"
        )
    else:
        lines.append("- 两者 recall@1 持平，差距体现在 MRR 与分数分离度上。")

    para_semantic = semantic["改写recall@1"]
    para_lexical = lexical["改写recall@1"]
    if para_semantic == para_semantic and para_lexical == para_lexical:  # 非 NaN
        if para_semantic > para_lexical:
            lines.append(
                f"- **口语化改写题**（{semantic['改写题数']} 题，不复用文档原词）：语义 recall@1 "
                f"{para_semantic} vs 字面 {para_lexical}——语义模型「换个说法也能召回」在这里体现出来了。"
            )
        elif para_semantic < para_lexical:
            lines.append(
                f"- **口语化改写题**（{semantic['改写题数']} 题，不复用文档原词）：语义 recall@1 "
                f"{para_semantic} vs 字面 {para_lexical}——**语义模型连改写题也没赢**。"
                "这与直觉相反，原因大概率是语料太小：小块语料里改写题与文档仍共享"
                "「两份」「主题」「标签」这类实词，BM25 抓得住；而 bge-small 在短句上的分数被压在"
                "0.4~0.6，区分度本就有限。**结论是当前语料下字面方案更划算**，"
                "语义模型的价值要在几十页以上的语料上重测（脚本已就位，换语料重跑即可）。"
            )
        else:
            lines.append(f"- 口语化改写题上两者持平（recall@1 = {para_semantic}）。")

    gap_semantic = semantic["库内最低分"] - semantic["库外最高分"]
    gap_lexical = lexical["库内最低分"] - lexical["库外最高分"]
    lines.append(
        f"- **阈值可分离性**：语义方案「库内最低分 − 库外最高分」= {round(gap_semantic, 3)}，"
        f"字面方案 = {round(gap_lexical, 3)}。"
        + (
            "为正说明存在一条能同时满足「库内不漏、库外不放过」的阈值。"
            if max(gap_semantic, gap_lexical) > 0
            else "为负说明无解：同领域但库里没有的问题（问 bge-large 维度、问 MySQL 支持）"
            "会拿到与库内问题重叠的分数，**单靠阈值拦不住**，必须靠生成侧的"
            "「上下文不足就明说」兜底，并让用户看到引用来源自己判断。"
        )
    )
    lines.append(
        f"- **落地阈值**：语义 `{calibrations['semantic']['best']['threshold']}`"
        f"（{calibrations['semantic']['best']['rule']}），"
        f"字面 `{calibrations['lexical']['best']['threshold']}`"
        f"（{calibrations['lexical']['best']['rule']}）。"
        "两条阈值分开存（`SCORE_THRESHOLD` / `LEXICAL_SCORE_THRESHOLD`），"
        "因为余弦相似度与 BM25 压缩分不是同一个量纲。"
    )
    lines.append(
        "- **口径说明**：BM25 的原始分用固定尺度压缩（`raw/(raw+8)`）而不是"
        "「除以本次查询最高分」——后者会让每条查询的第一名恒为 1.0，阈值永远拦不住任何东西"
        "（第一版评测里就是这么暴露的：扫描表全绿反而说明指标是坏的）。"
    )
    lines.append(
        "- **工程取舍**：语义模型要多背约 100MB 模型与 torch 运行时；"
        "字面方案零依赖、可作为降级路径与测试替身。生产默认仍走语义"
        "（它更抗改写与跨段关联，只是这份小语料量不出来），BM25 常驻用于对比与无模型环境。"
    )
    lines.append(
        "- **本次评测的局限**：语料只有 3 篇、题目 16 道，置信区间很宽；"
        "把 50 页 PDF（`scripts/make_eval_pdf.py` 产出）灌进去会更接近真实规模。"
    )
    return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _now() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M")


if __name__ == "__main__":
    raise SystemExit(main())
