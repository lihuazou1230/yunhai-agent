"""第三条路：语义 + 字面 RRF 融合，值不值得进生产？（第十阶段 10.2 的补充实验）

    .venv\\Scripts\\python scripts\\eval_hybrid.py

背景：`eval_retrieval.py` 量出语义与字面各有短板，于是自然会想到"两个都拿来用"。
标准做法是 RRF（Reciprocal Rank Fusion，`1/(k+rank)` 累加）——它不看分数只看名次，
天然免疫两种分数不同量纲的问题。

**结论：这份语料上融合没有增益，所以没有引入生产**（recall@4：语义 15/16、字面 16/16、融合 15/16）。
把它留成脚本而不是删掉，是因为"试过、量过、不行"也是结论——
面试里说得出"为什么没有上混合检索"，比含糊地说"我们用了混合检索"更站得住。

融合没用的原因也清楚：两种路子的排名高度重叠（12/16 道题两边都是第 1），
互补的只有个别题，而 RRF 对"一边第 1、另一边第 99"和"两边第 5"给的分数接近，反而会把稳的题拉低。
真要提升，方向应该是**换更强的 embedding**或**加查询改写（HyDE）**，而不是把两个弱检索器拌在一起。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.runtime import Runtime  # noqa: E402

RRF_K = 60


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rrf(rank_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    """名次融合：分数 = Σ 1/(k + rank)，与具体分数量纲无关。"""
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for position, chunk_id in enumerate(ranks, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def main() -> int:
    runtime = Runtime.build()
    rows = load_jsonl(ROOT / "eval" / "in_scope.jsonl")
    depth = 10  # 融合时两边各取多深（生产 top-k 是 4，这里放宽以便公平比较）

    print(f"{'#':<3}{'语义':<6}{'字面':<6}{'融合':<6} 问题")
    hits = {"semantic": 0, "lexical": 0, "hybrid": 0}
    for index, row in enumerate(rows, start=1):
        question, expected = row["question"], row["expected_source"]
        semantic = runtime.retriever.search(question, mode="semantic", top_k=depth)
        lexical = runtime.retriever.search(question, mode="lexical", top_k=depth)
        fused_ids = rrf([[h.chunk.chunk_id for h in semantic], [h.chunk.chunk_id for h in lexical]])
        pool = {h.chunk.chunk_id: h for h in semantic + lexical}

        def rank(hits_list, wanted=expected) -> int:
            # 默认参数绑定当前题目（B023）：闭包直接引用循环变量会被后一轮覆盖
            return next((p for p, h in enumerate(hits_list, start=1) if h.chunk.source == wanted), 99)

        semantic_rank = rank(semantic)
        lexical_rank = rank(lexical)
        hybrid_rank = next(
            (p for p, cid in enumerate(fused_ids, start=1) if pool[cid].chunk.source == expected), 99
        )
        hits["semantic"] += semantic_rank <= 4
        hits["lexical"] += lexical_rank <= 4
        hits["hybrid"] += hybrid_rank <= 4
        print(f"{index:<3}{semantic_rank:<6}{lexical_rank:<6}{hybrid_rank:<6} {question[:30]}")

    total = len(rows)
    print(
        f"\nrecall@4 —— 语义 {hits['semantic']}/{total} · 字面 {hits['lexical']}/{total} · "
        f"RRF 融合 {hits['hybrid']}/{total}"
    )
    best = max(hits["semantic"], hits["lexical"], hits["hybrid"])
    if hits["hybrid"] < best:
        print("[结论] 融合没有超过单路最好成绩，不引入第三条检索路径（见文件头的说明）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
