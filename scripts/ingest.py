"""CLI 入库：把文件/目录灌进知识库。

    python scripts/ingest.py eval/docs              # 入库一个目录
    python scripts/ingest.py eval/docs --reset      # 先清库再入库（换模型后必做）
    python scripts/ingest.py eval/docs --force      # 忽略内容哈希，强制重算
    python scripts/ingest.py --rebuild              # 用 data/uploads 里的原文件重建

`--rebuild` 是这套设计的一个副产品：上传时留了一份原文件，
所以调分块参数、换 embedding 模型都能离线重来，不用让用户重传。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import ALLOWED_EXTENSIONS  # noqa: E402
from app.runtime import Runtime  # noqa: E402


def collect_files(targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in ALLOWED_EXTENSIONS))
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"路径不存在：{target}")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="云海工作台知识库入库")
    parser.add_argument("paths", nargs="*", help="文件或目录")
    parser.add_argument("--reset", action="store_true", help="入库前清空向量库与字面索引")
    parser.add_argument("--force", action="store_true", help="忽略内容哈希，强制重新入库")
    parser.add_argument("--rebuild", action="store_true", help="用 data/uploads 下的原文件重建")
    args = parser.parse_args()

    runtime = Runtime.build()
    print(f"[env] embedder={runtime.embedder.name} collection={runtime.settings.collection}")
    if runtime.degraded_reason:
        print(f"[warn] {runtime.degraded_reason}")

    if args.reset:
        runtime.kb.reset()
        print("[reset] 向量库与字面索引已清空")

    if args.rebuild:
        files = sorted(runtime.settings.uploads_dir.glob("*"))
    else:
        if not args.paths:
            parser.error("请给出要入库的文件或目录（或使用 --rebuild）")
        files = collect_files(args.paths)

    if not files:
        print("[warn] 没有找到可入库的文件")
        return 1

    started = time.perf_counter()
    total_chunks = 0
    for path in files:
        data = path.read_bytes()
        try:
            result = runtime.kb.ingest(path.name, data, force=args.force or args.rebuild)
        except Exception as exc:  # noqa: BLE001 - CLI 要一条条报错继续跑
            print(f"  [fail] {path.name}: {exc}")
            continue
        flag = "skip" if result.skipped else "ok"
        total_chunks += 0 if result.skipped else result.chunks
        print(f"  [{flag}] {path.name}  条目 {result.items} / 块 {result.chunks} / {result.size_bytes} B")

    stats = runtime.kb.stats()
    print(
        f"[done] 新增块 {total_chunks}，库内文档 {stats['documents']} 篇 / 向量 {stats['chunks']} 块"
        f" / 字面索引 {stats['lexical_chunks']} 块，用时 {time.perf_counter() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
