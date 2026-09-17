"""下载中文向量模型（走 hf-mirror，国内直连）。

    set HF_ENDPOINT=https://hf-mirror.com
    .venv\\Scripts\\python scripts\\download_model.py

为什么要单独一个脚本：torch/transformers 只会去 huggingface.co 取模型，
而国内常常连不上；把镜像地址、目标目录、"关掉 xet 传输"这几件事写死在脚本里，
换台机器照着跑就行（xet 走 cas-server.xethub.hf.co，镜像不代理它，会 401）。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
PATTERNS = ["*.json", "*.txt", "*.bin", "*.model", "*.safetensors"]


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 bge 中文向量模型到本地")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target", default=str(ROOT / "models" / "bge-small-zh-v1.5"))
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()

    # 必须在 import huggingface_hub 之前设好环境变量
    os.environ.setdefault("HF_ENDPOINT", args.endpoint)
    # xet 传输会直连 cas-server.xethub.hf.co（镜像不代理）→ 关掉它，退回普通 HTTP 下载
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from huggingface_hub import snapshot_download

    target = Path(args.target)
    path = snapshot_download(args.model, local_dir=str(target), allow_patterns=PATTERNS)
    files = sorted(p.name for p in Path(path).glob("*") if p.is_file())
    print(f"[ok] 模型已下载到 {path}")
    print(f"[ok] 文件：{', '.join(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
