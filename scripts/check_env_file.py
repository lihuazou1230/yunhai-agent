"""一次性工具：检查 .env 的行尾并规范化（不打印任何值）。

背景：手工编辑 .env 时若被写成 CR-only（经典 Mac 行尾），
PowerShell 的 Get-Content 会把它整份读成一行、看起来像是"注释和 Key 挤在一起"，
而 python-dotenv 按通用换行解析仍然是好的——两边表现不一致，排查起来很费时间。
统一成 CRLF，顺手校验 LLM_API_KEY 是否真的解析得到。
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(sys.argv[1] if len(sys.argv) > 1 else ".env")

raw = TARGET.read_bytes()
crlf = raw.count(b"\r\n")
lf = raw.count(b"\n") - crlf
cr = raw.count(b"\r") - crlf
print(f"[bytes] 总长 {len(raw)}，CRLF={crlf}，纯 LF={lf}，纯 CR={cr}")

text = raw.decode("utf-8")
lines = text.splitlines()
print(f"[lines] 逻辑行数 {len(lines)}")
for index, line in enumerate(lines, start=1):
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        continue
    if "=" in stripped:
        name, _, value = stripped.partition("=")
        shown = f"<已填 {len(value.strip())} 字符>" if value.strip() else "<空>"
        print(f"  {index:>2}. {name.strip()} = {shown}")
    else:
        print(f"  {index:>2}. <无等号的一行，长度 {len(stripped)}>")

if cr or (crlf and lf):
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    with TARGET.open("w", encoding="utf-8", newline="\r\n") as handle:
        handle.write(normalized)
    print(f"[fix] 行尾混杂/含纯 CR，已统一为 CRLF 并写回 {TARGET}")
else:
    kind = "CRLF" if crlf else "LF"
    print(f"[fix] 行尾统一为 {kind}，无需改动")
