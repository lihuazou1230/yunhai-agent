# E2E smoke against a real uvicorn process (ASCII-only script: Windows PowerShell 5.1
# reads .ps1 as ANSI, so any CJK literal here would break parsing).
#
#   1) 另开一个窗口起服务（换一个端口，别占着 8000）：
#        $env:COLLECTION='yunhai_e2e'; .venv\Scripts\python -m uvicorn app.main:app --port 8011
#   2) 跑本脚本：
#        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\e2e_smoke.ps1
#
# 校验点：/api/health 自检 -> 上传 md 与 50 页 PDF -> 库内问题先出 citation
#（没配 Key 时随后报 llm_not_configured）-> 库外问题直接拒答 -> 会话历史落库。
$ErrorActionPreference = 'Continue'
$base = if ($env:E2E_BASE) { $env:E2E_BASE } else { 'http://127.0.0.1:8011' }
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$notes = (Get-ChildItem "$root\eval\docs\*.md" | Where-Object { $_.Name -like '*RAG*' } | Select-Object -First 1).FullName
$pdf = (Get-ChildItem "$root\eval\pdf\*.pdf" -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
$qIn = Join-Path $root 'eval\ask_in_scope.json'
$qOut = Join-Path $root 'eval\ask_out_of_scope.json'

function Show($title, $body) {
  Write-Output ""
  Write-Output "===== $title ====="
  Write-Output $body
}

Show 'GET /api/health' (curl.exe -s "$base/api/health")
Show 'GET /api/health/embedder' (curl.exe -s "$base/api/health/embedder")
Show 'POST /api/documents (markdown)' (curl.exe -s -X POST -F "file=@$notes" "$base/api/documents")
if ($pdf) {
  Show 'POST /api/documents (50-page PDF)' (curl.exe -s -X POST -F "file=@$pdf" "$base/api/documents")
} else {
  Write-Output "[skip] 没有 eval/pdf/*.pdf，先跑 scripts\make_eval_pdf.py"
}
Show 'GET /api/documents' (curl.exe -s "$base/api/documents")
Show 'POST /api/ask (in-scope)' (curl.exe -s -N -X POST -H "Content-Type: application/json" --data-binary "@$qIn" "$base/api/ask")
Show 'POST /api/ask (out-of-scope)' (curl.exe -s -N -X POST -H "Content-Type: application/json" --data-binary "@$qOut" "$base/api/ask")
Show 'GET /api/sessions' (curl.exe -s "$base/api/sessions")
Write-Output ""
Write-Output "=== DONE ==="
