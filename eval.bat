@echo off
chcp 65001 >nul
cd /d %~dp0
echo === 1/2 生成 50 页评测 PDF ===
.venv\Scripts\python scripts\make_eval_pdf.py
echo === 2/2 样本入库 + 检索评测 ===
.venv\Scripts\python scripts\ingest.py eval\docs
.venv\Scripts\python scripts\eval_retrieval.py
