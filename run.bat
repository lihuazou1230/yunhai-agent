@echo off
chcp 65001 >nul
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo [error] 还没有虚拟环境，先跑：
  echo   python -m venv .venv
  echo   .venv\Scripts\python -m pip install -r requirements.txt
  exit /b 1
)
if not exist .env (
  echo [warn] 没有 .env，先复制 .env.example 成 .env 并填 LLM_API_KEY
)
echo [start] http://127.0.0.1:8000/docs  （Ctrl+C 停止）
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
