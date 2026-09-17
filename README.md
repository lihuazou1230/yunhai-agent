# yunhai-agent · 云海工作台 AI 后端

云海工作台（Vue 3 前端，仓库 [`yunhai-workspace`](https://github.com/lihuazou1230/yunhai-workspace)）的 Python 后端。
两个仓库从第十阶段开始就是**独立提交**的：前端只做消费，模型 Key 与向量库都留在这一层。

> 当前进度：**第十阶段 · RAG 知识库** 已实现（文档管道 / 检索与阈值 / 流式问答 / 会话历史）。
> 第十一阶段的 ReAct 内核会在同一套 SSE 协议上加 `tool_call` / `tool_result` 事件，前端解析器不用改。

## 快速开始

```powershell
# 1) 建环境（Python 3.10+）
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# 2) 配 Key（后端独占，前端拿不到）
copy .env.example .env
#   编辑 .env：LLM_API_KEY=sk-xxxx（DeepSeek，OpenAI 兼容）

# 3) 拉中文向量模型（国内走 hf-mirror）
$env:HF_ENDPOINT = 'https://hf-mirror.com'
.venv\Scripts\python -c "from huggingface_hub import snapshot_download as d; d('BAAI/bge-small-zh-v1.5', local_dir='models/bge-small-zh-v1.5', allow_patterns=['*.json','*.txt','*.bin','*.model','*.safetensors'])"

# 4) 起服务
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

不想装 torch 也能跑：把 `.env` 里的 `EMBEDDER` 改成 `hash`（零依赖字面哈希向量，质量降级但管线完整），
或填 `EMBED_API_KEY` 用 OpenAI 兼容的 `/embeddings`（`EMBEDDER=api`）。

## 目录结构

```
app/
├── main.py            FastAPI 应用装配（CORS / 异常翻译 / 路由）
├── config.py          配置（.env -> Settings），Key 只在这里出现
├── runtime.py         运行时容器：模型、存储、管道，全局懒加载
├── sse.py             SSE 事件协议（token/tool_call/tool_result/citation/proposal/done/error）
├── schemas.py         接口出入参
├── sessions.py        会话与消息（SQLite）
├── jobs.py            异步入库任务登记表（线程池 + 任务 ID）
├── api/               health / documents / ask / sessions 四组路由
└── rag/
    ├── loader.py      解析分流：pdf 逐页、md/txt 整篇、jsonl 一行一条
    ├── chunker.py     递归字符分块（中文句末标点优先）
    ├── embedder.py    bge-small-zh 本地 / 字节哈希兜底 / OpenAI 兼容 API
    ├── lexical.py     BM25 字面索引（对比实验与降级路径）
    ├── store.py       Chroma 向量库（cosine + upsert 幂等）
    ├── retriever.py   top-k 检索 + 阈值拦截
    ├── prompt.py      生成侧约束（仅依据上下文、不足则明说、必须带来源）
    ├── generator.py   OpenAI 兼容流式客户端（httpx 手写 SSE 解析）
    └── pipeline.py    入库与问答的编排
scripts/
├── ingest.py          命令行入库（--reset / --force / --rebuild）
├── eval_retrieval.py  语义 vs 字面 对比 + 阈值校准，产出 eval/report.md
├── make_eval_pdf.py   生成 50 页中文评测 PDF
└── check_llm.py       LLM Key 冒烟测试
eval/                  评测语料（docs/ 三份样本 + 库内/库外问题集）
tests/                 pytest（默认用哈希向量 + 假 LLM，CI 不下载模型）
```

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 自检：Key 是否配、模型是什么、库里有几篇几块 |
| GET | `/api/health/embedder` | 真跑一次编码，确认向量模型可用 |
| POST | `/api/documents` | 上传入库（≤10MB；>2MB 转异步，202 + `job_id`） |
| GET | `/api/documents` | 文档清单（含块数、页数） |
| DELETE | `/api/documents/{doc_id}` | 删该文档的全部向量 |
| GET | `/api/jobs/{job_id}` | 异步入库任务状态 |
| POST | `/api/knowledge/reset` | 清库重建（换 embedding 模型后必做） |
| POST | `/api/ask` | **SSE 流式问答** |
| GET | `/api/sessions` | 会话历史列表 |
| GET | `/api/sessions/{id}` | 单个会话（含消息与引用） |
| DELETE | `/api/sessions/{id}` | 删除会话 |

### SSE 事件协议

```
event: citation
data: {"index":1,"source":"手册.md","source_type":"md","doc_id":"...","chunk_index":3,"page":null,"score":0.71,"snippet":"..."}

event: token
data: {"text":"分块默认 "}

event: done
data: {"session_id":"...","message_id":"...","citations":[...],"fallback":"kb","hit_count":3,"latency_ms":842}
```

`fallback` 取值：`kb`（命中知识库）、`refuse`（模式 A 拒答）、`bare`（模式 B 裸答并标注）、
`web`（模式 C 联网搜索，**本阶段未接入**，会额外发一条 `error` 说明）。
错误一律用 `event: error` + `{code, message}`，HTTP 状态码保持 200 —— 前端因此只需要一套流解析逻辑。

## 评测（第十阶段 10.2 / 10.3 的证据链）

```powershell
.venv\Scripts\python scripts\download_model.py         # 拉 bge-small-zh（走 hf-mirror）
.venv\Scripts\python scripts\make_eval_pdf.py          # 生成 50+ 页中文 PDF（验收用）
.venv\Scripts\python scripts\ingest.py eval/docs       # 样本入库
.venv\Scripts\python scripts\eval_retrieval.py         # 语义 vs 字面 + 阈值校准 → eval/report.md
```

评测集：库内 16 题（其中 **4 题是口语化改写**，专门用来暴露"只会字面匹配"的短板）、
库外 12 题（6 个完全无关 + 6 个"同领域但库里没有"）。
最后那 6 题是关键——它们会拿到中等偏高的相似度，单靠阈值拦不住，
必须靠生成侧的拒答约束兜底；报告里会把这个差额如实写出来。

### 实测结论（3 篇 / 22 块 / 2026-09-17，完整数据见 `eval/report.md`）

| 策略 | recall@1 | recall@4 | 改写题 recall@1 | MRR | 校准阈值 |
| --- | --- | --- | --- | --- | --- |
| 语义 bge-small-zh | 0.562 | 0.938 | 0.25 | 0.714 | 0.4 |
| 字面 BM25 | 0.750 | 1.000 | 0.50 | 0.875 | 0.6 |

- **这份小语料上字面方案更准**（含改写题），原因是语料只有 22 块、问题与文档仍共享实词；
  语义模型的分数被压在 0.4~0.6，区分度有限。**没有把它写成"语义更好"**——那是指标造假。
- **两条路都留**：语义是生产默认（更抗改写与跨段关联，只是这份语料量不出来），
  BM25 常驻用于对比、降级与单测。
- **阈值拦不住的那部分靠生成侧**：库内最低分 0.395 < 库外最高分 0.646，数学上就不存在
  一条"库内不漏且库外不放过"的线；所以 prompt 里写死"上下文不足就明说"，
  并且引用来源直接展示给用户判断。

## 设计上值得一提的几处

- **幂等**：`doc_id = sha256(内容)[:16]`，内容没变直接跳过；同名来源重传先清旧向量，不会新旧并存。
- **阈值分流**：语义用余弦相似度（绝对量纲，校准值 0.4）；BM25 用固定尺度压缩 `raw/(raw+8)`（独立阈值 0.6）
  ——**不能**按每次查询的最高分归一化，那样第一名恒为 1.0，阈值永远拦不住东西（评测第一版就是这么暴露的）。
- **上传留原文**：原文件留在 `data/uploads/`，换分块参数或换模型可 `--rebuild` 重建，不用重传。
- **不静默降级**：配了 `bge` 却没有依赖时，`/api/health` 会带 `degraded_reason` 明说，而不是悄悄换成哈希向量。
- **流式接口不检查 Key**：未配 Key 时也返回 200，把 `error` 事件发在流里，前端只有一条错误路径。

## 后续阶段

- 第十一阶段：ReAct 循环 + 工具注册表（`search_knowledge` / `task_crud` / `get_date` / `get_weather`），复用同一套 SSE 事件。
- 第十二阶段：晨报生成、主动提醒，复用前端的提醒体系。
- 第十三阶段：本地文件操作（`proposal` 事件 + 前端 `BaseDiffCard` 确认）。
