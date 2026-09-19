# yunhai-agent · 云海工作台 AI 后端

[![CI](https://github.com/lihuazou1230/yunhai-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/lihuazou1230/yunhai-agent/actions/workflows/ci.yml)

云海工作台（Vue 3 前端，仓库 [`yunhai-workspace`](https://github.com/lihuazou1230/yunhai-workspace)）的 Python 后端。
两个仓库从第十阶段开始就是**独立提交**的：前端只做消费，模型 Key 与向量库都留在这一层。

> 当前进度：**第十一阶段 · Agent 内核** 已实现（可插拔工具注册表 / 带护栏的 ReAct 循环 /
> 客户端工具回环 / 轻量对话兜底）。第十阶段的知识库在这里变成了 agent 的**一个工具**，
> 两条策略共存：`/api/ask` 带 `strategy=agent|rag`。

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

### 跑检查（与 CI 同一套命令）

```powershell
.venv\Scripts\python -m ruff check .          # lint
.venv\Scripts\python -m pytest -q             # 140 例单测：哈希向量 + 假 LLM，不下载模型
```

CI（`.github/workflows/ci.yml`）装的是 `requirements-dev.txt`——**刻意不含 torch / sentence-transformers**：
单测全程用零依赖哈希向量与假 LLM，把 400MB+ 的运行时拖进每次流水线只会让它慢三分钟，
而测的东西一点没变。真机要跑 bge 语义检索时仍按 `requirements.txt` 装。

## 部署到自有服务器（IIS 同源子应用 `/yhai`）

线上页面 `http://124.220.159.58/workspace/` 要能用 AI 助手，agent 就得跑在**服务器**上：
页面里配 `127.0.0.1:8000` 指的是**访问者自己的电脑**，而且 Chrome 142+ 的
[Local Network Access](https://developer.chrome.google.cn/blog/local-network-access?hl=zh-tw)
不允许公网 HTTP 页面访问回环地址（连申请权限的资格都没有）。
正解是复用 80 端口，把它挂成同源子路径。

```powershell
# 本机打部署包 → RDP 拖进服务器 → 右键 install.ps1 运行（管理员）
powershell -NoProfile -ExecutionPolicy Bypass -File deploy\build-agent-package.ps1
```

- 形态：`IIS 子应用 /yhai` →（URL Rewrite + ARR 反代）→ `127.0.0.1:8000` 的 uvicorn（计划任务常驻）
- 服务器上**不装 torch**：走 `requirements-server.txt`（去掉 sentence-transformers）+ `EMBEDDER=api`（SiliconFlow bge-m3）
- **不用 HttpPlatformHandler**：它 v1.2 自带 8KB 输出缓冲且无法关闭，会把 SSE 憋成"一次性吐"
- **Server 2012 R2 上 Python 最高 3.12**（3.13+ 要求 Win10+），且需要 UCRT
- 挂在子路径下由 `AGENT_URL_PREFIX` 剥前缀（`app/url_prefix.py`，纯 ASGI 中间件，不破坏流式）；本地不设 = 空操作

完整步骤、验收清单（含 SSE 端到端实测）与排障表见 [`deploy/部署说明.md`](deploy/部署说明.md)。

## 目录结构

```
app/
├── main.py            FastAPI 应用装配（CORS / 异常翻译 / 路由 / 子路径前缀）
├── config.py          配置（.env -> Settings），Key 只在这里出现
├── url_prefix.py      挂在子路径下时的前缀剥离（纯 ASGI 中间件，不缓冲流式响应）
├── runtime.py         运行时容器：模型、存储、管道、工具，全局懒加载
├── sse.py             SSE 事件协议（token/tool_call/tool_result/citation/proposal/done/error）
├── schemas.py         接口出入参
├── sessions.py        会话与消息（SQLite）
├── jobs.py            异步入库任务登记表（线程池 + 任务 ID）
├── api/               health / documents / ask / sessions 四组路由
├── agent/             第十一阶段：Agent 内核
│   ├── tools.py       工具注册表（名称+描述+JSON Schema+执行器，server|client）
│   ├── builtin.py     内置工具：search_knowledge / get_date / task_crud / get_weather
│   ├── react.py       ReAct 循环 + 护栏三件套 + 熔断 + 优雅收场
│   ├── runs.py        待续跑状态（client 工具回环用，SQLite + TTL）
│   └── prompts.py     Agent 系统提示（工具规矩 + 兜底策略 + 闲聊规矩）
└── rag/
    ├── loader.py      解析分流：pdf 逐页、md/txt 整篇、jsonl 一行一条
    ├── chunker.py     递归字符分块（Markdown 小节硬断 + 中文句末标点优先）
    ├── embedder.py    bge-small-zh 本地 / 字节哈希兜底 / OpenAI 兼容 API
    ├── lexical.py     BM25 字面索引（对比实验与降级路径）
    ├── store.py       Chroma 向量库（cosine + upsert 幂等）
    ├── retriever.py   top-k 检索 + 阈值拦截
    ├── prompt.py      生成侧约束（仅依据上下文、不足则明说、必须带来源）
    ├── generator.py   OpenAI 兼容流式客户端（含 tool_calls 分片累积）
    └── pipeline.py    入库与 RAG 直答（strategy=rag）的编排
scripts/
├── ingest.py          命令行入库（--reset / --force / --rebuild）
├── eval_retrieval.py  语义 vs 字面 对比 + 阈值校准，产出 eval/report.md
├── eval_answers.py    端到端答案验收（需真 Key）→ eval/answer_report.md
├── eval_agent.py      第十一阶段验收（工具轨迹 + 护栏）→ eval/agent_report.md
├── eval_hybrid.py     RRF 融合实验（负结论，留作可复现记录）
├── make_eval_pdf.py   生成 50 页中文评测 PDF
├── download_model.py  拉 bge 模型（走 hf-mirror，自动关掉 xet）
├── check_env_file.py  .env 行尾/字段自检（CR-only 行尾会坑 PowerShell）
└── check_llm.py       LLM Key 冒烟测试
eval/                  评测语料（docs/ 三份样本 + 库内/库外问题集）与三份报告
deploy/                服务器部署（IIS 同源子应用 /yhai）：install.ps1 + web.config + 打包脚本 + 部署说明.md
requirements-server.txt  服务器形态依赖（不含 torch，向量走 API）
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
| POST | `/api/ask` | **SSE 流式问答**（`strategy=agent`（默认）走 ReAct，`strategy=rag` 走第十阶段的直答链路） |
| POST | `/api/ask/resume` | 前端执行完 client 工具后回来续跑（同一个 SSE 协议） |
| GET | `/api/sessions` | 会话历史列表 |
| GET | `/api/sessions/{id}` | 单个会话（含消息与引用） |
| DELETE | `/api/sessions/{id}` | 删除会话 |

### SSE 事件协议

```
event: citation
data: {"index":1,"source":"手册.md","source_type":"md","doc_id":"...","chunk_index":3,"page":null,"score":0.71,"snippet":"..."}

event: token
data: {"text":"分块默认 "}

event: tool_call
data: {"id":"call_0","name":"task_crud","arguments":{"action":"create","title":"交周报"},"executor":"client","status":"awaiting_client"}

event: tool_result
data: {"id":"call_0","name":"task_crud","ok":true,"summary":"已创建任务「交周报」（id=t3，截止 2026-09-20）","error":"","citations":[],"meta":{}}

event: done
data: {"session_id":"...","run_id":"...","message_id":"...","citations":[...],"fallback":"kb","kind":"kb",
       "hit_count":3,"latency_ms":842,"status":"ok","rounds":1,"tokens":2103,"tools":[...],"pending":[]}
```

`fallback` 取值：`kb`（依据知识库）、`tool`（依据工具结果，如任务/日期/天气）、`chat`（轻量对话）、
`guardrail`（撞上限后收口），以及第十阶段的兜底三模式 `refuse` / `bare` / `web`
（知识库被查过但没命中时会如实显示成这三种之一，而不是假装只是一次普通工具调用）。
`status` 取值：`ok` / `awaiting_client`（等前端执行工具）/ `guardrail`。
错误一律用 `event: error` + `{code, message}`，HTTP 状态码保持 200 —— 前端因此只需要一套流解析逻辑。

## 🧠 Agent 内核（第十一阶段）

知识库在这一阶段**降级成了 agent 的一个工具**：问答不再固定"先检索再生成"，而是模型自己决定要不要查。

### 工具注册表（可插拔）

```
tool = 名称 + 描述 + JSON Schema + 执行器（server | client）
```

| 工具 | executor | 干什么 |
| --- | --- | --- |
| `search_knowledge` | server | 知识库检索（语义 + 字面两路并集，命中即产出 citation） |
| `get_date` | server | 当前日期 / 星期 / 时间（"今天/明天"的计算都先问它） |
| `task_crud` | **client** | 任务的查 / 建 / 改 / 完成 / 删 / 归档 |
| `get_weather` | **client** | 读用户本机已缓存的天气（不新增数据源） |

加一个工具只要 `registry.register(Tool(...))`；服务端工具缺 handler 会在注册时直接报错。

### 为什么 task_crud / get_weather 是「客户端工具」

任务与天气缓存只存在于**用户的工作台里**（localStorage + Supabase + Tauri 数据目录），
后端没有、也不该复制一份真相。所以做成两段式：

```
后端：模型决定调用 task_crud → 发 tool_call(executor=client, status=awaiting_client)
      → 把这一轮的 messages 存进 agent_runs（SQLite + TTL）→ 这一条流先结束
前端：在工作台数据上真的执行 → POST /api/ask/resume 带上观察结果
后端：把观察结果作为 tool 消息回填 → 循环继续 → 最终答案
```

这样后端始终"无状态地想"，前端始终是数据的唯一真相；事件流仍是同一条 SSE 协议，
前端只有一套解析器。

### 护栏三件套 + 两条工程护栏

| 护栏 | 默认 | 撞上之后 |
| --- | --- | --- |
| 最大轮数 | 8 | 摘掉工具、再问一次，让模型给诚实的收尾 |
| token 预算 | 12000（流里 usage 累计，缺失按字符估） | 同上 |
| 工具超时 | 检索 15s，其余 10s | 超时变成一条**观察结果**，循环继续 |
| 同一工具连续失败熔断 | 3 次 | 同上（换个参数重试合理，同样的失败重试 3 次就是空转） |
| 整轮墙钟超时 | 120s | 同上 |

**撞护栏 ≠ 抛错**：一律"把工具摘掉再问一次"，让模型说明完成了什么、卡在哪里——
用户看到的是一段完整回答，而不是断流或死循环（验收项 3 就是这么测的）。

### 轻量对话兜底（11.4）

没有命中任何工具时就是普通对话（`kind=chat`）。同时**用户那句和回答那句都会被标上 `chat`**，
下一轮回填上下文时整段剔除——"闲聊不进任务上下文"因此是代码行为，不是口头承诺。

### 本机验收（`scripts/eval_agent.py`，真 Key，2026-09-19）

| 验收项 | 结果 |
| --- | --- |
| "帮我加个明天交周报的任务"一句话落库 | ✅ 调 `task_crud(create)`，`due_date` 被换算成 `2026-09-20` |
| "今天还剩哪些活"回答准确 | ✅ 调 `task_crud(list)`，回答里出现工作台里真实的任务标题 |
| agent 自主决定"先查知识库再回答" | ✅ 自己调 `search_knowledge`，4 条引用，`kind=kb` |
| 必失败任务优雅收场 | ✅ 工具连续报错后 1 轮内收手并如实说明，没有死循环 |
| 问日期 / 天气秒回 | ✅ `get_date` 1.3s；`get_weather` 读缓存 1.1s |
| 闲聊不产生记忆残留 | ✅ 不调用工具、两侧都打 `chat` 标记 |

完整工具轨迹与回答原文见 `eval/agent_report.md`。

## 评测（第十阶段 10.2 / 10.3 的证据链）

```powershell
.venv\Scripts\python scripts\download_model.py         # 拉 bge-small-zh（走 hf-mirror）
.venv\Scripts\python scripts\make_eval_pdf.py          # 生成 50+ 页中文 PDF（验收用）
.venv\Scripts\python scripts\ingest.py eval/docs       # 样本入库
.venv\Scripts\python scripts\eval_retrieval.py --leak-cap 0.6   # 语义 vs 字面 + 阈值校准 → eval/report.md
.venv\Scripts\python scripts\eval_hybrid.py            # 第三条路（RRF 融合）值不值得上：不值得，见脚本头
.venv\Scripts\python scripts\eval_answers.py           # 端到端答案验收（需真 Key）→ eval/answer_report.md
```

评测集：库内 16 题（其中 **4 题是口语化改写**，专门用来暴露"只会字面匹配"的短板）、
库外 12 题（6 个完全无关 + 6 个"同领域但库里没有"）。
最后那 6 题是关键——它们会拿到中等偏高的相似度，单靠阈值拦不住，
必须靠生成侧的拒答约束兜底；报告里会把这个差额如实写出来。

### 实测结论（3 篇 / 30 块 / 2026-09-17，完整数据见 `eval/report.md`）

| 策略 | recall@1 | recall@4 | 改写题 recall@1 | MRR | 校准阈值 |
| --- | --- | --- | --- | --- | --- |
| 语义 bge-small-zh | 0.750 | 0.938 | 0.25 | 0.844 | 0.4 |
| 字面 BM25 | 0.750 | 1.000 | 0.50 | 0.854 | 0.55 |

- **两者 recall@1 打平，BM25 在 recall@4 上更满**（16/16 vs 15/16）。原因很具体：这份语料只有 30 块、
  问题与文档仍共享实词；**没有把它写成"语义更好"**——那是指标造假。
- **分块粒度是先被数据打回一次才修对的**：第一版按固定长度切，一份手册里"AI 边界 / 快捷键 / 故障排查"
  三个短小节被合并成一块，语义被平均掉——问"深色模式下图表发白是什么原因"只拿到 0.395 分（阈值 0.4）被拒答，
  而答案原文就在那一节里。改成**按 Markdown 小节硬断**（`app/rag/chunker.py`）后块数 23 → 30，
  语义 recall@1 从 0.562 提到 0.750。
- **两条路都留**：语义是生产默认（更抗改写与跨段关联），BM25 常驻用于降级、对比与单测；
  前端可以在「语义检索 / 字面检索」之间切换，方便现场对比。
- **第三条路量过了，没有上**：语义 + 字面的 RRF 融合（`scripts/eval_hybrid.py`）recall@4 只有 15/16，
  不比单路最好成绩好——两种路子的排名高度重叠，融合反而会把稳的题拉低。结论留在脚本里可复现。
- **阈值是偏召回的**：库内最低分 0.441 < 库外最高分 0.671，数学上不存在"库内不漏且库外不放过"的线。
  取 0.4 / 0.55 能保住全部 16 道库内题，代价是 12 道库外题漏网 7 道——敢这么选是因为端到端实测里
  漏网的题**全部由模型自己说明"知识库里没有"，没有一条编造**（见下）。阈值在这里的角色是省 token，不是最后一道防线。

### 端到端答案验收（`eval/answer_report.md`，需要真 Key，会花 token）

```powershell
.venv\\Scripts\\python scripts\\ingest.py eval\\docs --reset
.venv\\Scripts\\python scripts\\eval_answers.py          # 16 库内 + 12 库外，逐题留答案原文
```

| 指标 | 结果 |
| --- | --- |
| 库内：命中知识库并带引用 | **16/16** |
| 库内：检索到的块里含期望事实 | 13/16 |
| 库内：答案里出现期望事实 | **13/16**（直问直答 **11/12**、口语化改写 2/4） |
| 库外：没有编造 | **12/12**（5 题被阈值拦下、7 题由模型自己说明"库里没有"） |

三个口径分开报是有意的：检索没命中时模型会如实说"知识库里没有"，这不算编造，
但对库内题就是**没答出**——只报"带引用"会把这类题算成成功。验收要求（10 题 ≥8 正确且带引用）在
直问直答上是 11/12。

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
