# 第十一阶段验收报告（Agent 内核）

- 生成时间：2026-09-19 09:34
- 模型：`deepseek-chat`
- 工具：`search_knowledge`, `get_date`, `task_crud`, `get_weather`, `demo_boom`（客户端执行：task_crud, get_weather）
- 知识库：3 篇 / 30 块
- **说明**：CLI 里没有浏览器，client 工具由脚本里的假前端执行（任务存内存、天气给假缓存）；浏览器里那份真实执行由前端用例 `src/agent/clientTools.spec.ts` 覆盖。

## 一、验收清单

| 验收项 | 结果 | 判定依据 | 细节 |
| --- | --- | --- | --- |
| 一句话落库 | ✅ | agent 调用 task_crud(create) 且任务真的进了工作台数据 | 新建 [{'id': 't3', 'title': '交周报', 'due_date': '2026-09-20', 'priority': 'medium', 'status': 'active'}] |
| 日期换算 | ✅ | 「明天」被换算成绝对日期（YYYY-MM-DD）而不是原样透传 | due_date=['2026-09-20'] |
| 今天还剩哪些活 | ✅ | 回答里出现工作台里真实存在的任务标题 | 回答命中 ['写周报', '交季度报销'] |
| 自主查知识库 | ✅ | agent 自己决定调用 search_knowledge（不是后端预先检索），并带引用回答 | 引用 4 条，kind=kb |
| 答案正确 | ✅ | 答案包含库里的关键事实（500 / 80） | 默认块长 500 个字符，重叠 80 个字符，用递归字符切分 [1]。 依据《RAG技术笔记.md》：分隔符按「空行 → |
| 必失败任务优雅收场 | ✅ | 工具一直失败时在轮数上限（8）内停下并诚实收尾 | 状态 ok、失败 1 次、轮数 1、回答 108 字 |
| 日期秒回 | ✅ | 调用 get_date 并在答案里给出今天的日期 | 1294ms |
| 天气读缓存 | ✅ | 调用 client 工具 get_weather，答案里出现缓存里的天气信息 | 上海今天晴，24°C（体感 26°C，湿度 60%），数据更新于 10:20。 |
| 闲聊无记忆残留 | ✅ | 闲聊轮不调用工具，且用户与回答两侧都被打上 kind=chat（不进任务上下文） | chat 标记 2 条 |

## 二、逐用例的工具轨迹与回答

### task_create：帮我加个明天交周报的任务

- 状态 `ok` · kind `tool` · fallback `tool` · 轮数 2 · tokens 4561 · 引用 0 条 · 2380ms · 回环 1 次
- 🔧 `get_date`（server）参数 `{}`
- 🔧 `task_crud`（client）参数 `{"action": "create", "title": "交周报", "due_date": "2026-09-20", "priority": "medium"}`
- ✅ `get_date` → 今天是 2026-09-19（星期六），当前时间 09:34，ISO 周 2026-W38。

> 已加好：任务「交周报」，截止 2026-09-20（明天，周日），优先级默认中等。需要改优先级或日期跟我说。

### task_list：今天还剩哪些活？

- 状态 `ok` · kind `tool` · fallback `tool` · 轮数 1 · tokens 3031 · 引用 0 条 · 1502ms · 回环 1 次
- 🔧 `task_crud`（client）参数 `{"action": "list", "filter": "today"}`

> 今天还剩 3 条未完成任务： 1. 写周报 — 截止今天，高优先级 2. 交季度报销 — 截止今天，中优先级 3. 交周报 — 截止 2026-09-20，中优先级 前两条今天到期，建议优先处理。

### kb_search：分块默认的块长和重叠是多少？

- 状态 `ok` · kind `kb` · fallback `kb` · 轮数 1 · tokens 3506 · 引用 4 条 · 1829ms · 回环 0 次
- 🔧 `search_knowledge`（server）参数 `{"query": "分块默认块长和重叠"}`
- ✅ `search_knowledge` → 命中 4 块
- 引用：1.RAG技术笔记.md, 2.RAG技术笔记.md, 3.RAG技术笔记.md, 4.RAG技术笔记.md

> 默认块长 500 个字符，重叠 80 个字符，用递归字符切分 [1]。 依据《RAG技术笔记.md》：分隔符按「空行 → 换行 → 中文句末标点 → 逗号 → 空格 → 单字」逐级降级 [1]。

### guardrail：用 demo_boom 查演示数据；如果失败就再试，一直试到查到结果为止

- 状态 `ok` · kind `tool` · fallback `tool` · 轮数 1 · tokens 2935 · 引用 0 条 · 1918ms · 回环 0 次
- 🔧 `demo_boom`（server）参数 `{}`
- ❌ `demo_boom` → 演示用的必失败工具：外部接口 503

> 我先调用一次看看。这个工具是故意做成必定失败的（返回 503），重试多少次结果都一样，所以我不会一直试下去。 如果你确实需要演示数据，可以告诉我具体想看什么，我看看知识库里有没有；或者去仪表板那边确认下数据源状态。

### date：现在几点了？

- 状态 `ok` · kind `tool` · fallback `tool` · 轮数 1 · tokens 2871 · 引用 0 条 · 1294ms · 回环 0 次
- 🔧 `get_date`（server）参数 `{}`
- ✅ `get_date` → 今天是 2026-09-19（星期六），当前时间 09:34，ISO 周 2026-W38。

> 现在是 2026 年 9 月 19 日（星期六）09:34。

### weather：今天天气怎么样？

- 状态 `ok` · kind `tool` · fallback `tool` · 轮数 1 · tokens 2880 · 引用 0 条 · 1256ms · 回环 1 次
- 🔧 `get_weather`（client）参数 `{}`

> 上海今天晴，24°C（体感 26°C，湿度 60%），数据更新于 10:20。

### chat：讲个冷笑话吧

- 状态 `ok` · kind `chat` · fallback `chat` · 轮数 0 · tokens 1421 · 引用 0 条 · 966ms · 回环 0 次

> 为什么程序员总分不清万圣节和圣诞节？ 因为 Oct 31 等于 Dec 25。 （顺带说明：这类闲聊我不会记到你的工作台里。）
