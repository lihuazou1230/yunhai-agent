# 第十一阶段验收报告（Agent 内核）

- 生成时间：2026-09-19 09:50
- 模型：`deepseek-chat`
- 工具：`search_knowledge`, `get_date`, `task_crud`, `get_weather`, `demo_boom`（客户端执行：task_crud, get_weather）
- 知识库：3 篇 / 30 块
- **说明**：CLI 里没有浏览器，client 工具由脚本里的假前端执行（任务存内存、天气给假缓存）；浏览器里那份真实执行由前端用例 `src/agent/clientTools.spec.ts` 覆盖。

## 一、验收清单

| 验收项 | 结果 | 判定依据 | 细节 |
| --- | --- | --- | --- |
| 自主查知识库 | ✅ | agent 自己决定调用 search_knowledge（不是后端预先检索），并带引用回答 | 引用 4 条，kind=kb |
| 答案正确 | ✅ | 答案包含库里的关键事实（500 / 80） | 默认块长 500 个字符，重叠 80 个字符，用递归字符切分 [1]。 依据《RAG技术笔记.md》：分隔符按「空行 → |

## 二、逐用例的工具轨迹与回答

### kb_search：分块默认的块长和重叠是多少？

- 状态 `ok` · kind `kb` · fallback `kb` · 轮数 1 · tokens 3506 · 引用 4 条 · 1939ms · 回环 0 次
- 🔧 `search_knowledge`（server）参数 `{"query": "分块默认块长和重叠"}`
- ✅ `search_knowledge` → 命中 4 块
- 引用：1.RAG技术笔记.md, 2.RAG技术笔记.md, 3.RAG技术笔记.md, 4.RAG技术笔记.md

> 默认块长 500 个字符，重叠 80 个字符，用递归字符切分 [1]。 依据《RAG技术笔记.md》：分隔符按「空行 → 换行 → 中文句末标点 → 逗号 → 空格 → 单字」逐级降级 [1]。
