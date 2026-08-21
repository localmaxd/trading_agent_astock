# TradingAgents 项目架构指南

> TradingAgents: Multi-Agents LLM Financial Trading Framework (v0.2.4)
> 基于 LangGraph 的多智能体 LLM 金融交易框架

---

## 1. 项目概览

TradingAgents 是一个模仿真实交易公司运作的多智能体金融交易框架。它部署了多个由大语言模型（LLM）驱动的专业智能体，从基本面分析师、情绪专家、技术分析师到交易员、风险管理团队，协同评估市场状况并做出交易决策。

### 1.1 交互入口（2026-08 新增网页版）

- **CLI**：`tradingagents analyze`（Typer 交互式终端界面，实时刷新 agent 状态 / 工具活动 / token 统计）
- **网页版**：`api_server.py`（FastAPI）提供 Web UI + SSE 实时流，**注重事实可观测性**：
  - 输入**标的 + 日期**一键开启会话（分析师/深度可配）
  - 实时展示：Agent 进度、**工具执行过程**（开始/完成/耗时/数据量）、**LLM 调用与 token 消耗**
  - **节点级 LLM 可观测性**（调试用）：每个 LLM 调用归属到**图节点**（`langgraph_node`，如 `Analyst Team` / `Trader`），展示**输入/输出摘要**（点击展开）、tokens（↑↓）与耗时；**节点统计**面板聚合每节点的 LLM 次数 / Tokens / 耗时；进度面板实时显示"当前执行到图的哪个节点"
  - **事实校验面板**：每个分析师的逐条结论校验（Claim / 类型 fact|calculation / 报告值 vs 期望值 / 通过 / 差异原因 / 重试轮次 / 反馈）、结构化来源声明（claim ← source_tool）
  - 标的代码归一化事件（TickerGuard）、最终决策与完整报告

实现说明：SSE 使用 `stream_mode=["updates","messages"]` 双模式——updates 提供节点完成时序（节点耗时），messages 携带 `(message, langgraph_node)` 元数据；LLM 输入/输出/token 由 `SSEStatsHandler` 回调按 run_id（流式 chunk 无 run_id 时按完成顺序 FIFO）配对后与节点名合并为 `llm_call` 事件。

启动方式：
```bash
.venv/bin/python -m uvicorn api_server:app --host 0.0.0.0 --port 8007
# 打开 http://localhost:8007
```
可选环境变量：`WEB_LLM_PROVIDER`（默认 deepseek）、`WEB_QUICK_MODEL`/`WEB_DEEP_MODEL`（默认 deepseek-v4-flash）、`WEB_SEARCH_ENABLED`（默认 false）

### 核心特性
- **多智能体协作**：分析师团队 → 研究团队 → 交易员 → 风险管理 → 投资组合经理
- **多 LLM 提供商支持**：OpenAI、Google (Gemini)、Anthropic (Claude)、xAI (Grok)、DeepSeek、Qwen、GLM、OpenRouter、Ollama、Azure
- **结构化输出**：Research Manager、Trader、Portfolio Manager 使用 Pydantic Schema 输出
- **持久化与恢复**：决策日志（自动）+ LangGraph 检查点恢复（可选）
- **双数据源**：yfinance（免费）和 Alpha Vantage（需 API Key）
- **交互式 CLI**：基于 Rich/Typer 的终端界面

---

## 2. 目录结构

```
TradingAgents/
├── main.py                          # 入口示例：程序化调用 TradingAgentsGraph
├── pyproject.toml                   # 包配置 (setuptools)
├── docker-compose.yml               # Docker 支持
├── cli/                             # 交互式命令行界面
│   ├── main.py                      # Typer CLI 主入口（analyze 命令）
│   ├── config.py                    # CLI 配置模型
│   ├── models.py                    # CLI 枚举类型
│   ├── utils.py                     # CLI 工具函数
│   ├── announcements.py             # 远程公告获取
│   ├── stats_handler.py             # LLM/Tool 调用统计回调
│   └── static/welcome.txt           # ASCII 欢迎图
├── tradingagents/                   # 核心包
│   ├── default_config.py            # DEFAULT_CONFIG 默认配置
│   ├── graph/                       # LangGraph 图编排层
│   │   ├── trading_graph.py         # TradingAgentsGraph：主 orchestrator
│   │   ├── setup.py                 # GraphSetup：父图编排 5 个阶段子图
│   │   ├── subgraphs/               # 多子图架构（每个阶段独立编译的子图）
│   │   │   ├── states.py            # 各子图状态 Schema（AgentState 的严格子集）
│   │   │   ├── analyst_team.py      # 阶段1：分析师团队子图
│   │   │   ├── research_debate.py   # 阶段2：多空研究辩论子图
│   │   │   ├── trader.py            # 阶段3：交易员子图
│   │   │   ├── risk_debate.py       # 阶段4：风险辩论子图
│   │   │   └── portfolio_manager.py # 阶段5：投资组合经理子图
│   │   ├── conditional_logic.py     # ConditionalLogic：条件边路由
│   │   ├── propagation.py           # Propagator：状态初始化 + 图参数
│   │   ├── reflection.py            # Reflector：事后反思生成
│   │   ├── signal_processing.py     # SignalProcessor：信号解析（评分提取）
│   │   └── checkpointer.py          # SQLite 检查点管理（恢复/清理）
│   ├── agents/                      # 智能体定义
│   │   ├── __init__.py              # 统一导出
│   │   ├── schemas.py               # Pydantic 结构化输出 Schema
│   │   ├── analysts/                # 分析师团队
│   │   │   ├── market_analyst.py
│   │   │   ├── social_media_analyst.py
│   │   │   ├── news_analyst.py
│   │   │   └── fundamentals_analyst.py
│   │   ├── researchers/             # 研究团队
│   │   │   ├── bull_researcher.py
│   │   │   └── bear_researcher.py
│   │   ├── trader/                  # 交易员
│   │   │   └── trader.py
│   │   ├── risk_mgmt/               # 风险管理团队
│   │   │   ├── aggressive_debator.py
│   │   │   ├── conservative_debator.py
│   │   │   └── neutral_debator.py
│   │   ├── managers/                # 管理者
│   │   │   ├── research_manager.py
│   │   │   └── portfolio_manager.py
│   │   └── utils/                   # 智能体工具与状态
│   │       ├── agent_states.py      # AgentState / InvestDebateState / RiskDebateState
│   │       ├── agent_utils.py       # 工具导入 + 语言指令 + 消息清理
│   │       ├── structured.py        # 结构化输出绑定与降级
│   │       ├── memory.py            # TradingMemoryLog 决策日志
│   │       ├── rating.py            # 评分解析辅助
│   │       ├── core_stock_tools.py
│   │       ├── technical_indicators_tools.py
│   │       ├── fundamental_data_tools.py
│   │       └── news_data_tools.py
│   ├── dataflows/                   # 数据流层
│   │   ├── interface.py             # 统一路由接口（工具 → 供应商实现）
│   │   ├── config.py                # 数据流全局配置
│   │   ├── utils.py                 # 通用工具（safe_ticker_component 等）
│   │   ├── y_finance.py             # yfinance 实现
│   │   ├── yfinance_news.py         # yfinance 新闻实现
│   │   ├── alpha_vantage.py         # Alpha Vantage 统一入口
│   │   ├── alpha_vantage_stock.py
│   │   ├── alpha_vantage_indicator.py
│   │   ├── alpha_vantage_fundamentals.py
│   │   ├── alpha_vantage_news.py
│   │   ├── alpha_vantage_common.py  # 公共工具 + 限流错误
│   │   └── stockstats_utils.py      # stockstats 指标计算
│   └── llm_clients/                 # LLM 客户端层
│       ├── factory.py               # create_llm_client() 工厂
│       ├── base_client.py           # BaseLLMClient 抽象基类
│       ├── openai_client.py         # OpenAI 兼容客户端（OpenAI/xAI/DeepSeek/Qwen/GLM/Ollama/OpenRouter）
│       ├── anthropic_client.py
│       ├── google_client.py
│       ├── azure_client.py
│       ├── model_catalog.py         # CLI 模型选项目录
│       └── validators.py            # 模型验证
├── tests/                           # 测试套件
│   ├── conftest.py
│   ├── test_structured_agents.py
│   ├── test_signal_processing.py
│   ├── test_model_validation.py
│   ├── test_safe_ticker_component.py
│   ├── test_checkpoint_resume.py
│   ├── test_ticker_symbol_handling.py
│   ├── test_google_api_key.py
│   ├── test_deepseek_reasoning.py
│   └── test_memory_log.py
└── scripts/
    └── smoke_structured_output.py
```

---

## 3. 架构分层

### 3.1 执行流程（LangGraph 工作流）

```
START
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    I. Analyst Team                               │
│  Market Analyst → Social Analyst → News Analyst → Fundamentals   │
│  (每个分析师调用 tools → 生成报告 → 清理消息)                      │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│                  II. Research Team                               │
│  Bull Researcher ↔ Bear Researcher (多轮辩论)                    │
│         ↓                                                        │
│  Research Manager (结构化输出: ResearchPlan)                     │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│                  III. Trader                                     │
│  Trader (结构化输出: TraderProposal)                             │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│                 IV. Risk Management                              │
│  Aggressive ↔ Conservative ↔ Neutral (多轮辩论)                  │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│              V. Portfolio Manager                                │
│  Portfolio Manager (结构化输出: PortfolioDecision) → END         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 状态定义 (`AgentState`)

基于 LangGraph 的 `MessagesState`，扩展以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `company_of_interest` | str | 分析的股票代码 |
| `trade_date` | str | 交易日期 |
| `sender` | str | 发送消息的 Agent |
| `market_report` | str | 市场分析师报告 |
| `sentiment_report` | str | 社交媒体情绪报告 |
| `news_report` | str | 新闻分析报告 |
| `fundamentals_report` | str | 基本面分析报告 |
| `investment_debate_state` | InvestDebateState | 研究团队辩论状态 |
| `investment_plan` | str | 研究经理投资计划 |
| `trader_investment_plan` | str | 交易员计划 |
| `risk_debate_state` | RiskDebateState | 风险管理辩论状态 |
| `final_trade_decision` | str | 最终交易决策 |
| `past_context` | str | 记忆日志注入的上下文 |

### 3.3 核心类职责

| 类 | 文件 | 职责 |
|----|------|------|
| `TradingAgentsGraph` | `trading_graph.py` | 主编排器：初始化 LLM、工具节点、图组件；管理 propagate() 生命周期 |
| `GraphSetup` | `setup.py` | 构建 LangGraph `StateGraph`：添加节点、定义边、配置条件路由 |
| `ConditionalLogic` | `conditional_logic.py` | 条件边逻辑：分析师工具调用循环、辩论轮数控制、风险讨论流转 |
| `Propagator` | `propagation.py` | 状态初始化（`create_initial_state`）和图调用参数组装 |
| `Reflector` | `reflection.py` | 基于收益结果生成反思文本（2-4句简洁 prose） |
| `SignalProcessor` | `signal_processing.py` | 从 Portfolio Manager 决策中提取 5 级评分 |
| `TradingMemoryLog` | `memory.py` | 追加式 Markdown 决策日志：存储 → 延迟解析 → 批量更新 |

---

## 4. 关键模块详解

### 4.1 LLM 客户端 (`tradingagents/llm_clients/`)

**工厂模式**：`create_llm_client(provider, model, base_url, **kwargs)`

- **OpenAI 兼容组**：`openai`, `xai`, `deepseek`, `qwen`, `glm`, `ollama`, `openrouter` → `OpenAIClient`
- **独立实现**：`anthropic` → `AnthropicClient`, `google` → `GoogleClient`, `azure` → `AzureOpenAIClient`

**Thinking 配置**：
- Google: `google_thinking_level` ("high", "minimal")
- OpenAI: `openai_reasoning_effort` ("medium", "high", "low")
- Anthropic: `anthropic_effort` ("high", "medium", "low")

### 4.2 数据流层 (`tradingagents/dataflows/`)

**供应商路由系统** (`interface.py`)：
- 支持 **类别级** 和 **工具级** 供应商配置
- 自动 fallback：当主供应商返回 `AlphaVantageRateLimitError` 时自动切换到备选供应商
- 4 大类别：`core_stock_apis`, `technical_indicators`, `fundamental_data`, `news_data`

**数据源对比**：

| 类别 | yfinance | Alpha Vantage |
|------|----------|---------------|
| 核心股价 | ✓ | ✓ |
| 技术指标 | ✓ (stockstats) | ✓ |
| 基本面 | ✓ | ✓ |
| 新闻 | ✓ | ✓ |
| 内部交易 | ✓ | ✓ |
| 成本 | 免费 | 需 API Key |

### 4.3 结构化输出 (`tradingagents/agents/schemas.py`)

三个决策智能体使用 Pydantic Schema 实现结构化输出：

1. **ResearchPlan** (`Research Manager`)：
   - `recommendation`: Buy / Overweight / Hold / Underweight / Sell
   - `rationale`: 辩论要点总结
   - `strategic_actions`: 具体执行步骤

2. **TraderProposal** (`Trader`)：
   - `action`: Buy / Hold / Sell
   - `reasoning`, `entry_price`, `stop_loss`, `position_sizing`

3. **PortfolioDecision** (`Portfolio Manager`)：
   - `rating`: 5 级评分
   - `executive_summary`, `investment_thesis`, `price_target`, `time_horizon`

每个 Schema 配有 `render_*` 函数，将结构化实例渲染回 Markdown，保持下游兼容性。

**降级机制** (`structured.py`)：
- 如果 `with_structured_output` 不支持 → 使用自由文本
- 如果结构化调用失败 → 自动重试一次自由文本

### 4.4 记忆系统 (`tradingagents/agents/utils/memory.py`)

**两阶段设计**：
- **Phase A (存储)**：`propagate()` 结束时追加 pending 条目到 `~/.tradingagents/memory/trading_memory.md`
- **Phase B (解析)**：下次运行同一 ticker 时，获取实际收益 → 生成反思 → 批量原子更新

**注入内容**：
- 最近 5 条同 ticker 历史决策 + 反思
- 最近 3 条跨 ticker 教训

**文件格式**：Append-only Markdown，HTML 注释分隔符 `<!-- ENTRY_END -->`

### 4.5 检查点恢复 (`tradingagents/graph/checkpointer.py`)

- 基于 `langgraph-checkpoint-sqlite` 的 `SqliteSaver`
- **每 ticker 独立 SQLite DB**：`~/.tradingagents/cache/checkpoints/<TICKER>.db`
- **线程 ID**：`SHA256(ticker:date)[:16]` 确保同 ticker+date 可恢复
- 成功完成后自动清理检查点

### 4.6 多子图架构 (`tradingagents/graph/subgraphs/`)

v0.2.4 起流水线从单一 StateGraph 重构为 **5 个独立编译的阶段子图**，由薄父图顺序编排：

```
分析师团队 → 多空研究辩论 → 交易员 → 风险辩论 → 投资组合经理
(Analyst Team) (Research Debate) (Trader) (Risk Debate) (Portfolio Manager)
```

**核心设计**：
- **子图即节点**：每个阶段是独立编译的 StateGraph，作为父图节点嵌入（LangGraph subgraph-as-node）
- **状态隔离**：每个子图声明自己的状态 Schema（`subgraphs/states.py`），是 `AgentState` 的**严格子集**——子图只能读写自己声明的通道，父图负责合并回流
- **条件逻辑复用**：`ConditionalLogic` 原样复用；辩论子图内 "Portfolio Manager" 路由键映射到 END（真正 PM 由父图执行）
- **检查点恢复**：子图内部的中断点同样记录，崩溃后可从任意子图内部恢复（见 `tests/test_subgraph_structure.py`）
- **公共 API 不变**：`TradingAgentsGraph.propagate()` / CLI / API server 无需任何改动

### 4.7 标的代码一致性守卫 (`tradingagents/graph/ticker_guard.py`)

进入**每一个子图之前**都有一个 `TickerGuard-<Stage>` 节点，保障分析的标的代码准确：

| 检验 | 行为 |
|------|------|
| **格式校验** | 非法代码（路径穿越、空白、中文、超长等）→ 立即中止并指明阶段 |
| **一致性校验** | 状态中的 `company_of_interest` 必须与运行锚定值 `input_ticker`（`propagate()` 入口捕获，不可被改写）一致；任何环节篡改标的 → 在进入下一子图前中止 |
| **归一化写回** | `600519` → `600519.SH`（裸 A 股代码补全，规则与 akshare 数据层一致）、`600519.sh` → `600519.SH`（大小写）、`cnc.to` → `CNC.TO`（通用符号大写）；归一化值写回状态，图内外（状态日志、记忆库）全程一致 |

- `propagate()` 入口即归一化（fail-fast），非法输入在进入图之前就报错
- 锚定字段 `input_ticker` 定义在 `AgentState`，由 `Propagator.create_initial_state` 初始化，任何子图都不会写它
- 覆盖测试：`tests/test_ticker_guard.py`（含"子图篡改标的 → 下一子图前被拦截"的集成测试）

### 4.8 网络搜索与事实确认（0820 feature）

**网络搜索（可选，默认关闭）**——`tradingagents/agents/utils/web_search_tool.py`：

- 包装 DeepSeek Responses API 的**服务端内置 web_search 工具**（`/responses` 端点，`tools=[{"type":"web_search"}]`）
- 与主 LLM provider 解耦：使用独立 API Key（`web_search_api_key` 或 `DEEPSEEK_API_KEY`），本地 Qwen 主 LLM 也能使用
- 智能体**自主规划**：fundamentals / technical / game_theory 分析师（`web_search_analysts` 可配）的工具列表中可选 `web_search_tool`，可自主决定是否去东方财富（eastmoney.com）等网站补充信息
- 配置：`web_search_enabled`（总开关，默认 False）、`web_search_model`（默认 deepseek-chat）

**事实确认节点（默认开启）**——`tradingagents/graph/fact_checker.py`：

- 在 fundamentals / technical / game_theory **输出节点之后**插入 `FactChecker-<分析师>` 节点：
  1. **代码二次取数**：重新调用该分析师自己的工具（复算/复现）+ 按**数据重叠面**挑选的交叉来源工具（同一事实在多个独立端点出现时方可交叉，如 game_theory ↔ tool_risk 的内部人交易、technical ↔ tool_game_theory 的资金流向、fundamentals ↔ tool_news_sentiment 的公告）
  2. **多轮联网搜索**（`web_search_enabled` 开启时）：搜索规划 LLM 先分析报告/claims 生成 `VerificationSearchPlan`（0-N 条 query）→ 代码逐条执行 `web_search_tool`（单条失败自动重试一次）→ 基于已有结果进行第 2 轮补充规划（`verify_search_max_queries`=4 条/轮、`verify_search_max_rounds`=2 轮预算）；规划器不可用时回退为分析师主题模板 query；常规取数循环中跳过 web_search_tool，避免模板与规划搜索双重调用
  2. **结构化校验**：校验 LLM 输出 `FactVerificationReport`（Pydantic Schema），逐项给出 事实比对（fact）或 重新计算（calculation）结果；prompt 内置**逐分析师交叉指引**（哪些字段可跨源比对、哪些必须复算、哪些只能作锚点），防止用不相干数据源臆造比对
  3. **失败反馈闭环**：校验失败时把失败原因写入 `verification_state`，通过 `RetryClear` 节点把前置分析师**送回去重新组织材料**（feedback 注入其 prompt），最多 `max_verify_rounds`（默认 2）轮；超限后报告标记为未通过并继续，不阻塞流水线
- 分析师输出升级为 `AnalystFactualReport`（结论 + 来源清单：claim / value / source_tool / source_data），渲染回 Markdown 保持下游兼容；校验结果随状态日志持久化
- 配置：`verify_enabled`（默认 True）、`max_verify_rounds`（默认 2）
- 覆盖测试：`tests/test_fact_checker.py`、`tests/test_web_search_tool.py`

**各子图职责**：

| 子图 | 内部节点 | 产出通道 |
|------|---------|---------|
| Analyst Team | **4 个分析师并行**（START 扇出）：各自 分析师 ↔ 工具循环 + FactChecker（失败重试） | 4 份报告 |

**分析师并行**（2026-08）：分析师之间无依赖，从 START **并行扇出**，各自独立循环；报告字段（fundamentals_report 等）天然独立无竞争，每个分析师还有**独立的消息通道**（`messages_fundamentals` 等，`ToolNode` 通过 `messages_key` 写入各自通道）避免对话互相污染；由于通道隔离，**正常路径不再需要 Msg Clear 节点**（分析师的对话残留在自己通道中不会泄漏给任何下游），仅校验失败重试路径保留 RetryClear 防止上下文膨胀；`verification_state` 使用按分析师 key 合并的 reducer 支持校验节点并发写；全部完成自动汇聚后进入串行的辩论/交易/风控/组合经理阶段。
| Research Debate | Bull ↔ Bear 辩论循环 + Research Manager | `investment_plan` |
| Trader | Trader（position 工具 + 结构化输出） | `trader_investment_plan` |
| Risk Debate | Aggressive ↔ Conservative ↔ Neutral 辩论循环 | `risk_debate_state` |
| Portfolio Manager | PM（结构化输出） | `final_trade_decision` |

---

## 5. 配置系统

### 默认配置 (`default_config.py`)

```python
DEFAULT_CONFIG = {
    "results_dir": "~/.tradingagents/logs",
    "data_cache_dir": "~/.tradingagents/cache",
    "memory_log_path": "~/.tradingagents/memory/trading_memory.md",
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.4",
    "quick_think_llm": "gpt-5.4-mini",
    "backend_url": None,
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "output_language": "English",
    "data_vendors": {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    },
}
```

### 环境变量

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | OpenAI |
| `GOOGLE_API_KEY` | Google Gemini |
| `ANTHROPIC_API_KEY` | Anthropic Claude |
| `XAI_API_KEY` | xAI Grok |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `DASHSCOPE_API_KEY` | Qwen |
| `ZHIPU_API_KEY` | GLM |
| `OPENROUTER_API_KEY` | OpenRouter |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage |
| `TRADINGAGENTS_MEMORY_LOG_PATH` | 自定义决策日志路径 |
| `TRADINGAGENTS_CACHE_DIR` | 自定义缓存目录 |
| `TRADINGAGENTS_RESULTS_DIR` | 自定义结果目录 |

---

## 6. 使用方式

### 6.1 CLI 方式

```bash
# 安装
pip install .

# 启动交互式 CLI
tradingagents analyze

# 启用检查点恢复
tradingagents analyze --checkpoint

# 清除所有检查点
tradingagents analyze --clear-checkpoints
```

### 6.2 编程方式

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"
config["deep_think_llm"] = "gpt-5.4"
config["quick_think_llm"] = "gpt-5.4-mini"
config["max_debate_rounds"] = 2

ta = TradingAgentsGraph(debug=True, config=config)
final_state, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

---

## 7. 测试

```bash
# 运行所有测试
pytest

# 按标记筛选
pytest -m unit          # 单元测试
pytest -m integration   # 集成测试（需外部服务）
pytest -m smoke         # 快速冒烟测试
```

---

## 8. 扩展指南

### 添加新的 LLM 提供商

1. 在 `llm_clients/` 下创建新的客户端类（继承 `BaseLLMClient`）
2. 在 `factory.py` 的 `create_llm_client()` 中注册
3. 在 `model_catalog.py` 的 `MODEL_OPTIONS` 中添加模型选项
4. 在 `cli/main.py` 的 `select_llm_provider()` 中添加 CLI 支持

### 添加新的分析师

1. 在 `agents/analysts/` 下创建分析师工厂函数（返回 node function）
2. 在 `agents/__init__.py` 中导出
3. 在 `graph/setup.py` 的 `setup_graph()` 中添加节点和边
4. 在 `graph/trading_graph.py` 的 `_create_tool_nodes()` 中添加工具节点
5. 在 `graph/conditional_logic.py` 中添加条件边逻辑
6. 在 `agents/utils/agent_states.py` 的 `AgentState` 中添加报告字段

### 添加新的数据源

1. 在 `dataflows/` 下创建供应商模块
2. 在 `dataflows/interface.py` 的 `VENDOR_METHODS` 中注册
3. 在 `dataflows/config.py` 中配置默认供应商

---

## 9. 依赖栈

| 层 | 主要依赖 |
|----|----------|
| LLM 编排 | `langgraph`, `langchain-core`, `langchain-openai`, `langchain-anthropic`, `langchain-google-genai` |
| 检查点 | `langgraph-checkpoint-sqlite` |
| 数据 | `yfinance`, `stockstats`, `pandas`, `requests` |
| CLI | `typer`, `rich`, `questionary` |
| 回测 | `backtrader` |
| 其他 | `pydantic`, `typing-extensions`, `pytz` |

---

## 10. 设计原则

1. **LangGraph 优先**：所有 Agent 协作通过 LangGraph `StateGraph` 编排，状态流转透明可追溯
2. **供应商抽象**：LLM 和数据源均通过工厂/路由模式解耦，切换供应商只需改配置
3. **降级优雅**：结构化输出失败时自动降级为自由文本，从不阻塞 pipeline
4. **延迟反思**：交易决策先记录，待实际收益可获取后再生成反思，避免过早判断
5. **安全路径处理**：`safe_ticker_component` 防止 ticker 值中包含路径遍历字符
6. **原子写入**：决策日志使用 tmp + os.replace() 保证写操作原子性
