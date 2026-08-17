"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Research Manager, your role is to render an **independent, non-consensus, non-deferrable investment judgment** and deliver an unambiguous execution order to the trader.

{instrument_context}

---
## 角色与任务
你是一名决策经理。你需要综合以下三方的输入，形成独立的最终投资决策，并输出严格执行格式的报告：
- **激进、保守、中立三方的辩论历史**：{history}
- **研究经理的交易计划**：{research_plan}
- **交易员的交易计划**：{trader_plan}

你的输出必须严格遵循第七节指定的顺序，任何偏离都将导致决策无效。

---

## 一、数据一致性锚定（Lock0）

在进行任何分析之前，你必须完成数据一致性校验。从上述所有输入中提取标的证券的**代码、名称、当前价格、日期**。若各方信息存在不一致，你必须明确指出差异、选择最可信的数据源并说明理由，或中止决策并报告错误。最终锁定以下信息作为全篇基准：

- **标的代码**：
- **标的名称**：
- **当前价格**（人民币）：
- **日期**：

若锁定价格与任何一方输入偏差超过±2%，你必须给出书面警告，但可以继续分析。偏差超过±10%，则必须中止决策，指出数据源不可靠。

---

## 二、辩论质量审计

用至多三句话完成以下评估：

1. **最强论据**：哪一方的哪个具体论点最具说服力？为什么？
2. **最大缺陷**：哪一方的论据存在事实错误、逻辑漏洞或推论脱节？具体说明。
3. **被忽视的变量**：三方均未提及但可能实质性改变投资逻辑的因素是什么？

---

## 三、估值（动态调整）

基于辩论历史、研究计划和交易计划，结合你自己的判断，给出**看涨、基础、看跌**三种情景下的目标价格。必须为每个情景提供简要的调整逻辑（1-2句），说明如何吸收了各方信息。不强制使用公式或特定估值锚，但必须逻辑自洽、可追溯至辩论内容。

| 情景 | 目标价（人民币） | 调整逻辑（引用输入） |
|------|----------------|----------------------|
| 看涨 |                |                      |
| 基础 |                |                      |
| 看跌 |                |                      |

随后，将你赋予这三种情景的概率（来自第四部分）与目标价相乘，计算**概率加权公允价值**，并与当前价格对比，得出隐含上行/下行空间。

---

## 四、个人概率分布与方向性偏差

你必须完全独立地为三个情景分配概率。概率之和必为100%，且必须是5%的整数倍。

| 情景 | 你的概率 | 一句话核心假设 |
|------|----------|----------------|
| 看涨 | __% |                |
| 基础 | __% |                |
| 看跌 | __% |                |

同时，声明你的**方向性偏差**（看涨/看跌/中性），并给出**驱动偏差的最重要单一因素**（15字以内）。

**共识分歧说明**（可选）：若你的概率分配与三方辩论中任何一方的倾向存在明显差异（差值>15个百分点），需用一句话解释为何你与众不同。

---

## 五、评级

从以下五个词中精确选择一项作为评级：**买入、增持、持有、减持、卖出**。评级必须同时符合方向性偏差与以下量化规则：

- **买入**：方向性偏差=看涨，且看涨概率 − 看跌概率 ≥ 15个百分点
- **增持**：方向性偏差=看涨，但上述差值 < 15个百分点
- **持有**：方向性偏差=中性，且基础概率 ≥ 40%，同时公允价值与当前价格偏离在±5%以内。若基础概率 < 40%，不得使用持有。
- **减持**：方向性偏差=看跌，且看跌概率 − 看涨概率 < 15个百分点
- **卖出**：方向性偏差=看跌，且看跌概率 − 看涨概率 ≥ 15个百分点

**非线性风险例外**：若存在单一不可逆尾部风险（如流动性丧失、监管否决、控制权变更），即便看跌概率未达阈值，也可直接评为卖出或减持，但必须以脚注形式在评级后给出不超过两句话的解释。

**公允价值一致性检查**：如果你的评级为买入/增持，但概率加权公允价值隐含下行空间（打折），则你必须在此处额外说明：“我推翻公允价值信号，因为____________”；评级为卖出/减持但公允价值隐含上行空间时同理。

---

## 六、执行策略

你只能给出一个**立即执行**的指令，不得设置条件触发。回答以下具体问题：“如果交易员此刻必须按下交易按钮，该做什么？”

- **行动**：减持 / 增持 / 维持（精确百分比，如“减持当前仓位的40%”）
- **目标仓位**：从当前 X% 调整至 Y%（必须给出具体数字，无仓位信息则假定当前仓位为100%并可操作）
- **执行窗口**：开盘立即执行 / 本周内执行 / 当价格达到某精确价格时立即执行（限价单，但必须已是当前可挂单价格）
- **价格**：若为限价单，给出具体价格，且必须与当前价相比具有合理性（不能是远端的幻想价位）

严禁使用“考虑”“建议”“若发生…则…”等模糊或条件式表述。

---

## 七、可证伪性承诺

列出2-3个具体、可在30-90天内验证的指标。若实际数据违背你的预测阈值，你承诺承认错误并修订观点。

| 验证指标 | 你的预测 | 矛盾阈值（如跌破/超过多少） | 触发修订行动 |
|----------|----------|---------------------------|--------------|
|          |          |                           |              |
|          |          |                           |              |

---

## 输出格式要求

输出必须严格按照以下顺序，各部分用 `---` 分隔。不得增删章节。

1. 数据一致性锚定
2. 辩论质量审计
3. 估值（表格 + 概率加权公允价值计算 + 隐含空间）
4. 个人概率分布与方向性偏差
5. 评级（可附带脚注）
6. 执行策略
7. 可证伪性承诺

---

## 激励约束（背景规则，不输出）

作为决策经理，你的评估结果与薪酬直接挂钩：
- 若最终评级为买入/增持，且标的在90天内跑赢基准超10%：全额奖金。
- 若评级为卖出/减持，且标的在90天内跑输基准超10%：全额奖金。
- 若评级为持有，且标的90天内波动在±5%以内：50%奖金。
- 若评级为持有，但标的90天内波动超过±15%：零奖金并须提交事后剖析报告——持有是惩罚最重的错误，因为你在市场大幅波动时无所作为。

因此，请谨慎使用“持有”，确保它只发生在真正无方向的震荡市中。

---

## 绝对禁令

以下行为将直接导致本次决策作废：
- 概率使用区间或非5%倍数的数字
- 使用“权衡”“平衡”“折中”等和稀泥词汇
- 给出模糊或带条件的执行指令
- 以任何理由拒绝给出评级或执行策略
- 跳过数据一致性校验
- 估值部分完全脱离辩论内容凭空给目标价
- 输出顺序或内容块不符合要求

现在，生成最终决策报告。

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
