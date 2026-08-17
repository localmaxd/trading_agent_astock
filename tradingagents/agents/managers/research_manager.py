"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import build_instrument_context,get_astock_instruction
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]

        prompt = f"""As the Research Manager, you are the **sole owner of this investment decision**. No analyst, no process, and no discovered error can relieve you of the burden of judgment. You must render an **original, non-delegable, non-borrowable verdict**.

{instrument_context}

## 博弈底层逻辑声明（贯穿全锁）

本决策框架遵循A股博弈第一性原理：
1. **政策定方向** > 2. **资金定时机** > 3. **情绪定幅度** > 4. **基本面定下限**
任何论据若违背此优先级，其权重自动降级。你不是在评估公司价值，而是在评估筹码博弈的胜率。

---

## I. 预判锁定

### 锁 0：身份与博弈周期验证

| 验证项目 | 背景值 | 你使用的值 | 偏离度 | 解释 |
|---|---|---|---|---|
| 股票代码/名称 | | | | |
| ... | | | | |
| **博弈周期** | （如：政策蜜月/真空/退潮） | | | |
| **筹码性质** | （游资主导/机构主导/量化拥挤） | | | |

---

### 锁 1：数据可靠性绑定（博弈视角）

| 数据维度 | 可靠性 | 具体博弈缺陷 |
|---|---|---|
| 政策信号 | | （如"依据非官方自媒体传闻"则低） |
| 资金流与筹码 | | （如"北向数据滞后2天""股东户数未更新"则低） |
| 情绪面 | | （如"仅用涨停家数未结合炸板率"则中） |
| 基本面排雷 | | （如"未披露大股东质押率"则低） |

---

### 锁 4：估值推导（博弈锚替代估值锚）

#### 步骤1：锚点收编（只接受博弈锚）
- 禁止DCF、PE、PB目标价。有效锚：**筹码密集区、前高/前低、游资成本线、流动性溢价区间**。若分析师提供基本面估值锚，自动标注“基本面锚，降权50%”。

#### 步骤2：修正因子（四元修正）
| 修正来源 | 因子 | 影响方向 | 幅度 |
|---|---|---|---|
| 政策催化剂半衰期 | | ± | ±__% |
| 资金惯性/筹码压力 | | ± | ±__% |
| 情绪周期位置 | | ± | ±__% |
| 基本面故事素材 | | ± | ±__% |

---

### 锁 5：概率分布
核心假设必须绑定驱动层，如：“看涨情景由政策蜜月期（文件X） + 龙虎榜趋势锁仓资金驱动，低概率下情绪退潮终结此结构”。

---

### 锁 7：评级强度校准（政策否决权）
- 政策层标记为“退潮期”时，即使概率分布看涨，方向性偏好强制为中性或看跌，最高评级为“持有”。
- 大股东处于减持窗口，视为不可逆尾部风险，允许例外降级一档。

---

## 即时行动（嵌入博弈纪律）
- 买入指令默认附带：“若日内成交量<20日均量1.5倍，则本买入指令自动失效”。
- 卖出指令默认附带：“-7%硬止损触发时，以市价单立即执行，无商量”。
### 辩论历史:
{history}
"""

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
