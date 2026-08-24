from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from tradingagents.agents.utils.external_api_tools import tool_game_theory, tool_schema
from tradingagents.agents.utils.agent_utils import (
    REPORT_BUDGET_INSTRUCTION,
    STRUCTURED_CLAIMS_INSTRUCTION,
    build_instrument_context,
    compact_tool_messages,
    data_tool_done,
    get_language_instruction,
    get_verify_feedback,
)
from tradingagents.agents.schemas import AnalystClaimSet
from tradingagents.agents.utils.structured import bind_structured, invoke_factual_claims


def create_game_theory_analyst(llm, extra_tools=None):
    """Create the game-theory analyst node.

    Args:
        llm: The LLM to use.
        extra_tools: Optional additional tools (e.g. web_search_tool) the
            analyst may choose to call.
    """
    structured_llm = bind_structured(llm, AnalystClaimSet, "Game Theory Analyst")

    def game_theory_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        # The analyst runs in a parallel branch with its own message channel
        messages = state.get("messages_game_theory", []) or []
        prompt_messages = compact_tool_messages(messages)

        # 数据只取一轮：tool_game_theory 执行过一次后不再绑定工具，
        # 模型直接撰写报告（避免同一接口被反复调用/换日期重取）。
        tools = (
            []
            if data_tool_done(messages, "tool_game_theory")
            else [tool_schema, tool_game_theory] + (extra_tools or [])
        )

        verify_feedback = get_verify_feedback(state, "game_theory")

        system_message = (
            "你是一位博弈面研究员，从筹码分布、机构行为、内部人交易、风险信号角度分析市场博弈格局。\n"
            "请先调用 `tool_schema('game')` 查询该接口返回字段的含义（字段语义会随版本变化，"
            "务必先查询再解读数据），再调用 `tool_game_theory` 获取该股票的博弈面数据"
            "（含涨跌停统计、资金流向、龙虎榜、大宗交易、内部人交易、高管增减持、"
            "股权质押明细、融资融券明细、机构持仓等）。\n"
            "数据只取一次：`tool_game_theory` 拿到结果后直接撰写报告，不要重复取数或尝试其他 end_date。\n"
            "报告中的每个数据点/推理都必须有来源（工具返回原文）和计算方式（计算类数据给出公式与输入数值），"
            "后续将由代码逐条校验，无来源或无计算方式的内容不得写入报告。\n"
            "ts_code格式如 600519.SH 或 300394.SZ，end_date为当前日期。\n"
            "核心关注：谁在买、谁在卖、筹码在谁手里、成本是多少、风险暴露程度。\n"
            "从筹码集中度、机构行为方向、内部人信号、杠杆资金四个维度综合判断。\n"
            "报告末尾用Markdown表格整理关键博弈信号。"
            + REPORT_BUDGET_INSTRUCTION
            + get_language_instruction()
            + (
                "\n\n### 上一轮事实校验反馈（必须逐条修正后重新组织材料，不要重新取数）：\n" + verify_feedback
                if verify_feedback
                else ""
            )
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个AI助手，与其他助手协作。使用提供的工具推进任务。"
                    "可用工具: {tool_names}.\n{system_message}"
                    "当前日期: {current_date}。{instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        if tools:
            chain = prompt | llm.bind_tools(tools)
        else:
            # 数据已取：不再绑定工具，模型直接产出报告
            chain = prompt | RunnableLambda(lambda msgs: llm.invoke(msgs))
        result = chain.invoke(prompt_messages)

        if len(result.tool_calls) == 0:
            report = result.content
            history = list(prompt_messages) + [result]
            struct_input = [
                SystemMessage(content=STRUCTURED_CLAIMS_INSTRUCTION + get_language_instruction()),
                *history,
            ]
            claims = invoke_factual_claims(
                structured_llm, struct_input, "Game Theory Analyst"
            )
            return {
                "messages_game_theory": [AIMessage(content=report)],
                "game_theory_report": report,
                "game_theory_claims": claims,
            }

        return {
            "messages_game_theory": [result],
            "game_theory_report": "",
            "game_theory_claims": [],
        }

    return game_theory_analyst_node
