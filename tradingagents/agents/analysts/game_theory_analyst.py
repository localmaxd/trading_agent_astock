from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.external_api_tools import tool_game_theory
from tradingagents.agents.utils.agent_utils import (
    STRUCTURED_REPORT_INSTRUCTION,
    build_instrument_context,
    get_language_instruction,
    get_verify_feedback,
)
from tradingagents.agents.schemas import AnalystFactualReport
from tradingagents.agents.utils.structured import bind_structured, invoke_factual_report


def create_game_theory_analyst(llm, extra_tools=None):
    """Create the game-theory analyst node.

    Args:
        llm: The LLM to use.
        extra_tools: Optional additional tools (e.g. web_search_tool) the
            analyst may choose to call.
    """
    structured_llm = bind_structured(llm, AnalystFactualReport, "Game Theory Analyst")

    def game_theory_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])
        tools = [tool_game_theory] + (extra_tools or [])

        verify_feedback = get_verify_feedback(state, "game_theory")

        system_message = (
            "你是一位博弈面研究员，从筹码分布、机构行为、内部人交易、风险信号角度分析市场博弈格局。\n"
            "请调用 `tool_game_theory` 获取该股票的博弈面数据（含内部人交易、高管增减持、"
            "股权质押明细、融资融券明细、资金流向、龙虎榜、股东增减持，大宗交易，涨跌停价格等）。\n"
            "ts_code格式如 600519.SH 或 300394.SZ，end_date为当前日期。\n"
            "核心关注：谁在买、谁在卖、筹码在谁手里、成本是多少、风险暴露程度。\n"
            "从筹码集中度、机构行为方向、内部人信号、杠杆资金四个维度综合判断。\n"
            "报告末尾用Markdown表格整理关键博弈信号。"
            + get_language_instruction()
            + (
                "\n\n### 上一轮事实校验反馈（必须逐条修正后重新组织材料）：\n" + verify_feedback
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

        # The analyst runs in a parallel branch with its own message channel
        messages = state.get("messages_game_theory", []) or []
        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(messages)

        if len(result.tool_calls) == 0:
            history = list(messages) + [result]
            struct_input = [
                SystemMessage(content=STRUCTURED_REPORT_INSTRUCTION + get_language_instruction()),
                *history,
            ]
            report, claims = invoke_factual_report(
                structured_llm, llm, struct_input, "Game Theory Analyst"
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
