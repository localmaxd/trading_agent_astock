from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from tradingagents.agents.utils.external_api_tools import tool_market_environment
from tradingagents.agents.utils.agent_utils import (
    REPORT_BUDGET_INSTRUCTION,
    build_instrument_context,
    compact_tool_messages,
    data_tool_done,
    get_language_instruction,
)


def create_macro_analyst(llm, extra_tools=None):
    """Create the market environment (macro) analyst node.

    宏观环境分析师只使用 tool_market_environment（全市场维度，无 ts_code），
    不进行事实校验、无重试，也不输出结构化 claims。

    与其余分析师一致的标准 ReAct 模式：LLM **自己发起一次取数**（图上会走
    Analyze → tools_macro → Analyze 流转）。「每次运行恰好一次 HTTP」由三层
    守卫保证：data_tool_done（取数后不再绑定工具）+ once_per_run（同消息批量
    去重）+ 无事实校验复拉，因此不会重复取数。
    """
    def macro_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        # 宏观分析师在独立并行分支中运行，有自己的消息通道
        messages = state.get("messages_macro", []) or []
        prompt_messages = compact_tool_messages(messages)

        # 数据工具执行过一次后不再绑定：模型直接基于已有返回撰写报告
        tools = (
            []
            if data_tool_done(messages, "tool_market_environment")
            else [tool_market_environment]
        )

        system_message = (
            "你是一位市场宏观环境研究员，负责分析A股市场整体环境，为个股分析提供宏观背景。\n"
            "请**只调用一次** `tool_market_environment` 获取市场宏观环境数据"
            "（无需 ts_code，end_date 取当前日期即可），包含：市场趋势结构（指数均线/ADX/周线结构）、"
            "行业强弱分布、市场活跃度（涨跌停分布、主力资金流、沪深成交额）、隔夜美股行业表现等。\n"
            "一次取数完成后**立即撰写报告**，不要重复取数、不要尝试其他日期。\n"
            "从市场趋势（多空结构）、市场情绪（活跃度/涨跌停）、资金面（主力资金/成交额）、"
            "行业主线（强势行业）四个维度撰写宏观环境报告，最后给出对个股操作的市场环境结论："
            "顺风（可积极）/中性（谨慎）/逆风（回避）。\n"
            "报告末尾用Markdown表格整理关键宏观指标。"
            + REPORT_BUDGET_INSTRUCTION
            + get_language_instruction()
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

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages_macro": [result],
            "macro_environment_report": report,
        }

    return macro_analyst_node
