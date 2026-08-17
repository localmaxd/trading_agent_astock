from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.external_api_tools import tool_technical
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)


def create_technical_analyst(llm):
    def technical_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])
        tools = [tool_technical]

        system_message = (
            "你是一位技术面研究员，负责从量价和资金角度分析A股市场状态。\n"
            "请调用以下工具获取数据：\n"
            "- `tool_technical`: 日K线（前3个月）、分钟K线,各种技术指标\n"
            "ts_code格式如 600519.SH 或 300394.SZ，end_date为当前日期。\n"
            "从趋势、动量、波动、成交量、资金流向五个维度综合判断。原则：<结论必须有严格的数据支撑>\n"
            "报告末尾用Markdown表格整理关键技术指标和信号。"
            + get_language_instruction(),
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

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "technical_report": report,
        }

    return technical_analyst_node
