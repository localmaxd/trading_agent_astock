from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.external_api_tools import tool_risk
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)


def create_risk_analyst(llm):
    def risk_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])
        tools = [tool_risk]

        system_message = (
            "你是一位风险面研究员，负责识别和量化该股票的尾部风险。\n"
            "请调用 `tool_risk` 获取该股票的风险数据（含内部人交易、高管增减持、"
            "商誉余额、限售解禁等）。\n"
            "ts_code格式如 600519.SH 或 300394.SZ，end_date为当前日期。\n"
            "逐一评估每项风险因子的严重程度（高/中/低），给出综合风险评级。\n"
            "重点关注：内部人减持信号、限售股解禁抛压、商誉减值风险。\n"
            "报告末尾用Markdown表格整理风险矩阵。"
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
            "risk_report": report,
        }

    return risk_analyst_node
