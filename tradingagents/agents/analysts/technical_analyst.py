from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from tradingagents.agents.utils.external_api_tools import tool_technical, tool_schema
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


def create_technical_analyst(llm, extra_tools=None):
    """Create the technical analyst node.

    Args:
        llm: The LLM to use.
        extra_tools: Optional additional tools (e.g. web_search_tool) the
            analyst may choose to call.
    """
    structured_llm = bind_structured(llm, AnalystClaimSet, "Technical Analyst")

    def technical_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        # The analyst runs in a parallel branch with its own message channel
        messages = state.get("messages_technical", []) or []
        prompt_messages = compact_tool_messages(messages)

        # 数据只取一轮：tool_technical 执行过一次后不再绑定工具，
        # 模型直接撰写报告（避免同一接口被反复调用/换日期重取）。
        tools = (
            []
            if data_tool_done(messages, "tool_technical")
            else [tool_schema, tool_technical] + (extra_tools or [])
        )

        verify_feedback = get_verify_feedback(state, "technical")

        system_message = (
            "你是一位技术面研究员，负责从量价和资金角度分析A股市场状态。\n"
            "请先调用 `tool_schema('technical')` 查询该接口返回字段的含义（字段语义会随版本变化，"
            "务必先查询再解读数据），再调用以下工具获取数据：\n"
            "- `tool_technical`: 技术状态、形态与量价事件信号（趋势/RSI/KDJ/CCI/WR/布林/量价）\n"
            "数据只取一次：`tool_technical` 拿到结果后直接撰写报告，不要重复取数或尝试其他 end_date。\n"
            "报告中的每个数据点/推理都必须有来源（工具返回原文）和计算方式（计算类数据给出公式与输入数值），"
            "后续将由代码逐条校验，无来源或无计算方式的内容不得写入报告。\n"
            "ts_code格式如 600519.SH 或 300394.SZ，end_date为当前日期。\n"
            "从趋势、动量、波动、成交量、资金流向五个维度综合判断。原则：<结论必须有严格的数据支撑>\n"
            "报告末尾用Markdown表格整理关键技术指标和信号。"
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
                structured_llm, struct_input, "Technical Analyst"
            )
            return {
                "messages_technical": [AIMessage(content=report)],
                "technical_report": report,
                "technical_claims": claims,
            }

        return {
            "messages_technical": [result],
            "technical_report": "",
            "technical_claims": [],
        }

    return technical_analyst_node
