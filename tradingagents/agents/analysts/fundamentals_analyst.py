from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from tradingagents.agents.utils.external_api_tools import tool_fundamental, tool_schema
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


def create_fundamentals_analyst(llm, extra_tools=None):
    """Create the fundamentals analyst node.

    Args:
        llm: The LLM to use.
        extra_tools: Optional additional tools (e.g. web_search_tool) the
            analyst may choose to call.
    """
    structured_llm = bind_structured(llm, AnalystClaimSet, "Fundamentals Analyst")

    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        # The analyst runs in a parallel branch with its own message channel
        messages = state.get("messages_fundamentals", []) or []
        prompt_messages = compact_tool_messages(messages)

        # 数据只取一轮：tool_fundamental 执行过一次后不再绑定工具，
        # 模型直接撰写报告（避免同一接口被反复调用/换日期重取）。
        tools = (
            []
            if data_tool_done(messages, "tool_fundamental")
            else [tool_schema, tool_fundamental] + (extra_tools or [])
        )

        verify_feedback = get_verify_feedback(state, "fundamentals")

        system_message = (
            "你是一位基本面研究员，负责分析A股上市公司的基本面信息。\n"
            "请先调用 `tool_schema('fundamental')` 查询该接口返回字段的含义（字段语义会随版本变化，"
            "务必先查询再解读数据），然后调用 `tool_fundamental` 获取该股票的基本面数据"
            "（含经济利润/护城河估值主锚、财务指标、资产负债表、利润表、"
            "现金流量表、业绩预告/快报、机构预期、主营构成、行业PE等），"
            "然后从财务健康、盈利能力、成长性、三个维度撰写综合分析报告。\n"
            "数据只取一次：`tool_fundamental` 拿到结果后直接撰写报告，不要重复取数或尝试其他 end_date。\n"
            "报告中的每个数据点/推理都必须有来源（工具返回原文）和计算方式（计算类数据给出公式与输入数值），"
            "后续将由代码逐条校验，无来源或无计算方式的内容不得写入报告。\n"
            "原则：一：盈利能力需要剔除会计粉饰的，非经常性损益和账面幻觉，聚焦可持续的、能实实在在转化为现金的利润。\n"
            "二：财务健康坚持现金为王，结构性健康比总量健康更重要，盈利有根本依据，可持续盈利能力强，资产质量高，实在，没有水分，压力生存下能扛\n"
            "三：成长性要追问增长来源是否是内生的，追问盈利转化，追问现金转化，追问效率代价，追问有在积累未来能力吗？"
            "ts_code格式如 600519.SH 或 300394.SZ，end_date为当前日期。\n"
            "报告末尾用Markdown表格整理关键指标。"
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
            # The tool-aware response is already the final report. A compact
            # follow-up extracts provenance only instead of regenerating the
            # entire markdown report a second time.
            report = result.content
            history = list(prompt_messages) + [result]
            struct_input = [
                SystemMessage(content=STRUCTURED_CLAIMS_INSTRUCTION + get_language_instruction()),
                *history,
            ]
            claims = invoke_factual_claims(
                structured_llm, struct_input, "Fundamentals Analyst"
            )
            return {
                "messages_fundamentals": [AIMessage(content=report)],
                "fundamentals_report": report,
                "fundamentals_claims": claims,
            }

        return {
            "messages_fundamentals": [result],
            "fundamentals_report": "",
            "fundamentals_claims": [],
        }

    return fundamentals_analyst_node
