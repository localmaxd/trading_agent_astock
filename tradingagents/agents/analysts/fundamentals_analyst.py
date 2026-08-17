from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.external_api_tools import tool_fundamental
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])
        tools = [tool_fundamental]

        system_message = (
            "你是一位基本面研究员，负责分析A股上市公司的基本面信息。\n"
            "请调用 `tool_fundamental` 获取该股票的基本面数据（含财务指标、资产负债表、利润表、"
            "现金流量表、业绩预告/快报、机构预期、主营构成、行业PE等），"
            "然后从财务健康、盈利能力、成长性、三个维度撰写综合分析报告。\n"
            "原则：一：盈利能力需要剔除会计粉饰的，非经常性损益和账面幻觉，聚焦可持续的、能实实在在转化为现金的利润。\n"
            "二：财务健康坚持现金为王，结构性健康比总量健康更重要，盈利有根本依据，可持续盈利能力强，资产质量高，实在，没有水分，压力生存下能扛\n"
            "三：成长性要追问增长来源是否是内生的，追问盈利转化，追问现金转化，追问效率代价，追问有在积累未来能力吗？"
            "ts_code格式如 600519.SH 或 300394.SZ，end_date为当前日期。\n"
            "报告末尾用Markdown表格整理关键指标。"
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
                # 消息占位符，省去了自己build 上下文， 去invoke 里面取消息列表 message，用户的message输入就需要时messages
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        #  将基本不变的通过partial 先填起来
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
