from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.external_api_tools import tool_news_sentiment
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)


def create_news_sentiment_analyst(llm, extra_tools=None):
    """Create the news & sentiment analyst node.

    Args:
        llm: The LLM to use.
        extra_tools: Optional additional tools (e.g. web_search_tool) the
            analyst may choose to call.
    """
    def news_sentiment_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])
        tools = [tool_news_sentiment] + (extra_tools or [])

        system_message = (
            "你是一位新闻舆情研究员，负责分析该股票的新闻、公告和市场情绪。\n"
            "请调用 `tool_news_sentiment` 获取该股票的舆情数据（含个股研报前2条、"
            "个股新闻前10条、公告大全、互动易问答等）。\n"
            "ts_code格式如 600519.SH 或 300394.SZ，end_date为当前日期。\n"
            "从信息面边际变化、市场情绪方向、舆论热点三个维度分析。\n"
            "重点关注：正面/负面新闻占比、研报评级调整、公告重大事项。\n"
            "新闻核心看定量事实，而非定性结论，研报核心拆解逻辑，丢弃结论，情绪用来测温度"
            "报告末尾用Markdown表格整理关键舆情指标。"
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

        # The analyst runs in a parallel branch with its own message channel
        messages = state.get("messages_news_sentiment", []) or []
        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(messages)

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages_news_sentiment": [result],
            "news_sentiment_report": report,
        }

    return news_sentiment_analyst_node
