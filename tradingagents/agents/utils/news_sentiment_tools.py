"""新闻舆情工具 — 4个接口。"""

from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.akshare import (
    get_stock_news,
    get_research_report,
    get_notice_report,
    get_irm_answers,
)


@tool
def tool_stock_news(
    stock_code: Annotated[str, "股票代码"],
) -> str:
    """个股新闻：最近20条相关新闻，突发利好/利空第一入口。"""
    return get_stock_news(stock_code)


@tool
def tool_research_report(
    stock_code: Annotated[str, "股票代码"],
) -> str:
    """个股研报-前10条：机构观点、目标价、盈利预测。"""
    return get_research_report(stock_code)


@tool
def tool_notice_report(
    date: Annotated[str, "日期YYYYMMDD"] = "20240510",
) -> str:
    """公告大全：全市场监管公告。"""
    return get_notice_report(date)


@tool
def tool_irm_answers(
    stock_code: Annotated[str, "股票代码"],
) -> str:
    """互动易回答-前10条：公司官方对投资者关切的直接回应。"""
    return get_irm_answers(stock_code)


@tool
def tool_hot_rank() -> str:
    """个股人气榜：散户关注度与情绪热度排行。"""
    return get_hot_rank()


NEWS_SENTIMENT_TOOLS = [
    tool_stock_news,
    tool_research_report,
    tool_notice_report,
    tool_irm_answers,
]
