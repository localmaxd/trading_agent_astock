"""技术面工具 — 4个接口。"""

from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.akshare import (
    get_a_hist,
    get_a_hist_min,
    get_individual_fund_flow,
    get_market_congestion,
)


@tool
def tool_a_hist(
    stock_code: Annotated[str, "股票代码"],
    start_date: Annotated[str, "开始日期YYYYMMDD"],
    end_date: Annotated[str, "结束日期YYYYMMDD"],
    adjust: Annotated[str, "复权: qfq/hfq/空"] = "qfq",
) -> str:
    """历史日频K线：OHLCV数据，技术分析基础。"""
    return get_a_hist(stock_code, start_date, end_date, adjust)


@tool
def tool_a_hist_min(
    stock_code: Annotated[str, "股票代码"],
    start_date: Annotated[str, "开始日期YYYYMMDD"],
    end_date: Annotated[str, "结束日期YYYYMMDD"],
    period: Annotated[str, "周期: 1/5/15/30/60"] = "5",
) -> str:
    """分钟级分时：日内结构，支持多粒度。"""
    return get_a_hist_min(stock_code, start_date, end_date, period)


@tool
def tool_individual_fund_flow(
    stock_code: Annotated[str, "股票代码"],
) -> str:
    """个股资金流向：主力/超大单/大单/中单/小单流向。"""
    return get_individual_fund_flow(stock_code)


@tool
def tool_market_congestion() -> str:
    """大盘拥挤度：最近8条，系统性风险预警。"""
    return get_market_congestion()


TECHNICAL_TOOLS = [
    tool_a_hist,
    tool_a_hist_min,
    tool_individual_fund_flow,
    tool_market_congestion,
]
