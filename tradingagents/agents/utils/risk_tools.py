"""风险面工具 — 2个接口。"""

from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.akshare import (
    get_goodwill_balance,
    get_restricted_release_detail,
)


@tool
def tool_goodwill_balance(
    stock_code: Annotated[str, "股票代码"],
    date: Annotated[str, "报告期YYYYMMDD"] = "20260331",
) -> str:
    """个股商誉余额：减值潜力基数。高商誉+低现金流=高危信号。"""
    return get_goodwill_balance(stock_code, date)


@tool
def tool_restricted_release_detail(
    stock_code: Annotated[str, "股票代码"],
) -> str:
    """限售解禁详情-近三个月：该股票的解禁日期、数量、股东。抛压时点与规模计算。"""
    return get_restricted_release_detail(stock_code)


@tool
def tool_st_risk(
    stock_code: Annotated[str, "股票代码"],
) -> str:
    """ST风险警示：检查该股票是否在ST/*ST名单中。退市边缘硬性排除。"""
    return get_st_risk(stock_code)


RISK_TOOLS = [
    tool_goodwill_balance,
    tool_restricted_release_detail,
]
