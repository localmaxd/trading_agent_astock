"""博弈面工具 — 10个akshare接口（已删除：十大流通股东、流通股东变动）。"""

from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.akshare import (
    get_lhb_detail,
    get_lhb_institution_stat,
    get_management_change,
    get_institute_hold_detail,
    get_fund_report_stock,
    get_institute_research_stat,
    get_block_trade_detail,
    get_inner_trade,
)


@tool
def tool_lhb_detail(
    stock_code: Annotated[str, "股票代码"],
    start_date: Annotated[str, "开始日期YYYYMMDD(前7天)"],
    end_date: Annotated[str, "结束日期YYYYMMDD"],
) -> str:
    """龙虎榜详情-最近7天：该股票的游资/机构席位买卖明细。博弈分析核心工具。"""
    return get_lhb_detail(stock_code, start_date, end_date)


@tool
def tool_lhb_institution_stat(
    stock_code: Annotated[str, "股票代码"],
    period: Annotated[str, "周期: 近一月/近三月/近六月/近一年"] = "近一月",
) -> str:
    """机构席位追踪：该股票在指定周期的机构买卖统计。识别建仓/出货节奏。"""
    return get_lhb_institution_stat(stock_code, period)


@tool
def tool_management_change(
    stock_code: Annotated[str, "股票代码"],
) -> str:
    """高管增减持：内部人对公司价值的真实投票。"""
    return get_management_change(stock_code)


@tool
def tool_institute_hold_detail(
    stock_code: Annotated[str, "股票代码"],
) -> str:
    """机构持股详情-过去4个季度：基金/券商/保险/QFII持仓市值与变动。"""
    return get_institute_hold_detail(stock_code)


@tool
def tool_fund_report_stock(
    stock_code: Annotated[str, "股票代码"],
) -> str:
    """基金重仓股排名-近半年变化：该股票的基金覆盖家数排名趋势。"""
    return get_fund_report_stock(stock_code)


@tool
def tool_institute_research_stat(
    stock_code: Annotated[str, "股票代码"],
    date: Annotated[str, "开始日期YYYYMMDD"] = "20240101",
) -> str:
    """机构调研统计：该股票的调研次数和覆盖机构数。"""
    return get_institute_research_stat(stock_code, date)


@tool
def tool_block_trade_detail(
    stock_code: Annotated[str, "股票代码"],
    start_date: Annotated[str, "开始日期YYYYMMDD(一周前)"],
    end_date: Annotated[str, "结束日期YYYYMMDD"],
) -> str:
    """大宗交易明细-近一周：该股票的折溢价率和接盘方。"""
    return get_block_trade_detail(stock_code, start_date, end_date)


@tool
def tool_inner_trade(
    stock_code: Annotated[str, "股票代码"],
) -> str:
    """内部人交易（雪球）：该股票的内部人买卖记录。"""
    return get_inner_trade(stock_code)


GAME_THEORY_TOOLS = [
    tool_lhb_detail,
    tool_lhb_institution_stat,
    tool_management_change,
    tool_institute_hold_detail,
    tool_fund_report_stock,
    tool_institute_research_stat,
    tool_block_trade_detail,
    tool_inner_trade,
]
