"""基本面工具 — 9个akshare接口封装为LangChain tool。"""

from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.akshare import (
    get_financial_analysis_indicator,
    get_balance_sheet_report,
    get_profit_sheet_report,
    get_cash_flow_sheet_report,
    get_performance_preview,
    get_performance_express,
    get_profit_forecast,
    get_main_business_composition,
    get_industry_pe_ratio,
)


@tool
def tool_financial_analysis_indicator(
    stock_code: Annotated[str, "股票代码，纯数字如'000001'"],
) -> str:
    """财务核心指标速查：近两年ROE、毛利率、净利率、营收增速等。基本面分析首选。"""
    return get_financial_analysis_indicator(stock_code)


@tool
def tool_balance_sheet_report(
    stock_code: Annotated[str, "股票代码，纯数字如'000001'"],
) -> str:
    """资产负债表-近两年：偿债能力、资产质量、有息负债。"""
    return get_balance_sheet_report(stock_code)


@tool
def tool_profit_sheet_report(
    stock_code: Annotated[str, "股票代码，纯数字如'000001'"],
) -> str:
    """利润表-近两年：收入确认、成本结构、费用率。"""
    return get_profit_sheet_report(stock_code)


@tool
def tool_cash_flow_sheet_report(
    stock_code: Annotated[str, "股票代码，纯数字如'000001'"],
) -> str:
    """现金流量表-近两年：验证盈利质量，识别'利润好但现金流差'的信号。"""
    return get_cash_flow_sheet_report(stock_code)


@tool
def tool_performance_preview(
    stock_code: Annotated[str, "股票代码"],
    date: Annotated[str, "报告期YYYYMMDD"] = "20260331",
) -> str:
    """业绩预告：该股票在指定报告期的预告数据。预期差策略核心事件源。"""
    return get_performance_preview(stock_code, date)


@tool
def tool_performance_express(
    stock_code: Annotated[str, "股票代码"],
    date: Annotated[str, "报告期YYYYMMDD"] = "20260331",
) -> str:
    """业绩快报：该股票在指定报告期的快报数据。介于预告与正式财报之间。"""
    return get_performance_express(stock_code, date)


@tool
def tool_profit_forecast(
    stock_code: Annotated[str, "股票代码，纯数字如'000001'"],
) -> str:
    """机构一致预期：该股票的机构盈利预测共识。"""
    return get_profit_forecast(stock_code)


@tool
def tool_main_business_composition(
    stock_code: Annotated[str, "股票代码，纯数字如'000001'"],
) -> str:
    """主营构成-近两年：业务结构、收入占比、行业集中度。"""
    return get_main_business_composition(stock_code)


@tool
def tool_industry_pe_ratio(
    date: Annotated[str, "日期YYYYMMDD"] = "20260331",
) -> str:
    """行业估值锚：个股相对估值的基准线。"""
    return get_industry_pe_ratio(date)


FUNDAMENTALS_TOOLS = [
    tool_financial_analysis_indicator,
    tool_balance_sheet_report,
    tool_profit_sheet_report,
    tool_cash_flow_sheet_report,
    tool_performance_preview,
    tool_performance_express,
    tool_profit_forecast,
    tool_main_business_composition,
    tool_industry_pe_ratio,
]
